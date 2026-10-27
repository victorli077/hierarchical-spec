"""Native worker ownership for three-stage hierarchical speculation.

``DFlashWorkerV2`` is reused only as the A -> B execution kernel.  B is a
target-mode ``TpModelWorker`` owned by this module and has a KV pool that is
independent from C.  In particular, B is never installed in ``_draft_worker``:
that field continues to mean A for scheduler draft lifecycle operations.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

import torch
from sglang.kernels.ops.speculative.cache_locs import (
    assign_extend_cache_locs_func,
)
from sglang.srt.layers.logprob_processor import compute_spec_logprobs
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    compute_position,
)
from sglang.srt.model_executor.graph_memory_usage import (
    merge_graph_memory_usage,
    merge_graph_time_usage,
)
from sglang.srt.runtime_context import get_parallel, get_spec
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker, HiCacheDraftPlan
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
)
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2
from sglang.srt.speculative.hierarchical_config import cascade_verify_width
from sglang.srt.speculative.hierarchical_types import (
    append_pending_round_,
    greedy_anchor_block_accept,
)
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def build_intermediate_tp_worker(
    *,
    server_args: ServerArgs,
    gpu_id: int,
    nccl_port: int,
    target_model_config: ModelConfig,
) -> TpModelWorker:
    """Construct B through the native SGLang model-worker loader.

    The returned worker has its own model config and KV allocator.  It is not
    attached to the target worker's cache, which is an invariant required by
    the cascade coordinator.
    """
    cfg = get_spec()
    return TpModelWorker(
        server_args=server_args,
        gpu_id=gpu_id,
        nccl_port=nccl_port,
        is_draft_worker=False,
        is_auxiliary_target=True,
        context_length=target_model_config.context_len,
        draft_attention_backend=getattr(
            cfg, "speculative_draft_attention_backend", None
        ),
        model_path_override=cfg.speculative_intermediate_model_path,
        quantization_override=cfg.speculative_intermediate_quantization,
    )


class IntermediateStage:
    """Own B as an auxiliary target, including its private cache lifecycle."""

    def __init__(self, worker: TpModelWorker):
        if worker.is_draft_worker:
            raise ValueError("The hierarchical intermediate must be a target worker")
        if not worker.is_auxiliary_target:
            raise ValueError("The hierarchical intermediate must be auxiliary-target")
        self.worker = worker
        self._request_rows: dict[str, int] = {}
        self._cache_lengths: dict[str, int] = {}
        self._committed_lengths: dict[str, int] = {}
        self.tree_cache = None

    @property
    def model_runner(self):
        return self.worker.model_runner

    def alloc_memory_pool(self) -> None:
        # Passing no C-owned objects is intentional: ModelRunner creates B's
        # ReqToTokenPool, token allocator and physical KV tensors here.
        self.worker.alloc_memory_pool()

    def clear_cache_pool(self) -> None:
        req_pool, allocator = self.worker.get_memory_pool()
        if self.tree_cache is not None:
            self.tree_cache.reset()
        req_pool.clear()
        allocator.clear()
        req_pool.reset_aux_cache_allocator()
        self._request_rows.clear()
        self._cache_lengths.clear()
        self._committed_lengths.clear()

    def rows_for_requests(
        self, reqs, *, device: torch.device | str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return B-local GPU/CPU row tensors for a scheduler request batch."""
        req_pool, _ = self.worker.get_memory_pool()
        missing = [req.rid for req in reqs if req.rid not in self._request_rows]
        if missing:
            rows = req_pool.alloc_rows(len(missing))
            if rows is None:
                raise RuntimeError(
                    "HierarchicalSpec intermediate request pool is exhausted: "
                    f"needed {len(missing)} rows."
                )
            self._request_rows.update(zip(missing, rows))
            self._cache_lengths.update((rid, 0) for rid in missing)
            self._committed_lengths.update((rid, 0) for rid in missing)
        rows_cpu = torch.tensor(
            [self._request_rows[req.rid] for req in reqs], dtype=torch.int64
        )
        return rows_cpu.to(device=device, non_blocking=True), rows_cpu

    def update_cache_lengths(self, reqs, cache_lengths) -> None:
        """Record B allocated lengths from already-available CPU metadata."""
        if isinstance(cache_lengths, torch.Tensor):
            if cache_lengths.device.type != "cpu":
                raise ValueError("B cache length bookkeeping must use CPU metadata")
            cache_lengths = cache_lengths.tolist()
        if len(reqs) != len(cache_lengths):
            raise ValueError("request/cache-length batch dimensions do not match")
        for req, length in zip(reqs, cache_lengths):
            if req.rid not in self._request_rows:
                raise KeyError(f"B request row is not allocated for {req.rid!r}")
            self._cache_lengths[req.rid] = int(length)

    def update_committed_lengths(self, reqs, committed_lengths) -> None:
        if isinstance(committed_lengths, torch.Tensor):
            if committed_lengths.device.type != "cpu":
                raise ValueError("B committed lengths must use CPU metadata")
            committed_lengths = committed_lengths.tolist()
        for req, length in zip(reqs, committed_lengths):
            self._committed_lengths[req.rid] = int(length)

    def committed_lengths(self, reqs) -> list[int]:
        return [self._committed_lengths[req.rid] for req in reqs]

    def trim_to_lengths(self, reqs, committed_lengths: torch.Tensor) -> None:
        """Drop B's previous speculative tail before reserving a new window."""
        if committed_lengths.device.type != "cpu":
            raise ValueError("B trim lengths must be scheduler CPU metadata")
        req_pool, allocator = self.worker.get_memory_pool()
        stale_slots = []
        for req, committed in zip(reqs, committed_lengths.tolist()):
            row = self._request_rows[req.rid]
            allocated = self._cache_lengths.get(req.rid, 0)
            committed = int(committed)
            if allocated > committed:
                stale_slots.append(req_pool.req_to_token[row, committed:allocated])
            self._cache_lengths[req.rid] = committed
        if stale_slots:
            # free_segments() is for multiple logical ranges of one request;
            # positions from different requests overlap by construction. V1
            # requires page_size=1, so their physical slot ids can be returned
            # together without page-boundary bookkeeping.
            allocator.free(torch.cat(stale_slots))
            for slots in stale_slots:
                slots.zero_()

    def release_request(self, rid: str) -> None:
        row = self._request_rows.pop(rid, None)
        if row is not None:
            req_pool, allocator = self.worker.get_memory_pool()
            cache_len = self._cache_lengths.pop(rid, 0)
            self._committed_lengths.pop(rid, None)
            if cache_len:
                slots = req_pool.req_to_token[row, :cache_len]
                allocator.free_segment(slots, start_pos=0)
                req_pool.req_to_token[row, :cache_len].zero_()
            req_pool.free_rows([row])

    def reset_request(self, rid: str) -> None:
        row = self._request_rows[rid]
        req_pool, allocator = self.worker.get_memory_pool()
        allocated = self._cache_lengths.get(rid, 0)
        if allocated:
            slots = req_pool.req_to_token[row, :allocated]
            allocator.free_segment(slots, start_pos=0)
            req_pool.req_to_token[row, :allocated].zero_()
        self._cache_lengths[rid] = 0
        self._committed_lengths[rid] = 0

    def make_batch_view(self, batch):
        """Create a metadata-only B view without aliasing C cache fields."""
        if self.tree_cache is None:
            raise RuntimeError("B radix cache is not initialized")
        view = copy.copy(batch)
        req_pool, _ = self.worker.get_memory_pool()
        rows, rows_cpu = self.rows_for_requests(batch.reqs, device=batch.device)
        view.model_config = self.worker.model_config
        view.req_to_token_pool = req_pool
        view.tree_cache = self.tree_cache
        view.req_pool_indices = rows
        view.req_pool_indices_cpu = rows_cpu
        # These are per-forward mutable fields. Never let an inner DFlash
        # round overwrite the scheduler's C batch metadata.
        view.out_cache_loc = None
        view.spec_info = copy.copy(batch.spec_info)
        return view


class CascadeCoordinator:
    """Own the independent B stage and its injected A -> B execution kernel."""

    def __init__(
        self,
        *,
        intermediate_stage: IntermediateStage,
        inner_kernel: DFlashWorkerV2,
    ):
        if inner_kernel.target_worker is not intermediate_stage.worker:
            raise ValueError("A -> B kernel must target the coordinator-owned B worker")
        self.intermediate_stage = intermediate_stage
        self.inner_kernel = inner_kernel

    @property
    def intermediate_worker(self) -> TpModelWorker:
        return self.intermediate_stage.worker

    @property
    def draft_worker(self):
        return self.inner_kernel.draft_worker

    def release_request(self, rid: str) -> None:
        self.intermediate_stage.release_request(rid)

    def clear_cache_pool(self) -> None:
        self.intermediate_stage.clear_cache_pool()


class HierarchicalSpecWorker(BaseSpecWorker):
    """Own C, an independent B stage, and an A -> B DFlash kernel.

    The outer cascade forward is deliberately separate from DFlash's public
    generation method: that method returns a scheduler-committable result,
    while inner A -> B rounds must remain provisional until C verifies the
    accumulated block.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__()
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.nccl_port = nccl_port
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.device = target_worker.device
        self.cascade_rounds = int(get_spec().speculative_cascade_rounds)
        self.inner_block_size = int(get_spec().speculative_num_draft_tokens)
        self.outer_block_size = cascade_verify_width(
            self.inner_block_size, self.cascade_rounds
        )
        logger.info(
            "HierarchicalSpec resolved cascade configuration: rounds=%d, "
            "dflash_block_size=%d, target_verify_width=%d",
            self.cascade_rounds,
            self.inner_block_size,
            self.outer_block_size,
        )
        self._outer_position_offsets = torch.arange(
            self.outer_block_size, dtype=torch.int64, device=self.device
        )

        intermediate_worker = build_intermediate_tp_worker(
            server_args=server_args,
            gpu_id=gpu_id,
            nccl_port=nccl_port,
            target_model_config=target_worker.model_runner.model_config,
        )
        self._intermediate_memory_pool_config = None

        # Dependency injection is the ownership boundary: DFlash constructs
        # and owns only A, and treats the independently-owned B as its target.
        inner_dflash = DFlashWorkerV2(
            server_args=server_args,
            gpu_id=gpu_id,
            nccl_port=nccl_port,
            target_worker=intermediate_worker,
        )
        self.coordinator = CascadeCoordinator(
            intermediate_stage=IntermediateStage(intermediate_worker),
            inner_kernel=inner_dflash,
        )
        self._draft_worker = self.coordinator.draft_worker
        self.draft_parallel_layout = self.inner_dflash.draft_parallel_layout
        self._validate_model_contract()

    @property
    def intermediate_stage(self) -> IntermediateStage:
        return self.coordinator.intermediate_stage

    @property
    def inner_dflash(self) -> DFlashWorkerV2:
        return self.coordinator.inner_kernel

    @property
    def intermediate_worker(self) -> TpModelWorker:
        return self.intermediate_stage.worker

    def _validate_model_contract(self) -> None:
        a_vocab = int(self.inner_dflash.draft_model_runner.model_config.vocab_size)
        b_vocab = int(self.intermediate_worker.model_config.vocab_size)
        c_vocab = int(self.target_worker.model_config.vocab_size)
        if len({a_vocab, b_vocab, c_vocab}) != 1:
            raise ValueError(
                "HierarchicalSpec v1 requires one vocabulary for A/B/C; got "
                f"A={a_vocab}, B={b_vocab}, C={c_vocab}."
            )
        b_hidden = int(self.intermediate_worker.model_config.spec_hidden_size)
        c_hidden = int(self.target_worker.model_config.spec_hidden_size)
        if b_hidden != c_hidden:
            raise ValueError(
                "HierarchicalSpec requires B and C to expose the same DFlash "
                f"hidden width; got B={b_hidden}, C={c_hidden}."
            )

    def weight_update_runners(self):
        # Both are selected through the scheduler's draft selector, but B is
        # labelled intermediate so it is never confused with A in diagnostics.
        return [
            ("intermediate", self.intermediate_worker.model_runner),
            ("draft", self.inner_dflash.draft_model_runner),
        ]

    @property
    def graph_memory_usage(self):
        return merge_graph_memory_usage(
            self.intermediate_worker.graph_memory_usage,
            self.inner_dflash.graph_memory_usage,
        )

    @property
    def graph_time_usage(self):
        return merge_graph_time_usage(
            self.intermediate_worker.graph_time_usage,
            self.inner_dflash.graph_time_usage,
        )

    @property
    def weight_load_time(self) -> float:
        return (
            self.intermediate_worker.weight_load_time
            + self.inner_dflash.weight_load_time
        )

    @property
    def preloaded_weights_bytes(self) -> int:
        return (
            self.intermediate_worker.preloaded_weights_bytes
            + self.inner_dflash.preloaded_weights_bytes
        )

    @property
    def spec_v2_attn_backends(self) -> tuple:
        return (
            self.target_worker.model_runner.attn_backend,
            self.intermediate_worker.model_runner.attn_backend,
            self.inner_dflash.draft_model_runner.attn_backend,
        )

    def _draft_model_runners(self):
        # HiCache draft lifecycle applies only to A. B has target semantics and
        # is managed by IntermediateStage even though it is provisional to C.
        return (self.inner_dflash.draft_model_runner,)

    @property
    def primary_draft_kv_pool(self):
        # A's physical indices are allocated in B's slot space, not C's. The
        # scheduler's target-side draft pool consumers must not see this pool.
        return None

    def init_hicache_draft_plan(self) -> None:
        # A/B HiCache integration needs a cascade-aware backup namespace. A
        # normal target sidecar would interpret B slot ids as C slot ids.
        self._hicache_draft_plan = HiCacheDraftPlan()

    def init_intermediate_cache(self, *, page_size: int) -> None:
        """Build B's radix cache over B's independently allocated KV pool."""
        from sglang.srt.mem_cache import kv_cache_builder

        result = kv_cache_builder.build_kv_cache(
            server_args=self.server_args,
            model_config=self.intermediate_worker.model_config,
            tp_worker=self.intermediate_worker,
            page_size=page_size,
            spec_algorithm=self.model_runner.spec_algorithm,
            enable_metrics=False,
            enable_kv_cache_events=False,
            enable_hierarchical_cache=False,
            hicache_draft_plan=None,
            draft_swa_window=0,
        )
        b_pool, b_allocator = self.intermediate_worker.get_memory_pool()
        if result.req_to_token_pool is not b_pool:
            raise RuntimeError("B radix cache was built with a foreign request pool")
        if result.token_to_kv_pool_allocator is not b_allocator:
            raise RuntimeError("B radix cache was built with a foreign KV allocator")
        self.intermediate_stage.tree_cache = result.tree_cache

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

        if self._intermediate_memory_pool_config is None:
            raise RuntimeError(
                "HierarchicalSpec memory pools were not jointly planned before allocation"
            )
        self.intermediate_worker.alloc_memory_pool(
            memory_pool_config=self._intermediate_memory_pool_config,
            draft_parallel_layout=self.draft_parallel_layout,
        )
        b_pool, b_allocator = self.intermediate_worker.get_memory_pool()
        if b_pool is req_to_token_pool or b_allocator is token_to_kv_pool_allocator:
            raise RuntimeError("B and C must not share KV pool or allocator state")

        # A uses the normal DFlash cache materialization contract relative to
        # B. This does not make B a draft worker and does not involve C's pool.
        self.inner_dflash.alloc_memory_pool(
            memory_pool_config=self.intermediate_worker.model_runner.memory_pool_config,
            req_to_token_pool=b_pool,
            token_to_kv_pool_allocator=b_allocator,
        )

    def plan_memory_pool_configs(self, *, preloaded_weights_bytes: int):
        """Split one profiled KV budget between C and the independent B+A stack.

        C does not own A's KV in this algorithm. B's configurator does price A,
        because the inner DFlash draft uses B's virtual slot namespace. Both
        sides are capped to one logical token capacity so scheduler admission
        can never outrun the intermediate cache.
        """
        from sglang.srt.model_executor.pool_configurator import (
            create_memory_pool_configurator,
        )
        from sglang.srt.runtime_context import get_schedule

        c_runner = self.target_worker.model_runner
        b_runner = self.intermediate_worker.model_runner
        c_runner.account_preloaded_weights(preloaded_weights_bytes)
        c_runner.init_kv_cache_configurator()
        b_runner.init_kv_cache_configurator(self.draft_parallel_layout)
        c_kvc = c_runner.kv_cache_configurator
        b_kvc = b_runner.kv_cache_configurator

        total_budget = c_kvc._profile_available_bytes(c_runner.pre_model_load_memory)
        c_full = c_kvc.config_from_budget(total_budget)
        b_full = b_kvc.config_from_budget(total_budget)
        capacity_sum = c_full.max_total_num_tokens + b_full.max_total_num_tokens
        if capacity_sum <= 0:
            raise RuntimeError("HierarchicalSpec cannot size C and B KV pools")

        # Full-budget capacities are inverse per-token costs. This ratio gives
        # equal logical capacity to both stacks for ordinary paged KV models.
        c_budget = total_budget * b_full.max_total_num_tokens // capacity_sum
        b_budget = total_budget - c_budget
        c_config = c_kvc.config_from_budget(c_budget)
        b_config = b_kvc.config_from_budget(b_budget)
        shared_tokens = min(
            c_config.max_total_num_tokens, b_config.max_total_num_tokens
        )
        c_config = c_kvc.config_from_budget(c_budget, cap_tokens=shared_tokens)
        b_config = b_kvc.config_from_budget(b_budget, cap_tokens=shared_tokens)

        max_requests = min(
            c_kvc.resolve_max_num_reqs(shared_tokens),
            b_kvc.resolve_max_num_reqs(shared_tokens),
        )
        for runner, kvc, config in (
            (c_runner, c_kvc, c_config),
            (b_runner, b_kvc, b_config),
        ):
            config.max_running_requests = max_requests
            config = create_memory_pool_configurator(
                kvc
            ).finalize_with_max_running_requests(config)
            config.mem_fraction_static = get_schedule().mem_fraction_static
            if runner is c_runner:
                c_config = config
            else:
                b_config = config

        shared_tokens = min(
            c_config.max_total_num_tokens, b_config.max_total_num_tokens
        )
        if shared_tokens <= self.outer_block_size:
            raise RuntimeError(
                "HierarchicalSpec KV budget is too small for one cascade window: "
                f"capacity={shared_tokens}, window={self.outer_block_size}"
            )
        self._intermediate_memory_pool_config = b_config
        logger.info(
            "HierarchicalSpec KV plan: total=%d bytes, C=%d bytes, B+A=%d "
            "bytes, shared_capacity=%d, max_requests=%d",
            total_budget,
            c_budget,
            b_budget,
            shared_tokens,
            max_requests,
        )
        return c_config

    def init_attention_backends(self):
        self.intermediate_worker.init_attention_backends()
        self.inner_dflash.init_attention_backends()

    def init_cuda_graphs(self):
        self.intermediate_worker.init_cuda_graphs()
        self.inner_dflash.init_cuda_graphs()

    def clear_cache_pool(self):
        self.coordinator.clear_cache_pool()

    def note_request_finished(self, *, rid: str, natural_stop: bool) -> None:
        del natural_stop
        self.coordinator.release_request(rid)

    def note_request_retracted(self, *, rid: str) -> None:
        # C has already released its request KV and reset kv_committed_len.
        # Drop B's independent row as well; re-prefill allocates a fresh row
        # and reconstructs B/A state from the scheduler's authoritative tokens.
        self.coordinator.release_request(rid)

    def note_request_aborted(self, *, rid: str) -> None:
        self.coordinator.release_request(rid)

    def _verify_with_c(
        self,
        batch,
        *,
        candidate_chain: torch.Tensor,
        pending_lens: torch.Tensor,
        b_cache_locs: torch.Tensor,
    ):
        """Run C once over ``[anchor, B pending...]`` and accept greedily."""
        from sglang.srt.managers.scheduler import GenerationBatchResult

        bs, width = candidate_chain.shape
        if width != self.outer_block_size:
            raise ValueError(
                f"expected outer block width {self.outer_block_size}, got {width}"
            )
        prefix_lens = batch.seq_lens
        positions_2d = prefix_lens[:, None] + self._outer_position_offsets[None, :]
        end_offsets = prefix_lens + width
        c_cache_locs = assign_extend_cache_locs_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=self.model_runner.req_to_token_pool.req_to_token,
            start_offset=prefix_lens,
            end_offset=end_offsets,
            batch_size=bs,
            draft_token_num=width,
            device=self.device,
        )
        kv_loc_plan = self.model_runner.kv_index_translator.plan(
            req_pool_indices=batch.req_pool_indices,
            seq_lens=prefix_lens,
            seq_lens_cpu=batch.seq_lens_cpu,
            write_virtual=c_cache_locs,
            read_extent=width,
        )
        verify_input = DFlashVerifyInput(
            draft_token=candidate_chain.reshape(-1),
            positions=positions_2d.reshape(-1),
            draft_token_num=width,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            kv_loc_plan=kv_loc_plan,
        )
        batch.out_cache_loc = c_cache_locs
        seq_lens_cpu_backup = batch.seq_lens_cpu
        seq_lens_sum_backup = batch.seq_lens_sum
        if seq_lens_cpu_backup is not None:
            verify_cpu_lens = seq_lens_cpu_backup + width
            batch.seq_lens_cpu = verify_cpu_lens
            batch.seq_lens_sum = int(verify_cpu_lens.sum())
        verify_forward_batch, _ = verify_input.prepare_for_verify(
            batch,
            self.target_worker,
            defer_metadata_init=get_parallel().attn_dp_enabled,
        )
        batch.seq_lens_cpu = seq_lens_cpu_backup
        batch.seq_lens_sum = seq_lens_sum_backup
        target_out = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=(None if get_parallel().attn_dp_enabled else True),
        )
        logits_output = target_out.logits_output
        logits = logits_output.next_token_logits.view(bs, width, -1)
        if batch.sampling_info is not None:
            apply_dflash_verify_logits_adjustments(
                next_token_logits=logits_output.next_token_logits,
                sampling_info=batch.sampling_info,
                draft_token_num=width,
            )
        target_top1 = torch.argmax(logits, dim=-1)

        _, commit_lens, bonus, out_tokens = greedy_anchor_block_accept(
            candidate_chain, target_top1, pending_lens
        )
        new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)

        if batch.return_logprob:
            compute_spec_logprobs(
                batch,
                logits_output,
                out_tokens.reshape(-1),
                chain_stride=width,
            )

        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError("HierarchicalSpec C verify returned no hidden states")
        hidden = hidden.view(bs, width, -1)
        # A uses B's slot namespace. Only C-accepted input rows become valid;
        # rejected B hidden states are overwritten or remain beyond seq_lens.
        self.inner_dflash._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=hidden.reshape(-1, hidden.shape[-1]),
            cache_loc=b_cache_locs.reshape(-1),
            cache_loc_2d=b_cache_locs,
            positions=positions_2d.reshape(-1),
            commit_lens=commit_lens,
        )
        logits_output.hidden_states = None
        next_draft_input = make_draft_input_v2(
            bonus_tokens=bonus,
            new_seq_lens=new_seq_lens,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            proposed_lens=pending_lens,
            can_run_cuda_graph=target_out.can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=width,
            speculative_output_stride=width,
            new_seq_lens=new_seq_lens,
            routed_experts_output=target_out.routed_experts_output,
            indexer_topk_output=target_out.indexer_topk_output,
        )

    def _prepare_b_decode_window(self, batch):
        b_batch = self.intermediate_stage.make_batch_view(batch)
        if batch.seq_lens_cpu is None:
            raise RuntimeError("HierarchicalSpec requires CPU seq_lens metadata")
        self.intermediate_stage.trim_to_lengths(batch.reqs, batch.seq_lens_cpu)
        self.intermediate_stage.update_committed_lengths(batch.reqs, batch.seq_lens_cpu)
        _, allocator = self.intermediate_worker.get_memory_pool()
        if allocator.page_size != 1:
            raise NotImplementedError(
                "HierarchicalSpec v1 currently requires page_size=1 for the "
                "independent B rollback allocator"
            )
        bs = batch.batch_size()
        slots = allocator.alloc(bs * self.outer_block_size)
        if slots is None:
            raise RuntimeError(
                "HierarchicalSpec B KV cache is out of memory while reserving "
                f"{bs * self.outer_block_size} tokens"
            )
        slots_2d = slots.view(bs, self.outer_block_size)
        positions = batch.seq_lens[:, None] + self._outer_position_offsets[None, :]
        b_pool, _ = self.intermediate_worker.get_memory_pool()
        b_pool.req_to_token[b_batch.req_pool_indices[:, None], positions] = slots_2d.to(
            torch.int32
        )
        allocated_cpu = batch.seq_lens_cpu + self.outer_block_size
        self.intermediate_stage.update_cache_lengths(batch.reqs, allocated_cpu)

        draft_input = batch.spec_info
        if draft_input is None or not hasattr(draft_input, "bonus_tokens"):
            raise RuntimeError("HierarchicalSpec decode is missing the C anchor token")
        b_batch.spec_info = make_draft_input_v2(
            bonus_tokens=draft_input.bonus_tokens,
            new_seq_lens=batch.seq_lens,
        )
        b_batch.spec_info.nxt_kv_lens_cpu = allocated_cpu.to(torch.int32)
        b_batch.spec_info.nxt_kv_lens_sum = int(allocated_cpu.sum())
        # The fixed allocation bound above replaces exact per-round host
        # lengths. Device seq_lens remain exact and advance without D2H sync.
        b_batch.seq_lens_cpu = None
        b_batch.seq_lens_sum = int(allocated_cpu.sum())
        return b_batch, slots_2d

    def _prepare_b_prefill_batch(self, batch):
        b_batch = self.intermediate_stage.make_batch_view(batch)
        _, allocator = self.intermediate_worker.get_memory_pool()
        if allocator.page_size != 1:
            raise NotImplementedError(
                "HierarchicalSpec v1 currently requires page_size=1 for B"
            )
        prefix_lens = self.intermediate_stage.committed_lengths(batch.reqs)
        seq_lens_cpu = batch.seq_lens_cpu
        if seq_lens_cpu is None:
            raise RuntimeError("HierarchicalSpec prefill requires CPU seq_lens")
        end_lens = seq_lens_cpu.tolist()
        for req, prefix, end in zip(batch.reqs, prefix_lens, end_lens):
            if prefix > int(end):
                self.intermediate_stage.reset_request(req.rid)
        prefix_lens = self.intermediate_stage.committed_lengths(batch.reqs)
        prefix_tensor = torch.tensor(prefix_lens, dtype=torch.int64)
        self.intermediate_stage.trim_to_lengths(batch.reqs, prefix_tensor)
        extend_lens = [
            int(end) - int(prefix) for end, prefix in zip(end_lens, prefix_lens)
        ]
        if any(length < 0 for length in extend_lens):
            raise RuntimeError("B committed prefix is ahead of C during prefill")
        input_ids = []
        for req, prefix, end in zip(batch.reqs, prefix_lens, end_lens):
            input_ids.extend(req.get_fill_ids()[prefix : int(end)])
        total = sum(extend_lens)
        slots = allocator.alloc(total)
        if slots is None:
            raise RuntimeError(
                f"HierarchicalSpec B KV cache cannot allocate {total} prefill tokens"
            )
        prefix_gpu = torch.tensor(prefix_lens, dtype=torch.int64, device=self.device)
        end_gpu = seq_lens_cpu.to(self.device, non_blocking=True)
        b_pool, _ = self.intermediate_worker.get_memory_pool()
        assign_req_to_token_pool_func(
            b_batch.req_pool_indices,
            b_pool.req_to_token,
            prefix_gpu,
            end_gpu,
            slots,
            batch.batch_size(),
        )
        b_batch.prefix_lens = prefix_lens
        b_batch.extend_lens = extend_lens
        b_batch.extend_num_tokens = total
        b_batch.seq_lens = end_gpu
        b_batch.seq_lens_cpu = seq_lens_cpu
        b_batch.seq_lens_sum = int(seq_lens_cpu.sum())
        b_batch.input_ids = torch.tensor(
            input_ids, dtype=torch.int64, device=self.device
        )
        b_batch.prefill_input_ids_cpu = None
        b_batch.out_cache_loc = slots
        b_batch.return_logprob = False
        b_batch.extend_logprob_start_lens = [0] * batch.batch_size()
        self.intermediate_stage.update_cache_lengths(batch.reqs, seq_lens_cpu)
        self.intermediate_stage.update_committed_lengths(batch.reqs, seq_lens_cpu)
        return b_batch, slots

    def _forward_prefill(self, batch, *, pp_proxy_tensors=None):
        # C remains authoritative for the sampled token and scheduler-visible
        # result. Capture its hidden states before B runs.
        c_result = self.target_worker.forward_batch_generation(
            batch,
            pp_proxy_tensors=pp_proxy_tensors,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            return_kv_loc_plan=True,
        )
        b_batch, _ = self._prepare_b_prefill_batch(batch)
        self.inner_dflash.forward_provisional(
            b_batch, pp_proxy_tensors=pp_proxy_tensors
        )

        c_hidden = c_result.logits_output.hidden_states
        if c_hidden is None:
            raise RuntimeError("HierarchicalSpec C prefill returned no hidden states")
        ctx_lens = torch.tensor(
            batch.extend_lens, dtype=torch.int32, device=self.device
        )
        c_prefix = torch.tensor(
            batch.prefix_lens, dtype=torch.int32, device=self.device
        )
        positions, _ = compute_position(
            self.model_runner.prefill_attention_backend_str,
            c_prefix,
            ctx_lens,
            int(sum(batch.extend_lens)),
        )
        b_pool, _ = self.intermediate_worker.get_memory_pool()
        b_c_slots = torch.cat(
            [
                b_pool.req_to_token[int(row), int(start) : int(start) + int(length)]
                for row, start, length in zip(
                    b_batch.req_pool_indices_cpu.tolist(),
                    batch.prefix_lens,
                    batch.extend_lens,
                )
            ]
        ).to(torch.int64)
        self.inner_dflash._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=c_hidden,
            cache_loc=b_c_slots,
            positions=positions,
            extend_lens=ctx_lens,
        )
        c_result.logits_output.hidden_states = None
        c_result.new_seq_lens = batch.seq_lens
        c_result.next_draft_input = make_draft_input_v2(
            bonus_tokens=c_result.next_token_ids,
            new_seq_lens=batch.seq_lens,
        )
        return c_result

    def _forward_decode(self, batch, *, grammar_barrier=None, pp_proxy_tensors=None):
        if batch.sampling_info is not None and not batch.sampling_info.is_all_greedy:
            raise ValueError("HierarchicalSpec v1 supports greedy decoding only")
        if batch.has_grammar:
            raise ValueError(
                "HierarchicalSpec v1 does not support constrained decoding"
            )
        b_batch, b_cache_locs = self._prepare_b_decode_window(batch)
        bs = batch.batch_size()
        candidate_chain = torch.zeros(
            (bs, self.outer_block_size), dtype=torch.int64, device=self.device
        )
        candidate_chain[:, 0].copy_(b_batch.spec_info.bonus_tokens)
        pending_lens = torch.zeros((bs,), dtype=torch.int64, device=self.device)
        round_offsets = torch.arange(
            self.inner_block_size, dtype=torch.int64, device=self.device
        )[None, :]

        # One extra column per row absorbs masked scatter writes without a
        # token-wise Python loop or a device-to-host length read.
        scatter_width = self.outer_block_size + 1
        scatter_storage = torch.zeros(
            (bs, scatter_width), dtype=torch.int64, device=self.device
        )
        scatter_storage[:, : self.outer_block_size].copy_(candidate_chain)
        row_offsets = (
            torch.arange(bs, device=self.device, dtype=torch.int64)[:, None]
            * scatter_width
        )
        for _ in range(self.cascade_rounds):
            inner_result = self.inner_dflash.forward_provisional(
                b_batch,
                grammar_barrier=grammar_barrier,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            round_tokens = inner_result.next_token_ids.view(bs, self.inner_block_size)
            round_lens = inner_result.accept_lens.to(torch.int64)
            append_pending_round_(
                scatter_storage,
                pending_lens,
                round_tokens,
                round_lens,
                row_offsets=row_offsets,
                round_offsets=round_offsets,
                overflow_col=self.outer_block_size,
            )
            b_batch.seq_lens = inner_result.new_seq_lens
            b_batch.spec_info = inner_result.next_draft_input
            allocated_cpu = batch.seq_lens_cpu + self.outer_block_size
            b_batch.spec_info.nxt_kv_lens_cpu = allocated_cpu.to(torch.int32)
            b_batch.spec_info.nxt_kv_lens_sum = int(allocated_cpu.sum())
            b_batch.seq_lens_cpu = None

        self._materialize_b_final_recovery(b_batch)
        candidate_chain.copy_(scatter_storage[:, : self.outer_block_size])
        return self._verify_with_c(
            batch,
            candidate_chain=candidate_chain,
            pending_lens=pending_lens,
            b_cache_locs=b_cache_locs,
        )

    def _materialize_b_final_recovery(self, b_batch) -> None:
        """Write the Kth B recovery token into B KV before C may accept it."""
        bs = b_batch.batch_size()
        positions = b_batch.seq_lens.to(torch.int64)
        b_pool, _ = self.intermediate_worker.get_memory_pool()
        cache_locs = assign_extend_cache_locs_func(
            req_pool_indices=b_batch.req_pool_indices,
            req_to_token=b_pool.req_to_token,
            start_offset=b_batch.seq_lens,
            end_offset=b_batch.seq_lens + 1,
            batch_size=bs,
            draft_token_num=1,
            device=self.device,
        )
        plan = self.intermediate_worker.model_runner.kv_index_translator.plan(
            req_pool_indices=b_batch.req_pool_indices,
            seq_lens=b_batch.seq_lens,
            seq_lens_cpu=b_batch.spec_info.nxt_kv_lens_cpu,
            write_virtual=cache_locs,
            read_extent=1,
        )
        verify_input = DFlashVerifyInput(
            draft_token=b_batch.spec_info.bonus_tokens,
            positions=positions,
            draft_token_num=1,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            kv_loc_plan=plan,
        )
        b_batch.out_cache_loc = cache_locs
        forward_batch, _ = verify_input.prepare_for_verify(
            b_batch, self.intermediate_worker
        )
        self.intermediate_worker.forward_batch_generation(
            batch=None,
            forward_batch=forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )

    def forward_batch_generation(
        self,
        batch,
        on_publish=None,
        grammar_barrier=None,
        pp_proxy_tensors=None,
    ):
        if on_publish is not None:
            raise RuntimeError(
                "HierarchicalSpec v1 requires --disable-overlap-schedule"
            )
        if batch.return_hidden_states:
            raise ValueError(
                "HierarchicalSpec v1 does not support return_hidden_states"
            )
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return self._forward_prefill(batch, pp_proxy_tensors=pp_proxy_tensors)
        return self._forward_decode(
            batch,
            grammar_barrier=grammar_barrier,
            pp_proxy_tensors=pp_proxy_tensors,
        )

    def __getattr__(self, name):
        if name == "_target_worker":
            raise AttributeError(name)
        return getattr(self.target_worker, name)
