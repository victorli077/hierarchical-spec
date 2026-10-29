import logging
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import _handle_hierarchical
from sglang.srt.environ import envs
from sglang.srt.layers.attention.flashinfer_backend import (
    _resolve_fast_verify_host_lens,
)
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig
from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
    resolve_hierarchical_graph_flags,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_worker_v2 import _DFlashTimingReporter
from sglang.srt.speculative.hierarchical_config import (
    cascade_pending_width,
    cascade_verify_width,
    hierarchical_b_uses_dynamic_verify_plan,
    hierarchical_model_verify_width,
    hierarchical_uses_dynamic_verify_plan,
    parse_device_map,
    validate_hierarchical_request,
)
from sglang.srt.speculative.hierarchical_types import (
    CascadeCoordinator,
    CascadeRequestState,
    append_pending_round_,
    greedy_anchor_block_accept,
    greedy_block_verify,
    pending_lens_for_b_round,
    should_run_b_verify,
)
from sglang.srt.speculative.hierarchical_worker import (
    CascadeCoordinator as RuntimeCascadeCoordinator,
)
from sglang.srt.speculative.hierarchical_worker import (
    HierarchicalSpecWorker,
    IntermediateStage,
    _validate_hierarchical_kv_copy_strides,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_dflash_ac_timing_reporter_aggregates_a_and_c(caplog):
    stages = [
        "a_draft",
        "c_verify_setup",
        "c_model_verify",
        "c_accept_postprocess",
        "c_hidden_publish",
        "c_result_finalize",
    ]
    reporter = _DFlashTimingReporter(enabled=True, interval=2, stage_names=stages)
    sample = {
        "a_draft": 2.0,
        "c_verify_setup": 1.0,
        "c_model_verify": 3.0,
        "c_accept_postprocess": 1.0,
        "c_hidden_publish": 1.0,
        "c_result_finalize": 0.5,
        "host_launch_sync_residual": 1.5,
        "total": 10.0,
    }

    with caplog.at_level(logging.INFO):
        reporter.add(sample, batch_size=4, a_graph_replay=True, c_graph_replay=False)
        assert "DFlash A+C timing:" not in caplog.text
        reporter.add(sample, batch_size=8, a_graph_replay=False, c_graph_replay=True)

    assert "DFlash A+C timing: samples=2, latest_bs=8" in caplog.text
    assert "a_total=2.000 ms (20.0%)" in caplog.text
    assert "c_total=6.500 ms (65.0%)" in caplog.text
    assert "a_graph_replays=1 a_graph_fallbacks=1" in caplog.text
    assert "c_graph_replays=1 c_graph_fallbacks=1" in caplog.text
    assert reporter._count == 0


def _state(batch=2, width=4):
    return CascadeRequestState(
        committed_len=torch.tensor([10] * batch),
        pending_token_ids=torch.full((batch, width), 99, dtype=torch.long),
        pending_rounds=torch.ones(batch, dtype=torch.int32),
        cache_lengths={s: torch.tensor([10] * batch) for s in "ABC"},
        checkpoints={s: torch.tensor([10] * batch) for s in "ABC"},
    )


def test_greedy_block_verify_accept_all_and_reject_boundary():
    candidates = torch.tensor([[1, 2, 3], [4, 5, 6]])
    logits = torch.full((2, 3, 8), -100.0)
    logits[0, 0, 1] = logits[0, 1, 2] = logits[0, 2, 3] = 1
    logits[1, 0, 4] = 1
    logits[1, 1, 7] = 1
    result = greedy_block_verify(candidates, logits)
    assert result.accepted_len.tolist() == [3, 1]
    assert result.recovery_token_ids.tolist() == [3, 7]
    assert result.committed_token_ids[1].tolist() == [4, 7, 6]

    bonus_logits = torch.full((1, 4, 8), -100.0)
    bonus_logits[0, 0, 1] = bonus_logits[0, 1, 2] = bonus_logits[0, 2, 3] = 1
    bonus_logits[0, 3, 7] = 2
    bonus = greedy_block_verify(candidates[:1], bonus_logits)
    assert bonus.accepted_len.tolist() == [3]
    assert bonus.recovery_token_ids.tolist() == [7]
    assert bonus.committed_token_ids[0].tolist() == [1, 2, 3, 7]


def test_anchor_block_accept_all_partial_first_and_ragged():
    chains = torch.tensor(
        [
            [90, 1, 2, 3, 0],
            [91, 4, 5, 6, 7],
            [92, 8, 9, 0, 0],
        ]
    )
    # Row 0 accepts all three pending tokens, row 1 rejects after token 4,
    # row 2 rejects its first pending token. Last columns are recovery logits.
    target = torch.tensor(
        [
            [1, 2, 3, 10, 11],
            [4, 12, 6, 7, 13],
            [14, 9, 15, 16, 17],
        ]
    )
    accepted, committed, bonus, outputs = greedy_anchor_block_accept(
        chains, target, torch.tensor([3, 4, 2])
    )
    assert accepted.tolist() == [3, 1, 0]
    assert committed.tolist() == [4, 2, 1]
    assert bonus.tolist() == [10, 12, 14]
    assert outputs[0, :4].tolist() == [1, 2, 3, 10]
    assert outputs[1, :2].tolist() == [4, 12]
    assert outputs[2, 0].item() == 14


def test_ragged_round_accumulation_is_contiguous_and_drops_padding():
    batch, outer_width, round_width = 3, 10, 3
    storage = torch.zeros((batch, outer_width + 1), dtype=torch.int64)
    storage[:, 0] = torch.tensor([90, 91, 92])
    pending = torch.zeros(batch, dtype=torch.int64)
    row_offsets = torch.arange(batch)[:, None] * storage.shape[1]
    round_offsets = torch.arange(round_width)[None, :]

    append_pending_round_(
        storage,
        pending,
        torch.tensor([[1, 2, 3], [4, 99, 99], [5, 6, 99]]),
        torch.tensor([3, 1, 2]),
        row_offsets=row_offsets,
        round_offsets=round_offsets,
        overflow_col=outer_width,
    )
    append_pending_round_(
        storage,
        pending,
        torch.tensor([[7, 99, 99], [8, 9, 10], [11, 12, 99]]),
        torch.tensor([1, 3, 2]),
        row_offsets=row_offsets,
        round_offsets=round_offsets,
        overflow_col=outer_width,
    )

    assert pending.tolist() == [4, 4, 4]
    assert storage[0, :5].tolist() == [90, 1, 2, 3, 7]
    assert storage[1, :5].tolist() == [91, 4, 8, 9, 10]
    assert storage[2, :5].tolist() == [92, 5, 6, 11, 12]


def test_only_final_b_round_drops_its_bonus():
    commit_lens = torch.tensor([4, 2, 1], dtype=torch.int64)
    assert pending_lens_for_b_round(commit_lens, is_final_round=False).tolist() == [
        4,
        2,
        1,
    ]
    assert pending_lens_for_b_round(commit_lens, is_final_round=True).tolist() == [
        3,
        1,
        0,
    ]


def test_c_to_b_handoff_runs_only_k_minus_one_b_verifies():
    assert [
        should_run_b_verify(i, rounds=3, use_c_to_b_kv_handoff=True) for i in range(3)
    ] == [True, True, False]
    assert [
        should_run_b_verify(i, rounds=3, use_c_to_b_kv_handoff=False) for i in range(3)
    ] == [True, True, True]
    assert not should_run_b_verify(0, rounds=1, use_c_to_b_kv_handoff=True)
    with pytest.raises(ValueError, match="round_index"):
        should_run_b_verify(3, rounds=3, use_c_to_b_kv_handoff=True)


def test_c_to_b_kv_copy_requires_identical_row_strides():
    """A layout mismatch must fail before launching a cross-pool copy kernel."""
    strides = torch.tensor([128, 256], dtype=torch.int64)
    _validate_hierarchical_kv_copy_strides(strides, strides.clone())
    with pytest.raises(RuntimeError, match="identical K/V row strides"):
        _validate_hierarchical_kv_copy_strides(
            strides, torch.tensor([128, 512], dtype=torch.int64)
        )


class TestHierarchicalKvCopyMetadata(CustomTestCase):
    def test_resolves_kv_pool_through_allocator(self):
        def make_worker():
            k_buffer = [torch.zeros((4, 2), dtype=torch.bfloat16)]
            v_buffer = [torch.zeros((4, 2), dtype=torch.bfloat16)]
            kv_pool = SimpleNamespace(
                k_buffer=k_buffer,
                v_buffer=v_buffer,
                k_data_ptrs=torch.tensor([k_buffer[0].data_ptr()], dtype=torch.uint64),
                v_data_ptrs=torch.tensor([v_buffer[0].data_ptr()], dtype=torch.uint64),
                kv_cache_layout="nhd",
                is_quantized_kv_cache=False,
            )
            allocator = SimpleNamespace(get_kvcache=lambda: kv_pool)
            request_pool = object()
            return (
                SimpleNamespace(
                    get_memory_pool=lambda: (request_pool, allocator),
                ),
                kv_pool,
            )

        target_worker, c_pool = make_worker()
        intermediate_worker, b_pool = make_worker()
        worker = object.__new__(HierarchicalSpecWorker)
        worker.target_worker = target_worker
        worker.coordinator = SimpleNamespace(
            intermediate_stage=SimpleNamespace(worker=intermediate_worker)
        )

        metadata = worker._build_c_to_b_kv_copy_metadata()

        self.assertEqual(metadata["src_ptrs"][0].item(), c_pool.k_buffer[0].data_ptr())
        self.assertEqual(metadata["dst_ptrs"][0].item(), b_pool.k_buffer[0].data_ptr())
        self.assertEqual(metadata["row_bytes"], 4)


def test_coordinator_commit_clears_rejected_suffix_and_resyncs_all_stages():
    state = _state()
    state.pending_token_ids.copy_(torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]))
    coordinator = CascadeCoordinator(state)
    logits = torch.full((2, 4, 16), -100.0)
    logits[0, 0, 1] = logits[0, 1, 2] = 1
    logits[0, 2, 9] = logits[0, 3, 0] = 1
    logits[1, 0, 5] = logits[1, 1, 9] = 1
    result = greedy_block_verify(
        state.pending_token_ids,
        logits,
        valid_lengths=torch.tensor([2, 1]),
    )
    coordinator.commit_and_resync(result)
    assert state.committed_len.tolist() == [13, 12]
    assert state.pending_token_ids[0, 2:].tolist() == [0, 0]
    assert state.pending_token_ids[1, 1:].tolist() == [0, 0, 0]
    for length in state.cache_lengths.values():
        assert length.tolist() == [13, 12]


def test_device_map_requires_all_stages():
    assert parse_device_map("A:0,B:1,C:2") == {"A": (0,), "B": (1,), "C": (2,)}
    assert parse_device_map("A:0,B:1;2,C:3")["B"] == (1, 2)


def test_cascade_width_drops_only_final_b_bonus():
    # Each non-final round contributes L-1 A proposals plus its B bonus. The
    # final round contributes only L-1 A proposals, while C supplies the final
    # bonus. C's anchor makes its verify width exactly K*L.
    assert cascade_pending_width(inner_block_size=4, rounds=3) == 11
    assert cascade_verify_width(inner_block_size=4, rounds=3) == 12
    assert cascade_pending_width(inner_block_size=4, rounds=1) == 3
    assert cascade_verify_width(inner_block_size=4, rounds=1) == 4


def _hierarchical_server_args(**overrides):
    values = dict(
        model_path="C",
        speculative_draft_model_path="A",
        speculative_intermediate_model_path="B",
        speculative_cascade_rounds=3,
        page_size=1,
        attention_backend="flashinfer",
        cuda_graph_config=CudaGraphConfig(),
    )
    values.update(overrides)
    return ServerArgs(**values)


def test_hierarchical_dflash_block_size_reaches_worker_field():
    args = _hierarchical_server_args(speculative_dflash_block_size=8)
    _handle_hierarchical(args)
    assert resolution_result(args, "speculative_num_draft_tokens") == 8
    assert (
        cascade_verify_width(
            resolution_result(args, "speculative_num_draft_tokens"),
            resolution_result(args, "speculative_cascade_rounds"),
        )
        == 24
    )


def test_hierarchical_model_graph_widths():
    kwargs = {"inner_block_size": 8, "rounds": 3}
    assert (
        hierarchical_model_verify_width(
            **kwargs, is_auxiliary_target=False, is_draft_worker=True
        )
        == 8
    )
    assert (
        hierarchical_model_verify_width(
            **kwargs, is_auxiliary_target=True, is_draft_worker=False
        )
        == 8
    )
    assert (
        hierarchical_model_verify_width(
            **kwargs, is_auxiliary_target=False, is_draft_worker=False
        )
        == 24
    )


@pytest.mark.parametrize(
    ("is_hierarchical", "is_auxiliary_target", "is_draft_worker", "expected"),
    [
        (True, True, False, True),
        (True, False, False, False),
        (True, False, True, False),
        (False, True, False, False),
    ],
)
def test_hierarchical_b_uses_dynamic_verify_plan(
    is_hierarchical, is_auxiliary_target, is_draft_worker, expected
):
    assert (
        hierarchical_b_uses_dynamic_verify_plan(
            is_hierarchical=is_hierarchical,
            is_auxiliary_target=is_auxiliary_target,
            is_draft_worker=is_draft_worker,
        )
        is expected
    )


def test_hierarchical_dflash_block_size_defaults_and_rejects_conflict():
    default_args = _hierarchical_server_args()
    _handle_hierarchical(default_args)
    assert resolution_result(default_args, "speculative_num_draft_tokens") == 4

    conflict_args = _hierarchical_server_args(
        speculative_dflash_block_size=8,
        speculative_num_draft_tokens=4,
    )
    with pytest.raises(ValueError, match="they differ"):
        _handle_hierarchical(conflict_args)


def test_hierarchical_b_cuda_graph_is_opt_in():
    eager_args = _hierarchical_server_args()
    _handle_hierarchical(eager_args)
    assert resolution_result(eager_args, "disable_cuda_graph") is True

    graph_args = _hierarchical_server_args(device="cuda")
    with envs.SGLANG_ENABLE_HIERARCHICAL_B_CUDA_GRAPH.override(True):
        _handle_hierarchical(graph_args)
    assert resolution_result(graph_args, "disable_cuda_graph") is False
    assert resolution_result(graph_args, "disable_decode_cuda_graph") is False
    assert resolution_result(graph_args, "disable_prefill_cuda_graph") is False


def test_hierarchical_b_cuda_graph_requires_cuda():
    args = _hierarchical_server_args(device="cpu")
    with envs.SGLANG_ENABLE_HIERARCHICAL_B_CUDA_GRAPH.override(True):
        _handle_hierarchical(args)
    assert resolution_result(args, "disable_cuda_graph") is True


@pytest.mark.parametrize(
    ("is_auxiliary_target", "is_draft_worker", "expected"),
    [
        (False, False, (False, False)),  # C stays fully eager.
        (True, False, (True, False)),  # B keeps only target-verify decode graph.
        (False, True, (False, False)),  # A is independently opt-in.
    ],
)
def test_hierarchical_role_graph_worker_roles(
    is_auxiliary_target, is_draft_worker, expected
):
    with envs.SGLANG_ENABLE_HIERARCHICAL_B_CUDA_GRAPH.override(True):
        actual = resolve_hierarchical_graph_flags(
            is_hierarchical=True,
            is_auxiliary_target=is_auxiliary_target,
            is_draft_worker=is_draft_worker,
            capture_decode_cuda_graph=True,
        )
    assert actual == expected


@pytest.mark.parametrize(
    ("is_auxiliary_target", "is_draft_worker", "expected"),
    [(True, False, True), (False, True, True), (False, False, False)],
)
def test_hierarchical_a_or_b_uses_dynamic_verify_plan(
    is_auxiliary_target, is_draft_worker, expected
):
    assert (
        hierarchical_uses_dynamic_verify_plan(
            is_hierarchical=True,
            is_auxiliary_target=is_auxiliary_target,
            is_draft_worker=is_draft_worker,
        )
        is expected
    )


def test_hierarchical_c_cuda_graph_is_opt_in():
    args = _hierarchical_server_args(device="cuda")
    with envs.SGLANG_ENABLE_HIERARCHICAL_C_CUDA_GRAPH.override(True):
        _handle_hierarchical(args)
    assert resolution_result(args, "disable_cuda_graph") is False
    assert resolution_result(args, "disable_decode_cuda_graph") is False
    assert resolution_result(args, "disable_prefill_cuda_graph") is False


def test_hierarchical_c_graph_role_is_decode_only():
    with envs.SGLANG_ENABLE_HIERARCHICAL_C_CUDA_GRAPH.override(True):
        actual = resolve_hierarchical_graph_flags(
            is_hierarchical=True,
            is_auxiliary_target=False,
            is_draft_worker=False,
            capture_decode_cuda_graph=True,
            capture_prefill_cuda_graph=True,
        )
    assert actual == (True, False)


def test_hierarchical_role_graph_policy_is_opt_in():
    with (
        envs.SGLANG_ENABLE_HIERARCHICAL_A_CUDA_GRAPH.override(False),
        envs.SGLANG_ENABLE_HIERARCHICAL_B_CUDA_GRAPH.override(False),
        envs.SGLANG_ENABLE_HIERARCHICAL_C_CUDA_GRAPH.override(False),
    ):
        actual = resolve_hierarchical_graph_flags(
            is_hierarchical=True,
            is_auxiliary_target=False,
            is_draft_worker=False,
            capture_decode_cuda_graph=True,
        )
    assert actual == (True, True)


@pytest.mark.parametrize(
    ("a_enabled", "b_enabled", "c_enabled", "expected"),
    [
        (True, False, False, ((False, False), (True, True), (False, False))),
        (True, True, False, ((False, False), (True, True), (True, False))),
        (False, False, True, ((True, False), (False, False), (False, False))),
    ],
)
def test_hierarchical_graph_role_selection(a_enabled, b_enabled, c_enabled, expected):
    with (
        envs.SGLANG_ENABLE_HIERARCHICAL_A_CUDA_GRAPH.override(a_enabled),
        envs.SGLANG_ENABLE_HIERARCHICAL_B_CUDA_GRAPH.override(b_enabled),
        envs.SGLANG_ENABLE_HIERARCHICAL_C_CUDA_GRAPH.override(c_enabled),
    ):
        c_flags = resolve_hierarchical_graph_flags(
            is_hierarchical=True,
            is_auxiliary_target=False,
            is_draft_worker=False,
            capture_decode_cuda_graph=True,
        )
        a_flags = resolve_hierarchical_graph_flags(
            is_hierarchical=True,
            is_auxiliary_target=False,
            is_draft_worker=True,
            capture_decode_cuda_graph=True,
        )
        b_flags = resolve_hierarchical_graph_flags(
            is_hierarchical=True,
            is_auxiliary_target=True,
            is_draft_worker=False,
            capture_decode_cuda_graph=True,
        )
    assert (c_flags, a_flags, b_flags) == expected


def test_hierarchical_b_graph_capture_error_installs_complete_eager_fallback():
    eager_runner = object()
    model_runner = SimpleNamespace(decode_cuda_graph_captured=True)

    class IntermediateWorker:
        def __init__(self):
            self.model_runner = model_runner
            self.capture_attempts = []

        def init_cuda_graphs(self, capture_decode_cuda_graph=True):
            self.capture_attempts.append(capture_decode_cuda_graph)
            if capture_decode_cuda_graph:
                raise RuntimeError("synthetic B graph capture error")
            model_runner.eager_runner = eager_runner
            model_runner.prefill_cuda_graph_runner = eager_runner
            model_runner.decode_cuda_graph_runner = eager_runner

    class InnerDFlash:
        graph_initialized = False

        def init_cuda_graphs(self):
            self.graph_initialized = True

    worker = object.__new__(HierarchicalSpecWorker)
    worker.intermediate_worker = IntermediateWorker()
    worker.inner_dflash = InnerDFlash()
    worker._b_cuda_graph_capture_failed = False

    worker.init_cuda_graphs()

    assert worker._b_cuda_graph_capture_failed
    assert worker.intermediate_worker.capture_attempts == [True, False]
    assert model_runner.prefill_cuda_graph_runner is eager_runner
    assert model_runner.decode_cuda_graph_runner is eager_runner
    assert model_runner.decode_cuda_graph_captured is False
    assert worker.inner_dflash.graph_initialized


def test_hierarchical_b_fast_verify_plan_uses_live_prefix_plus_width():
    live_prefix = torch.tensor([10, 13], device="cpu", dtype=torch.int64)
    allocation_bound = torch.tensor([34, 37], dtype=torch.int32)
    result = _resolve_fast_verify_host_lens(
        paged_kernel_lens=live_prefix,
        seq_lens_cpu=allocation_bound,
        num_tokens_per_req=8,
        hierarchical_dynamic=True,
    )
    assert result.tolist() == [18, 21]


def test_v1_request_validation_rejects_unsupported_features():
    def req(**sampling):
        defaults = dict(
            top_k=1,
            json_schema=None,
            regex=None,
            ebnf=None,
            structural_tag=None,
        )
        defaults.update(sampling)
        return SimpleNamespace(
            sampling_params=SimpleNamespace(**defaults),
            return_hidden_states=False,
        )

    assert validate_hierarchical_request(req()) is None
    assert "greedy" in validate_hierarchical_request(req(top_k=20))
    assert "constrained" in validate_hierarchical_request(req(regex="[0-9]+"))
    hidden_req = req()
    hidden_req.return_hidden_states = True
    assert "return_hidden_states" in validate_hierarchical_request(hidden_req)


def test_intermediate_stage_owns_and_clears_private_pool():
    class Pool:
        def __init__(self):
            self.cleared = self.aux_reset = False
            self.next_rows = [7, 8, 9]
            self.freed = []
            self.req_to_token = torch.zeros((10, 6), dtype=torch.int32)

        def clear(self):
            self.cleared = True

        def reset_aux_cache_allocator(self):
            self.aux_reset = True

        def alloc_rows(self, count):
            rows, self.next_rows = self.next_rows[:count], self.next_rows[count:]
            return rows

        def free_rows(self, rows):
            self.freed.extend(rows)

    class Allocator:
        def __init__(self):
            self.cleared = False
            self.freed = []
            self.freed_together = []
            self.freed_segments = []

        def clear(self):
            self.cleared = True

        def free_segment(self, slots, *, start_pos):
            self.freed.append((slots.clone(), start_pos))

        def free(self, slots):
            self.freed_together.append(slots.clone())

        def free_segments(self, segments):
            self.freed_segments.extend(
                (slots.clone(), start_pos) for slots, start_pos in segments
            )

    class Worker:
        is_draft_worker = False
        is_auxiliary_target = True
        model_runner = object()
        model_config = object()

        def __init__(self):
            self.pool = Pool()
            self.allocator = Allocator()
            self.alloc_args = None

        def alloc_memory_pool(self, *args, **kwargs):
            self.alloc_args = (args, kwargs)

        def get_memory_pool(self):
            return self.pool, self.allocator

    worker = Worker()
    stage = IntermediateStage(worker)
    stage.alloc_memory_pool()
    assert worker.alloc_args == ((), {})
    reqs = [type("Req", (), {"rid": rid})() for rid in ("r0", "r1")]
    gpu_rows, cpu_rows = stage.rows_for_requests(reqs, device="cpu")
    assert gpu_rows.tolist() == cpu_rows.tolist() == [7, 8]
    # Dynamic re-batching must preserve B-local rows without allocating again.
    assert stage.rows_for_requests(reqs[::-1], device="cpu")[0].tolist() == [8, 7]

    class TreeCache:
        def __init__(self):
            self.reset_called = False

        def reset(self):
            self.reset_called = True

    stage.tree_cache = TreeCache()
    c_batch = SimpleNamespace(
        reqs=reqs,
        device="cpu",
        model_config=object(),
        req_to_token_pool=object(),
        tree_cache=object(),
        req_pool_indices=torch.tensor([1, 2]),
        req_pool_indices_cpu=torch.tensor([1, 2]),
        out_cache_loc=torch.tensor([99]),
        spec_info=SimpleNamespace(seed=1),
    )
    b_batch = stage.make_batch_view(c_batch)
    assert b_batch is not c_batch
    assert b_batch.req_to_token_pool is worker.pool
    assert b_batch.tree_cache is stage.tree_cache
    assert b_batch.req_pool_indices.tolist() == [7, 8]
    assert b_batch.out_cache_loc is None and c_batch.out_cache_loc.tolist() == [99]
    worker.pool.req_to_token[7, :3] = torch.tensor([11, 12, 13])
    stage.update_cache_lengths(reqs, torch.tensor([3, 0]))
    stage.release_request("r0")
    assert worker.pool.freed == [7]
    freed_slots, start_pos = worker.allocator.freed[0]
    assert freed_slots.tolist() == [11, 12, 13] and start_pos == 0
    assert worker.pool.req_to_token[7, :3].tolist() == [0, 0, 0]

    # A rejected provisional suffix is removed from both allocator ownership
    # and B's effective req->token mapping before the next outer window.
    worker.pool.req_to_token[8, :5] = torch.tensor([21, 22, 23, 24, 25])
    stage.update_cache_lengths(reqs[1:], torch.tensor([5]))
    stage.trim_to_lengths(reqs[1:], torch.tensor([2]))
    assert worker.allocator.freed_together[-1].tolist() == [23, 24, 25]
    assert worker.pool.req_to_token[8, :5].tolist() == [21, 22, 0, 0, 0]

    # Logical positions overlap across requests, so their physical slots must
    # be freed together rather than passed to free_segments(), whose contract
    # is limited to disjoint ranges belonging to one request.
    req2 = type("Req", (), {"rid": "r2"})()
    stage.rows_for_requests([req2], device="cpu")
    worker.pool.req_to_token[8, :4] = torch.tensor([31, 32, 33, 34])
    worker.pool.req_to_token[9, :5] = torch.tensor([41, 42, 43, 44, 45])
    stage.update_cache_lengths([reqs[1], req2], torch.tensor([4, 5]))
    stage.trim_to_lengths([reqs[1], req2], torch.tensor([2, 3]))
    assert worker.allocator.freed_together[-1].tolist() == [33, 34, 44, 45]
    assert worker.allocator.freed_segments == []

    # Preemption discards the remaining B prefix; a resumed request rebuilds
    # it from C-authoritative request tokens during prefill.
    stage.reset_request("r1")
    assert worker.pool.req_to_token[8, :2].tolist() == [0, 0]
    stage.release_request("missing")
    stage.clear_cache_pool()
    assert stage.tree_cache.reset_called
    assert worker.pool.cleared and worker.pool.aux_reset
    assert worker.allocator.cleared


def test_runtime_coordinator_owns_b_and_checks_kernel_injection():
    b_worker = SimpleNamespace(is_draft_worker=False, is_auxiliary_target=True)
    stage = IntermediateStage(b_worker)
    draft_worker = object()
    kernel = SimpleNamespace(target_worker=b_worker, draft_worker=draft_worker)
    coordinator = RuntimeCascadeCoordinator(
        intermediate_stage=stage, inner_kernel=kernel
    )
    assert coordinator.intermediate_worker is b_worker
    assert coordinator.draft_worker is draft_worker

    wrong_kernel = SimpleNamespace(
        target_worker=SimpleNamespace(), draft_worker=draft_worker
    )
    try:
        RuntimeCascadeCoordinator(intermediate_stage=stage, inner_kernel=wrong_kernel)
    except ValueError as exc:
        assert "coordinator-owned B" in str(exc)
    else:
        raise AssertionError("mismatched A -> B target injection must be rejected")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
