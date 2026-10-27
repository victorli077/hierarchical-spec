"""Tensor-only state and acceptance primitives for three-stage speculation.

The coordinator deliberately contains no model calls.  Model runners provide
packed tensors to these functions, which keeps the commit/rollback contract
testable independently of a particular attention backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class TargetReadyProposal:
    candidate_token_ids: torch.Tensor
    candidate_logprobs: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    parent_indices: torch.Tensor | None = None
    slot_mapping: torch.Tensor | None = None
    aux_state: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.candidate_token_ids.ndim != 2:
            raise ValueError("candidate_token_ids must have shape [batch, width]")
        if self.candidate_token_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("candidate_token_ids must be an integer tensor")


@dataclass
class CascadeRequestState:
    committed_len: torch.Tensor
    pending_token_ids: torch.Tensor
    pending_rounds: torch.Tensor
    cache_lengths: dict[str, torch.Tensor]
    checkpoints: dict[str, torch.Tensor]
    rng_state: torch.Tensor | None = None

    def validate(self) -> None:
        if self.committed_len.ndim != 1:
            raise ValueError("committed_len must be one-dimensional")
        batch = self.committed_len.numel()
        if self.pending_token_ids.ndim != 2 or self.pending_token_ids.shape[0] != batch:
            raise ValueError("pending_token_ids must have shape [batch, pending_width]")
        if self.pending_rounds.ndim != 1 or self.pending_rounds.numel() != batch:
            raise ValueError("pending_rounds must have one entry per request")
        for stage in ("A", "B", "C"):
            if stage not in self.cache_lengths or stage not in self.checkpoints:
                raise ValueError(f"missing {stage} cache/checkpoint state")


@dataclass
class CascadeVerifyResult:
    accepted_len: torch.Tensor
    committed_token_ids: torch.Tensor
    recovery_token_ids: torch.Tensor
    accepted_end_hidden: torch.Tensor | None = None


def append_pending_round_(
    storage: torch.Tensor,
    pending_lens: torch.Tensor,
    round_tokens: torch.Tensor,
    round_lens: torch.Tensor,
    *,
    row_offsets: torch.Tensor,
    round_offsets: torch.Tensor,
    overflow_col: int,
) -> None:
    """Append one ragged B result into fixed GPU storage in place.

    ``storage`` has one overflow column. Invalid round positions all scatter to
    that column and are discarded by the caller, avoiding a host length read or
    token-wise Python loop.
    """
    if storage.ndim != 2 or round_tokens.ndim != 2:
        raise ValueError("storage and round_tokens must be two-dimensional")
    if pending_lens.shape != round_lens.shape or pending_lens.ndim != 1:
        raise ValueError("pending_lens and round_lens must align with the batch")
    if (
        storage.shape[0] != round_tokens.shape[0]
        or storage.shape[0] != pending_lens.numel()
    ):
        raise ValueError("round accumulation batch dimensions do not match")
    if overflow_col != storage.shape[1] - 1:
        raise ValueError("overflow_col must name storage's final column")

    valid = round_offsets < round_lens[:, None]
    dst = 1 + pending_lens[:, None] + round_offsets
    dst = torch.where(valid, dst, torch.full_like(dst, overflow_col))
    storage.view(-1).scatter_(
        0,
        (row_offsets + dst).reshape(-1),
        torch.where(valid, round_tokens, torch.zeros_like(round_tokens)).reshape(-1),
    )
    pending_lens.add_(round_lens)


def greedy_anchor_block_accept(
    candidate_chain: torch.Tensor,
    target_top1: torch.Tensor,
    pending_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Accept a DFlash-style chain whose first column is an input anchor.

    Returns ``(accepted_pending, commit_lens, bonus, padded_outputs)``. The
    output excludes the anchor and includes exactly one C-generated bonus at
    the first mismatch (or after the valid pending suffix).
    """
    if candidate_chain.ndim != 2 or target_top1.shape != candidate_chain.shape:
        raise ValueError("candidate_chain and target_top1 must share [batch,width]")
    batch, width = candidate_chain.shape
    if pending_lens.shape != (batch,):
        raise ValueError("pending_lens must have one entry per request")
    proposals = candidate_chain[:, 1:]
    proposal_positions = torch.arange(
        width - 1, device=candidate_chain.device, dtype=pending_lens.dtype
    )[None, :]
    valid = proposal_positions < pending_lens[:, None]
    matches = target_top1[:, :-1].eq(proposals) | ~valid
    rejected = (~matches) & valid
    first_reject = torch.argmax(rejected.to(torch.int32), dim=1)
    accepted = torch.where(rejected.any(dim=1), first_reject, pending_lens)
    bonus = target_top1.gather(1, accepted.to(torch.int64)[:, None]).squeeze(1)
    out_tokens = torch.zeros_like(candidate_chain)
    out_tokens[:, :-1].copy_(proposals)
    out_tokens.scatter_(1, accepted.to(torch.int64)[:, None], bonus[:, None])
    return accepted, accepted.to(torch.int32) + 1, bonus, out_tokens


def greedy_block_verify(
    candidate_token_ids: torch.Tensor,
    logits: torch.Tensor,
    valid_lengths: torch.Tensor | None = None,
    hidden_states: torch.Tensor | None = None,
) -> CascadeVerifyResult:
    """Verify a packed greedy block without a per-token Python loop.

    ``logits[:, i]`` predicts the token at candidate position ``i``.  An
    optional final logits row (``width + 1``) is used for the recovery token
    when every candidate is accepted.  The returned committed block contains
    the accepted prefix and the first target-generated recovery token.
    """
    if candidate_token_ids.ndim != 2 or logits.ndim != 3:
        raise ValueError(
            "expected candidates [batch,width] and logits [batch,width,vocab]"
        )
    batch, width = candidate_token_ids.shape
    if logits.shape[0] != batch or logits.shape[1] not in (width, width + 1):
        raise ValueError("candidate/logit block dimensions do not match")
    predicted = logits[:, :width].argmax(dim=-1).to(candidate_token_ids.dtype)
    matches = predicted.eq(candidate_token_ids)
    if valid_lengths is None:
        valid_lengths = torch.full(
            (batch,), width, dtype=torch.int64, device=candidate_token_ids.device
        )
    valid_lengths = valid_lengths.to(
        device=candidate_token_ids.device, dtype=torch.int64
    )
    positions = torch.arange(width, device=candidate_token_ids.device).unsqueeze(0)
    matches &= positions < valid_lengths.unsqueeze(1)
    # Number of true values before the first false value, vectorized.
    rejected = (~matches).to(torch.int64)
    first_reject = torch.where(
        rejected.any(dim=1), rejected.argmax(dim=1), valid_lengths.clamp_max(width)
    )
    accepted = torch.minimum(first_reject, valid_lengths.clamp_min(0))
    recovery_index = accepted.clamp_max(max(width - 1, 0))
    recovery_logits = logits.argmax(dim=-1).to(candidate_token_ids.dtype)
    has_bonus_row = recovery_logits.shape[1] == width + 1
    recovery_pos = accepted if has_bonus_row else recovery_index
    recovery = recovery_logits.gather(
        1, recovery_pos.clamp_max(recovery_logits.shape[1] - 1).unsqueeze(1)
    ).squeeze(1)
    committed = torch.zeros(
        (batch, width + int(has_bonus_row)),
        dtype=candidate_token_ids.dtype,
        device=candidate_token_ids.device,
    )
    committed[:, :width].copy_(candidate_token_ids)
    committed.scatter_(1, recovery_pos.unsqueeze(1), recovery.unsqueeze(1))
    if hidden_states is not None:
        if hidden_states.shape[:2] != (batch, width):
            raise ValueError("hidden_states must align with the verify block")
        hidden = hidden_states.gather(
            1, recovery_index[:, None, None].expand(-1, 1, hidden_states.shape[-1])
        ).squeeze(1)
    else:
        hidden = None
    return CascadeVerifyResult(accepted, committed, recovery, hidden)


class CascadeCoordinator:
    """Owns C-authoritative commit and stage rollback bookkeeping."""

    def __init__(self, state: CascadeRequestState):
        state.validate()
        self.state = state

    def commit_and_resync(self, result: CascadeVerifyResult) -> None:
        batch = self.state.committed_len.numel()
        if result.accepted_len.shape != (batch,):
            raise ValueError("accepted_len must have one entry per request")
        accepted = result.accepted_len.to(
            device=self.state.committed_len.device, dtype=self.state.committed_len.dtype
        )
        self.state.committed_len = self.state.committed_len + accepted + 1
        # Rejected suffixes are never visible after this operation.
        width = self.state.pending_token_ids.shape[1]
        positions = torch.arange(width, device=self.state.pending_token_ids.device)
        keep = positions.unsqueeze(0) < accepted.to(positions.device).unsqueeze(1)
        self.state.pending_token_ids = torch.where(
            keep,
            self.state.pending_token_ids,
            torch.zeros_like(self.state.pending_token_ids),
        )
        self.state.pending_rounds.zero_()
        for stage, length in self.state.cache_lengths.items():
            checkpoint = self.state.checkpoints[stage]
            length.copy_(checkpoint + accepted.to(length.device) + 1)

    def rollback_to_committed(self) -> None:
        for stage, length in self.state.cache_lengths.items():
            length.copy_(self.state.checkpoints[stage])
        self.state.pending_token_ids.zero_()
        self.state.pending_rounds.zero_()

    def run_window(self, drafter, intermediate, target, *, rounds: int, width: int):
        """Run a logical cascade window using native stage callbacks.

        Callbacks receive and return GPU tensors.  This method only loops over
        the configured number of rounds; it never iterates over individual
        tokens or converts stage data to Python lists.
        """
        if rounds <= 0 or width <= 0:
            raise ValueError("rounds and width must be positive")
        prefix = self.state.pending_token_ids
        round_outputs = []
        for round_index in range(rounds):
            proposal = drafter.propose(prefix, round_index)
            refined = intermediate.verify_and_refine(proposal, round_index)
            if not isinstance(refined, TargetReadyProposal):
                raise TypeError("intermediate stage must return TargetReadyProposal")
            if refined.candidate_token_ids.shape[1] > width:
                raise ValueError("intermediate proposal exceeds configured block width")
            round_outputs.append(refined.candidate_token_ids)
            prefix = refined.candidate_token_ids
        pending = torch.cat(round_outputs, dim=1)
        result = target.verify_block(pending)
        if not isinstance(result, CascadeVerifyResult):
            raise TypeError("target stage must return CascadeVerifyResult")
        self.state.pending_token_ids = pending
        self.commit_and_resync(result)
        return result
