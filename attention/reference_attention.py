"""Small, runnable reference implementations for the attention article.

These are deliberately clear rather than fused or production-optimised.  They
operate on a whole causal sequence and are useful for checking tensor shapes.
"""

import torch
from dataclasses import dataclass
from torch import Tensor, nn


def causal_mask(scores: Tensor, key_positions: Tensor) -> Tensor:
    """Mask keys that occur after each query position."""
    query_positions = torch.arange(scores.size(1), device=scores.device)
    return scores.masked_fill(key_positions[None, None, :] > query_positions[None, :, None], float("-inf"))


def attend_selected(q: Tensor, k: Tensor, v: Tensor, indices: Tensor, valid: Tensor | None = None) -> Tensor:
    """Attention over per-query selected entries.

    q is [batch, time, heads, width], k and v are [batch, entries, width],
    and indices is [batch, time, selected].
    """
    batch, time, heads, width = q.shape
    selected = indices.size(-1)
    gather = indices[:, :, :, None].expand(batch, time, selected, width)
    selected_k = k[:, None].expand(-1, time, -1, -1).gather(2, gather)
    selected_v = v[:, None].expand(-1, time, -1, -1).gather(2, gather)
    logits = torch.einsum("bthd,btkd->bthk", q, selected_k) * width ** -0.5
    if valid is not None:
        logits = logits.masked_fill(~valid[:, :, None, :], float("-inf"))
    weights = logits.softmax(dim=-1)
    if valid is not None:
        weights = torch.where(valid.any(dim=-1)[:, :, None, None], weights, torch.zeros_like(weights))
    return torch.einsum("bthk,btkd->bthd", weights, selected_v)


class DeepSeekSparseAttention(nn.Module):
    """DSA-style token selection over a shared (MQA-like) KV cache."""

    def __init__(self, dim: int, heads: int = 4, index_heads: int = 2, top_k: int = 8):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.width, self.top_k = heads, dim // heads, top_k
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * self.width, bias=False)  # one KV set, shared by heads
        self.index_q = nn.Linear(dim, index_heads * self.width, bias=False)
        self.index_k = nn.Linear(dim, index_heads * self.width, bias=False)
        self.index_weight = nn.Parameter(torch.ones(index_heads))
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, time, dim = x.shape
        q = self.q(x).view(batch, time, self.heads, self.width)
        k, v = self.kv(x).chunk(2, dim=-1)
        iq = self.index_q(x).view(batch, time, -1, self.width)
        ik = self.index_k(x).view(batch, time, -1, self.width)
        # The inexpensive indexer ranks every earlier token before main attention.
        index_scores = torch.einsum("btjd,bsjd->btsj", iq, ik).relu()
        index_scores = (index_scores * self.index_weight).sum(dim=-1)
        positions = torch.arange(time, device=x.device)
        index_scores = index_scores.masked_fill(positions[None, None, :] > positions[None, :, None], float("-inf"))
        selected = index_scores.topk(min(self.top_k, time), dim=-1).indices
        valid = selected <= positions[None, :, None]
        y = attend_selected(q, k, v, selected, valid)
        return self.out(y.reshape(batch, time, dim))


class CompressedSparseAttention(nn.Module):
    """CSA-style learned block compression, sparse global read, and local window."""

    def __init__(self, dim: int, heads: int = 4, block_size: int = 4, top_k: int = 4, window: int = 8):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.width = heads, dim // heads
        self.block_size, self.top_k, self.window = block_size, top_k, window
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * self.width, bias=False)
        self.pool_logits = nn.Linear(dim, 1, bias=False)
        self.index_q = nn.Linear(dim, self.width, bias=False)
        self.index_k = nn.Linear(self.width, self.width, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def compress(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return one learned KV pair per complete-or-padded sequence block."""
        batch, time, _ = x.shape
        pad = (-time) % self.block_size
        if pad:
            x = torch.cat((x, torch.zeros(batch, pad, x.size(-1), device=x.device, dtype=x.dtype)), dim=1)
        blocks = x.view(batch, -1, self.block_size, x.size(-1))
        logits = self.pool_logits(blocks).squeeze(-1)
        weights = logits.softmax(dim=-1)
        k, v = self.kv(blocks).chunk(2, dim=-1)
        return (weights[..., None] * k).sum(dim=2), (weights[..., None] * v).sum(dim=2), logits

    def _global_indices(self, x: Tensor, compressed_k: Tensor) -> tuple[Tensor, Tensor]:
        batch, time, _ = x.shape
        score = torch.einsum("btd,brd->btr", self.index_q(x), self.index_k(compressed_k))
        # A global block is visible only once its final source token is in the past.
        block_ends = torch.arange(compressed_k.size(1), device=x.device) * self.block_size + self.block_size - 1
        positions = torch.arange(time, device=x.device)
        score = score.masked_fill(block_ends[None, None, :] > positions[None, :, None], float("-inf"))
        selected_scores, indices = score.topk(min(self.top_k, compressed_k.size(1)), dim=-1)
        # Early tokens may have no completed global block. Their global contribution is zero.
        return indices, selected_scores.isfinite()

    def forward(self, x: Tensor) -> Tensor:
        batch, time, dim = x.shape
        q = self.q(x).view(batch, time, self.heads, self.width)
        global_k, global_v, _ = self.compress(x)
        indices, valid = self._global_indices(x, global_k)
        global_y = attend_selected(q, global_k, global_v, indices, valid)

        # A direct local path retains token-level detail near the current token.
        local_k, local_v = self.kv(x).chunk(2, dim=-1)
        local_scores = torch.einsum("bthd,bsd->bths", q, local_k) * self.width ** -0.5
        positions = torch.arange(time, device=x.device)
        local_scores = local_scores.masked_fill(positions[None, None, None, :] > positions[None, :, None, None],
                                                float("-inf"))
        local_scores = local_scores.masked_fill(
            positions[None, None, None, :] < (positions - self.window + 1)[None, :, None, None], float("-inf"))
        local_y = torch.einsum("bths,bsd->bthd", local_scores.softmax(-1), local_v)
        return self.out((global_y + local_y).reshape(batch, time, dim))


class HeavilyCompressedAttention(CompressedSparseAttention):
    """HCA reads every heavily compressed block instead of selecting Top-K blocks."""

    def forward(self, x: Tensor) -> Tensor:
        batch, time, dim = x.shape
        q = self.q(x).view(batch, time, self.heads, self.width)
        k, v, _ = self.compress(x)
        scores = torch.einsum("bthd,brd->bthr", q, k) * self.width ** -0.5
        ends = torch.arange(k.size(1), device=x.device) * self.block_size + self.block_size - 1
        positions = torch.arange(time, device=x.device)
        scores = scores.masked_fill(ends[None, None, None, :] > positions[None, :, None, None], float("-inf"))
        valid = ends[None, None, :] <= positions[None, :, None]
        weights = scores.softmax(-1)
        weights = torch.where(valid[:, :, None, :].any(dim=-1, keepdim=True), weights, torch.zeros_like(weights))
        y = torch.einsum("bthr,brd->bthd", weights, v)
        return self.out(y.reshape(batch, time, dim))


@dataclass
class CSA2State:
    keys: Tensor
    values: Tensor
    indices: Tensor
    valid: Tensor


class CSA2Layer(CompressedSparseAttention):
    """Reference for CSA2's Full, Reindex, and Reuse preparation modes."""

    def forward(self, x: Tensor, mode: str = "full", previous: CSA2State | None = None) -> tuple[Tensor, CSA2State]:
        if mode == "full":
            keys, values, _ = self.compress(x)
            indices, valid = self._global_indices(x, keys)
        elif mode == "reindex" and previous is not None:
            keys, values = previous.keys, previous.values
            indices, valid = self._global_indices(x, keys)
        elif mode == "reuse" and previous is not None:
            keys, values, indices, valid = previous.keys, previous.values, previous.indices, previous.valid
        else:
            raise ValueError("reindex and reuse need the preceding CSA2State")
        q = self.q(x).view(x.size(0), x.size(1), self.heads, self.width)
        y = attend_selected(q, keys, values, indices, valid)
        return self.out(y.reshape_as(x)), CSA2State(keys, values, indices, valid)


if __name__ == "__main__":
    x = torch.randn(2, 16, 32)
    assert DeepSeekSparseAttention(32)(x).shape == x.shape
    assert CompressedSparseAttention(32)(x).shape == x.shape
    assert HeavilyCompressedAttention(32, block_size=8)(x).shape == x.shape
    layer = CSA2Layer(32)
    y, state = layer(x, "full")
    assert layer(y, "reindex", state)[0].shape == x.shape
    assert layer(y, "reuse", state)[0].shape == x.shape
