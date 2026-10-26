"""Text-line recognizer of indecis's OCR, in PyTorch: the reference for the Go
implementation (internal/linerec).

The architecture is that of Indra OCR's rec.onnx (v1.1.0), read from its
graph: a stem of 32x4 patches over the middle band of a three-line window, a
context branch that pools the whole window over its rows, depthwise 1D
blocks with masked squeeze-excitation, and a CTC head.  Parameter names are
those of rec.onnx's named initializers, under "rec." removed.

Inputs: x [B, 1, window*height, W] ink maps (0 = paper, 1 = ink), W a
multiple of the stride; widths [B], the true width of each line in pixels.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    height: int = 32  # line height in pixels
    window: int = 3  # line heights in the input window
    stride: int = 4  # pixels per step
    dim: int = 128
    expand: int = 256
    se: int = 64
    blocks: int = 10
    kernel: int = 9
    ctx_channels: int = 32
    ctx_dim: int = 64
    ctx_heads: int = 4
    vocab: int = 1698  # blank = 0
    norm: str = "batch"  # stem normalization: "batch" (Indra) or "layer"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def from_json(cls, s: str) -> "Config":
        return cls(**json.loads(s))


def hardswish(x):
    return x * torch.clamp(x + 3, 0, 6) / 6


class Context(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        s = c.stride
        self.c = c
        self.patch = nn.Conv2d(1, c.ctx_channels, (s, s), stride=(s, s))
        self.dw = nn.Conv2d(
            c.ctx_channels, c.ctx_channels, 3, padding=1, groups=c.ctx_channels, bias=False
        )
        self.pw = nn.Conv2d(c.ctx_channels, c.ctx_dim, 1)
        self.score = nn.Conv2d(c.ctx_dim, c.ctx_heads, 1)
        rows = c.window * c.height // s
        self.row_prior = nn.Parameter(torch.zeros(1, c.ctx_heads, rows, 1))
        self.proj = nn.Conv1d(c.ctx_dim, c.dim, 1)

    def forward(self, x):
        c = self.c
        f = hardswish(self.patch(x))  # [B, cc, R, T]
        f = hardswish(self.pw(self.dw(f)))  # [B, cd, R, T]
        a = torch.softmax(self.score(f) + self.row_prior, dim=2)  # [B, heads, R, T]
        B, D, R, T = f.shape
        f = f.view(B, c.ctx_heads, D // c.ctx_heads, R, T)
        pooled = (f * a.unsqueeze(2)).sum(3).reshape(B, D, T)
        return self.proj(pooled)


class SE(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.fc1 = nn.Conv1d(c.expand, c.se, 1)
        self.fc2 = nn.Conv1d(c.se, c.expand, 1)

    def forward(self, u, mask):
        m = (u * mask).sum(2, keepdim=True) / mask.sum(2, keepdim=True).clamp(min=1)
        return u * torch.sigmoid(self.fc2(F.relu(self.fc1(m))))


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.dw = nn.Conv1d(c.dim, c.dim, c.kernel, padding=c.kernel // 2, groups=c.dim)
        self.expand = nn.Conv1d(c.dim, c.expand, 1)
        self.se = SE(c)
        self.project = nn.Conv1d(c.expand, c.dim, 1)

    def forward(self, h, mask):
        u = hardswish(self.expand(self.dw(h)))
        return (h + self.project(self.se(u, mask))) * mask


class Recognizer(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.stem = nn.Conv2d(1, c.dim, (c.height, c.stride), stride=(c.height, c.stride))
        self.context = Context(c)
        if c.norm == "batch":
            self.stem_bn = nn.BatchNorm1d(c.dim)
        else:
            self.stem_ln = nn.LayerNorm(c.dim)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.blocks))
        self.head = nn.Conv1d(c.dim, c.vocab, 1)

    def lengths(self, widths, T):
        return torch.clamp((widths + self.c.stride - 1) // self.c.stride, max=T)

    def forward(self, x, widths):
        """Logits [B, T, vocab] and valid steps [B]."""
        c = self.c
        top = (c.window // 2) * c.height
        h = self.stem(x[:, :, top : top + c.height, :]).squeeze(2)  # [B, dim, T]
        h = h + self.context(x)
        T = h.shape[2]
        lengths = self.lengths(widths, T)
        mask = (torch.arange(T)[None, :] < lengths[:, None]).to(h.dtype)[:, None, :]
        if c.norm == "batch":
            h = self.stem_bn(h)
        else:
            h = self.stem_ln(h.transpose(1, 2)).transpose(1, 2)
        h = hardswish(h) * mask
        for b in self.blocks:
            h = b(h, mask)
        return self.head(h).transpose(1, 2), lengths


def greedy(logits, lengths, blank=0):
    """Best path: per line, the collapsed symbol ids (blanks and repeats removed)."""
    out = []
    ids = logits.argmax(-1)
    for b in range(ids.shape[0]):
        seq, prev = [], blank
        for t in range(int(lengths[b])):
            i = int(ids[b, t])
            if i != blank and i != prev:
                seq.append(i)
            prev = i
        out.append(seq)
    return out
