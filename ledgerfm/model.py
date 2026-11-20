"""LedgerFM encoder: field-embedding transaction tokens -> causal transformer over an account's history.

Several accounts can be packed into one row; attention is causal *within* a segment only.
Output at position t = embedding of transaction t given its own fields and the account's past.
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F
from .tokens import INT_FIELDS, FLOAT_FIELDS, MASK_TARGETS, NEXT_TARGETS

FIELDS = list(INT_FIELDS)
FIDX = {k: i for i, k in enumerate(FIELDS)}
FLIDX = {k: i for i, k in enumerate(FLOAT_FIELDS)}
# float columns that would leak a masked int field
MASK_FLOAT_LEAK = {"amt_b": ["lamt", "s_dev"], "hour": ["hsin", "hcos"]}


class Config:
    def __init__(self, d=256, layers=6, heads=8, ff=4, max_len=128, dropout=0.1):
        self.d, self.layers, self.heads, self.ff, self.max_len, self.dropout = d, layers, heads, ff, max_len, dropout

    def to_dict(self):
        return dict(vars(self))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.d), nn.LayerNorm(c.d)
        self.qkv, self.proj = nn.Linear(c.d, 3 * c.d), nn.Linear(c.d, c.d)
        self.ff = nn.Sequential(nn.Linear(c.d, c.ff * c.d), nn.GELU(), nn.Linear(c.ff * c.d, c.d))
        self.h, self.drop = c.heads, c.dropout

    def forward(self, x, mask):
        B, L, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.drop if self.training else 0.0)
        x = x + F.dropout(self.proj(a.transpose(1, 2).reshape(B, L, D)), self.drop, self.training)
        return x + F.dropout(self.ff(self.ln2(x)), self.drop, self.training)


class LedgerFM(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.emb = nn.ModuleDict({k: nn.Embedding(v, c.d) for k, v in INT_FIELDS.items()})
        self.flt = nn.Linear(len(FLOAT_FIELDS), c.d)
        self.pos = nn.Embedding(c.max_len, c.d)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.ln = nn.LayerNorm(c.d)
        self.mask_heads = nn.ModuleDict({k: nn.Linear(c.d, INT_FIELDS[k]) for k in MASK_TARGETS})
        self.next_heads = nn.ModuleDict({k: nn.Linear(c.d, INT_FIELDS[k]) for k in NEXT_TARGETS})
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def encode(self, ints, floats, seg, pos):
        """ints [B,L,Fi] long, floats [B,L,Ff], seg [B,L] (0 = padding), pos [B,L]. Returns [B,L,d]."""
        x = sum(self.emb[k](ints[..., i]) for i, k in enumerate(FIELDS)) + self.flt(floats) + self.pos(pos)
        L = ints.shape[1]
        causal = torch.ones(L, L, dtype=torch.bool, device=ints.device).tril()
        same = (seg[:, :, None] == seg[:, None, :]) & (seg[:, :, None] > 0)
        mask = (same & causal)[:, None]
        mask = mask | torch.eye(L, dtype=torch.bool, device=ints.device)  # padding rows attend to self (no NaN)
        for b in self.blocks:
            x = b(x, mask)
        return self.ln(x)

    def pretrain_loss(self, ints, floats, seg, pos, p_mask=0.15):
        """Masked-field + next-transaction losses. Returns total, dict of parts."""
        valid = seg > 0
        m = (torch.rand(ints.shape[:2] + (len(MASK_TARGETS),), device=ints.device) < p_mask) & valid[..., None]
        ints_in, fl_in = ints.clone(), floats.clone()
        for j, k in enumerate(MASK_TARGETS):
            ints_in[..., FIDX[k]] = ints_in[..., FIDX[k]].masked_fill(m[..., j], 0)
            for fk in MASK_FLOAT_LEAK.get(k, []):
                fl_in[..., FLIDX[fk]] = fl_in[..., FLIDX[fk]].masked_fill(m[..., j], 0)
        h = self.encode(ints_in, fl_in, seg, pos)
        parts = {}
        for j, k in enumerate(MASK_TARGETS):
            tgt = ints[..., FIDX[k]]
            sel = m[..., j] & (tgt > 0)
            if sel.any():
                parts[f"m_{k}"] = F.cross_entropy(self.mask_heads[k](h[sel]).float(), tgt[sel])
        nxt = valid[:, :-1] & (seg[:, 1:] == seg[:, :-1])
        hn = h[:, :-1][nxt]
        for k in NEXT_TARGETS:
            tgt = ints[:, 1:, FIDX[k]][nxt]
            ok = (tgt > 0) if k != "flags" else torch.ones_like(tgt, dtype=torch.bool)
            if ok.any():
                parts[f"n_{k}"] = F.cross_entropy(self.next_heads[k](hn[ok]).float(), tgt[ok])
        return sum(parts.values()) / max(len(parts), 1), parts


def n_params(m):
    return sum(p.numel() for p in m.parameters())
