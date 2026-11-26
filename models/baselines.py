"""
Compact re-implementations of the external restoration architectures requested by the
reviewer (NAFNet, Restormer, SwinIR), adapted to this task: 1-channel LR input
[B,1,H,W] -> 1-channel x2 output [B,1,2H,2W] with a global bilinear residual and a
PixelShuffle upsampler.

IMPORTANT (state this in the paper): these are faithful-in-design but *re-implemented,
size-reduced* versions written for this repository, NOT the official code/weights.
They are trained from scratch with exactly the same split, augmentation, optimiser,
schedule, budget and checkpoint-selection rule as the proposed model.  For a
camera-ready comparison you may swap in the official repos through BASELINE_REGISTRY
(same call signature), and report both.

Sizes are chosen so that 'small' ~ comparable to the proposed model's budget and
'base' is a stronger, larger reference; parameters / FLOPs are always reported.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- common
class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(c)); self.b = nn.Parameter(torch.zeros(c)); self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.w.view(1, -1, 1, 1) + self.b.view(1, -1, 1, 1)


class _SRWrap(nn.Module):
    """stem -> body(LR) -> PixelShuffle x2 -> head, + bilinear global residual."""
    def __init__(self, body, width, scale=2):
        super().__init__()
        self.scale = scale
        self.stem = nn.Conv2d(1, width, 3, padding=1)
        self.body = body
        self.up = nn.Sequential(nn.Conv2d(width, width * scale * scale, 3, padding=1), nn.PixelShuffle(scale))
        self.head = nn.Conv2d(width, 1, 3, padding=1)

    def forward(self, x):
        f = self.stem(x)
        f = self.body(f)
        out = self.head(F.gelu(self.up(f)))
        return F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False) + out


# ----------------------------------------------------------------------------- NAFNet
class _NAFBlock(nn.Module):
    """Official NAFNet block (LayerNorm2d, SimpleGate, Simplified Channel Attention)."""
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw = c * dw_expand
        self.n1 = LayerNorm2d(c)
        self.c1 = nn.Conv2d(c, dw, 1); self.c2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        self.c3 = nn.Conv2d(dw // 2, c, 1)
        ffn = c * ffn_expand
        self.n2 = LayerNorm2d(c)
        self.c4 = nn.Conv2d(c, ffn, 1); self.c5 = nn.Conv2d(ffn // 2, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1)); self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    @staticmethod
    def _gate(x):
        a, b = x.chunk(2, 1); return a * b

    def forward(self, x):
        y = self._gate(self.c2(self.c1(self.n1(x))))
        y = self.c3(y * self.sca(y))
        x = x + y * self.beta
        y = self.c5(self._gate(self.c4(self.n2(x))))
        return x + y * self.gamma


class _NAFUNet(nn.Module):
    def __init__(self, w, enc=(1, 1, 2), mid=2, dec=(1, 1, 1)):
        super().__init__()
        self.enc, self.dec, self.down, self.up = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        c = w
        for n in enc:
            self.enc.append(nn.Sequential(*[_NAFBlock(c) for _ in range(n)]))
            self.down.append(nn.Conv2d(c, 2 * c, 2, 2)); c *= 2
        self.mid = nn.Sequential(*[_NAFBlock(c) for _ in range(mid)])
        for n in dec:
            self.up.append(nn.Sequential(nn.Conv2d(c, c * 2, 1, bias=False), nn.PixelShuffle(2))); c //= 2
            self.dec.append(nn.Sequential(*[_NAFBlock(c) for _ in range(n)]))
        self.levels = len(enc)

    def forward(self, x):
        H, W = x.shape[-2:]
        m = 2 ** self.levels
        ph, pw = (m - H % m) % m, (m - W % m) % m
        x = F.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x
        skips = []
        for e, d in zip(self.enc, self.down):
            x = e(x); skips.append(x); x = d(x)
        x = self.mid(x)
        for u, d, s in zip(self.up, self.dec, reversed(skips)):
            x = d(u(x) + s)
        return x[..., :H, :W]


def NAFNetSR(size="small"):
    w, enc, mid, dec = {"small": (16, (1, 1), 2, (1, 1)), "base": (24, (1, 1), 2, (1, 1))}[size]
    return _SRWrap(_NAFUNet(w, enc, mid, dec), w)


# ----------------------------------------------------------------------------- Restormer
class _MDTA(nn.Module):
    def __init__(self, c, heads):
        super().__init__()
        self.h = heads; self.t = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(c, c * 3, 1, bias=False)
        self.dw = nn.Conv2d(c * 3, c * 3, 3, padding=1, groups=c * 3, bias=False)
        self.out = nn.Conv2d(c, c, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.dw(self.qkv(x)).chunk(3, 1)
        q, k, v = [t.reshape(B, self.h, C // self.h, H * W) for t in (q, k, v)]
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        a = (q @ k.transpose(-2, -1)) * self.t
        return self.out((a.softmax(-1) @ v).reshape(B, C, H, W))


class _GDFN(nn.Module):
    def __init__(self, c, e=2.0):
        super().__init__()
        h = int(c * e)
        self.pi = nn.Conv2d(c, h * 2, 1, bias=False)
        self.dw = nn.Conv2d(h * 2, h * 2, 3, padding=1, groups=h * 2, bias=False)
        self.po = nn.Conv2d(h, c, 1, bias=False)

    def forward(self, x):
        a, b = self.dw(self.pi(x)).chunk(2, 1)
        return self.po(F.gelu(a) * b)


class _RBlock(nn.Module):
    def __init__(self, c, heads):
        super().__init__()
        self.n1, self.n2 = LayerNorm2d(c), LayerNorm2d(c)
        self.attn, self.ffn = _MDTA(c, heads), _GDFN(c)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.ffn(self.n2(x))


def RestormerSR(size="small"):
    w, heads, n = {"small": (32, 2, 6), "base": (48, 4, 8)}[size]
    return _SRWrap(nn.Sequential(*[_RBlock(w, heads) for _ in range(n)]), w)


# ----------------------------------------------------------------------------- SwinIR
def _win_part(x, ws):
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)


def _win_rev(w, ws, H, W):
    B = w.shape[0] // ((H // ws) * (W // ws))
    x = w.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)


class _SwinBlock(nn.Module):
    def __init__(self, c, heads, ws, shift):
        super().__init__()
        self.ws, self.shift, self.h = ws, shift, heads
        self.n1, self.n2 = nn.LayerNorm(c), nn.LayerNorm(c)
        self.qkv, self.proj = nn.Linear(c, 3 * c), nn.Linear(c, c)
        self.mlp = nn.Sequential(nn.Linear(c, 2 * c), nn.GELU(), nn.Linear(2 * c, c))
        self.rpb = nn.Parameter(torch.zeros((2 * ws - 1) ** 2, heads))
        nn.init.trunc_normal_(self.rpb, std=0.02)
        co = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij")).flatten(1)
        rel = (co[:, :, None] - co[:, None, :]).permute(1, 2, 0) + (ws - 1)
        self.register_buffer("rpi", rel[..., 0] * (2 * ws - 1) + rel[..., 1], persistent=False)

    def _mask(self, H, W, dev):
        ws, s = self.ws, self.shift
        img = torch.zeros(1, H, W, 1, device=dev); cnt = 0
        for hs in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
            for wsl in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
                img[:, hs, wsl, :] = cnt; cnt += 1
        mw = _win_part(img, ws).squeeze(-1)
        m = mw[:, None, :] - mw[:, :, None]
        return m.masked_fill(m != 0, -100.0).masked_fill(m == 0, 0.0)

    def forward(self, x, H, W):          # x: [B,H*W,C]
        B, L, C = x.shape
        y = self.n1(x).view(B, H, W, C)
        if self.shift:
            y = torch.roll(y, (-self.shift, -self.shift), (1, 2))
        w = _win_part(y, self.ws)
        N = self.ws * self.ws
        qkv = self.qkv(w).reshape(-1, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * (C // self.h) ** -0.5, qkv[1], qkv[2]
        a = q @ k.transpose(-2, -1)
        a = a + self.rpb[self.rpi.view(-1)].view(N, N, -1).permute(2, 0, 1).unsqueeze(0)
        if self.shift:
            m = self._mask(H, W, x.device)
            nW = m.shape[0]
            a = (a.view(-1, nW, self.h, N, N) + m.unsqueeze(1).unsqueeze(0).to(a.dtype)).view(-1, self.h, N, N)
        o = (a.softmax(-1) @ v).transpose(1, 2).reshape(-1, N, C)
        y = _win_rev(self.proj(o), self.ws, H, W)
        if self.shift:
            y = torch.roll(y, (self.shift, self.shift), (1, 2))
        x = x + y.reshape(B, L, C)
        return x + self.mlp(self.n2(x))


class _SwinBody(nn.Module):
    def __init__(self, c, heads, depth, ws):
        super().__init__()
        self.ws = ws
        self.blocks = nn.ModuleList([_SwinBlock(c, heads, ws, 0 if i % 2 == 0 else ws // 2) for i in range(depth)])
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, f):
        B, C, H, W = f.shape
        ph, pw = (self.ws - H % self.ws) % self.ws, (self.ws - W % self.ws) % self.ws
        g = F.pad(f, (0, pw, 0, ph), mode="reflect") if (ph or pw) else f
        Hp, Wp = g.shape[-2:]
        t = g.flatten(2).transpose(1, 2)
        for b in self.blocks:
            t = b(t, Hp, Wp)
        t = t.transpose(1, 2).reshape(B, C, Hp, Wp)[..., :H, :W]
        return f + self.conv(t)          # residual Swin transformer block (RSTB) style


def SwinIRSR(size="small", window=8):
    c, heads, depth = {"small": (32, 2, 6), "base": (48, 4, 8)}[size]
    return _SRWrap(_SwinBody(c, heads, depth, window), c)


BASELINE_REGISTRY = {
    "NAFNet-small": lambda: NAFNetSR("small"), "NAFNet-base": lambda: NAFNetSR("base"),
    "Restormer-small": lambda: RestormerSR("small"), "Restormer-base": lambda: RestormerSR("base"),
    "SwinIR-small": lambda: SwinIRSR("small"), "SwinIR-base": lambda: SwinIRSR("base"),
}
