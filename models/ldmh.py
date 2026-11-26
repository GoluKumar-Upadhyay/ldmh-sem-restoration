"""
Local Distribution Mixture Head (LDMH) -- v2.

Changes vs. the submitted version (each one addresses a finding of the audit):
  * beta_mode: 'sigmoid' (corrected, default) or 'clamp' (the original, buggy
    parameterisation) so the collapse claim can be tested *experimentally*
    instead of asserted.
  * init_scale: the prototype scales are initialised near the empirical residual
    scale (the old softplus(0)=0.69 is ~15x larger than the residual std).
  * NLL no longer uses `(r/scale).clamp(min=eps)`: that clamp is conceptually the same defect
    (zero gradient inside |r| < eps*scale) although its measure is negligible in practice
    (see notebook 02).  Replaced by an additive eps; the NLL is evaluated in float32 outside
    autocast (lgamma/pow are not safe in fp16).
  * `param_groups()` exposes the prototype parameters so weight decay is NOT
    applied to log_beta / log_scale.
  * `beta_hit_bound` diagnostic: reports whether a beta sits against beta_max /
    beta_min (sigmoid saturation), which the original paper did not check.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalDistributionMixtureHead(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, n_components=3, beta_min=0.3, beta_max=2.5,
                 temperature=1.5, beta_mode="sigmoid", init_scale=0.05):
        super().__init__()
        assert beta_mode in ("sigmoid", "clamp")
        self.K = n_components
        self.beta_min, self.beta_max = beta_min, beta_max
        self.T = temperature
        self.beta_mode = beta_mode
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1), nn.GroupNorm(8, base_ch), nn.GELU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.GroupNorm(8, base_ch), nn.GELU(),
        )
        self.weight_head = nn.Conv2d(base_ch, n_components, 1)

        init_betas = torch.linspace(beta_min + 0.1, beta_max - 0.1, n_components)
        if beta_mode == "sigmoid":
            frac = (init_betas - beta_min) / (beta_max - beta_min)
            self.log_beta = nn.Parameter(torch.logit(frac))
        else:  # original: beta = clamp(exp(l))
            self.log_beta = nn.Parameter(torch.log(init_betas))
        # inverse softplus so that softplus(log_scale) + 1e-3 == init_scale
        s = max(init_scale - 1e-3, 1e-4)
        self.log_scale = nn.Parameter(torch.full((n_components,), math.log(math.expm1(s))))

    # ------------------------------------------------------------------
    def betas(self):
        if self.beta_mode == "sigmoid":
            return self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(self.log_beta)
        return torch.exp(self.log_beta).clamp(self.beta_min, self.beta_max)

    def forward(self, x):
        feat = self.encoder(x)
        logits = self.weight_head(feat)
        mix_weights = F.softmax(logits / self.T, dim=1)       # Eq. (3): temperature T
        beta = self.betas()
        scale = F.softplus(self.log_scale) + 1e-3
        return mix_weights, beta, scale, feat

    def proto_params(self):
        return [self.log_beta, self.log_scale]

    @torch.no_grad()
    def get_component_stats(self, mix_weights, beta, scale):
        K = beta.shape[0]
        stats = {"beta": beta.detach().cpu().tolist(), "scale": scale.detach().cpu().tolist()}
        stats["mean_usage"] = mix_weights.mean(dim=(0, 2, 3)).detach().cpu().tolist()
        dom = mix_weights.argmax(dim=1)
        stats["dominant_frac"] = (torch.bincount(dom.flatten(), minlength=K).float() / dom.numel()).cpu().tolist()
        stats["min_beta_pairwise_dist"] = torch.pdist(beta.unsqueeze(1)).min().item() if K > 1 else float("nan")
        stats["beta_hit_bound"] = [bool((b - self.beta_min) < 0.01 or (self.beta_max - b) < 0.01) for b in beta.tolist()]
        return stats


def mixture_gen_gaussian_nll(residual, mix_weights, beta, scale, eps=1e-6):
    """Mixture of zero-mean generalised Gaussians, evaluated in fp32."""
    with torch.autocast(device_type=residual.device.type, enabled=False):
        residual, mix_weights = residual.float(), mix_weights.float()
        beta, scale = beta.float(), scale.float()
        K = beta.shape[0]
        r = residual.abs()
        if r.dim() == 3:
            r = r.unsqueeze(1)
        b = beta.view(1, K, 1, 1)
        s = scale.view(1, K, 1, 1)
        log_norm = torch.log(b) - torch.log(2 * s) - torch.lgamma(1.0 / b)
        log_comp = log_norm - ((r + eps) / s).pow(b)          # no gradient-killing clamp
        log_mix = torch.logsumexp(torch.log(mix_weights + eps) + log_comp, dim=1)
        return -log_mix.mean()


def beta_diversity_penalty(beta, margin=0.3):
    if beta.shape[0] < 2:
        return beta.new_zeros(())
    return F.relu(margin - torch.pdist(beta.unsqueeze(1))).mean()


def load_balance_penalty_entropy(mix_weights):
    K = mix_weights.shape[1]
    u = mix_weights.float().mean(dim=(0, 2, 3)).clamp(min=1e-8)
    entropy = -(u * u.log()).sum()
    return math.log(K) - entropy


class MixtureFiLMProjector(nn.Module):
    def __init__(self, n_components, feat_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_components, feat_ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(feat_ch, feat_ch * 2, 3, padding=1),
        )

    def forward(self, mix_weights):
        gamma, shift = self.net(mix_weights).chunk(2, dim=1)
        return gamma, shift
