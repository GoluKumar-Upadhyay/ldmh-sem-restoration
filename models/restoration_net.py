"""
Model zoo for the ablation ladder (all share the SAME backbone):

  baseline        BaselineNet                                  -- backbone only
  film_control    DistributionMixtureRestorationNet(use_film=True, trained WITHOUT mixture NLL)
                  -> same extra parameters / FLOPs as the full model, but no
                     distribution modelling.  This is the missing capacity-matched control.
  ldmh_loss_only  DistributionMixtureRestorationNet(use_film=False)
                  -> NOTE: backbone never sees the LDMH output, so this is *functionally the
                     baseline* plus a disconnected auxiliary head.  Kept only as a seed-noise
                     sanity check (it must be statistically indistinguishable from baseline).
  full            DistributionMixtureRestorationNet(use_film=True) + mixture NLL (+ penalties)
"""
import torch.nn as nn
from models.backbone import RestorationBackboneLite
from models.ldmh import LocalDistributionMixtureHead, MixtureFiLMProjector


class BaselineNet(nn.Module):
    def __init__(self, base_ch=32, n_lr_blocks=4, n_hr_blocks=2, scale=2):
        super().__init__()
        self.backbone = RestorationBackboneLite(base_ch, n_lr_blocks, n_hr_blocks, scale)

    def forward(self, x):
        return self.backbone(x, film_gamma=None, film_shift=None)


class DistributionMixtureRestorationNet(nn.Module):
    def __init__(self, base_ch=32, n_components=3, n_lr_blocks=4, n_hr_blocks=2, scale=2,
                 use_film=True, beta_mode="sigmoid", init_scale=0.05, temperature=1.5):
        super().__init__()
        self.use_film = use_film
        self.ldmh = LocalDistributionMixtureHead(1, base_ch, n_components, beta_mode=beta_mode,
                                                 init_scale=init_scale, temperature=temperature)
        self.film_proj = MixtureFiLMProjector(n_components, base_ch)
        self.backbone = RestorationBackboneLite(base_ch, n_lr_blocks, n_hr_blocks, scale)

    def forward(self, x):
        mix_weights, beta, scale, _ = self.ldmh(x)
        if self.use_film:
            gamma, shift = self.film_proj(mix_weights)
        else:
            gamma, shift = None, None
        return self.backbone(x, gamma, shift), mix_weights, beta, scale
