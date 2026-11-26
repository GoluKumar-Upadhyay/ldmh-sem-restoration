"""Fixed (non-adaptive) reconstruction losses used as baselines."""
import torch


def charbonnier_loss(pred, target, eps=1e-3):
    return torch.sqrt((pred - target) ** 2 + eps ** 2).mean()


def gen_charbonnier_loss(pred, target, p=0.845, eps=1e-3):
    return ((pred - target) ** 2 + eps ** 2).pow(p / 2).mean()
