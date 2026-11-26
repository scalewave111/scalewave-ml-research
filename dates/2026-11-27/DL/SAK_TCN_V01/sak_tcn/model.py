"""Causal multi-rate direction model and same-information controls."""
from dataclasses import asdict, dataclass
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    features: int = 146
    channels: int = 8
    kernel: int = 3
    dilations: tuple = (1, 2, 4, 8)
    rates: tuple = (1, 4)
    dropout: float = 0.1

    @property
    def steps(self):
        return 1 + 2 * (self.kernel - 1) * sum(self.dilations)

    def validate(self):
        if self.features < 1 or self.channels < 1 or self.kernel < 1:
            raise ValueError("Positive model dimensions required.")
        if not self.dilations or min(self.dilations) < 1:
            raise ValueError("Positive dilations required.")
        if not self.rates or self.rates[0] != 1 or min(self.rates) < 1:
            raise ValueError("Rates must begin with the one-minute branch.")
        if not 0 <= self.dropout < 1:
            raise ValueError("Invalid dropout.")


class CausalConv(nn.Conv1d):
    def __init__(self, channels, kernel, dilation):
        super().__init__(channels, channels, kernel, dilation=dilation)
        self.left = (kernel - 1) * dilation

    def forward(self, x):
        return super().forward(F.pad(x, (self.left, 0)))


class ResidualBlock(nn.Module):
    def __init__(self, config, dilation):
        super().__init__()
        self.c1 = CausalConv(config.channels, config.kernel, dilation)
        self.c2 = CausalConv(config.channels, config.kernel, dilation)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        h = self.drop(F.gelu(self.c1(x)))
        h = self.drop(F.gelu(self.c2(h)))
        return F.gelu(x + h)


class DirectionTCN(nn.Module):
    """Input [batch, rates, 2*features, history]; values then validity masks."""
    def __init__(self, config, prior_up=0.5):
        super().__init__()
        config.validate()
        self.config = config
        self.project = nn.Conv1d(2 * config.features, config.channels, 1)
        self.branches = nn.ModuleList([
            nn.Sequential(*[ResidualBlock(config, d) for d in config.dilations])
            for _ in config.rates
        ])
        self.head = nn.Linear(len(config.rates) * config.channels, 1)
        self.skip = nn.Linear(2 * config.features, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.skip.weight)
        nn.init.constant_(self.skip.bias, math.log(np.clip(prior_up, 1e-6, 1-1e-6) /
                                                 (1-np.clip(prior_up, 1e-6, 1-1e-6))))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (
            len(self.config.rates), 2*self.config.features, self.config.steps
        ):
            raise ValueError("Unexpected window shape.")
        h = [branch(F.gelu(self.project(x[:, i])))[:, :, -1]
             for i, branch in enumerate(self.branches)]
        return (self.head(torch.cat(h, dim=1)) + self.skip(x[:, 0, :, -1])).squeeze(-1)


class LinearControl(nn.Module):
    """Snapshot or pooled-history logistic control; identical input windows."""
    def __init__(self, config, mode, prior_up=0.5):
        super().__init__()
        if mode not in ("snapshot", "summary"):
            raise ValueError(mode)
        self.config, self.mode = config, mode
        width = 2*config.features
        if mode == "summary":
            width += 3*config.features*len(config.rates)
        self.linear = nn.Linear(width, 1)
        nn.init.zeros_(self.linear.weight)
        prior = np.clip(prior_up, 1e-6, 1-1e-6)
        nn.init.constant_(self.linear.bias, math.log(prior/(1-prior)))

    def forward(self, x):
        parts = [x[:, 0, :, -1]]
        if self.mode == "summary":
            n = self.config.features
            for i in range(len(self.config.rates)):
                values, masks = x[:, i, :n], x[:, i, n:]
                count = masks.sum(-1).clamp_min(1)
                mean = (values*masks).sum(-1)/count
                variance = ((values-mean.unsqueeze(-1)).square()*masks).sum(-1)/count
                parts.extend([mean, variance.clamp_min(0).sqrt(), masks.mean(-1)])
        return self.linear(torch.cat(parts, dim=1)).squeeze(-1)


def build_model(config, kind="tcn", prior_up=0.5):
    return DirectionTCN(config, prior_up) if kind == "tcn" else LinearControl(config, kind, prior_up)
