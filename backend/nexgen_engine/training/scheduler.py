from __future__ import annotations

import math


class WarmRestartScheduler:
    def __init__(self, base_lr: float = 1e-3, min_lr: float = 1e-6, t_0: int = 10, t_mult: int = 2, warmup_steps: int = 1000) -> None:
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.t_0 = t_0
        self.t_mult = t_mult
        self.warmup_steps = warmup_steps

    def learning_rate(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        cycle_length = self.t_0
        cycle_step = step - self.warmup_steps
        while cycle_step >= cycle_length:
            cycle_step -= cycle_length
            cycle_length *= self.t_mult
        cosine = 0.5 * (1.0 + math.cos(math.pi * cycle_step / cycle_length))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine
