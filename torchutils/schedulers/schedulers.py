import math

from typing import List

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


__all__ = ['WarmupCosineAnnealingLR']


class CosineAnnealingLinearWarmup(_LRScheduler):
    """
    Implements a warmup cosine annealing learning rate scheduler.

    Attributes:
        warmup_epochs (int): The number of warmup epochs.
        max_epochs (int): The total number of epochs.
        initial_lr (float): The initial learning rate.
        cosine_annealing_epochs (int): The number of epochs for the cosine annealing phase.

    Methods:
        get_lr(): Get the learning rate for each parameter group.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        initial_lr: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        """
        Initialize a new WarmupCosineAnnealingLR instance.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
            warmup_epochs (int): The number of warmup epochs.
            max_epochs (int): The total number of epochs.
            initial_lr (float, optional): The initial learning rate. Default is 1e-6.
            last_epoch (int, optional): The index of the last epoch. Default is -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.initial_lr = initial_lr
        self.cosine_annealing_epochs = self.max_epochs - self.warmup_epochs

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Get the learning rate for each parameter group.

        Returns:
            List[float]: The learning rate for each parameter group.
        """
        if self.last_epoch < self.warmup_epochs:
            return [self.initial_lr + (base_lr - self.initial_lr) * (self.last_epoch) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            cos_input = math.pi * (self.last_epoch - self.warmup_epochs) / self.cosine_annealing_epochs
            return [base_lr * (1 + math.cos(cos_input)) / 2 for base_lr in self.base_lrs]
