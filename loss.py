import torch
import torch.nn as nn


class ClippedL2Loss(nn.Module):
    def __init__(self, delta: float):
        """
        Clipped L2 loss: behaves like L2 loss when |x - y| < delta,
        but takes a constant value outside.

        Args:
            delta (float): The threshold value.
        """
        super().__init__()
        self.delta = delta

    def forward(self, x, y):
        diff = torch.abs(x - y)
        loss = torch.where(
            diff < self.delta,
            0.5 * diff**2,  # Quadratic inside delta
            0.5 * self.delta**2,  # Constant outside
        )
        return loss.mean()
