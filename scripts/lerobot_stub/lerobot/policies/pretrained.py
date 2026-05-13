"""Minimal PreTrainedPolicy stub for lingbotvla eval."""
import torch.nn as nn


class PreTrainedPolicy(nn.Module):
    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = config
