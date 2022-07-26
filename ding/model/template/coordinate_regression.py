"""Vanilla DFO and EBM are adapted from https://github.com/kevinzakka/ibc.
   MCMC is adapted from https://github.com/google-research/ibc. 
"""

from dataclasses import replace
from typing import Callable, Union, Dict, Optional, Tuple
from xmlrpc.client import Boolean

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod

from ding.utils import MODEL_REGISTRY, STOCHASTIC_OPTIMIZER_REGISTRY, SequenceType, squeeze
from ding.torch_utils import unsqueeze_repeat, fold_batch, unfold_batch
from ding.torch_utils.network.gtrxl import PositionalEmbedding
from ..common import RegressionHead, ConvEncoder


def create_stochastic_optimizer(stochastic_optimizer_config):
    return STOCHASTIC_OPTIMIZER_REGISTRY.build(stochastic_optimizer_config.pop("type"), **stochastic_optimizer_config)


def no_ebm_grad():
    """Wrapper that disables energy based model gradients"""

    def ebm_disable_grad_wrapper(func: Callable):

        def wrapper(*args, **kwargs):
            # make sure ebm is the last positional arguments
            ebm = args[-1]
            ebm.requires_grad_(False)
            result = func(*args, **kwargs)
            ebm.requires_grad_(True)
            return result

        return wrapper

    return ebm_disable_grad_wrapper


class SpatialSoftArgmax(nn.Module):
    """Spatial softmax as defined in https://arxiv.org/abs/1504.00702.
    Concretely, the spatial softmax of each feature map is used to compute a weighted
    mean of the pixel locations, effectively performing a soft arg-max over the feature
    dimension.
    """

    def __init__(self, normalize: bool = True) -> None:
        super().__init__()

        self.normalize = normalize

    def _coord_grid(
            self,
            h: int,
            w: int,
            device: torch.device,
    ) -> torch.Tensor:
        if self.normalize:
            return torch.stack(
                torch.meshgrid(
                    torch.linspace(-1, 1, w, device=device),
                    torch.linspace(-1, 1, h, device=device),
                )
            )
        return torch.stack(torch.meshgrid(
            torch.arange(0, w, device=device),
            torch.arange(0, h, device=device),
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, "Expecting a tensor of shape (B, C, H, W)."

        # Compute a spatial softmax over the input:
        # Given an input of shape (B, C, H, W), reshape it to (B*C, H*W) then apply the
        # softmax operator over the last dimension.
        _, c, h, w = x.shape
        softmax = F.softmax(x.view(-1, h * w), dim=-1)

        # Create a meshgrid of normalized pixel coordinates.
        xc, yc = self._coord_grid(h, w, x.device)

        # Element-wise multiply the x and y coordinates with the softmax, then sum over
        # the h*w dimension. This effectively computes the weighted mean x and y
        # locations.
        x_mean = (softmax * xc.flatten()).sum(dim=1, keepdims=True)
        y_mean = (softmax * yc.flatten()).sum(dim=1, keepdims=True)

        # Concatenate and reshape the result to (B, C*2) where for every feature we have
        # the expected x and y pixel locations.
        return torch.cat([x_mean, y_mean], dim=1).view(-1, c * 2)


class ResidualBlock(nn.Module):

    def __init__(
            self,
            depth: int,
            activation_fn: Optional[nn.Module] = nn.ReLU(),
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(depth, depth, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(depth, depth, 3, padding=1, bias=True)
        self.activation = activation_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(x)
        out = self.conv2(out)
        return out + x


@MODEL_REGISTRY.register('coordinate_model')
class EBM(nn.Module):

    def __init__(
        self,
        obs_shape: Union[int, SequenceType],
        action_shape: Union[int, SequenceType],
        hidden_size: int = 64,
        hidden_layer_num: int = 1,
        cnn_block: Union[int, SequenceType] = [16],
        implicit: Boolean = True,
        **kwargs,
    ):
        super().__init__()
        self.implicit = implicit
        self.obs_shape, self.action_shape = squeeze(obs_shape), squeeze(action_shape)
        layers = []
        depth_in = self.obs_shape[0]
        for depth_out in cnn_block:
            layers.extend([
                nn.Conv2d(depth_in, depth_out, 3, padding=1),
                ResidualBlock(depth_out),
            ])
            depth_in = depth_out

        self.cnnnet = nn.Sequential(*layers)
        self.activate = nn.ReLU()
        self.conv = nn.Conv2d(cnn_block[-1], 16, 1)
        self.reducer = SpatialSoftArgmax()
        input_size = 16 * 2
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            RegressionHead(
                hidden_size,
                2,
                hidden_layer_num,
                final_tanh=False,
            )
        )

    def forward(self, obs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # obs: (B, O)
        # action: (B, A)
        x = self.cnnnet(obs)
        x = self.activate(self.conv(x))
        x = self.reducer(x)
        x = self.net(x)
        return {'action': x['pred']}
