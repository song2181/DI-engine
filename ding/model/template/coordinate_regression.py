"""Explicit BC and implicit BC model.
   Vanilla DFO and EBM are adapted from https://github.com/kevinzakka/ibc.
   MCMC is adapted from https://github.com/google-research/ibc. 
"""

from ast import IsNot
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


class StochasticOptimizer(ABC):

    def _sample(self, obs: torch.Tensor, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper method for drawing action samples from the uniform random distribution
        and tiling observations to the same shape as action samples.
        obs: (B, O)
        return: (B, N, O), (B, N, A).
        """
        # TODO: do not use np.random
        action_bounds = self.action_bounds.cpu().numpy()
        size = (obs.shape[0], num_samples, action_bounds.shape[1])
        action_samples = np.random.uniform(action_bounds[0, :], action_bounds[1, :], size=size)
        action_samples = torch.as_tensor(action_samples, dtype=torch.float32).to(self.device)
        tiled_obs = unsqueeze_repeat(obs, num_samples, 1)
        return tiled_obs, action_samples

    @staticmethod
    @torch.no_grad()
    def _get_best_action_sample(obs: torch.Tensor, action_samples: torch.Tensor, ebm: nn.Module):
        """Return target with highest probability (lowest energy).
        obs: (B, N, O), action_samples: (B, N, A)
        return: (B, A).
        """
        # (B, N)
        energies = ebm.forward(obs, action_samples)
        probs = F.softmax(-1.0 * energies, dim=-1)
        # (B, )
        best_idxs = probs.argmax(dim=-1)
        return action_samples[torch.arange(action_samples.size(0)), best_idxs, :]

    def set_action_bounds(self, action_bounds: np.ndarray):
        self.action_bounds = torch.as_tensor(action_bounds, dtype=torch.float32).to(self.device)

    @abstractmethod
    def sample(self, obs: torch.Tensor, ebm: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create tiled observations and sample counter-negatives for feeding to the InfoNCE objective.
        obs: (B, O)
        return: (B, N, O), (B, N, A).
        """
        raise NotImplementedError

    @abstractmethod
    def infer(self, obs: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """Optimize for the best action conditioned on the current observation.
        obs: (B, O)
        return: (B, A).
        """
        raise NotImplementedError


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
class CoordModel(nn.Module):

    def __init__(
        self,
        obs_shape: Union[int, SequenceType],
        action_shape: Union[int, SequenceType],
        hidden_size: int = 64,
        hidden_layer_num: int = 1,
        cnn_block: Union[int, SequenceType] = [16],
        implicit: Boolean = False,
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
        if implicit:
            input_size = 16 * 2 + self.action_shape
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_size), nn.ReLU(),
                RegressionHead(
                    hidden_size,
                    1,
                    hidden_layer_num,
                    final_tanh=False,
                )
            )
        else:
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

    def forward(self, obs: torch.Tensor, action: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # obs: (B, O)
        # action: (B, A)
        if self.implicit:
            assert action is not None, "For implicit BC, action can not be none."
            B, N = obs.shape[0], obs.shape[1]
            obs = obs.reshape((B * N, ) + self.obs_shape)
            action = action.reshape(B * N, self.action_shape)
            x = self.cnnnet(obs)
            x = self.activate(self.conv(x))
            x = self.reducer(x)
            x = torch.cat([x, action], -1)
            x = self.net(x)
            x['pred'] = x['pred'].reshape(B, N)
            return x['pred']
        else:
            x = self.cnnnet(obs)
            x = self.activate(self.conv(x))
            x = self.reducer(x)
            x = self.net(x)
            return {'action': x['pred']}


@MODEL_REGISTRY.register('coord_arebm')
class AutoregressiveEBM(nn.Module):

    def __init__(
        self,
        obs_shape: int,
        action_shape: int,
        hidden_size: int = 512,
        hidden_layer_num: int = 4,
        cnn_block: Union[int, SequenceType] = [16],
        implicit: Boolean = False,
        **kwargs,
    ):
        super().__init__()
        self._implicit = implicit
        self.ebm_list = nn.ModuleList()
        for i in range(action_shape):
            self.ebm_list.append(CoordModel(obs_shape, i + 1, hidden_size, hidden_layer_num, cnn_block, implicit))

    def forward(self, obs, action):
        # obs: (B, N, O)
        # action: (B, N, A)
        # return: (B, N, A)

        # (B, N)
        output_list = []
        for i, ebm in enumerate(self.ebm_list):
            output_list.append(ebm(obs, action[..., :i + 1]))
        # (B, N, A)
        return torch.stack(output_list, axis=-1)
