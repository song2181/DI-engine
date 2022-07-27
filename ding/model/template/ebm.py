"""Vanilla DFO and EBM are adapted from https://github.com/kevinzakka/ibc.
   MCMC is adapted from https://github.com/google-research/ibc. 
"""

from dataclasses import replace
from typing import Callable, Union, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod

from ding.utils import MODEL_REGISTRY, STOCHASTIC_OPTIMIZER_REGISTRY
from ding.torch_utils import unsqueeze_repeat, fold_batch, unfold_batch
from ding.torch_utils.network.gtrxl import PositionalEmbedding
from ..common import RegressionHead


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


@STOCHASTIC_OPTIMIZER_REGISTRY.register('dfo')
class DFO(StochasticOptimizer):

    def __init__(
        self,
        # action_bounds: np.ndarray,
        noise_scale: float = 0.1,
        noise_shrink: float = 0.5,
        iters: int = 3,
        train_samples: int = 256,
        inference_samples: int = 512,
        cuda: bool = False,
    ):
        # set later by `set_action_bounds`
        self.action_bounds = None
        self.noise_scale = noise_scale
        self.noise_shrink = noise_shrink
        self.iters = iters
        self.train_samples = train_samples
        self.inference_samples = inference_samples
        self.device = torch.device('cuda' if cuda else "cpu")

    def sample(self, obs: torch.Tensor, ebm: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create tiled observations and sample counter-negatives for feeding to the InfoNCE objective.
        obs: (B, O)
        return: (B, N, O), (B, N, A).
        """
        del ebm
        return self._sample(obs, self.train_samples)

    @torch.no_grad()
    def infer(self, obs: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """Optimize for the best action conditioned on the current observation.
        obs: (B, O)
        return: (B, A).
        """
        noise_scale = self.noise_scale

        # (B, N, O), (B, N, A)
        obs, action_samples = self._sample(obs, self.inference_samples)

        for i in range(self.iters):
            # (B, N)
            energies = ebm.forward(obs, action_samples)
            probs = F.softmax(-1.0 * energies, dim=-1)

            # Resample with replacement.
            idxs = torch.multinomial(probs, self.inference_samples, replacement=True)
            action_samples = action_samples[torch.arange(action_samples.size(0)).unsqueeze(-1), idxs]

            # Add noise and clip to target bounds.
            action_samples = action_samples + torch.randn_like(action_samples) * noise_scale
            #  clamp:argument 'min' and 'max' must be Number, not Tensor
            for j in range(action_samples.shape[-1]):
                action_samples[:, :, j] = action_samples[:, :, j].clamp(
                    min=self.action_bounds[0, j], max=self.action_bounds[1, j]
                )

            noise_scale *= self.noise_shrink

        # Return target with highest probability.
        return self._get_best_action_sample(obs, action_samples, ebm)


@STOCHASTIC_OPTIMIZER_REGISTRY.register('ardfo')
class AutoRegressiveDFO(DFO):

    def __init__(
        self,
        # action_bounds: np.ndarray,
        noise_scale: float = 0.1,
        noise_shrink: float = 0.5,
        iters: int = 3,
        train_samples: int = 256,
        inference_samples: int = 512,
        cuda: bool = False,
    ):
        super().__init__(noise_scale, noise_shrink, iters, train_samples, inference_samples, cuda)

    @torch.no_grad()
    def infer(self, obs: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """Optimize for the best action conditioned on the current observation.
        obs: (B, O)
        return: (B, A).
        """
        noise_scale = self.noise_scale

        obs, action_samples = self._sample(obs, self.inference_samples)

        for i in range(self.iters):

            # (B, N, A)
            energies = ebm.forward(obs, action_samples)
            probs = F.softmax(-1 * energies, dim=1)

            for j in range(energies.shape[-1]):
                # TODO: move `energies = ebm.forward(obs, action_samples)` into inner loop?
                _action_samples = action_samples[:, :, j]
                _probs = probs[:, :, j]
                _idxs = torch.multinomial(_probs, self.inference_samples, replacement=True)
                _action_samples = _action_samples[torch.arange(_action_samples.size(0)).unsqueeze(-1), _idxs]

                _action_samples = _action_samples + torch.randn_like(_action_samples) * noise_scale
                _action_samples = _action_samples.clamp(min=self.action_bounds[0, j], max=self.action_bounds[1, j])

                action_samples[:, :, j] = _action_samples

            noise_scale *= self.noise_shrink

        # (B, N, A)
        energies = ebm.forward(obs, action_samples)
        probs = F.softmax(-1 * energies, dim=1)
        # (B)
        best_idxs = probs[:, :, -1].argmax(dim=1)
        return action_samples[torch.arange(action_samples.size(0)), best_idxs, :]


@STOCHASTIC_OPTIMIZER_REGISTRY.register('mcmc')
class MCMC(StochasticOptimizer):

    class BaseScheduler(ABC):

        @abstractmethod
        def get_rate(self, index):
            raise NotImplementedError

    class ExponentialScheduler:
        """Exponential learning rate schedule for Langevin sampler."""

        def __init__(self, init, decay):
            self._decay = decay
            self._latest_lr = init

        def get_rate(self, index):
            """Get learning rate. Assumes calling sequentially."""
            del index
            lr = self._latest_lr
            self._latest_lr *= self._decay
            return lr

    class PolynomialScheduler:
        """Polynomial learning rate schedule for Langevin sampler."""

        def __init__(self, init, final, power, num_steps):
            self._init = init
            self._final = final
            self._power = power
            self._num_steps = num_steps

        def get_rate(self, index):
            """Get learning rate for index."""
            if index == -1:
                return self._init
            return (
                (self._init - self._final) * ((1 - (float(index) / float(self._num_steps - 1))) ** (self._power))
            ) + self._final

    def __init__(
        self,
        # action_bounds: np.ndarray,
        iters: int = 25,
        use_langevin_negative_samples: bool = False,
        train_samples: int = 256,
        inference_samples: int = 512,
        stepsize_scheduler: dict = dict(
            init=0.5,
            decay=0.8,
        ),
        optimize_again: bool = True,
        again_stepsize_scheduler: dict = dict(
            init=1e-4,
            final=1e-5,
            power=2.0,
            # num_steps,
        ),
        cuda: bool = False,
        ## langevin_step
        noise_scale: float = 0.5,
        grad_clip=None,
        delta_action_clip: float = 0.1,
        add_grad_penalty: bool = False,
        grad_norm_type: str = 'inf',
        grad_margin: float = 1.0,
        grad_loss_weight: float = 1.0,
        **kwargs,
    ):
        self.iters = iters
        self.use_langevin_negative_samples = use_langevin_negative_samples
        # TODO(zzh): multigpu pipeline, langevin negative sampling is slow on single gpu.
        assert not self.use_langevin_negative_samples, "MULTIGPU NotImplemented"
        self.train_samples = train_samples
        self.inference_samples = inference_samples
        self.stepsize_scheduler = stepsize_scheduler
        self.optimize_again = optimize_again
        self.again_stepsize_scheduler = again_stepsize_scheduler

        self.noise_scale = noise_scale
        self.grad_clip = grad_clip
        self.delta_action_clip = delta_action_clip
        self.add_grad_penalty = add_grad_penalty
        self.grad_norm_type = grad_norm_type
        self.grad_margin = grad_margin
        self.grad_loss_weight = grad_loss_weight
        self.device = torch.device('cuda' if cuda else "cpu")

    @staticmethod
    def _gradient_wrt_act(
            obs: torch.Tensor,
            action: torch.Tensor,
            ebm: nn.Module,
            is_train: bool = False,
    ) -> torch.Tensor:
        """
        Calculate gradient w.r.t action.
        obs: (B, N, O), action: (B, N, A)
        return: (B, N, A).
        """
        action = nn.Parameter(action)
        energy = ebm.forward(obs, action).sum()
        return torch.autograd.grad(energy, action, create_graph=is_train)[0]

    def grad_penalty(self, obs: torch.Tensor, action: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """
        Calculate grad_penalty.
        Make sure `torch.is_grad_enabled()==True` to calculate second order derivatives.
        obs: (B, N+1, O), action: (B, N+1, A)
        return: loss
        """
        if not self.add_grad_penalty:
            return torch.tensor(0.)
        # (B, N+1, A), this gradient is differentiable w.r.t model parameters
        de_dact = MCMC._gradient_wrt_act(obs, action, ebm, is_train=True)

        def compute_grad_norm(grad_norm_type, de_dact) -> torch.Tensor:
            # de_deact: B, N+1, A
            # return:   B, N+1
            grad_norm_type_to_ord = {
                '1': 1,
                '2': 2,
                'inf': float('inf'),
            }
            ord = grad_norm_type_to_ord[grad_norm_type]
            return torch.linalg.norm(de_dact, ord, dim=-1)

        # (B, N+1)
        grad_norms = compute_grad_norm(self.grad_norm_type, de_dact)
        grad_norms = grad_norms - self.grad_margin
        grad_norms = grad_norms.clamp(min=0., max=1e10)
        grad_norms = grad_norms.pow(2)

        grad_loss = grad_norms.mean()
        return grad_loss * self.grad_loss_weight

    # can not use @torch.no_grad() even during the inference
    # because we need to calculate gradient w.r.t inputs as MCMC updates.
    @no_ebm_grad()
    def _langevin_step(self, obs: torch.Tensor, action: torch.Tensor, stepsize: float, ebm: nn.Module) -> torch.Tensor:
        """
        Run one langevin MCMC step.
        obs: (B, N, O), action: (B, N, A)
        return: (B, N, A).
        """
        l_lambda = 1.0
        de_dact = MCMC._gradient_wrt_act(obs, action, ebm)

        if self.grad_clip:
            de_dact = de_dact.clamp(min=-self.grad_clip, max=self.grad_clip)

        gradient_scale = 0.5
        de_dact = (gradient_scale * l_lambda * de_dact + torch.randn_like(de_dact) * l_lambda * self.noise_scale)

        delta_action = stepsize * de_dact
        delta_action_clip = self.delta_action_clip * 0.5 * (self.action_bounds[1] - self.action_bounds[0])
        delta_action = delta_action.clamp(min=-delta_action_clip, max=delta_action_clip)

        action = action - delta_action
        action = action.clamp(min=self.action_bounds[0], max=self.action_bounds[1])

        return action

    @no_ebm_grad()
    def _langevin_action_gives_obs(
            self,
            obs: torch.Tensor,
            action: torch.Tensor,
            ebm: nn.Module,
            scheduler: BaseScheduler = None
    ) -> torch.Tensor:
        """
        Run langevin MCMC for `self.iters` steps.
        obs: (B, N, O), action: (B, N, A)
        return: (B, N, A)
        """
        if not scheduler:
            scheduler = MCMC.ExponentialScheduler(**self.stepsize_scheduler)
        stepsize = scheduler.get_rate(-1)
        for i in range(self.iters):
            action = self._langevin_step(obs, action, stepsize, ebm)
            stepsize = scheduler.get_rate(i)
        return action

    @no_ebm_grad()
    def sample(self, obs: torch.Tensor, ebm: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create tiled observations and sample counter-negatives for feeding to the InfoNCE objective.
        obs: (B, O)
        return: (B, N, O), (B, N, A)
        """
        obs, uniform_action_samples = self._sample(obs, self.train_samples)
        if not self.use_langevin_negative_samples:
            return obs, uniform_action_samples
        langevin_action_samples = self._langevin_action_gives_obs(obs, uniform_action_samples, ebm)
        return obs, langevin_action_samples

    @no_ebm_grad()
    def infer(self, obs: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """Optimize for the best action conditioned on the current observation.
        obs: (B, O)
        return: (B, A).
        """
        # (B, N, O), (B, N, A)
        obs, uniform_action_samples = self._sample(obs, self.inference_samples)
        action_samples = self._langevin_action_gives_obs(
            obs,
            uniform_action_samples,
            ebm,
        )

        # Run a second optimization, a trick for more precise inference
        if self.optimize_again:
            self.again_stepsize_scheduler['num_steps'] = self.iters
            action_samples = self._langevin_action_gives_obs(
                obs,
                action_samples,
                ebm,
                scheduler=MCMC.PolynomialScheduler(**self.again_stepsize_scheduler),
            )

        # action_samples: B, N, A
        return self._get_best_action_sample(obs, action_samples, ebm)


@MODEL_REGISTRY.register('ebm')
class EBM(nn.Module):

    def __init__(
        self,
        obs_shape: int,
        action_shape: int,
        hidden_size: int = 64,
        hidden_layer_num: int = 1,
        **kwargs,
    ):
        super().__init__()
        input_size = obs_shape + action_shape
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            RegressionHead(
                hidden_size,
                1,
                hidden_layer_num,
                final_tanh=False,
            )
        )

    def forward(self, obs, action):
        # obs: (B, N, O)
        # action: (B, N, A)
        # return: (B, N)
        x = torch.cat([obs, action], -1)
        x = self.net(x)
        return x['pred']


@MODEL_REGISTRY.register('arebm')
class AutoregressiveEBM(nn.Module):

    def __init__(
        self,
        obs_shape: int,
        action_shape: int,
        d_model: int = 64,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 64,
        cuda: bool = False,
        **kwargs,
    ):
        # treat obs_dim, and action_dim as sequence_dim
        super().__init__()
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = torch.device('cuda' if cuda else "cpu")
        self.obs_embed_layer = nn.Linear(1, d_model)
        self.action_embed_layer = nn.Linear(1, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.action_mask = self.transformer.generate_square_subsequent_mask(action_shape).to(self.device)
        self.output_layer = nn.Linear(d_model, 1)
        self._generate_positional_encoding(d_model)

    def _generate_positional_encoding(self, d_model):
        positional_encoding_layer = PositionalEmbedding(d_model)
        # batch_first
        self.obs_pe = positional_encoding_layer(PositionalEmbedding.generate_pos_seq(self.obs_shape)
                                                ).permute(1, 0, 2).contiguous().to(self.device)
        self.action_pe = positional_encoding_layer(PositionalEmbedding.generate_pos_seq(self.action_shape)
                                                   ).permute(1, 0, 2).contiguous().to(self.device)

    def forward(self, obs, action):
        # obs: (B, N, O)
        # action: (B, N, A)
        # return: (B, N, A)

        # obs: (B*N, O)
        # action: (B*N, A)
        obs, batch_dims = fold_batch(obs)
        action, _ = fold_batch(action)

        # obs: (B*N, O, 1)
        # action: (B*N, A, 1)
        # the second dimension (O, A) is now interpreted as sequence dimension
        # so that `obs`, `action` can be used as `src` and `tgt` to `nn.Transformer`
        # block with `batch_first=False`
        obs = self.obs_embed_layer(obs.unsqueeze(-1)) + self.obs_pe.to(obs.device)
        action = self.action_embed_layer(action.unsqueeze(-1)) + self.action_pe.to(obs.device)

        output = self.transformer(src=obs, tgt=action, tgt_mask=self.action_mask)

        # output: (B*N, A)
        output = self.output_layer(output).squeeze(-1)

        # output(energy): (B, N, A)
        output = unfold_batch(output, batch_dims)

        return output
