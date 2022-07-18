from re import X
from typing import Union, Optional, List, Any, Tuple
import os
import torch
from tqdm import tqdm
from functools import partial
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, MetricSerialEvaluator, IMetric
from ding.config import read_config, compile_config
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ding.utils.data import create_dataset
from dizoo.coordinate_regression.dataset import CoordinateRegression
from ding.policy.ibc import IBCPolicy
from ding.model.template.ebm import EBM
from dizoo.coordinate_regression.plot import plot, eval


class CoordinateRegressionMetric(IMetric):

    def __init__(self) -> None:
        self.loss = nn.MSELoss()

    def eval(self, inputs: dict, label: torch.Tensor) -> dict:
        """
        Returns:
            - eval_result (:obj:`dict`): {'loss': xxx, 'acc1': xxx, 'acc5': xxx}
        """
        loss = self.loss(inputs['action'], label)
        return {'loss': loss.item()}

    def reduce_mean(self, inputs: List[dict]) -> dict:
        L = len(inputs)
        output = {}
        for k in inputs[0].keys():
            output[k] = sum([t[k] for t in inputs]) / L
        return output

    def gt(self, metric1: dict, metric2: dict) -> bool:
        if metric2 is None:
            return True
        for k in metric1:
            if metric1[k] < metric2[k]:
                return False
        return True


def serial_pipeline_offline(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        max_train_iter: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline entry.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[List[Any]]`): A list with 3 elements: \
            ``BaseEnv`` subclass, collector env config, and evaluator env config.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - max_train_iter (:obj:`Optional[int]`): Maximum policy update iterations in training.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = input_cfg
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)

    # Dataset
    train_dataset = CoordinateRegression(cfg.train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        cfg.policy.learn.batch_size,
        shuffle=False,
        collate_fn=lambda x: x,
        pin_memory=cfg.policy.cuda,
    )
    test_dataset = CoordinateRegression(cfg.test_dataset)
    test_dataset.exclude(train_dataset.coordinates)
    test_dataloader = DataLoader(
        test_dataset,
        cfg.policy.eval.batch_size,
        # cfg.test_dataset.dataset_size,
        shuffle=True,
        pin_memory=cfg.policy.cuda,
    )
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'eval'])
    # Normalization for state in offlineRL dataset.
    # if cfg.policy.collect.get('normalize_states', None):
    #     policy.set_norm_statistics({'action_bounds': train_dataset.get_target_bounds})
    policy._stochastic_optimizer.set_action_bounds(train_dataset.get_target_bounds())

    # Main components
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    eval_metric = CoordinateRegressionMetric()
    evaluator = MetricSerialEvaluator(
        cfg.policy.eval.evaluator, [test_dataloader, eval_metric], policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    # evaluator = InteractionSerialEvaluator(
    #     cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    # )
    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    learner.call_hook('before_run')
    stop = False

    for epoch in tqdm(range(cfg.policy.learn.train_epoch)):
        # Evaluate policy per epoch

        # stop, reward = evaluator.eval(learner.save_checkpoint, epoch, 0)

        for train_data in tqdm(train_dataloader):
            learner.train(train_data)

        for test_data in tqdm(test_dataloader):
            stop, reward, output = evaluator.eval(learner.save_checkpoint, learner.train_iter)

    policy.eval_mode.reset()
    test_coords = test_dataset.coordinates
    train_coords = train_dataset.coordinates
    train_dataloader = DataLoader(
        train_dataset,
        cfg.policy.learn.batch_size,
        shuffle=False,
        # collate_fn=lambda x: x,
        pin_memory=cfg.policy.cuda,
    )
    test_dataloader = DataLoader(
        test_dataset,
        # cfg.policy.learn.batch_size,
        cfg.test_dataset.dataset_size,
        shuffle=True,
        pin_memory=cfg.policy.cuda,
    )
    errors = eval(train_dataloader, test_dataloader, policy.eval_mode)
    plot(train_coords, test_coords, errors, cfg.test_dataset.resolution, 'img.png', 100, 140)
    learner.call_hook('after_run')
    return policy, stop
