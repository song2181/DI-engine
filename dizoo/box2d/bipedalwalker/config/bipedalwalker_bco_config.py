from easydict import EasyDict

bipedalwalker_bco_config = dict(
    exp_name='bipedalwalker_bco_rl0.001_noeps',
    env=dict(
        env_id='BipedalWalker-v3',
        collector_env_num=8,
        evaluator_env_num=5,
        # (bool) Scale output action into legal range.
        act_scale=True,
        n_evaluator_episode=5,
        stop_value=300,
        rew_clip=True,
        # The path to save the game replay
        replay_path=None,
    ),
    policy=dict(
        # Whether to use cuda for network.
        cuda=False,
        continuous=True,
        loss_type='l1_loss',
        model=dict(
            obs_shape=24,
            action_shape=4,
            # action_space='reparameterization',
            action_space='regression',
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
        ),
        learn=dict(
            train_epoch=200,  # If train_epoch is 1, the algorithm will be BCO(0)
            batch_size=128,
            learning_rate=0.001,
            weight_decay=1e-4,
        ),
        idm_learn=dict(
            idm_batch_size=256,
            idm_learning_rate=0.001,
            idm_train_epoch=10,
            idm_encoder_hidden_size_list=[60, 80, 100, 40],
        ),
        collect=dict(
            n_episode=100,
            # control the number (alpha*n_episode) of post-demonstration environment interactions at each iteration.
            # Notice: alpha * n_episode > collector_env_num
            alpha=0.1,
            model_path='/home/DI-engine/dizoo/box2d/bipedalwalker/config/expert_sac/ckpt_best.pth.tar',
            data_path='abs data path',
        ),
        eval=dict(evaluator=dict(eval_freq=40, )),
        other=dict(replay_buffer=dict(replay_buffer_size=100000, ), ),
    ),
)

bipedalwalker_bco_config = EasyDict(bipedalwalker_bco_config)
main_config = bipedalwalker_bco_config

bipedalwalker_bco_create_config = dict(
    env=dict(
        type='bipedalwalker',
        import_names=['dizoo.box2d.bipedalwalker.envs.bipedalwalker_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='bc'),
    collector=dict(type='episode'),
)
bipedalwalker_bco_create_config = EasyDict(bipedalwalker_bco_create_config)
create_config = bipedalwalker_bco_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c lunarlander_dqn_config.py -s 0`
    from ding.entry import serial_pipeline_bco
    from dizoo.box2d.bipedalwalker.config import bipedalwalker_sac_config, bipedalwalker_sac_create_config
    expert_main_config = bipedalwalker_sac_config
    expert_create_config = bipedalwalker_sac_create_config
    serial_pipeline_bco(
        [main_config, create_config], [expert_main_config, expert_create_config], seed=0, max_env_step=1000000
    )
