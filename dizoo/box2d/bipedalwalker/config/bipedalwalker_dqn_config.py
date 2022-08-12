from easydict import EasyDict

nstep = 3
bipedalwalker_dqn_config = dict(
    exp_name='bipedalwalker_dqn_test',
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
        each_dim_disc_size=5,
        discrete=True,
    ),
    policy=dict(
        # Whether to use cuda for network.
        cuda=True,
        model=dict(
            obs_shape=24,
            action_shape=int(5 ** 4),
            encoder_hidden_size_list=[512, 64],
            # Whether to use dueling head.
            dueling=True,
        ),
        # Reward's future discount factor, aka. gamma.
        discount_factor=0.99,
        # How many steps in td error.
        nstep=nstep,
        # learn_mode config
        learn=dict(
            update_per_collect=10,
            batch_size=64,
            learning_rate=0.001,
            # Frequency of target network update.
            target_update_freq=100,
        ),
        # collect_mode config
        collect=dict(
            # You can use either "n_sample" or "n_episode" in collector.collect.
            # Get "n_sample" samples per collect.
            n_sample=64,
            # Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
        ),
        # command_mode config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # Decay type. Support ['exp', 'linear'].
                type='exp',
                start=0.95,
                end=0.1,
                decay=50000,
            ),
            replay_buffer=dict(replay_buffer_size=100000, )
        ),
    ),
)
bipedalwalker_dqn_config = EasyDict(bipedalwalker_dqn_config)
main_config = bipedalwalker_dqn_config

bipedalwalker_dqn_create_config = dict(
    env=dict(
        type='bipedalwalker',
        import_names=['dizoo.box2d.bipedalwalker.envs.bipedalwalker_env'],
    ),
    # env_manager=dict(type='subprocess'),
    env_manager=dict(type='base'),
    policy=dict(type='dqn'),
)
bipedalwalker_dqn_create_config = EasyDict(bipedalwalker_dqn_create_config)
create_config = bipedalwalker_dqn_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline
    serial_pipeline([main_config, create_config], seed=0)
