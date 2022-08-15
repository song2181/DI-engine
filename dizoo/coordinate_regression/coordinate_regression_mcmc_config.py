from easydict import EasyDict

cuda = True
multi_gpu = True

main_config = dict(
    exp_name='coor_regre_mcmc_multi_gpu',
    policy=dict(
        cuda=cuda,
        implicit=True,
        model=dict(
            obs_shape=[3, 96, 96],
            action_shape=2,
            cnn_block=[16, 32, 32],
            hidden_size=256,
            hidden_layer_num=2,
            implicit=True,
            stochastic_optim=dict(
                type='mcmc',
                cuda=cuda,
                train_samples=64,
                noise_scale=0.33,
                inference_samples=2 ** 11,
            )
        ),
        learn=dict(
            multi_gpu=multi_gpu,
            train_epoch=200,
            batch_size=8,
            optim=dict(learning_rate=1e-3, ),
            learner=dict(hook=dict(log_show_after_iter=10)),
        ),
        collect=dict(
            normalize_states=False,
            data_path=None,
        ),
        eval=dict(
            batch_size=1,
            evaluator=dict(eval_freq=10, multi_gpu=multi_gpu, cfg_type='MetricSerialEvaluatorDict', stop_value=None)
        ),
    ),
    train_dataset=dict(
        dataset_size=10,
        resolution=(96, 96),
        pixel_size=7,
        pixel_color=(0, 255, 0),
        seed=0,
    ),
    eval_dataset=dict(
        dataset_size=100,
        resolution=(96, 96),
        pixel_size=7,
        pixel_color=(0, 255, 0),
        seed=0,
    ),
    test_dataset=dict(
        dataset_size=500,
        resolution=(96, 96),
        pixel_size=7,
        pixel_color=(0, 255, 0),
        seed=0,
    )
)
main_config = EasyDict(main_config)
main_config = main_config
create_config = dict(
    policy=dict(
        type='ibc',
        import_names=['ding.policy.ibc'],
        model=dict(
            type='coordinate_model',
            import_names=['ding.model.template.coordinate_regression'],
        ),
    ),
)
create_config = EasyDict(create_config)
create_config = create_config


# if __name__ == "__main__":
#     from dizoo.coordinate_regression.entry import serial_pipeline_offline
#     main_config.exp_name = "coordinate_regression_mcmc_ts_" + str(main_config.train_dataset.dataset_size)+"-1"
#     serial_pipeline_offline([main_config, create_config], seed=0)
def train(args):
    from dizoo.coordinate_regression.entry import serial_pipeline_offline
    import copy
    main_config.train_dataset.dataset_size = args.train_data
    main_config.exp_name = "CoorReg_mcmc_ts" + str(args.train_data)
    serial_pipeline_offline([copy.deepcopy(main_config), copy.deepcopy(create_config)], seed=0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', '-t', type=int, default=10)
    parser.add_argument('--seed', '-s', type=int, default=0)
    # parser.add_argument('--multi_gpu', '-m', type=bool, default=False)
    args = parser.parse_args()
    train(args)