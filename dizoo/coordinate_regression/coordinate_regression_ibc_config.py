from easydict import EasyDict

cuda = True
multi_gpu = False

main_config = dict(
    policy=dict(
        cuda=False,
        implicit=True,
        model=dict(
            type='coordinate_model',
            import_names=['ding.model.template.coordinate_regression'],
            obs_shape=[3, 96, 96],
            action_shape=2,
            cnn_block=[16, 32, 32],
            hidden_size=256,
            hidden_layer_num=2,
            implicit=True,
            stochastic_optim=dict(
                type='dfo',
                cuda=False,
                train_samples=64,
                inference_samples=2 ** 11,
                noise_scale=0.33,
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
    ),
)
create_config = EasyDict(create_config)
create_config = create_config

if __name__ == "__main__":
    from dizoo.coordinate_regression.entry import serial_pipeline_offline
    main_config.exp_name = "coordinate_regression_ibc_ts_" + str(main_config.train_dataset.dataset_size)
    serial_pipeline_offline([main_config, create_config], seed=0)
