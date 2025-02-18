config = {
    "scheduler_config": {
        "gpu": ["0", "1", "2"],
        "config_string_value_maxlen": 1000,
        "result_root_folder": "../../../results/"
    },

    "global_config": {
        "batch_size": 1000,
        "vis_freq": 200,
        "vis_num_sample": 5,
        "d_rounds": 1,
        "g_rounds": 1,
        "num_packing": 1,
        "noise": True,
        "feed_back": False,
        "g_lr": 0.001,
        "d_lr": 0.001,
        "d_gp_coe": 10.0,
        "gen_feature_num_layers": 1,
        "gen_feature_num_units": 100,
        "gen_attribute_num_layers": 3,
        "gen_attribute_num_units": 100,
        "disc_num_layers": 5,
        "disc_num_units": 200,
        "initial_state": "random",

        "attr_d_lr": 0.001,
        "attr_d_gp_coe": 10.0,
        "g_attr_d_coe": 1.0,
        "attr_disc_num_layers": 5,
        "attr_disc_num_units": 200,
    },

    "test_config": [
        #{
        #    "dataset": ["google"],
        #    "epoch": [400],
        #    "run": [0, 1, 2],
        #    "sample_len": [1, 5, 10],
        #    "extra_checkpoint_freq": [5],
        #    "epoch_checkpoint_freq": [1],
        #    "aux_disc": [False],
        #    "self_norm": [False]
        #},
        {
            "dataset": ["web"],
            "epoch": [200],
            "run": [0, 1, 2],
            "sample_len": [5],
            "extra_checkpoint_freq": [5],
            "epoch_checkpoint_freq": [1],
            "aux_disc": [True],
            "self_norm": [True]
        }#,
        #{
        #    "dataset": ["FCC_MBA"],
        #    "epoch": [17000],
        #    "run": [0, 1, 2],
        #    "sample_len": [1, 4, 8],
        #    "extra_checkpoint_freq": [850],
        #    "epoch_checkpoint_freq": [70],
        #    "aux_disc": [False],
        #    "self_norm": [False]
        #}
    ]
}
