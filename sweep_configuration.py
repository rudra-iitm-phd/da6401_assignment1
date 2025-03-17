sweep_config = {
            "method": "random",
            "metric": {"name": "Accuracy", "goal": "maximize"},
            "parameters": {
                "activation": {"values": ["relu", "sigmoid", 'tanh', 'identity']},
                "batch_size": {"values": [512, 1024, 2048]},
                "learning_rate": {"distribution": "log_uniform_values", "min": 1e-4, "max": 0.1},
                "optimizer": {"values": ["adam", "sgd", "momentum", "nag", "rmsprop"]},
                "momentum": {"distribution": "uniform", "min": 0.5, "max": 0.99},
                "beta1": {"distribution": "uniform", "min": 0.8, "max": 0.95},
                "beta2": {"distribution": "uniform", "min": 0.9, "max": 0.999},
                "weight_decay": {"distribution": "uniform", "min": 0, "max": 1e-3},
                "loss": {"values": ["cross_entropy"]},
                "epochs": {"values": [500, 800, 1000]},
                "hidden_size": {"values": [[784], [32, 64, 128],  [128, 256], [64, 128, 256], [784, 32], [512, 64], [32, 64, 128, 256]]},
                "weight_init": {"values": ["random", "xavier"]}
            }
        }