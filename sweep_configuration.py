sweep_config = {
            "method": "bayes",
            "metric": {"name": "Accuracy", "goal": "maximize"},
            "parameters": {
                "activation": {"values": ["relu", "sigmoid"]},
                "batch_size": {"values": [256, 512, 1024]},
                "learning_rate": {"distribution": "log_uniform_values", "min": 1e-4, "max": 0.1},
                "optimizer": {"values": ["adam", "sgd", "momentum", "nag", "rmsprop"]},
                "momentum": {"distribution": "uniform", "min": 0.5, "max": 0.99},
                "beta1": {"distribution": "uniform", "min": 0.8, "max": 0.95},
                "beta2": {"distribution": "uniform", "min": 0.9, "max": 0.999},
                "weight_decay": {"distribution": "uniform", "min": 0, "max": 1e-3},
                "loss": {"values": ["cross_entropy", "mean_squared_error"]},
                "epochs": {"values": [300, 500, 800, 1000]},
                "hidden_size": {"values": [[784], [32, 64, 128], [784, 10], [128, 256], [64, 128, 256], [784, 32]]},
                "weight_init": {"values": ["random", "xavier"]}
            }
        }