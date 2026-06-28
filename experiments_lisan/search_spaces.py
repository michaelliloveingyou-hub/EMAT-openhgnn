from typing import Any


def build_search_space(model: str, task: str, max_epoch: int, patience: int):
    """Return an Optuna search-space callable using OpenHGNN runtime parameter names.

    OpenHGNN config.ini often stores learning rates as `learning_rate`, while the
    Experiment runtime uses `args.lr`; these spaces therefore emit `lr`.
    For SimpleHGN, `dropout` maps to `feats_drop_rate` because the model reads
    feature-dropout from that runtime field.
    """

    def suggest(trial) -> dict[str, Any]:
        params: dict[str, Any] = {
            "lr": trial.suggest_categorical("lr", [1e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2]),
            "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 5e-4, 1e-3]),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128, 256]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.7),
            "max_epoch": max_epoch,
            "patience": patience,
            "mini_batch_flag": False,
            "graphbolt": False,
        }

        if model == "RGCN":
            params.update(
                {
                    "num_layers": trial.suggest_int("num_layers", 2, 3),
                    "n_bases": trial.suggest_categorical("n_bases", [20, 40, 80]),
                }
            )
        elif model == "HAN":
            params.update(
                {
                    "num_heads": [trial.suggest_categorical("han_num_heads", [2, 4, 8])],
                    "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
                }
            )
        elif model == "HGT":
            params.update(
                {
                    "num_layers": trial.suggest_int("num_layers", 2, 3),
                    "num_heads": trial.suggest_categorical("num_heads", [1, 2, 4, 8]),
                    "norm": trial.suggest_categorical("norm", [True, False]),
                }
            )
        elif model == "SimpleHGN":
            dropout = params["dropout"]
            params.update(
                {
                    "num_layers": trial.suggest_int("num_layers", 2, 3),
                    "num_heads": trial.suggest_categorical("num_heads", [4, 8, 16]),
                    "edge_dim": trial.suggest_categorical("edge_dim", [32, 64, 128]),
                    "slope": trial.suggest_categorical("slope", [0.05, 0.1, 0.2]),
                    "beta": trial.suggest_categorical("beta", [0.01, 0.05, 0.1]),
                    "residual": trial.suggest_categorical("residual", [True, False]),
                    "feats_drop_rate": dropout,
                }
            )
        elif model == "GCN":
            params.update(
                {
                    "gnn_type": "gcnconv",
                    "layers_gnn": trial.suggest_int("layers_gnn", 2, 4),
                    "layers_pre_mp": 1,
                    "layers_post_mp": 1,
                    "stage_type": "stack",
                    "num_heads": 1,
                    "has_bn": trial.suggest_categorical("has_bn", [True, False]),
                    "has_l2norm": trial.suggest_categorical("has_l2norm", [True, False]),
                }
            )
        elif model == "GAT":
            params.update(
                {
                    "gnn_type": "gatconv",
                    "layers_gnn": trial.suggest_int("layers_gnn", 2, 4),
                    "layers_pre_mp": 1,
                    "layers_post_mp": 1,
                    "stage_type": "stack",
                    "num_heads": trial.suggest_categorical("num_heads", [2, 4, 8]),
                    "has_bn": trial.suggest_categorical("has_bn", [True, False]),
                    "has_l2norm": trial.suggest_categorical("has_l2norm", [True, False]),
                }
            )

        if task == "link_prediction":
            params["score_fn"] = "distmult"
        return params

    return suggest
