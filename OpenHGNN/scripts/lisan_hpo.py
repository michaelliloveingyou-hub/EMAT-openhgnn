# OpenHGNN 脚本文件功能说明：
# 本文件为 lisan-acm、lisan-dblp 数据集提供命令行超参数搜索入口。
# 它调用 OpenHGNN Experiment 的 hpo_search_space/hpo_trials 能力，
# 支持 node_classification 和 link_prediction 两个任务，可用于 RGCN、HAN、HGT、SimpleHGN 等模型的基础搜索。

import argparse

from openhgnn import Experiment


def build_search_space(model, task, fixed_max_epoch=None, fixed_patience=None):
    def search_space(trial):
        params = {
            "lr": trial.suggest_categorical("lr", [1e-3, 3e-3, 5e-3, 1e-2]),
            "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3]),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.6),
            "patience": fixed_patience
            if fixed_patience is not None
            else trial.suggest_categorical("patience", [20, 50]),
            "max_epoch": fixed_max_epoch
            if fixed_max_epoch is not None
            else trial.suggest_categorical("max_epoch", [50, 100]),
        }
        if model in {"RGCN", "HGT", "SimpleHGN"}:
            params["num_layers"] = trial.suggest_int("num_layers", 2, 3)
        if model == "HAN":
            params["num_heads"] = [trial.suggest_categorical("han_num_heads", [2, 4, 8])]
        if model == "SimpleHGN":
            params["edge_dim"] = trial.suggest_categorical("edge_dim", [32, 64])
            params["slope"] = trial.suggest_categorical("slope", [0.05, 0.1, 0.2])
            params["feats_drop_rate"] = params["dropout"]
        if task == "link_prediction":
            params["score_fn"] = "distmult"
        return params

    return search_space


def main():
    parser = argparse.ArgumentParser(description="Run OpenHGNN HPO on lisan-acm or lisan-dblp.")
    parser.add_argument("--model", "-m", default="RGCN")
    parser.add_argument("--dataset", "-d", choices=["lisan-acm", "lisan-dblp"], required=True)
    parser.add_argument("--task", "-t", choices=["node_classification", "link_prediction"], required=True)
    parser.add_argument(
        "--feature_mode",
        "--feature-mode",
        choices=[
            "A", "B", "C", "D", "E",
            "raw", "emat", "emat_3025", "raw_emat_3025",
            "emat_svd_64", "emat_svd_128", "emat_svd_256",
            "raw_emat_svd_64", "raw_emat_svd_128", "raw_emat_svd_256",
            "emat_tfidf", "emat_tfidf_3025", "raw_emat_tfidf_3025",
            "emat_tfidf_svd_64", "emat_tfidf_svd_128", "emat_tfidf_svd_256",
            "raw_emat_tfidf_svd_64", "raw_emat_tfidf_svd_128", "raw_emat_tfidf_svd_256",
            "emat_sparse_encoder", "raw_emat_sparse_encoder",
        ],
        default="raw",
        help="A=raw, B=EMat, C=EMat-SVD-128, D=EMat-TFIDF-3025, E=raw+EMat sparse encoder.",
    )
    parser.add_argument("--gpu", "-g", type=int, default=-1)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    args = parser.parse_args()

    # 与 main.py 保持一致的默认运行字段，避免 HPO 内部重新构建 flow 时缺少属性。
    runtime_defaults = {
        "graphbolt": False,
        "mini_batch_flag": False,
        "use_ashin": False,
        "ashin_version": None,
        "ashin_base_version": "B",
        "ashin_base_version_user_set": False,
        "ashin_dim": 128,
        "ashin_dim_user_set": False,
        "ashin_norm": "log1p_zscore",
        "ashin_norm_user_set": False,
        "ashin_attr_agg": "mean",
        "ashin_attr_agg_user_set": False,
        "ashin_common_op": "max",
        "ashin_common_op_user_set": False,
        "ashin_common_norm": "row",
        "ashin_common_norm_user_set": False,
        "ashin_common_topk": 0,
        "ashin_common_topk_user_set": False,
        "ashin_cache_dir": "./openhgnn/output/ashin_cache",
        "ashin_rebuild": False,
        "ashin_log_dir": "./openhgnn/output/ashin_logs",
        "ashin_best_config_dir": "./openhgnn/output/optuna",
        "ashin_best_params_path": None,
        "run_name": None,
        "feature_mode": args.feature_mode,
    }

    experiment = Experiment(
        model=args.model,
        dataset=args.dataset,
        task=args.task,
        gpu=args.gpu,
        seed=args.seed,
        hpo_search_space=build_search_space(args.model, args.task, args.max_epoch, args.patience),
        hpo_trials=args.trials,
        **runtime_defaults,
    )
    experiment.run()


if __name__ == "__main__":
    main()
