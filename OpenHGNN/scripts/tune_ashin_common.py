"""
tune_ashin_common.py

Optuna 公共入口。

RGCN、SimpleHGN、HAN、RSHN、HPN、HGT 的 ASHIN 调参 wrapper 都会走这里。
这里定义搜索空间、启动 main.py、读取 valid 分数，并把最优参数写到 best_params.json。
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_score(metrics, log_text):
    # 优先读 metrics.json；如果某个 trainer 没正常写 metrics，再从日志里兜底抓 valid 分数。
    # Optuna 只能看验证集分数，这里不要读 test。
    metric = metrics.get("metric", {}) if isinstance(metrics, dict) else {}
    valid = metric.get("valid", {})
    for key in ("Accuracy", "Micro_f1", "Mirco_f1", "Macro_f1"):
        if key in valid:
            return float(valid[key])
    matches = re.findall(r"Mode:valid,.*?(?:Accuracy|Micro_f1|Mirco_f1|Macro_f1):\s*([0-9.]+)", log_text)
    if matches:
        return float(matches[-1])
    return 0.0


def latest_run_dir(run_name):
    # main.py 会给同一个 run_name 追加时间戳目录。
    # trial 结束后取最新目录，才能拿到这次 subprocess 的 metrics/log。
    root = ROOT / "openhgnn" / "output" / "ashin_logs"
    matches = sorted(root.glob(f"{run_name}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def add_model_space(trial, model, params):
    if model == "RGCN":
        params["num_layers"] = trial.suggest_categorical("num_layers", [1, 2, 3])
        params["n_bases"] = trial.suggest_categorical("n_bases", [-1, 4, 8, 16])
    elif model == "SimpleHGN":
        params["num_layers"] = trial.suggest_categorical("num_layers", [1, 2, 3])
        params["num_heads"] = trial.suggest_categorical("num_heads", [2, 4, 8])
        params["edge_dim"] = trial.suggest_categorical("edge_dim", [16, 32, 64])
        params["negative_slope"] = trial.suggest_categorical("negative_slope", [0.01, 0.05, 0.2])
        params["residual"] = trial.suggest_categorical("residual", [True, False])
        params["beta"] = trial.suggest_float("beta", 0.0, 1.0)
    elif model == "HAN":
        params["han_num_heads"] = trial.suggest_categorical("han_num_heads", [2, 4, 8])
        params["num_layers"] = 1
    elif model == "RSHN":
        params["rw_len"] = trial.suggest_categorical("rw_len", [2, 4, 6])
        params["num_node_layer"] = trial.suggest_categorical("num_node_layer", [1, 2, 3])
        params["num_edge_layer"] = trial.suggest_categorical("num_edge_layer", [1, 2, 3])
        params["batch_size"] = trial.suggest_categorical("batch_size", [512, 1000, 2048])
    elif model == "HPN":
        params["k_layer"] = trial.suggest_categorical("k_layer", [1, 2, 3, 4])
        params["alpha"] = trial.suggest_float("alpha", 0.05, 0.5)
        params["edge_drop"] = trial.suggest_categorical("edge_drop", [0.0, 0.1, 0.3, 0.5])
    elif model == "HGT":
        params["num_heads"] = trial.suggest_categorical("num_heads", [2, 4, 8])
        params["num_layers"] = trial.suggest_categorical("num_layers", [2, 3])
        params["batch_size"] = trial.suggest_categorical("batch_size", [1024, 2048, 5120])
        params["fanout"] = trial.suggest_categorical("fanout", [2, 5, 10])
        params["norm"] = trial.suggest_categorical("norm", [True, False])


def add_ashin_space(trial, version, params):
    # 这里的参数名会直接写进 best_params.json。
    # 新增 ASHIN 参数时，要同步检查 main.py 和 best_config.py 是否能接住。
    if version in ("C", "D"):
        params["ashin_dim"] = trial.suggest_categorical("ashin_dim", [32, 64, 128, 256])
        params["ashin_norm"] = trial.suggest_categorical("ashin_norm", ["none", "zscore", "log1p_zscore"])
    elif version == "G":
        params["ashin_dim"] = trial.suggest_categorical("ashin_dim", [32, 64, 128, 256])
        params["ashin_norm"] = trial.suggest_categorical("ashin_norm", ["none", "zscore", "log1p_zscore"])
        params["ashin_attr_agg"] = trial.suggest_categorical("ashin_attr_agg", ["mean", "sum", "max"])
    elif version == "H":
        params["ashin_dim"] = trial.suggest_categorical("ashin_dim", [32, 64, 128, 256])
        params["ashin_norm"] = trial.suggest_categorical("ashin_norm", ["none", "zscore", "log1p_zscore"])
        params["ashin_common_op"] = trial.suggest_categorical("ashin_common_op", ["max", "sum"])
        params["ashin_common_norm"] = trial.suggest_categorical("ashin_common_norm", ["row", "binary"])
        params["ashin_common_topk"] = trial.suggest_categorical("ashin_common_topk", [0, 10, 20, 50])
    elif version in ("E", "F"):
        params["ashin_base_version"] = trial.suggest_categorical("ashin_base_version", ["B", "C"])
        if params["ashin_base_version"] == "C":
            params["ashin_dim"] = trial.suggest_categorical("ashin_dim", [32, 64, 128, 256])
        params["ashin_norm"] = trial.suggest_categorical("ashin_norm", ["none", "zscore", "log1p_zscore"])
    else:
        params["ashin_norm"] = trial.suggest_categorical("ashin_norm", ["none", "zscore"])


def objective_factory(cli_args, optuna):
    def objective(trial):
        # 一个 trial 就是一次完整 main.py 训练。
        # 这样调参和最终测试走同一条训练入口，少一些“调参脚本能跑、正式入口不能跑”的问题。
        params = {
            "lr": trial.suggest_float("lr", 1e-4, 5e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.7),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128, 256]),
            "patience": trial.suggest_categorical("patience", [30, 50, 100]),
        }
        add_model_space(trial, cli_args.model, params)
        add_ashin_space(trial, cli_args.ashin_version, params)

        run_name = f"optuna_{cli_args.model}_{cli_args.dataset}_ashin{cli_args.ashin_version}_trial{trial.number}_{int(time.time())}"
        cmd = [
            sys.executable,
            "main.py",
            "-m",
            cli_args.model,
            "-d",
            cli_args.dataset,
            "-t",
            "node_classification",
            "-g",
            str(cli_args.gpu),
            "--seed",
            str(cli_args.seed),
            "--use_ashin",
            "--ashin_version",
            cli_args.ashin_version,
            "--run_name",
            run_name,
        ]
        if cli_args.max_epoch is not None:
            cmd.extend(["--max_epoch", str(cli_args.max_epoch)])
        for key, value in params.items():
            cmd.extend([f"--{key}", str(value)])

        completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        run_dir = latest_run_dir(run_name)
        log_text = completed.stdout + "\n" + completed.stderr
        metrics = {}
        if run_dir and (run_dir / "metrics.json").exists():
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            log_text += "\n" + (run_dir / "train.log").read_text(encoding="utf-8", errors="ignore")
        if completed.returncode != 0:
            trial.set_user_attr("returncode", completed.returncode)
            trial.set_user_attr("stderr_tail", completed.stderr[-2000:])
            return 0.0
        trial.set_user_attr("run_dir", str(run_dir) if run_dir else "")
        return parse_score(metrics, log_text)

    return objective


def main(default_model):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ohgbn-acm")
    parser.add_argument("--model", default=default_model, choices=["RGCN", "SimpleHGN", "HAN", "RSHN", "HPN", "HGT"])
    parser.add_argument("--ashin_version", required=True, choices=["B", "C", "D", "E", "F", "G", "H"])
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--study_name", default=None)
    parser.add_argument("--storage", default=None)
    parser.add_argument("--max_epoch", type=int, default=None, help="optional smoke-test epoch override")
    args = parser.parse_args()

    try:
        import optuna
    except Exception as exc:
        raise SystemExit(f"Optuna is required for tuning: {exc}")

    storage = args.storage
    if storage is None:
        db = ROOT / "openhgnn" / "output" / "optuna" / f"ashin_{args.model.lower()}_{args.dataset}.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{db.as_posix()}"
    study_name = args.study_name or f"ashin_{args.model}_{args.dataset}_{args.ashin_version}"
    study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage, load_if_exists=True)
    study.optimize(objective_factory(args, optuna), n_trials=args.n_trials)

    out_dir = ROOT / "openhgnn" / "output" / "optuna" / study_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "best_params.json").write_text(json.dumps(study.best_params, indent=2), encoding="utf-8")
    with (out_dir / "optuna_trials.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["number", "value", "params", "user_attrs", "state"])
        for trial in study.trials:
            writer.writerow([trial.number, trial.value, json.dumps(trial.params), json.dumps(trial.user_attrs), str(trial.state)])
    (out_dir / "study_summary.json").write_text(
        json.dumps({"best_value": study.best_value, "best_params": study.best_params}, indent=2),
        encoding="utf-8",
    )
    print(f"Best value: {study.best_value}")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main(default_model="RGCN")
