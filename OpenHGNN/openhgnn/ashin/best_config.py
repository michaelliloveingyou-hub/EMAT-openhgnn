"""
best_config.py

ASHIN 自己的 best config 加载逻辑。

OpenHGNN 原来的 --use_best_config 会走 utils/best_config.py；ASHIN 的参数是 Optuna 写出来的，
所以这里单独读 openhgnn/output/optuna/.../best_params.json。

这里只做一件事：把 best_params.json 里存在的字段覆盖到 args 上。没搜过的参数继续用 config.ini。
不要在这里按 test 指标挑参数，也不要碰 label/mask。
"""

import json
from pathlib import Path


def default_best_params_path(args):
    # 默认路径必须和 tune_ashin_common.py 的 study_name 规则一致。
    root = Path(getattr(args, "ashin_best_config_dir", "./openhgnn/output/optuna"))
    study_name = "ashin_{}_{}_{}".format(args.model_name, args.dataset_name, args.ashin_version)
    return root / study_name / "best_params.json"


def _mapped_params(args, params):
    # 调参脚本用的是更直观的参数名，OpenHGNN 某些模型内部用的是旧字段名。
    # 这里集中做一次翻译，避免在调参脚本和训练入口里各写一套兼容逻辑。
    mapped = dict(params)
    if "negative_slope" in mapped and "slope" not in mapped:
        mapped["slope"] = mapped.pop("negative_slope")
    if args.model_name == "SimpleHGN" and "dropout" in mapped:
        mapped["feats_drop_rate"] = mapped["dropout"]
    return mapped


def apply_ashin_best_config(args):
    # 在 Experiment 中调用：如果启用了 ASHIN + --use_best_config，就加载 ASHIN best_params。
    if not getattr(args, "use_ashin", False):
        return args
    if getattr(args, "ashin_version", None) not in ("B", "C", "D", "E", "F", "G", "H"):
        raise ValueError("--use_best_config with --use_ashin requires --ashin_version {B,C,D,E,F,G,H}.")

    path = getattr(args, "ashin_best_params_path", None)
    path = Path(path) if path else default_best_params_path(args)
    if not path.exists():
        raise FileNotFoundError(
            "ASHIN best config was requested, but best_params.json was not found: {}. "
            "Run Optuna first, or pass --ashin_best_params_path.".format(path)
        )

    with path.open("r", encoding="utf-8") as f:
        params = json.load(f)
    for key, value in _mapped_params(args, params).items():
        setattr(args, key, value)
    args.ashin_best_params_path = str(path)
    args.ashin_best_params = params
    print("Load ASHIN best config from {}.".format(path))
    return args
