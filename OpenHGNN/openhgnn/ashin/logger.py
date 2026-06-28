"""
logger.py

ASHIN 实验日志。

每次 run 单独建目录，保存 args、环境、图信息、ASHIN 元数据、metrics 和 train.log。
这些文件主要是给自己回头查“这次到底跑了什么”用的。

这里不参与训练决策，也不根据 test 指标改参数。
"""

import contextlib
import datetime
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


class _Tee:
    # 同时把控制台输出写到原始 stdout/stderr 和 train.log。
    def __init__(self, stream, file_obj):
        self.stream = stream
        self.file_obj = file_obj

    def write(self, data):
        self.stream.write(data)
        self.file_obj.write(data)

    def flush(self):
        self.stream.flush()
        self.file_obj.flush()


class TeeCapture:
    # 管理 stdout/stderr 捕获的开始与结束。
    def __init__(self, path):
        self.path = Path(path)
        self.file_obj = None
        self.stdout = None
        self.stderr = None

    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file_obj = self.path.open("a", encoding="utf-8")
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = _Tee(sys.stdout, self.file_obj)
        sys.stderr = _Tee(sys.stderr, self.file_obj)

    def stop(self):
        if self.file_obj is None:
            return
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file_obj.close()
        self.file_obj = None


def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def default_run_name(args):
    # 如果用户没有指定 run_name，就自动生成一个包含模型、数据集、任务和 seed 的名字。
    base = "_".join([
        str(getattr(args, "model_name", getattr(args, "model", "model"))),
        str(getattr(args, "dataset_name", getattr(args, "dataset", "dataset"))),
        str(getattr(args, "task", "task")),
        "ashin" + str(getattr(args, "ashin_version", "NA")),
        "seed" + str(getattr(args, "seed", 0)),
    ])
    return base


def ensure_run_logger(args):
    # 只有启用 ASHIN 时才创建专用 run 日志，避免影响原始 OpenHGNN。
    if not getattr(args, "use_ashin", False):
        return None
    existing = getattr(args, "ashin_run_dir", None)
    if existing:
        return Path(existing)
    run_name = getattr(args, "run_name", None) or default_run_name(args)
    run_dir = Path(getattr(args, "ashin_log_dir", "./openhgnn/output/ashin_logs")) / f"{run_name}_{timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    args.run_name = run_name
    args.ashin_run_dir = str(run_dir)
    _write_json(run_dir / "args.json", _jsonable(vars(args)))
    _write_text(run_dir / "environment.txt", environment_text())
    write_changed_files(run_dir)
    capture = TeeCapture(run_dir / "train.log")
    capture.start()
    args._ashin_tee_capture = capture
    return run_dir


def close_run_logger(args):
    # 关闭 train.log 捕获，并恢复原始 stdout/stderr。
    capture = getattr(args, "_ashin_tee_capture", None)
    if capture is not None:
        capture.stop()
        args._ashin_tee_capture = None


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)


def _write_text(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _run_command(cmd):
    try:
        completed = subprocess.run(cmd, cwd=Path.cwd(), text=True, capture_output=True, check=False)
        if completed.returncode == 0:
            return completed.stdout.strip()
        return f"UNAVAILABLE ({completed.stderr.strip()})"
    except Exception as exc:
        return f"UNAVAILABLE ({exc})"


def environment_text():
    # 把运行环境写下来，后面复现实验时能对上依赖版本。
    lines = [
        f"python: {sys.version}",
        f"platform: {platform.platform()}",
    ]
    for name in ["torch", "dgl", "sklearn", "optuna"]:
        try:
            module = __import__(name)
            version = getattr(module, "__version__", "unknown")
        except Exception as exc:
            version = f"UNAVAILABLE ({exc})"
        lines.append(f"{name}: {version}")
    try:
        import torch
        lines.append(f"cuda_available: {torch.cuda.is_available()}")
    except Exception as exc:
        lines.append(f"cuda_available: UNAVAILABLE ({exc})")
    lines.append(f"git_commit: {_run_command(['git', 'rev-parse', 'HEAD'])}")
    return "\n".join(lines) + "\n"


def write_changed_files(run_dir):
    # 如果 git 可用，记录真实 diff；如果 git 不可用，记录 ASHIN 相关候选文件。
    run_dir = Path(run_dir)
    names = _run_command(["git", "diff", "--name-only"])
    diff = _run_command(["git", "diff"])
    if not names.startswith("UNAVAILABLE"):
        _write_text(run_dir / "changed_files.md", "# Changed files\n\n" + "\n".join(f"- {n}" for n in names.splitlines()))
        _write_text(run_dir / "git_diff.patch", diff + "\n")
        return
    candidates = []
    for root in ["openhgnn/ashin", "scripts", "configs/ashin", "docs"]:
        path = Path(root)
        if path.exists():
            for item in path.rglob("*"):
                if item.is_file():
                    candidates.append(str(item))
    _write_text(
        run_dir / "changed_files.md",
        "# Changed files\n\nGit is unavailable; candidate ASHIN files:\n\n"
        + "\n".join(f"- {n}" for n in sorted(candidates))
        + "\n",
    )
    _write_text(run_dir / "git_diff.patch", "Git is unavailable in this runtime.\n")


def write_graph_info(args, g, target_ntype, raw_dim, ashin_dim, final_dim):
    # 保存图结构和特征维度信息，方便确认 ASHIN 是否正确拼接。
    run_dir = getattr(args, "ashin_run_dir", None)
    if not run_dir:
        return
    payload = {
        "dataset": getattr(args, "dataset_name", getattr(args, "dataset", None)),
        "task": getattr(args, "task", None),
        "model": getattr(args, "model_name", getattr(args, "model", None)),
        "ntypes": sorted([str(n) for n in g.ntypes]),
        "canonical_etypes": [list(map(str, e)) for e in sorted(g.canonical_etypes, key=lambda e: (str(e[0]), str(e[1]), str(e[2])))],
        "num_nodes": {str(n): int(g.num_nodes(n)) for n in g.ntypes},
        "num_edges": {str(e): int(g.num_edges(etype=e)) for e in g.canonical_etypes},
        "target_ntype": str(target_ntype),
        "raw_feature_dim": int(raw_dim),
        "ashin_feature_dim": int(ashin_dim),
        "final_feature_dim": int(final_dim),
    }
    _write_json(Path(run_dir) / "graph_info.json", payload)


def write_ashin_metadata(args, metadata):
    run_dir = getattr(args, "ashin_run_dir", None)
    if run_dir:
        _write_json(Path(run_dir) / "ashin_metadata.json", _jsonable(metadata))


def save_metrics(args, result):
    # 保存训练流程返回的最终指标；该函数不选择 best 参数。
    if not getattr(args, "use_ashin", False):
        return
    run_dir = getattr(args, "ashin_run_dir", None)
    if run_dir:
        _write_json(Path(run_dir) / "metrics.json", _jsonable(result or {}))


@contextlib.contextmanager
def managed_run(args):
    try:
        ensure_run_logger(args)
        yield
    finally:
        close_run_logger(args)
