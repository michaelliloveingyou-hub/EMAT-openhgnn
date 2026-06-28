from __future__ import annotations

import contextlib
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from .config import OPENHGNN_ROOT, ExperimentSpec, is_supported, normalize_feature_mode, openhgnn_model_name
from .io_utils import append_log, ensure_dir, read_csv_rows, read_json, write_csv_rows, write_exception, write_json
from .metrics import flatten_result_metrics, objective_from_result
from .search_spaces import build_search_space
from .seed_utils import set_deterministic_seed
from .summary import build_final_row


COMMON_RUNTIME_DEFAULTS = {
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
}


NODE_SEED_COLUMNS = [
    "seed",
    "status",
    "valid_macro_f1",
    "valid_micro_f1",
    "test_macro_f1",
    "test_micro_f1",
    "test_accuracy",
    "notes",
]

LINK_SEED_COLUMNS = [
    "seed",
    "status",
    "valid_auc",
    "valid_loss",
    "test_auc",
    "test_loss",
    "test_ap",
    "test_mrr",
    "test_hits10",
    "notes",
]

TRIAL_COLUMNS = [
    "number",
    "state",
    "objective",
    "valid_metric",
    "duration_sec",
    "params_json",
    "notes",
]


def ensure_openhgnn_importable() -> None:
    root = str(OPENHGNN_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def git_commit_hash(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "not_a_git_repo"


def environment_metadata(project_root: Path, argv: list[str], args_dict: dict[str, Any], spec: ExperimentSpec) -> dict[str, Any]:
    try:
        import dgl

        dgl_version = dgl.__version__
    except Exception as exc:
        dgl_version = f"unavailable: {exc}"
    cuda_info = {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        cuda_info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {
        "command": " ".join(argv),
        "args": args_dict,
        "dataset": spec.dataset,
        "task": spec.task,
        "model": spec.model,
        "feature_mode": normalize_feature_mode(spec.feature_mode),
        "git_commit": git_commit_hash(project_root),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "dgl": dgl_version,
        "cuda": cuda_info,
    }


def run_openhgnn_once(
    spec: ExperimentSpec,
    params: dict[str, Any],
    seed: int,
    gpu: int,
    output_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    ensure_openhgnn_importable()
    from openhgnn import Experiment

    ensure_dir(output_dir)
    runtime_model = openhgnn_model_name(spec.model)
    ensure_dir(output_dir / runtime_model)
    set_deterministic_seed(seed)
    kwargs = dict(COMMON_RUNTIME_DEFAULTS)
    kwargs.update(params)
    kwargs["feature_mode"] = normalize_feature_mode(spec.feature_mode)
    kwargs["seed"] = seed
    kwargs["run_name"] = f"{spec.dataset}_{spec.task}_{spec.model}_{normalize_feature_mode(spec.feature_mode)}_seed{seed}"

    with log_path.open("a", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            with working_directory(OPENHGNN_ROOT):
                experiment = Experiment(
                    model=runtime_model,
                    dataset=spec.dataset,
                    task=spec.task,
                    gpu=gpu,
                    output_dir=str(output_dir),
                    **kwargs,
                )
                result = experiment.run()
    if result is None:
        raise RuntimeError("OpenHGNN returned no result")
    return result


@contextlib.contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def successful_seed_rows(path: Path) -> dict[int, dict[str, str]]:
    rows = read_csv_rows(path)
    completed = {}
    for row in rows:
        try:
            seed = int(row.get("seed", ""))
        except ValueError:
            continue
        if row.get("status") == "completed":
            completed[seed] = row
    return completed


def run_hpo(
    spec: ExperimentSpec,
    combo_dir: Path,
    n_trials: int,
    seeds: list[int],
    max_epoch: int,
    patience: int,
    gpu: int,
    resume: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import optuna

    best_params_path = combo_dir / "best_params.json"
    hpo_summary_path = combo_dir / "hpo_summary.json"
    if resume and best_params_path.exists():
        best_params = read_json(best_params_path, {})
        hpo_summary = read_json(hpo_summary_path, {})
        if best_params and hpo_summary:
            append_log(combo_dir / "run.log", "Skipping HPO because best_params.json already exists.")
            return best_params, hpo_summary

    search_space = build_search_space(spec.model, spec.task, max_epoch=max_epoch, patience=patience)
    trial_rows: list[dict[str, Any]] = []
    best: dict[str, Any] = {"score": None, "params": None, "valid_metric": None}
    hpo_seed = seeds[0]

    def objective(trial) -> float:
        params = search_space(trial)
        started = time.time()
        try:
            result = run_openhgnn_once(
                spec=spec,
                params=params,
                seed=hpo_seed,
                gpu=gpu,
                output_dir=combo_dir / "openhgnn_output" / f"trial_{trial.number}",
                log_path=combo_dir / "run.log",
            )
            score = objective_from_result(spec.task, result)
            duration = time.time() - started
            trial_rows.append(
                {
                    "number": trial.number,
                    "state": "COMPLETE",
                    "objective": score,
                    "valid_metric": result.get("metric", {}).get("valid", {}),
                    "duration_sec": f"{duration:.3f}",
                    "params_json": params,
                    "notes": "",
                }
            )
            if best["score"] is None or score > best["score"]:
                best.update({"score": score, "params": params, "valid_metric": result.get("metric", {}).get("valid", {})})
            return score
        except Exception as exc:
            duration = time.time() - started
            write_exception(combo_dir / "error.log", exc)
            trial_rows.append(
                {
                    "number": trial.number,
                    "state": "FAIL",
                    "objective": "",
                    "valid_metric": "",
                    "duration_sec": f"{duration:.3f}",
                    "params_json": params,
                    "notes": str(exc),
                }
            )
            raise

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=1, catch=(Exception,))

    trials_for_csv = []
    for row in trial_rows:
        csv_row = dict(row)
        csv_row["valid_metric"] = _json_string(csv_row["valid_metric"])
        csv_row["params_json"] = _json_string(csv_row["params_json"])
        trials_for_csv.append(csv_row)
    write_csv_rows(combo_dir / "hpo_trials.csv", trials_for_csv, TRIAL_COLUMNS)

    if best["params"] is None:
        raise RuntimeError("all HPO trials failed")
    write_json(best_params_path, best["params"])
    hpo_summary = {
        "best_score": best["score"],
        "best_valid_metric": best["valid_metric"],
        "n_trials": n_trials,
        "hpo_seed": hpo_seed,
    }
    write_json(hpo_summary_path, hpo_summary)
    return best["params"], hpo_summary


def run_seed_evaluations(
    spec: ExperimentSpec,
    combo_dir: Path,
    best_params: dict[str, Any],
    seeds: list[int],
    gpu: int,
    resume: bool,
) -> list[dict[str, Any]]:
    seed_results_path = combo_dir / "seed_results.csv"
    completed = successful_seed_rows(seed_results_path) if resume else {}
    rows: list[dict[str, Any]] = [completed[seed] for seed in seeds if seed in completed]

    for seed in seeds:
        if seed in completed:
            append_log(combo_dir / "run.log", f"Skipping seed {seed} because it is already completed.")
            continue
        try:
            result = run_openhgnn_once(
                spec=spec,
                params=best_params,
                seed=seed,
                gpu=gpu,
                output_dir=combo_dir / "openhgnn_output" / f"seed_{seed}",
                log_path=combo_dir / "run.log",
            )
            rows.append(flatten_result_metrics(spec.task, seed, result, "completed"))
        except Exception as exc:
            write_exception(combo_dir / "error.log", exc)
            rows.append(flatten_result_metrics(spec.task, seed, {}, "failed", notes=str(exc)))

        write_csv_rows(seed_results_path, rows, NODE_SEED_COLUMNS if spec.task == "node_classification" else LINK_SEED_COLUMNS)

    write_csv_rows(seed_results_path, rows, NODE_SEED_COLUMNS if spec.task == "node_classification" else LINK_SEED_COLUMNS)
    return rows


def run_combination(
    spec: ExperimentSpec,
    output_root: Path,
    project_root: Path,
    n_trials: int,
    seeds: list[int],
    max_epoch: int,
    patience: int,
    gpu: int,
    resume: bool,
    argv: list[str],
    args_dict: dict[str, Any],
) -> dict[str, Any]:
    combo_dir = spec.combo_dir(output_root)
    ensure_dir(combo_dir)
    start_time = time.time()
    started = datetime.now().isoformat(timespec="seconds")
    metadata = environment_metadata(project_root, argv, args_dict, spec)
    metadata.update({"started_at": started, "max_epoch": max_epoch, "patience": patience})
    write_json(combo_dir / "metadata.json", metadata)

    supported, reason = is_supported(spec)
    best_params_path = combo_dir / "best_params.json"
    if not supported:
        notes = reason
        summary = {
            "status": "skipped",
            "notes": notes,
            "started_at": started,
            "ended_at": datetime.now().isoformat(timespec="seconds"),
        }
        write_json(combo_dir / "summary.json", summary)
        return build_final_row(
            spec.task,
            spec.dataset,
            spec.model,
            normalize_feature_mode(spec.feature_mode),
            n_trials,
            seeds,
            best_params_path,
            "skipped",
            notes,
            [],
            {},
        )

    if resume:
        prior_summary = read_json(combo_dir / "summary.json", {})
        if prior_summary.get("status") == "completed":
            seed_rows = read_csv_rows(combo_dir / "seed_results.csv")
            best_values = prior_summary.get("best_values", {})
            notes = prior_summary.get("notes", "")
            return build_final_row(
                spec.task,
                spec.dataset,
                spec.model,
                normalize_feature_mode(spec.feature_mode),
                n_trials,
                seeds,
                best_params_path,
                "completed",
                notes,
                seed_rows,
                best_values,
            )

    try:
        error_log = combo_dir / "error.log"
        if error_log.exists():
            error_log.write_text("", encoding="utf-8")
        append_log(combo_dir / "run.log", f"Started {spec.combo_id} at {started}")
        best_params, hpo_summary = run_hpo(spec, combo_dir, n_trials, seeds, max_epoch, patience, gpu, resume)
        seed_rows = run_seed_evaluations(spec, combo_dir, best_params, seeds, gpu, resume)
        completed_count = sum(1 for row in seed_rows if row.get("status") == "completed")
        status = "completed" if completed_count == len(seeds) else "failed"
        notes = "" if status == "completed" else f"{completed_count}/{len(seeds)} seeds completed"
        if spec.task == "link_prediction":
            notes = (notes + "; " if notes else "") + "OpenHGNN link metrics only: roc_auc/loss; AP/MRR/Hits@10 not computed"
        best_values = _best_values(spec.task, hpo_summary)
        ended = datetime.now().isoformat(timespec="seconds")
        summary = {
            "status": status,
            "notes": notes,
            "started_at": started,
            "ended_at": ended,
            "duration_sec": round(time.time() - start_time, 3),
            "best_values": best_values,
            "hpo_summary": hpo_summary,
            "seeds": seeds,
        }
        write_json(combo_dir / "summary.json", summary)
        append_log(combo_dir / "run.log", f"Finished {spec.combo_id} at {ended} with status={status}")
        return build_final_row(
            spec.task,
            spec.dataset,
            spec.model,
            normalize_feature_mode(spec.feature_mode),
            n_trials,
            seeds,
            best_params_path,
            status,
            notes,
            seed_rows,
            best_values,
        )
    except Exception as exc:
        write_exception(combo_dir / "error.log", exc)
        notes = str(exc)
        summary = {
            "status": "failed",
            "notes": notes,
            "started_at": started,
            "ended_at": datetime.now().isoformat(timespec="seconds"),
        }
        write_json(combo_dir / "summary.json", summary)
        seed_rows = read_csv_rows(combo_dir / "seed_results.csv")
        return build_final_row(
            spec.task,
            spec.dataset,
            spec.model,
            normalize_feature_mode(spec.feature_mode),
            n_trials,
            seeds,
            best_params_path,
            "failed",
            notes,
            seed_rows,
            {},
        )


def _best_values(task: str, hpo_summary: dict[str, Any]) -> dict[str, Any]:
    valid = hpo_summary.get("best_valid_metric", {}) or {}
    if task == "node_classification":
        return {"best_val_macro_f1": valid.get("Macro_f1") or valid.get("macro_f1")}
    return {"best_val_auc": valid.get("roc_auc") or valid.get("AUC")}


def _json_string(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
