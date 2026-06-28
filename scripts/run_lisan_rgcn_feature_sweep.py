from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments_lisan.config import DATASETS, FEATURE_MODES, TASKS, ExperimentSpec, is_supported, normalize_feature_mode
from experiments_lisan.io_utils import ensure_dir
from experiments_lisan.summary import write_final_summaries


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "lisan_rgcn_feature_sweep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one RGCN HPO/evaluation workflow for each Lisan feature mode, "
            "dataset, and task combination."
        )
    )
    parser.add_argument("--feature_modes", nargs="+", default=["A", "B", "C", "D", "E"], choices=FEATURE_MODES)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    parser.add_argument("--n_trials", type=int, default=1, help="Optuna trials per combination.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="Evaluation seeds after HPO.")
    parser.add_argument("--max_epoch", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def build_specs(args: argparse.Namespace) -> list[ExperimentSpec]:
    specs = []
    for feature_mode in args.feature_modes:
        canonical_mode = normalize_feature_mode(feature_mode)
        for dataset in args.datasets:
            for task in args.tasks:
                specs.append(
                    ExperimentSpec(
                        dataset=dataset,
                        task=task,
                        model="RGCN",
                        feature_mode=canonical_mode,
                    )
                )
    return specs


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    specs = build_specs(args)

    print("Planned RGCN feature sweep combinations:")
    for spec in specs:
        supported, reason = is_supported(spec)
        status = "run" if supported else f"skip: {reason}"
        print(f"{spec.dataset}\t{spec.task}\tRGCN\t{spec.feature_mode}\t{status}")

    if args.dry_run:
        return 0

    ensure_dir(output_root)
    from experiments_lisan.runner import run_combination

    rows = []
    args_dict = vars(args).copy()
    args_dict["output_root"] = str(output_root)
    for index, spec in enumerate(specs, start=1):
        print(
            f"[{index}/{len(specs)}] Running {spec.dataset} / {spec.task} / "
            f"RGCN / {spec.feature_mode}"
        )
        row = run_combination(
            spec=spec,
            output_root=output_root,
            project_root=PROJECT_ROOT,
            n_trials=args.n_trials,
            seeds=args.seeds,
            max_epoch=args.max_epoch,
            patience=args.patience,
            gpu=args.gpu,
            resume=args.resume,
            argv=sys.argv,
            args_dict=args_dict,
        )
        rows.append(row)
        write_final_summaries(output_root, rows)
        print(
            f"[{index}/{len(specs)}] Finished {spec.dataset} / {spec.task} / "
            f"RGCN / {spec.feature_mode}: {row.get('status')}"
        )

    write_final_summaries(output_root, rows)
    print(f"Final summaries written to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
