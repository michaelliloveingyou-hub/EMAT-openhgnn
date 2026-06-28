import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments_lisan.config import (
    DATASETS,
    DEFAULT_EMAT_OUTPUT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    FEATURE_MODES,
    MODELS,
    TASKS,
    ExperimentSpec,
    is_supported,
    normalize_feature_mode,
)
from experiments_lisan.io_utils import ensure_dir
from experiments_lisan.summary import write_final_summaries


def parse_args():
    parser = argparse.ArgumentParser(description="Run Lisan OpenHGNN baseline experiments.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument(
        "--feature_mode",
        "--feature-mode",
        choices=FEATURE_MODES,
        default="raw",
        help="raw uses original node.dat features; emat uses Dataset_Emat graph files.",
    )
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    canonical_feature_mode = normalize_feature_mode(args.feature_mode)
    default_output_root = DEFAULT_OUTPUT_ROOT if canonical_feature_mode == "raw" else DEFAULT_EMAT_OUTPUT_ROOT
    output_root = (args.output_root or default_output_root).resolve()
    specs = [
        ExperimentSpec(dataset=dataset, task=task, model=model, feature_mode=args.feature_mode)
        for dataset in args.datasets
        for task in args.tasks
        for model in args.models
    ]

    if args.dry_run:
        for spec in specs:
            supported, reason = is_supported(spec)
            status = "run" if supported else f"skip: {reason}"
            print(f"{spec.dataset}\t{spec.task}\t{spec.model}\t{status}")
        return 0

    ensure_dir(output_root)
    from experiments_lisan.runner import run_combination

    rows = []
    args_dict = vars(args).copy()
    args_dict["output_root"] = str(output_root)
    for spec in specs:
        print(f"Running {spec.dataset} / {spec.task} / {spec.model}")
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
        print(f"Finished {spec.dataset} / {spec.task} / {spec.model}: {row.get('status')}")

    write_final_summaries(output_root, rows)
    print(f"Final summaries written to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
