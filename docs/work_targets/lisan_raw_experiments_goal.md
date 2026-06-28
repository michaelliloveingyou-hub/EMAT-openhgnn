# Lisan Raw OpenHGNN Baseline Experiments Goal

## Scope

Build an automated raw-feature baseline experiment system for `lisan-acm` and `lisan-dblp` in the project-local OpenHGNN checkout.

Only the existing raw node features are used in this phase. Do not implement `raw_random`, `aug_only`, or `raw_aug`.

## Entry Point

Create:

- `scripts/run_lisan_raw_experiments.py`
- `experiments_lisan/`

Default output root:

- `experiments/lisan_raw_runs/`

Required command:

```powershell
python scripts/run_lisan_raw_experiments.py `
  --datasets lisan-acm lisan-dblp `
  --tasks node_classification link_prediction `
  --models RGCN HAN HGT SimpleHGN `
  --n_trials 50 `
  --seeds 0 1 2 3 4 `
  --max_epoch 200 `
  --patience 20 `
  --gpu 0 `
  --resume
```

## Supported Matrix

Node classification:

- RGCN
- HAN
- HGT
- SimpleHGN

Link prediction:

- RGCN
- HGT
- SimpleHGN

`HAN` with `link_prediction` is outside this raw-baseline matrix and is recorded as `skipped` if requested.

## HPO Rules

Use Optuna per `dataset x task x model`.

The objective must use validation metrics only.

- Node classification objective: `valid Macro_f1`
- Link prediction objective: `valid roc_auc`
- Never use test metrics for HPO selection

Save per combination:

- `hpo_trials.csv`
- `best_params.json`
- `seed_results.csv`
- `summary.json`
- `metadata.json`
- `run.log`
- `error.log` on failure

## Final Evaluation

After HPO, train with best params on seeds `[0, 1, 2, 3, 4]`.

Seed control must cover:

- Python random
- NumPy
- PyTorch CPU/CUDA
- DGL
- cuDNN deterministic settings where available

Report mean and std from test metrics only.

## Final Summaries

Write:

- `experiments/lisan_raw_runs/final_summary_node_classification.csv`
- `experiments/lisan_raw_runs/final_summary_link_prediction.csv`

Node classification columns:

`task,dataset,model,n_trials,seeds,best_val_macro_f1,test_macro_f1_mean,test_macro_f1_std,test_micro_f1_mean,test_micro_f1_std,test_accuracy_mean,test_accuracy_std,best_params_path,status,notes`

Link prediction columns:

`task,dataset,model,n_trials,seeds,best_val_auc,best_val_ap,test_auc_mean,test_auc_std,test_ap_mean,test_ap_std,test_mrr_mean,test_mrr_std,test_hits10_mean,test_hits10_std,best_params_path,status,notes`

The current OpenHGNN link-prediction path reports `roc_auc` and `loss`; AP, MRR, and Hits@10 columns remain present but blank with notes.

## Resume And Failure Behavior

With `--resume`:

- skip completed HPO if `best_params.json` and a successful `summary.json` exist
- skip seed runs already recorded as successful in `seed_results.csv`
- rerun missing or failed pieces

A failed combination must not stop the full run. Record stack traces in that combination's `error.log`.

## Metadata

Save metadata for every combination:

- full command
- parsed arguments
- dataset/task/model
- feature mode: `raw`
- Git commit hash or `not_a_git_repo`
- Python version
- PyTorch version
- DGL version
- CUDA availability/device info
- OpenHGNN config overrides
- start/end time and duration

## Documentation

Add `experiments_lisan/README.md`.

Update root `README.md` with:

- raw-baseline workflow
- configuration notes
- output directory
- PowerShell and CMD examples
- resume examples
- single dataset/model examples
