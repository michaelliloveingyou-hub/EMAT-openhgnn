# Lisan OpenHGNN Experiments

This package runs raw-feature and EMat-feature OpenHGNN baselines for `lisan-acm` and `lisan-dblp`.

## Goal

Run reproducible baseline experiments for:

- datasets: `lisan-acm`, `lisan-dblp`
- tasks: `node_classification`, `link_prediction`
- node-classification models: `RGCN`, `HAN`, `HGT`, `SimpleHGN`, `GCN`, `GAT`
- link-prediction models: `RGCN`, `HGT`, `SimpleHGN`, `GCN`, `GAT`

Use `--feature_mode` to run the same workflow on the selected static feature graph. The HPO search space, seeds, tasks, models, and split logic are intentionally shared across feature modes.

Feature mode mapping:

- `A`: raw `node.dat` features.
- `B`: direct EMat features.
- `C`: EMat-SVD-128 features.
- `D`: EMat-TFIDF full-basis features.
- `E`: `raw_emat_sparse_encoder`; this is a runtime learnable encoder mode and is skipped by the generic OpenHGNN/RGCN runner until a wrapper injects `EMatSparseEncoder`.

## Workflow

For every `dataset x task x model` combination, the runner:

1. searches hyperparameters with Optuna;
2. selects best parameters using validation metrics only;
3. retrains with best parameters on the requested seeds;
4. writes per-trial, per-seed, per-combination, and final summary outputs.

Node classification uses validation `Macro_f1` as the HPO objective. Link prediction uses OpenHGNN validation `roc_auc`.

## Output

Default raw output root:

```text
experiments/lisan_raw_runs/
```

Default EMat output root:

```text
experiments/lisan_emat_runs/
```

Each combination writes:

- `hpo_trials.csv`
- `best_params.json`
- `seed_results.csv`
- `summary.json`
- `metadata.json`
- `run.log`
- `error.log` if failures occur

Final tables:

- `experiments/lisan_raw_runs/final_summary_node_classification.csv`
- `experiments/lisan_raw_runs/final_summary_link_prediction.csv`
- `experiments/lisan_emat_runs/final_summary_node_classification.csv`
- `experiments/lisan_emat_runs/final_summary_link_prediction.csv`

The current OpenHGNN link-prediction path reports `roc_auc` and `loss`; AP, MRR, and Hits@10 summary columns are kept blank.

`GCN` and `GAT` are implemented through OpenHGNN's `homo_GNN` backend. The experiment output directories and summaries still use the user-facing names `GCN` and `GAT`; internally the runner sets `gnn_type=gcnconv` or `gnn_type=gatconv`.

## PowerShell

```powershell
conda activate lisan-openhgnn
cd E:\Lisan_project
python .\scripts\run_lisan_raw_experiments.py `
  --feature_mode A `
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

EMat feature version:

```powershell
python .\scripts\run_lisan_raw_experiments.py `
  --feature_mode raw_emat_tfidf_svd_128 `
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

HGT, SimpleHGN, GCN, and GAT feature sweep:

```powershell
python .\scripts\run_lisan_model_feature_sweep.py `
  --n_trials 50 `
  --seeds 0 1 2 3 4 `
  --max_epoch 200 `
  --patience 20 `
  --gpu 0 `
  --resume
```

The default sweep covers `HGT SimpleHGN GCN GAT x A B C D E x lisan-acm lisan-dblp x node_classification link_prediction`. Feature mode `E` is recorded as skipped until the runtime `EMatSparseEncoder` wrapper is implemented.

## CMD

```cmd
conda activate lisan-openhgnn
cd /d E:\Lisan_project
python scripts\run_lisan_raw_experiments.py --feature_mode A --datasets lisan-acm lisan-dblp --tasks node_classification link_prediction --models RGCN HAN HGT SimpleHGN --n_trials 50 --seeds 0 1 2 3 4 --max_epoch 200 --patience 20 --gpu 0 --resume
python scripts\run_lisan_raw_experiments.py --feature_mode raw_emat_tfidf_svd_128 --datasets lisan-acm lisan-dblp --tasks node_classification link_prediction --models RGCN HAN HGT SimpleHGN --n_trials 50 --seeds 0 1 2 3 4 --max_epoch 200 --patience 20 --gpu 0 --resume
```

## Dry Run

```powershell
python .\scripts\run_lisan_raw_experiments.py --feature_mode C --dry_run
```

## Resume

```powershell
python .\scripts\run_lisan_raw_experiments.py --resume
```

Completed combinations are skipped. Completed seed rows are skipped; missing or failed seed rows are rerun.

## Single Dataset Or Model

```powershell
python .\scripts\run_lisan_raw_experiments.py --datasets lisan-acm --tasks node_classification --models RGCN --n_trials 10 --seeds 0 1 2 3 4 --gpu -1 --resume
python .\scripts\run_lisan_raw_experiments.py --feature_mode C --datasets lisan-acm --tasks node_classification --models RGCN --n_trials 10 --seeds 0 1 2 3 4 --gpu -1 --resume
```
