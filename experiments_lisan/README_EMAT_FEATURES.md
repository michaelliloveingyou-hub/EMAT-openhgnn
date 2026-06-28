# Lisan EMat Feature Modes

This document describes the EMat feature pipeline registered for `lisan-acm` and `lisan-dblp`.

## What EMat Is

`Dataset_Emat/<ACM|DBLP>/EMat_<type>.pt` stores basis-coordinate features with shape `num_basis x num_nodes_of_type`. OpenHGNN graph files store node features as `num_nodes_of_type x feature_dim`, so direct EMat modes transpose the tensor before writing `g.nodes[ntype].data["h"]`.

## Preprocessing

Run diagnostics and static preprocessing first:

```powershell
conda activate lisan-openhgnn
cd E:\Lisan_project
python .\scripts\diagnose_emat_features.py --datasets lisan-acm lisan-dblp --output_dir .\data\lisan_processed_features
python .\scripts\preprocess_emat_svd.py --datasets lisan-acm lisan-dblp --svd_dims 64 128 256 --output_dir .\data\lisan_processed_features --normalize zscore
python .\scripts\preprocess_emat_tfidf.py --datasets lisan-acm lisan-dblp --output_dir .\data\lisan_processed_features --tf_mode log1p
python .\scripts\preprocess_emat_tfidf_svd.py --datasets lisan-acm lisan-dblp --svd_dims 64 128 256 --output_dir .\data\lisan_processed_features --tf_mode log1p --normalize zscore
python .\scripts\preprocess_emat_sparse_inputs.py --datasets lisan-acm lisan-dblp --output_dir .\data\lisan_processed_features
```

Outputs are saved under `data/lisan_processed_features/<dataset>/`.

## Feature Modes

| Mode | Canonical name | Meaning |
|---|---|---|
| `A` | `raw` | Original `node.dat` features. |
| `B` | `emat` | Direct EMat basis coordinates. |
| `C` | `emat_svd_128` | Per-node-type TruncatedSVD from preprocessed EMat, 128 dimensions with z-score normalization. |
| `D` | `emat_tfidf_3025` | log1p TF-IDF/IDF weighted EMat, preserving the full basis dimension. DBLP uses 14314 basis dimensions. |
| `E` | `raw_emat_sparse_encoder` | Runtime learnable sparse encoder mode. It is not a static `graph.bin` feature file yet. |

Additional static modes are available, including `raw_emat_3025`, `raw_emat_svd_128`, `emat_tfidf_svd_128`, and `raw_emat_tfidf_svd_128`.

## Generate Static Graph Files

```powershell
python .\scripts\convert_lisan_to_openhgnn.py --datasets-root .\Datasets --emat-root .\Dataset_Emat --openhgnn-root .\OpenHGNN --processed-feature-root .\data\lisan_processed_features --dataset all --feature_mode emat_svd_128
python .\scripts\convert_lisan_to_openhgnn.py --datasets-root .\Datasets --emat-root .\Dataset_Emat --openhgnn-root .\OpenHGNN --processed-feature-root .\data\lisan_processed_features --dataset all --feature_mode raw_emat_tfidf_svd_128
```

Static graph files are written to `OpenHGNN/openhgnn/dataset/lisan_hgb/`. Raw graph files are not required to be regenerated.

## Run Experiments

Single OpenHGNN run:

```powershell
cd E:\Lisan_project\OpenHGNN
python .\main.py -m RGCN -d lisan-acm -t node_classification -g -1 --feature_mode C --max_epoch 50 --patience 10
python .\main.py -m RGCN -d lisan-acm -t node_classification -g -1 --feature_mode raw_emat_tfidf_svd_128 --max_epoch 50 --patience 10
```

Batch HPO and seed evaluation:

```powershell
cd E:\Lisan_project
python .\scripts\run_lisan_raw_experiments.py --feature_mode raw_emat_tfidf_svd_128 --datasets lisan-acm lisan-dblp --tasks node_classification link_prediction --models RGCN HAN HGT SimpleHGN --n_trials 50 --seeds 0 1 2 3 4 --max_epoch 200 --patience 20 --gpu 0 --resume
```

RGCN A-E sweep:

```powershell
python .\scripts\run_lisan_rgcn_feature_sweep.py --resume
```

The sweep now runs A-D and records E as skipped because `raw_emat_sparse_encoder` requires a runtime wrapper around the OpenHGNN backbone.

## Validate Feature Graphs

```powershell
python .\scripts\validate_lisan_feature_graphs.py --datasets lisan-acm lisan-dblp --feature_modes A B C D raw_emat_3025 raw_emat_svd_128 emat_tfidf_svd_128 raw_emat_tfidf_svd_128 E --output .\data\lisan_processed_features\feature_graph_validation.json
```

The validator checks edge order, node `global_id` order, labels, masks, feature shape, dtype, finite values, nonzero counts, and zero ratio. Runtime sparse encoder mode is skipped because no static graph file should exist for it.

## Current Limits

- Link prediction may have leakage risk because EMat was generated from the complete graph before train/valid/test edge splitting.
- `emat_tfidf_3025` means full basis dimension; DBLP's full basis dimension is 14314.
- `EMatSparseEncoder` exists in `experiments_lisan/models/emat_sparse_encoder.py`, and CSR inputs are saved by `scripts/preprocess_emat_sparse_inputs.py`, but the generic OpenHGNN/RGCN runner does not yet inject this trainable module.
