from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENHGNN_ROOT = PROJECT_ROOT / "OpenHGNN"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "lisan_raw_runs"
DEFAULT_EMAT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "lisan_emat_runs"
FEATURE_MODE_ALIASES = {
    "A": "raw",
    "B": "emat",
    "C": "emat_svd_128",
    "D": "emat_tfidf_3025",
    "E": "raw_emat_sparse_encoder",
    "raw": "raw",
    "emat": "emat",
    "emat_3025": "emat",
    "raw_emat_3025": "raw_emat_3025",
    "emat_svd_64": "emat_svd_64",
    "emat_svd_128": "emat_svd_128",
    "emat_svd_256": "emat_svd_256",
    "raw_emat_svd_64": "raw_emat_svd_64",
    "raw_emat_svd_128": "raw_emat_svd_128",
    "raw_emat_svd_256": "raw_emat_svd_256",
    "emat_tfidf": "emat_tfidf_3025",
    "emat_tfidf_3025": "emat_tfidf_3025",
    "raw_emat_tfidf_3025": "raw_emat_tfidf_3025",
    "emat_tfidf_svd_64": "emat_tfidf_svd_64",
    "emat_tfidf_svd_128": "emat_tfidf_svd_128",
    "emat_tfidf_svd_256": "emat_tfidf_svd_256",
    "raw_emat_tfidf_svd_64": "raw_emat_tfidf_svd_64",
    "raw_emat_tfidf_svd_128": "raw_emat_tfidf_svd_128",
    "raw_emat_tfidf_svd_256": "raw_emat_tfidf_svd_256",
    "emat_sparse_encoder": "emat_sparse_encoder",
    "raw_emat_sparse_encoder": "raw_emat_sparse_encoder",
}
FEATURE_MODES = tuple(FEATURE_MODE_ALIASES.keys())
CANONICAL_FEATURE_MODES = tuple(dict.fromkeys(FEATURE_MODE_ALIASES.values()))
SPARSE_ENCODER_FEATURE_MODES = {"emat_sparse_encoder", "raw_emat_sparse_encoder"}


def normalize_feature_mode(feature_mode: str) -> str:
    return FEATURE_MODE_ALIASES.get(feature_mode, feature_mode)

DATASETS = ("lisan-acm", "lisan-dblp")
TASKS = ("node_classification", "link_prediction")
MODELS = ("RGCN", "HAN", "HGT", "SimpleHGN", "GCN", "GAT")

SUPPORTED_MODELS_BY_TASK = {
    "node_classification": {"RGCN", "HAN", "HGT", "SimpleHGN", "GCN", "GAT"},
    "link_prediction": {"RGCN", "HGT", "SimpleHGN", "GCN", "GAT"},
}

OPENHGNN_MODEL_ALIASES = {
    "GCN": "homo_GNN",
    "GAT": "homo_GNN",
}


def openhgnn_model_name(model: str) -> str:
    return OPENHGNN_MODEL_ALIASES.get(model, model)

NODE_SUMMARY_COLUMNS = [
    "task",
    "dataset",
    "model",
    "feature_mode",
    "n_trials",
    "seeds",
    "best_val_macro_f1",
    "test_macro_f1_mean",
    "test_macro_f1_std",
    "test_micro_f1_mean",
    "test_micro_f1_std",
    "test_accuracy_mean",
    "test_accuracy_std",
    "best_params_path",
    "status",
    "notes",
]

LINK_SUMMARY_COLUMNS = [
    "task",
    "dataset",
    "model",
    "feature_mode",
    "n_trials",
    "seeds",
    "best_val_auc",
    "best_val_ap",
    "test_auc_mean",
    "test_auc_std",
    "test_ap_mean",
    "test_ap_std",
    "test_mrr_mean",
    "test_mrr_std",
    "test_hits10_mean",
    "test_hits10_std",
    "best_params_path",
    "status",
    "notes",
]


@dataclass(frozen=True)
class ExperimentSpec:
    dataset: str
    task: str
    model: str
    feature_mode: str = "raw"

    @property
    def combo_id(self) -> str:
        return f"{self.dataset}__{normalize_feature_mode(self.feature_mode)}__{self.task}__{self.model}"

    def combo_dir(self, output_root: Path) -> Path:
        return output_root / self.task / self.dataset / self.model / normalize_feature_mode(self.feature_mode)


def is_supported(spec: ExperimentSpec) -> tuple[bool, str]:
    feature_mode = normalize_feature_mode(spec.feature_mode)
    if feature_mode not in CANONICAL_FEATURE_MODES:
        return False, f"unsupported feature mode: {spec.feature_mode}"
    if feature_mode in SPARSE_ENCODER_FEATURE_MODES:
        return False, (
            f"{feature_mode} needs a runtime EMatSparseEncoder wrapper and is not supported by the "
            "generic OpenHGNN experiment runner yet"
        )
    if spec.dataset not in DATASETS:
        return False, f"unsupported dataset: {spec.dataset}"
    if spec.task not in TASKS:
        return False, f"unsupported task: {spec.task}"
    if spec.model not in MODELS:
        return False, f"unsupported model: {spec.model}"
    if spec.model not in SUPPORTED_MODELS_BY_TASK[spec.task]:
        return False, f"{spec.model} is outside the raw-baseline support matrix for {spec.task}"
    return True, ""
