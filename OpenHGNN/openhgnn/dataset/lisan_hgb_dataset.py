# OpenHGNN 数据集注册文件功能说明：
# 本文件注册 Lisan_project 转换得到的 lisan-acm、lisan-dblp 数据集。
# 它分别支持 node_classification 和 link_prediction 两个任务：
# 节点分类读取转换脚本写入的 label/train_mask/val_mask/test_mask；
# 链接预测读取同一个 graph.bin，但按目标边类型构造标准 70/10/20 边划分，并处理反向边防止泄漏。

import os

import dgl
import torch as th
from dgl.data.utils import load_graphs

from . import register_dataset
from .LinkPredictionDataset import LinkPredictionDataset
from .NodeClassificationDataset import NodeClassificationDataset


LISAN_DATASETS = {
    "lisan-acm": {
        "files": {
            "raw": "lisan-acm.bin",
            "emat": "lisan-acm-emat.bin",
            "emat_svd_128": "lisan-acm-emat-svd-128.bin",
            "emat_tfidf_3025": "lisan-acm-emat-tfidf-3025.bin",
        },
        "category": "paper",
        "num_classes": 3,
        "node_type": ["author", "paper", "subject", "term"],
        "target_link": [("paper", "paper-ref-paper", "paper")],
        "target_link_r": [("paper", "paper-cite-paper", "paper")],
        "meta_paths_dict": {
            "PAP": [("paper", "paper-author", "author"), ("author", "author-paper", "paper")],
            "PSP": [("paper", "paper-subject", "subject"), ("subject", "subject-paper", "paper")],
            "PcPAP": [
                ("paper", "paper-cite-paper", "paper"),
                ("paper", "paper-author", "author"),
                ("author", "author-paper", "paper"),
            ],
            "PcPSP": [
                ("paper", "paper-cite-paper", "paper"),
                ("paper", "paper-subject", "subject"),
                ("subject", "subject-paper", "paper"),
            ],
            "PrPAP": [
                ("paper", "paper-ref-paper", "paper"),
                ("paper", "paper-author", "author"),
                ("author", "author-paper", "paper"),
            ],
            "PrPSP": [
                ("paper", "paper-ref-paper", "paper"),
                ("paper", "paper-subject", "subject"),
                ("subject", "subject-paper", "paper"),
            ],
        },
    },
    "lisan-dblp": {
        "files": {
            "raw": "lisan-dblp.bin",
            "emat": "lisan-dblp-emat.bin",
            "emat_svd_128": "lisan-dblp-emat-svd-128.bin",
            "emat_tfidf_3025": "lisan-dblp-emat-tfidf-3025.bin",
        },
        "category": "author",
        "num_classes": 4,
        "node_type": ["author", "paper", "term", "venue"],
        "target_link": [("author", "author-paper", "paper")],
        "target_link_r": [("paper", "paper-author", "author")],
        "meta_paths_dict": {
            "APA": [("author", "author-paper", "paper"), ("paper", "paper-author", "author")],
            "APTPA": [
                ("author", "author-paper", "paper"),
                ("paper", "paper-term", "term"),
                ("term", "term-paper", "paper"),
                ("paper", "paper-author", "author"),
            ],
            "APVPA": [
                ("author", "author-paper", "paper"),
                ("paper", "paper-venue", "venue"),
                ("venue", "venue-paper", "paper"),
                ("paper", "paper-author", "author"),
            ],
            "PAP": [("paper", "paper-author", "author"), ("author", "author-paper", "paper")],
            "PTP": [("paper", "paper-term", "term"), ("term", "term-paper", "paper")],
            "PVP": [("paper", "paper-venue", "venue"), ("venue", "venue-paper", "paper")],
        },
    },
}

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
SPARSE_ENCODER_MODES = {"emat_sparse_encoder", "raw_emat_sparse_encoder"}


def _feature_mode_from_kwargs(kwargs):
    args = kwargs.get("args")
    feature_mode = getattr(args, "feature_mode", "raw") if args is not None else "raw"
    if feature_mode not in FEATURE_MODE_ALIASES:
        raise ValueError("Unsupported Lisan feature_mode {}. Use A/B/C/D/E or a descriptive mode.".format(feature_mode))
    return FEATURE_MODE_ALIASES[feature_mode]


def _graph_path(dataset_name, feature_mode):
    if feature_mode in SPARSE_ENCODER_MODES:
        raise NotImplementedError(
            "{} is a learnable runtime encoder mode and is not stored as a static graph.bin. "
            "Build sparse inputs with scripts/preprocess_emat_sparse_inputs.py and use a model wrapper that "
            "instantiates experiments_lisan.models.EMatSparseEncoder.".format(feature_mode)
        )
    spec = LISAN_DATASETS[dataset_name]
    filename = spec["files"].get(feature_mode)
    if filename is None:
        stem = spec["files"]["raw"][:-4]
        filename = "{}-{}.bin".format(stem, feature_mode.replace("_", "-"))
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "lisan_hgb", filename)


def _load_lisan_graph(dataset_name, feature_mode):
    path = _graph_path(dataset_name, feature_mode)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Missing converted Lisan graph file: {}. Run scripts\\convert_lisan_to_openhgnn.py "
            "--feature-mode {} in E:\\Lisan_project first.".format(path, feature_mode)
        )
    graph = load_graphs(path)[0][0].long()
    return graph


@register_dataset("lisan_hgb_node_classification")
class LisanHGBNodeClassification(NodeClassificationDataset):
    def __init__(self, dataset_name, *args, **kwargs):
        super(LisanHGBNodeClassification, self).__init__(*args, **kwargs)
        if dataset_name not in LISAN_DATASETS:
            raise ValueError("Unsupported Lisan dataset {}".format(dataset_name))
        spec = LISAN_DATASETS[dataset_name]
        self.feature_mode = _feature_mode_from_kwargs(kwargs)
        self.dataset_name = dataset_name
        self.g = _load_lisan_graph(dataset_name, self.feature_mode)
        self.category = spec["category"]
        self.num_classes = spec["num_classes"]
        self.multi_label = False
        self.has_feature = True
        self.meta_paths_dict = spec["meta_paths_dict"]
        self.in_dim = self.g.nodes[self.category].data["h"].shape[1]


@register_dataset("lisan_hgb_link_prediction")
class LisanHGBLinkPrediction(LinkPredictionDataset):
    def __init__(self, dataset_name, *args, **kwargs):
        super(LisanHGBLinkPrediction, self).__init__(*args, **kwargs)
        if dataset_name not in LISAN_DATASETS:
            raise ValueError("Unsupported Lisan dataset {}".format(dataset_name))
        spec = LISAN_DATASETS[dataset_name]
        self.feature_mode = _feature_mode_from_kwargs(kwargs)
        self.dataset_name = dataset_name
        self.g = _load_lisan_graph(dataset_name, self.feature_mode)
        self.has_feature = True
        self.target_link = spec["target_link"]
        self.target_link_r = spec["target_link_r"]
        self.node_type = spec["node_type"]
        self.meta_paths_dict = spec["meta_paths_dict"]

    def get_split(self, val_ratio=0.1, test_ratio=0.2):
        val_edge_dict = {}
        test_edge_dict = {}
        out_ntypes = []
        train_graph = self.g

        for i, etype in enumerate(self.target_link):
            num_edges = self.g.num_edges(etype)
            random_int = th.randperm(num_edges)
            val_count = int(num_edges * val_ratio)
            test_count = int(num_edges * test_ratio)
            val_index = random_int[:val_count]
            test_index = random_int[val_count:val_count + test_count]
            remove_index = th.cat((val_index, test_index))

            val_edge = self.g.find_edges(val_index, etype)
            test_edge = self.g.find_edges(test_index, etype)
            val_edge_dict[etype] = val_edge
            test_edge_dict[etype] = test_edge
            out_ntypes.extend([etype[0], etype[2]])

            train_graph = dgl.remove_edges(train_graph, remove_index, etype)
            if self.target_link_r is not None:
                reverse_edge = self.target_link_r[i]
                train_graph = dgl.remove_edges(train_graph, th.arange(train_graph.num_edges(reverse_edge)), reverse_edge)
                train_edges = train_graph.edges(etype=etype)
                train_graph = dgl.add_edges(train_graph, train_edges[1], train_edges[0], etype=reverse_edge)

        self.out_ntypes = set(out_ntypes)
        num_nodes_dict = {ntype: self.g.number_of_nodes(ntype) for ntype in self.out_ntypes}
        val_graph = dgl.heterograph(val_edge_dict, num_nodes_dict)
        test_graph = dgl.heterograph(test_edge_dict, num_nodes_dict)
        return train_graph, val_graph, test_graph, None, None
