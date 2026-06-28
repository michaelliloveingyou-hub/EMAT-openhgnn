"""
validator.py

ASHIN 构造检查。

这里放一些小测试：toy graph、B 的 0/1 incidence、dense 版本的维度和 NaN/Inf、
signature 可复现、cache 命中、写回图后的维度。

这些检查只是为了确认构造逻辑没坏，不参与训练。
"""

import argparse
import tempfile
from types import SimpleNamespace

import dgl
import torch

from .builder import apply_ashin_features, build_ashin_features
from .signature import build_node_signatures, sorted_canonical_etypes, sorted_ntypes


def make_toy_graph():
    # 手工构造一个很小的异质图，用于确认 ASHIN-B 不是普通 degree feature，
    # 而是“每个 paper 一个 unit signature”的 incidence 特征。
    data = {
        ("author", "writes", "paper"): (torch.tensor([0, 1, 1]), torch.tensor([0, 0, 1])),
        ("paper", "has_term", "term"): (torch.tensor([0, 1]), torch.tensor([0, 0])),
        ("paper", "published_conf", "conf"): (torch.tensor([0, 1]), torch.tensor([0, 0])),
    }
    g = dgl.heterograph(data, num_nodes_dict={"paper": 2, "author": 2, "term": 1, "conf": 1})
    g.nodes["paper"].data["h"] = torch.eye(2)
    return g


def default_args(dataset="toy", version="B", ashin_dim=128, cache_dir=None):
    # 构造一个最小 args 对象，模拟 main.py 传入 ASHIN builder 的参数。
    return SimpleNamespace(
        use_ashin=True,
        ashin_version=version,
        ashin_dim=ashin_dim,
        ashin_norm="none" if version == "B" else "log1p_zscore",
        ashin_norm_user_set=True,
        ashin_cache_dir=cache_dir or tempfile.mkdtemp(prefix="ashin_cache_"),
        ashin_rebuild=True,
        ashin_log_dir=tempfile.mkdtemp(prefix="ashin_logs_"),
        run_name=None,
        dataset=dataset,
        dataset_name=dataset,
        model="validator",
        model_name="validator",
        task="node_classification",
        seed=0,
        ashin_attr_agg="mean",
        ashin_common_op="max",
        ashin_common_norm="row",
        ashin_common_topk=0,
    )


def assert_basic(payload, g, target_ntype, version, ashin_dim):
    # 基础数值和形状检查：行数、维度、NaN/Inf。
    # B 是稀疏 one-hot，C/D/E/F/H 是固定 ashin_dim，G 会按属性类型拼成多个 block。
    x = payload["x_ashin"]
    assert x.shape[0] == g.num_nodes(target_ntype), "ASHIN row count does not match target node count."
    assert not torch.isnan(x).any(), "ASHIN feature contains NaN."
    assert not torch.isinf(x).any(), "ASHIN feature contains Inf."
    if version == "B":
        assert x.shape[1] == len(payload["signature_to_id"]), "Version B dim != signature vocabulary size."
        assert torch.all((x == 0) | (x == 1)), "Version B must be 0/1 incidence."
        assert torch.all(x.sum(dim=1) >= 1), "Every Version B row must have at least one nonzero."
    elif version == "G":
        assert x.shape[1] >= ashin_dim, "Version G dim should contain at least one attribute block."
        assert x.shape[1] % ashin_dim == 0, "Version G dim should be attribute_count * ashin_dim."
        assert x.abs().sum() > 0, "Version G must not be all zero."
    elif version == "H":
        assert x.shape[1] == ashin_dim, "Version H dim must equal ashin_dim."
    else:
        assert x.shape[1] == ashin_dim, "Version C/D/E/F dim must equal ashin_dim for one target build."
        assert x.abs().sum() > 0, "Version C/D/E/F must not be all zero."


def run_toy_test():
    # toy graph 是最重要的单元测试，用来验证 signature vocabulary 的基本行为。
    g = make_toy_graph()
    dataset = SimpleNamespace(category="paper")
    args = default_args(version="B")
    payload = build_ashin_features(g, dataset, args)
    assert_basic(payload, g, "paper", "B", 128)
    signatures, signature_to_id, _ = build_node_signatures(g, "paper")
    assert len(signature_to_id) == 2, "Toy graph should produce two distinct paper signatures."
    assert signatures[0] != signatures[1], "Toy graph manual expectation: p0 and p1 signatures differ."
    assert sorted_canonical_etypes(g) == sorted(sorted_canonical_etypes(g), key=lambda e: (str(e[0]), str(e[1]), str(e[2])))
    assert sorted_ntypes(g) == ["author", "conf", "paper", "term"]
    print("[ASHIN validate] toy graph test passed.")


def run_repro_test(g, dataset, args):
    args.ashin_rebuild = True
    first = build_ashin_features(g, dataset, args)
    second = build_ashin_features(g, dataset, args)
    assert first["signature_to_id"] == second["signature_to_id"], "signature_to_id is not reproducible."
    if args.ashin_version in ("C", "D", "E", "F", "G", "H"):
        assert torch.allclose(first["x_ashin"], second["x_ashin"], atol=1e-5), "ASHIN dense version is not reproducible."
    print("[ASHIN validate] reproducibility test passed.")


def run_cache_test(g, dataset, args):
    args.ashin_rebuild = True
    first = build_ashin_features(g, dataset, args)
    args.ashin_rebuild = False
    second = build_ashin_features(g, dataset, args)
    assert second["metadata"]["cache_hit"], "Second build should hit cache."
    args.ashin_rebuild = True
    third = build_ashin_features(g, dataset, args)
    assert not third["metadata"]["cache_hit"], "ashin_rebuild should force rebuild."
    print(f"[ASHIN validate] cache test passed: {first['metadata']['cache_path']}")


def run_concat_test(g, dataset, args):
    # 函数名沿用早期叫法。现在 D/F/G/H 后续会做 gated，
    # 但图上写回时仍然是 raw 与 ASHIN 先拼起来，所以这里检查的仍是写回维度。
    target = dataset.category
    raw_dim = g.nodes[target].data.get("h", g.nodes[target].data.get("feat")).shape[1]
    args.ashin_rebuild = True
    g2 = apply_ashin_features(g, dataset, args)
    new_dim = g2.nodes[target].data.get("h", g2.nodes[target].data.get("feat")).shape[1]
    assert new_dim == raw_dim + args.ashin_feature_dim, "Concatenated dim mismatch."
    print("[ASHIN validate] concatenation test passed.")
