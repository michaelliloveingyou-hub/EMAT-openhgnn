"""
builder.py

ASHIN 接到 OpenHGNN 训练流程里的主文件。

这里管几件事：选 B/C/D/E/F/G/H 版本、读写 cache、把特征写回图、写实验日志。
B/C 是目标节点 concat；D/F/G/H 是目标节点 gated；E 会把所有节点类型都跑一遍。
新增版本时优先在这里接入，不要把版本判断散到 trainerflow 里。
"""

import time

import torch

from .cache import cache_path, load_cache, save_cache
from .logger import ensure_run_logger, write_ashin_metadata, write_graph_info
from .version_b import build_ashin_b
from .version_c import build_ashin_c
from .version_g import build_ashin_g
from .version_h import build_ashin_h


def resolve_target_ntype(dataset, args):
    for source in (dataset, args):
        target = getattr(source, "category", None)
        if target is not None:
            return target
    raise ValueError("ASHIN requires target node type, but dataset.category/args.category is unavailable.")


def _base_version(args, version):
    # D/G/H 的结构特征都以 C 的低维 dense 表示为基础。
    # E/F 是可切换版本，base_version 才真正来自命令行或 best_params。
    if version in ("D", "G", "H"):
        return "C"
    if version in ("E", "F"):
        base_version = getattr(args, "ashin_base_version", "B")
        if base_version not in ("B", "C"):
            raise ValueError("ASHIN-E/F requires --ashin_base_version {B,C}.")
        return base_version
    return version


def _cache_version(version, base_version):
    return version if version in ("B", "C") else f"{version}_{base_version}"


def effective_norm(args, version):
    # B 的默认形式应该保留严格 0/1 incidence。
    # 只有用户显式传了 --ashin_norm，才允许覆盖这个默认行为。
    norm = getattr(args, "ashin_norm", "log1p_zscore")
    user_set = getattr(args, "ashin_norm_user_set", False)
    base_version = _base_version(args, version)
    if base_version == "B" and not user_set:
        return "none"
    return norm


def _feature_field(g, ntype):
    # OpenHGNN 数据集里常见字段是 h；少数数据会叫 feat。
    # 如果某类节点没有特征，给一个 1 维零向量，保证后续拼接逻辑能走通。
    data = g.nodes[ntype].data
    if "h" in data:
        return "h", data["h"].detach().cpu()
    if "feat" in data:
        return "feat", data["feat"].detach().cpu()
    return "h", torch.zeros((g.num_nodes(ntype), 1), dtype=torch.float32)


def _density(x):
    if x.numel() == 0:
        return 0.0
    return float((x != 0).sum().item()) / float(x.numel())


def build_ashin_features(g, dataset, args, target_ntype=None):
    target_ntype = target_ntype or resolve_target_ntype(dataset, args)
    version = getattr(args, "ashin_version", None)
    if version not in ("B", "C", "D", "E", "F", "G", "H"):
        raise ValueError("ASHIN is enabled, but ashin_version must be one of {B,C,D,E,F,G,H}.")

    base_version = _base_version(args, version)
    norm = effective_norm(args, version)
    cache_version = _cache_version(version, base_version)
    path = cache_path(args, g, target_ntype, cache_version, norm)

    if not getattr(args, "ashin_rebuild", False):
        payload = load_cache(path)
        if payload is not None:
            payload.setdefault("metadata", {})
            payload["metadata"]["cache_hit"] = True
            payload["metadata"]["cache_path"] = str(path)
            payload["metadata"].setdefault("ashin_base_version", base_version)
            payload["metadata"].setdefault("ashin_cache_version", cache_version)
            return payload

    start = time.time()
    # 这里是所有版本真正分流的位置。新增版本时尽量只往这里接入，
    # 后面的写回、日志、cache 元信息保持共用，减少实验记录不一致。
    if version == "G":
        version_result = build_ashin_g(
            g,
            target_ntype,
            norm=norm,
            ashin_dim=getattr(args, "ashin_dim", 128),
            seed=getattr(args, "seed", 0),
            attr_agg=getattr(args, "ashin_attr_agg", "mean"),
        )
    elif version == "H":
        version_result = build_ashin_h(
            g,
            target_ntype,
            norm=norm,
            ashin_dim=getattr(args, "ashin_dim", 128),
            seed=getattr(args, "seed", 0),
            common_op=getattr(args, "ashin_common_op", "max"),
            common_norm=getattr(args, "ashin_common_norm", "row"),
            common_topk=getattr(args, "ashin_common_topk", 0),
        )
    elif base_version == "B":
        version_result = build_ashin_b(g, target_ntype, norm=norm)
    else:
        version_result = build_ashin_c(
            g,
            target_ntype,
            norm=norm,
            ashin_dim=getattr(args, "ashin_dim", 128),
            seed=getattr(args, "seed", 0),
        )

    x_ashin = version_result["x_ashin"]
    metadata = {
        "ashin_version": version,
        "ashin_base_version": base_version,
        "ashin_cache_version": cache_version,
        "unit_definition": "Each selected node is a unit core; all one-hop neighbors under sorted canonical edge types are included.",
        "signature_construction_rules": (
            "For each sorted canonical edge type: src type, relation, dst type, direction, "
            "core degree, neighbor count, neighbor degree sum, neighbor degree mean x1000, neighbor degree max."
        ),
        "target_ntype": str(target_ntype),
        "num_target_nodes": int(g.num_nodes(target_ntype)),
        "num_signatures": int(version_result["num_signatures"]),
        "signature_dim": int(x_ashin.shape[1]),
        "feature_density": _density(x_ashin),
        "normalization": norm,
        "cache_path": str(path),
        "cache_hit": False,
        "build_time_seconds": float(time.time() - start),
        "top10_signature_frequency": version_result["signature_freq"][:10],
        "reduction": version_result["reduction"],
    }
    payload = {
        "x_ashin": x_ashin.cpu().float(),
        "target_ntype": str(target_ntype),
        "ashin_version": version,
        "ashin_base_version": base_version,
        "signature_to_id": version_result["signature_to_id"],
        "signature_freq": version_result["signature_freq"],
        "metadata": metadata,
    }
    save_cache(path, payload)
    return payload


def _apply_one_ntype(g, dataset, args, ntype):
    field, x_raw = _feature_field(g, ntype)
    payload = build_ashin_features(g, dataset, args, target_ntype=ntype)
    x_ashin = payload["x_ashin"].cpu().float()
    if x_raw.shape[0] != x_ashin.shape[0]:
        raise ValueError(
            f"ASHIN row mismatch: raw rows={x_raw.shape[0]}, ASHIN rows={x_ashin.shape[0]}, target_ntype={ntype}."
        )
    x_new = torch.cat([x_raw.float(), x_ashin.float()], dim=1)
    g.nodes[ntype].data[field] = x_new
    return {
        "field": field,
        "raw_dim": int(x_raw.shape[1]),
        "ashin_dim": int(x_ashin.shape[1]),
        "final_dim": int(x_new.shape[1]),
        "payload": payload,
    }


def apply_ashin_features(g, dataset, args):
    if not getattr(args, "use_ashin", False):
        return g

    ensure_run_logger(args)
    target_ntype = resolve_target_ntype(dataset, args)
    version = getattr(args, "ashin_version", None)

    if version == "E":
        # E 的特殊点是“所有节点类型都增强”。每个 ntype 都会被临时当作 target_ntype
        # 构造一次 ASHIN，因此日志里需要按节点类型分别保存 metadata。
        results = {}
        metadata = {
            "ashin_version": version,
            "ashin_base_version": getattr(args, "ashin_base_version", "B"),
            "target_ntype": str(target_ntype),
            "mode": "all_node_type_feature_augmentation",
            "node_types": {},
        }
        for ntype in sorted(g.ntypes, key=str):
            result = _apply_one_ntype(g, dataset, args, ntype)
            results[str(ntype)] = result
            metadata["node_types"][str(ntype)] = result["payload"]["metadata"]

        target_result = results[str(target_ntype)]
        setattr(args, "ashin_target_ntype", target_ntype)
        setattr(args, "ashin_enhanced_ntypes", sorted([str(n) for n in g.ntypes]))
        setattr(args, "ashin_feature_dims", {ntype: result["ashin_dim"] for ntype, result in results.items()})
        setattr(args, "ashin_feature_dim", target_result["ashin_dim"])
        setattr(args, "ashin_fusion", "concat")
        setattr(args, "ashin_cache_path", metadata["node_types"][str(target_ntype)].get("cache_path"))
        write_graph_info(args, g, target_ntype, target_result["raw_dim"], target_result["ashin_dim"], target_result["final_dim"])
        write_ashin_metadata(args, metadata)
        print(
            "[ASHIN] version=E, base={}, enhanced_ntypes={}, target={}, target_ashin_dim={}".format(
                getattr(args, "ashin_base_version", "B"),
                ",".join(sorted([str(n) for n in g.ntypes])),
                target_ntype,
                target_result["ashin_dim"],
            )
        )
        return g

    result = _apply_one_ntype(g, dataset, args, target_ntype)
    payload = result["payload"]
    if version in ("D", "F", "G", "H"):
        # 这几个版本仍然先把 raw 和 ASHIN 拼在图特征里，
        # 但会把切分维度交给 HeteroFeature，后面用 gated 融合而不是普通线性层。
        setattr(args, "ashin_fusion", "gated")
        setattr(
            args,
            "ashin_split_dims",
            {str(target_ntype): {"raw_dim": result["raw_dim"], "ashin_dim": result["ashin_dim"]}},
        )
    else:
        setattr(args, "ashin_fusion", "concat")

    setattr(args, "ashin_target_ntype", target_ntype)
    setattr(args, "ashin_feature_dim", result["ashin_dim"])
    setattr(args, "ashin_cache_path", payload["metadata"].get("cache_path"))
    write_graph_info(args, g, target_ntype, result["raw_dim"], result["ashin_dim"], result["final_dim"])
    write_ashin_metadata(args, payload["metadata"])
    print(
        "[ASHIN] version={}, base={}, target_ntype={}, nodes={}, signatures={}, ashin_dim={}, density={:.6f}, cache_hit={}, fusion={}".format(
            version,
            payload["metadata"].get("ashin_base_version", version),
            target_ntype,
            g.num_nodes(target_ntype),
            payload["metadata"].get("num_signatures"),
            result["ashin_dim"],
            payload["metadata"].get("feature_density", 0.0),
            payload["metadata"].get("cache_hit"),
            getattr(args, "ashin_fusion", "concat"),
        )
    )
    return g
