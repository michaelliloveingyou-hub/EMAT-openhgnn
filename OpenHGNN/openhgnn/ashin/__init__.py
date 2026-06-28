"""
openhgnn/ashin/__init__.py

ASHIN 包入口。
外面只从这里拿两个函数就够了，内部版本分发都放在 builder.py。

apply_ashin_features 会把增强特征写回图；build_ashin_features 只返回某个节点类型的 ASHIN 特征。
"""

from .builder import apply_ashin_features, build_ashin_features

__all__ = ["apply_ashin_features", "build_ashin_features"]
