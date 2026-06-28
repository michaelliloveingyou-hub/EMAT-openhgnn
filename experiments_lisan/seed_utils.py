import os
import random

import numpy as np
import torch

try:
    import dgl
except Exception:  # pragma: no cover - DGL import is environment dependent.
    dgl = None


def set_deterministic_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if dgl is not None:
        dgl.seed(seed)
        try:
            dgl.random.seed(seed)
        except AttributeError:
            pass

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)

