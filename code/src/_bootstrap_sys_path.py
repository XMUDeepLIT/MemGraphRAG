"""
在 Conda 环境下从 sys.path 去掉 user-site（~/.local/...），避免
`~/.local` 的 torch 与环境内 torchvision 混用导致
RuntimeError: operator torchvision::nms does not exist。

若需在 Conda 中继续使用 pip install --user 的包，可设置环境变量：
    HIPPO_KEEP_USER_SITE=1
"""

from __future__ import annotations

import os
import site
import sys


def drop_user_site_from_sys_path() -> None:
    if os.environ.get("HIPPO_KEEP_USER_SITE") == "1":
        return
    conda_meta = os.path.join(sys.prefix, "conda-meta")
    if not os.path.isdir(conda_meta):
        return
    try:
        user_root = os.path.abspath(site.getusersitepackages())
    except Exception:
        return
    if not user_root:
        return
    sep = os.sep
    kept = []
    for p in sys.path:
        if not p:
            kept.append(p)
            continue
        try:
            ap = os.path.abspath(p)
        except Exception:
            kept.append(p)
            continue
        if ap == user_root or ap.startswith(user_root + sep):
            continue
        kept.append(p)
    sys.path[:] = kept


drop_user_site_from_sys_path()
