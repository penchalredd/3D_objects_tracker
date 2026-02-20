from __future__ import annotations

import importlib
from typing import Any


def import_symbol(path: str) -> Any:
    if ":" in path:
        mod_name, sym_name = path.split(":", 1)
    else:
        parts = path.split(".")
        if len(parts) < 2:
            raise ValueError(f"Invalid import path: {path}")
        mod_name = ".".join(parts[:-1])
        sym_name = parts[-1]
    mod = importlib.import_module(mod_name)
    if not hasattr(mod, sym_name):
        raise AttributeError(f"Module '{mod_name}' has no symbol '{sym_name}'")
    return getattr(mod, sym_name)
