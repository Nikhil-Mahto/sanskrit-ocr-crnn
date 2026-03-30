from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: Path,
    model_state: Dict[str, Any],
    optimizer_state: Optional[Dict[str, Any]],
    epoch: int,
    charset: list[str],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "charset": charset,
        "extra": extra or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)
