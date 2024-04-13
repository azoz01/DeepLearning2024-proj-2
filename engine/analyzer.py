from pathlib import Path
from typing import Any, Dict

import yaml

from engine import attention, lstm
from engine.model_base import LightningBaseModule


def get_logs_info(root_path: str = "lightning_logs") -> Dict[str, Any]:
    root_path = Path(root_path)
    res = {}

    for path in root_path.glob("*"):
        desc = {}
        res[str(path)] = desc
        with open(path / "hparams.yaml") as f:
            desc["hp"] = yaml.safe_load(f)
        desc["model_paths"] = list(path.rglob("*.ckpt"))
        desc["cls"] = desc["hp"]["cls_name"]

    return res


def get_model_from_log(log_dict: Dict[str, Any]) -> LightningBaseModule:
    model_path = log_dict["model_paths"]
    assert len(model_path) == 1

    try:
        Model = getattr(lstm, log_dict["cls"])
    except:
        Model = getattr(attention, log_dict["cls"])

    model = Model.load_from_checkpoint(model_path[0])

    return model
