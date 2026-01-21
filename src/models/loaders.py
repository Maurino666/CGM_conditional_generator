import importlib
import torch
from typing import Any
from .module_interfaces import BaseTrainableModule

def load_any_model(path: str, device: torch.device) -> BaseTrainableModule:
    ckpt: dict[str, Any] = torch.load(path, map_location=device)

    module_path = ckpt["module_path"]       # es. "src.models.time_gan.conditional_module"
    class_name = ckpt["model_class"]       # es. "ConditionalTimeGanModule"

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    # cls è la classe corretta, che implementa from_checkpoint
    model: BaseTrainableModule = cls.from_checkpoint(ckpt, map_location=device)
    model.eval()
    return model
