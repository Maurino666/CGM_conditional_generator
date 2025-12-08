from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn, Tensor

class BaseTrainableModule(nn.Module, ABC):
    """
    Interface-like base class for trainable generative models.

    Any subclass must implement:
      - training_step(batch): compute training loss for a batch
    """

    @abstractmethod
    def training_step(self, batch: Tensor) -> Tensor:
        """
        Compute the training loss for a single batch.

        This method must be overridden in subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """
        Restituisce un dict di iperparametri sufficiente per ricostruire il modello
        con  __class__(**config).

        Ogni sottoclasse deve implementarla.
        """
        raise NotImplementedError

    def to_checkpoint(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Crea un dict pronto per torch.save, con:
          - info classe
          - config
          - state_dict
          - eventuali info extra (es. history)
        """
        if extra is None:
            extra = {}

        ckpt: dict[str, Any] = {
            "model_class": self.__class__.__name__,
            "module_path": self.__class__.__module__,
            "config": self.get_config(),
            "state_dict": self.state_dict(),
        }
        ckpt.update(extra)
        return ckpt

    @classmethod
    def from_checkpoint(
            cls,
            checkpoint: dict[str, Any],
            map_location: str | torch.device | None = None,
    ) -> "BaseTrainableModule":
        """
        Ricostruisce un'istanza della *stessa classe* a partire da un checkpoint
        creato con to_checkpoint().

        Nota: qui si assume che cls sia effettivamente la classe giusta.
        """
        config = checkpoint["config"]
        state_dict = checkpoint["state_dict"]

        model = cls(**config)  # type: ignore[arg-type]
        model.load_state_dict(state_dict)

        if map_location is not None:
            model.to(map_location)

        return model