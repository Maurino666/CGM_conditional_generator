from abc import ABC, abstractmethod
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

    def validation_step(self, batch: Any) -> dict[str, float] | None:
        """
        Computes validation metrics (optional).
        If model does not support standard validation,
        this step can be ignored.
        """
        return None

    @property
    def should_validate(self) -> bool:
        """
        Indicates whether this module should validate or not.
        Subclasses can override this to implement custom checks.

        Default: True.
        """
        return True

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """
        Returns a dictionary of hyperparameters for this model.
        Every subclass must implement this method.
        """
        raise NotImplementedError

    def to_checkpoint(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Creates a dict ready for torch.save() having:
          - info class
          - config
          - state_dict
          - eventual info extra (es. history)
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
        Reconstructs an instance of the same class from checkpoint, made with .to_checkpoint().

        Note: assumes right class.
        """
        config = checkpoint["config"]
        state_dict = checkpoint["state_dict"]

        model = cls(**config)  # type: ignore[arg-type]
        model.load_state_dict(state_dict)

        if map_location is not None:
            model.to(map_location)

        return model