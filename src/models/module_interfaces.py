from abc import ABC, abstractmethod
from dataclasses import dataclass

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