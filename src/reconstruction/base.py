from __future__ import annotations
from dataclasses import dataclass
from data_management.normalization import Normalizer

@dataclass(frozen=True)
class ReconstructionConfig:
    """
    Configuration object for reconstruction logic.
    """
    target_col: str
    cond_cols: list[str]
    synth_col: str = "glucose_synth"
    include_true_target: bool = True


class BaseReconstructor:
    """
    Base class for Reconstructors.

    Responsibilities:
    1. Holds the configuration (target/conditional column names).
    2. Holds the Normalizer instance used for de-normalization.

    It does NOT implement the specific `reconstruct` method, as the input data
    structure differs between strategies (Stacked Arrays vs. Lists).
    """

    def __init__(
            self,
            cfg: ReconstructionConfig,
            normalizer: Normalizer | None = None
    ) -> None:
        """
        Args:
            cfg: Configuration dataclass.
            normalizer: The fitted normalizer instance.
        """
        self.cfg = cfg
        self.normalizer = normalizer
