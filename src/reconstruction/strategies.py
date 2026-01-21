from __future__ import annotations

import numpy as np


class ReconstructionStrategy:
    """
    Protocol/Base class for reconstruction strategies.

    The reconstruction process has three phases:
    1. initialize_buffers: Allocate necessary memory (values, counts, sums, etc.)
    2. place: Accumulate a window into the buffers.
    3. finalize: Compute the final array (e.g., divide sum by count).
    """

    def initialize_buffers(self, length: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def place(self, buffers: dict[str, np.ndarray], window: np.ndarray, start_row: int) -> None:
        raise NotImplementedError

    def finalize(self, buffers: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError


class OverwriteStrategy(ReconstructionStrategy):
    """
    Classic 'Last-Write-Wins' strategy.

    Best for: Non-overlapping windows (Validation/Test).
    Behavior: If windows overlap, the later window completely overwrites the earlier one.
    """

    def initialize_buffers(self, length: int) -> dict[str, np.ndarray]:
        # We only need one buffer for the values, initialized to NaN
        return {"values": np.full((length, 1), np.nan, dtype=np.float32)}

    def place(self, buffers: dict[str, np.ndarray], window: np.ndarray, start_row: int) -> None:
        target = buffers["values"]
        end_row = start_row + len(window)

        # Safety clip to avoid index out of bounds
        limit = len(target)
        actual_end = min(end_row, limit)
        win_len = actual_end - start_row

        if win_len > 0:
            target[start_row:actual_end] = window[:win_len]

    def finalize(self, buffers: dict[str, np.ndarray]) -> np.ndarray:
        return buffers["values"]


class AverageStrategy(ReconstructionStrategy):
    """
    Averaging strategy for overlapping windows.

    Best for: Overlapping windows (Training Data Reconstruction).
    Behavior: Accumulates sum and counts for every time step, then computes mean.
    Result: Smooth transitions between windows, reduces generation noise.
    """

    def initialize_buffers(self, length: int) -> dict[str, np.ndarray]:
        # We need two buffers: one for the sum of predictions, one for the count of hits
        return {
            "sum": np.zeros((length, 1), dtype=np.float32),
            "count": np.zeros((length, 1), dtype=np.float32)
        }

    def place(self, buffers: dict[str, np.ndarray], window: np.ndarray, start_row: int) -> None:
        sum_buf = buffers["sum"]
        count_buf = buffers["count"]

        end_row = start_row + len(window)

        # Safety clip
        limit = len(sum_buf)
        actual_end = min(end_row, limit)
        win_len = actual_end - start_row

        if win_len > 0:
            # Accumulate values
            sum_buf[start_row:actual_end] += window[:win_len]
            # Increment counters
            count_buf[start_row:actual_end] += 1.0

    def finalize(self, buffers: dict[str, np.ndarray]) -> np.ndarray:
        sums = buffers["sum"]
        counts = buffers["count"]

        # Avoid division by zero
        # Where count is 0, we leave it as NaN (no prediction made for that point)
        mask = counts > 0

        out = np.full_like(sums, np.nan)
        np.divide(sums, counts, out=out, where=mask)

        return out