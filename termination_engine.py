"""
TerminationEngine — determines when an episode ends.

An episode terminates when any of these conditions is met:
  - max_steps reached
  - all active fraud routes have been deactivated (defender wins)
  - total laundered amount exceeds the episode threshold (fraudster wins)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from scam_detection.hidden_world_state import HiddenWorldState


@dataclass
class TerminationResult:
    done: bool
    reason: Optional[str] = None   # "max_steps" | "all_routes_blocked" | "fraud_threshold_exceeded"


class TerminationEngine:
    """
    Checks whether the episode should end after a step.

    Parameters
    ----------
    fraud_threshold_ratio:
        Episode ends when laundered / total_fraud_amount exceeds this ratio.
        Represents the fraudster winning by extracting a significant portion.
    """

    def __init__(self, fraud_threshold_ratio: float = 0.8) -> None:
        self._threshold = fraud_threshold_ratio

    def check(self, world: HiddenWorldState) -> TerminationResult:
        """Return a TerminationResult after the current step."""
        # 1. Max steps
        if world.step >= world.max_steps:
            return TerminationResult(done=True, reason="max_steps")

        # 2. All fraud routes deactivated — defender wins
        if world.fraud_routes and all(not r.is_active for r in world.fraud_routes):
            return TerminationResult(done=True, reason="all_routes_blocked")

        # 3. Fraudster laundered a large fraction — fraudster wins
        if world.total_fraud_amount > 0:
            ratio = world.total_laundered / world.total_fraud_amount
            if ratio >= self._threshold:
                return TerminationResult(done=True, reason="fraud_threshold_exceeded")

        return TerminationResult(done=False)
