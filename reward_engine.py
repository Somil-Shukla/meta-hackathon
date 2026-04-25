"""
RewardEngine — computes per-step shaped rewards for both agents.

Defender reward is shaped to encourage:
  - Fraud prevention (true positives)
  - Early, accurate intervention
  - Low false positive rate
  - Low customer friction
  - Low investigation cost

Fraudster reward is shaped to encourage:
  - Successful money movement (cashout)
  - Undetected activity
  - Adaptive evasion

Rewards are per-step and intended to drive online RL learning,
NOT the same as episode-level grading metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from scam_detection.action_processor import ActionResult
from scam_detection.constants import (
    DEFENDER_CUSTOMER_FRICTION_PENALTY,
    DEFENDER_EARLY_DETECTION_BONUS,
    DEFENDER_FALSE_POSITIVE_PENALTY,
    DEFENDER_FRAUD_PREVENTED_REWARD,
    DEFENDER_INVESTIGATION_COST,
    DEFENDER_MISSED_FRAUD_PENALTY,
    DEFENDER_UNNECESSARY_FREEZE_PENALTY,
    FRAUDSTER_BLOCK_PENALTY,
    FRAUDSTER_EVASION_REWARD,
    FRAUDSTER_FAILED_CASHOUT_PENALTY,
    FRAUDSTER_FREEZE_PENALTY,
    FRAUDSTER_ROUTE_LOSS_PENALTY,
    FRAUDSTER_SUCCESSFUL_CASHOUT_REWARD,
    FRAUDSTER_UNDETECTED_ACTIVITY_REWARD,
    MAX_STEPS,
)
from scam_detection.hidden_world_state import HiddenWorldState


@dataclass
class StepReward:
    """Per-step reward breakdown for both agents."""
    defender_reward: float
    fraudster_reward: float
    defender_breakdown: Dict[str, float]
    fraudster_breakdown: Dict[str, float]


class RewardEngine:
    """
    Computes per-step rewards for defender and fraudster.

    Accepts the ``ActionResult`` from both agents plus the current
    ``HiddenWorldState`` (post-transition).
    """

    def compute(
        self,
        world: HiddenWorldState,
        fraudster_result: ActionResult,
        defender_result: ActionResult,
    ) -> StepReward:
        """Compute rewards for the current step."""
        defender_breakdown: Dict[str, float] = {}
        fraudster_breakdown: Dict[str, float] = {}

        # ── Defender reward ───────────────────────────────────────────────────
        d_reward = 0.0

        # Direct reward signal from action result
        if defender_result.action_type == "freeze":
            if defender_result.details.get("is_fraud"):
                d_reward += DEFENDER_FRAUD_PREVENTED_REWARD
                defender_breakdown["fraud_prevented"] = DEFENDER_FRAUD_PREVENTED_REWARD
                # Early detection bonus
                early_bonus = self._early_detection_bonus(world)
                d_reward += early_bonus
                defender_breakdown["early_detection"] = early_bonus
            else:
                d_reward += DEFENDER_FALSE_POSITIVE_PENALTY
                defender_breakdown["false_positive"] = DEFENDER_FALSE_POSITIVE_PENALTY
                d_reward += DEFENDER_CUSTOMER_FRICTION_PENALTY
                defender_breakdown["customer_friction"] = DEFENDER_CUSTOMER_FRICTION_PENALTY
                d_reward += DEFENDER_UNNECESSARY_FREEZE_PENALTY
                defender_breakdown["unnecessary_freeze"] = DEFENDER_UNNECESSARY_FREEZE_PENALTY

        elif defender_result.action_type == "hold":
            if defender_result.details.get("is_fraud"):
                d_reward += DEFENDER_FRAUD_PREVENTED_REWARD * 0.8
                defender_breakdown["fraud_held"] = DEFENDER_FRAUD_PREVENTED_REWARD * 0.8
            else:
                d_reward += DEFENDER_FALSE_POSITIVE_PENALTY * 0.5
                defender_breakdown["false_positive_hold"] = DEFENDER_FALSE_POSITIVE_PENALTY * 0.5
                d_reward += DEFENDER_CUSTOMER_FRICTION_PENALTY
                defender_breakdown["customer_friction"] = DEFENDER_CUSTOMER_FRICTION_PENALTY

        elif defender_result.action_type == "block_merchant":
            if defender_result.details.get("is_colluding"):
                d_reward += DEFENDER_FRAUD_PREVENTED_REWARD * 0.9
                defender_breakdown["colluding_merchant_blocked"] = DEFENDER_FRAUD_PREVENTED_REWARD * 0.9
            else:
                d_reward += DEFENDER_FALSE_POSITIVE_PENALTY
                defender_breakdown["false_positive_block"] = DEFENDER_FALSE_POSITIVE_PENALTY

        elif defender_result.action_type in ("monitor", "challenge", "investigate_neighborhood"):
            d_reward += DEFENDER_INVESTIGATION_COST
            defender_breakdown["investigation_cost"] = DEFENDER_INVESTIGATION_COST

        # Penalty for fraud that slipped through this step
        missed = fraudster_result.reward_signal
        if fraudster_result.action_type == "cashout_attempt" and fraudster_result.success:
            d_reward += DEFENDER_MISSED_FRAUD_PENALTY * 0.5
            defender_breakdown["missed_fraud"] = DEFENDER_MISSED_FRAUD_PENALTY * 0.5

        # ── Fraudster reward ──────────────────────────────────────────────────
        f_reward = 0.0

        if fraudster_result.action_type == "cashout_attempt":
            if fraudster_result.success:
                f_reward += FRAUDSTER_SUCCESSFUL_CASHOUT_REWARD
                fraudster_breakdown["successful_cashout"] = FRAUDSTER_SUCCESSFUL_CASHOUT_REWARD
            else:
                f_reward += FRAUDSTER_FAILED_CASHOUT_PENALTY
                fraudster_breakdown["failed_cashout"] = FRAUDSTER_FAILED_CASHOUT_PENALTY

        elif fraudster_result.action_type in ("split_payment", "refund_abuse"):
            if fraudster_result.success:
                # Small reward for undetected activity
                f_reward += FRAUDSTER_UNDETECTED_ACTIVITY_REWARD
                fraudster_breakdown["undetected_activity"] = FRAUDSTER_UNDETECTED_ACTIVITY_REWARD
            else:
                f_reward += FRAUDSTER_FAILED_CASHOUT_PENALTY * 0.5
                fraudster_breakdown["failed_activity"] = FRAUDSTER_FAILED_CASHOUT_PENALTY * 0.5

        elif fraudster_result.action_type in ("rotate_mule", "switch_merchant", "rotate_device", "delay"):
            # Evasion reward — good if alert level was high
            if world.fraudster_alert_level > 0.5:
                f_reward += FRAUDSTER_EVASION_REWARD
                fraudster_breakdown["evasion"] = FRAUDSTER_EVASION_REWARD
            else:
                # Wasted action if alert was low
                f_reward -= 0.02
                fraudster_breakdown["wasted_evasion"] = -0.02

        # Penalty when defender freezes fraudster's accounts
        if defender_result.action_type == "freeze" and defender_result.details.get("is_fraud"):
            f_reward += FRAUDSTER_FREEZE_PENALTY
            fraudster_breakdown["account_frozen"] = FRAUDSTER_FREEZE_PENALTY

        if defender_result.action_type == "block_merchant" and defender_result.details.get("is_colluding"):
            f_reward += FRAUDSTER_BLOCK_PENALTY
            fraudster_breakdown["merchant_blocked"] = FRAUDSTER_BLOCK_PENALTY

        # Penalty for route loss
        n_newly_inactive = sum(
            1 for r in world.fraud_routes
            if not r.is_active and r.steps_active > 0
        )
        if n_newly_inactive > 0:
            f_reward += FRAUDSTER_ROUTE_LOSS_PENALTY * n_newly_inactive
            fraudster_breakdown["route_loss"] = FRAUDSTER_ROUTE_LOSS_PENALTY * n_newly_inactive

        defender_breakdown["total"] = round(d_reward, 4)
        fraudster_breakdown["total"] = round(f_reward, 4)

        return StepReward(
            defender_reward=round(d_reward, 4),
            fraudster_reward=round(f_reward, 4),
            defender_breakdown=defender_breakdown,
            fraudster_breakdown=fraudster_breakdown,
        )

    def _early_detection_bonus(self, world: HiddenWorldState) -> float:
        """
        Bonus for catching fraud early in the episode.
        Linearly decays from EARLY_DETECTION_BONUS to 0 over the episode.
        """
        if world.max_steps <= 0:
            return 0.0
        fraction_remaining = (world.max_steps - world.step) / world.max_steps
        return round(DEFENDER_EARLY_DETECTION_BONUS * fraction_remaining, 4)
