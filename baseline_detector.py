"""
BaselineRuleDetector — a rule-based defender policy for comparison.

Provides a non-RL baseline so PPO performance can be benchmarked.

Rules (in priority order):
  1. If a transaction has very high amount (>2000) and the account has
     high device-reuse (>3), freeze the account.
  2. If a merchant has refund_rate > 0.35, block it.
  3. If an account has risk_score > 0.7 and is not yet monitored, monitor it.
  4. If an account is monitored and risk_score > 0.85, freeze it.
  5. If any alert is "high_velocity" with count >= 4, challenge the account.
  6. Otherwise, do nothing.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from scam_detection.constants import DEFENDER_ACTION_TYPES
from scam_detection.models import DefenderActionType


class BaselineRuleDetector:
    """
    Deterministic rule-based defender policy.

    ``select_action(defender_obs, legal_actions, action_targets)``
    returns ``(action_type_str, target_id_or_None)``.
    """

    def select_action(
        self,
        defender_obs: Optional[Dict[str, Any]],
        legal_actions: Optional[List[str]] = None,
        action_targets: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[str, Optional[str]]:
        """
        Choose defender action using a fixed rule hierarchy.
        """
        if not defender_obs:
            return DefenderActionType.DO_NOTHING.value, None

        legal   = set(legal_actions or DEFENDER_ACTION_TYPES)
        targets = action_targets or {}

        accounts  = defender_obs.get("accounts", [])
        merchants = defender_obs.get("merchants", [])
        alerts    = defender_obs.get("alerts", [])

        # Rule 1: freeze high-risk accounts with device reuse
        if DefenderActionType.FREEZE.value in legal:
            for acc in accounts:
                if (
                    acc.get("risk_score", 0) > 0.75
                    and acc.get("device_reuse_count", 0) >= 3
                    and not acc.get("is_frozen")
                    and acc["id"] in targets.get(DefenderActionType.FREEZE.value, [])
                ):
                    return DefenderActionType.FREEZE.value, acc["id"]

        # Rule 2: block colluding/suspicious merchants
        if DefenderActionType.BLOCK_MERCHANT.value in legal:
            for merch in merchants:
                if (
                    merch.get("refund_rate", 0) > 0.4
                    and merch.get("anomaly_score", 0) > 0.6
                    and not merch.get("is_blocked")
                    and merch["id"] in targets.get(DefenderActionType.BLOCK_MERCHANT.value, [])
                ):
                    return DefenderActionType.BLOCK_MERCHANT.value, merch["id"]

        # Rule 3: monitor high-risk accounts
        if DefenderActionType.MONITOR.value in legal:
            for acc in accounts:
                if (
                    acc.get("risk_score", 0) > 0.65
                    and not acc.get("is_monitored")
                    and acc["id"] in targets.get(DefenderActionType.MONITOR.value, [])
                ):
                    return DefenderActionType.MONITOR.value, acc["id"]

        # Rule 4: freeze monitored accounts that escalated
        if DefenderActionType.FREEZE.value in legal:
            for acc in accounts:
                if (
                    acc.get("is_monitored")
                    and acc.get("risk_score", 0) > 0.85
                    and not acc.get("is_frozen")
                    and acc["id"] in targets.get(DefenderActionType.FREEZE.value, [])
                ):
                    return DefenderActionType.FREEZE.value, acc["id"]

        # Rule 5: challenge accounts flagged in high-velocity alerts
        if DefenderActionType.CHALLENGE.value in legal:
            hv_accounts = {
                a["account_id"]
                for a in alerts
                if a.get("type") == "high_velocity" and a.get("count", 0) >= 4
            }
            for acc_id in hv_accounts:
                if acc_id in targets.get(DefenderActionType.CHALLENGE.value, []):
                    return DefenderActionType.CHALLENGE.value, acc_id

        # Rule 6: investigate if flagged accounts exist
        if DefenderActionType.INVESTIGATE_NEIGHBORHOOD.value in legal:
            inv_targets = targets.get(DefenderActionType.INVESTIGATE_NEIGHBORHOOD.value, [])
            if inv_targets:
                return DefenderActionType.INVESTIGATE_NEIGHBORHOOD.value, random.choice(inv_targets)

        return DefenderActionType.DO_NOTHING.value, None
