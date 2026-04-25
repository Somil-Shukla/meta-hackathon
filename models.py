"""
Data models for the Fraud Detection RL Environment.

Design rationale
----------------
OpenEnv's ``create_app`` accepts exactly **one** Action class and **one**
Observation class.  For the multi-agent setup (defender + fraudster), we
use a **unified superset** approach:

* ``FraudAction``       — one class containing the actions for BOTH agents.
                          The defender submits ``defender_action`` and the
                          fraudster submits ``fraudster_action`` in each step.
* ``FraudObservation``  — one class carrying partial observations for both
                          agents (``defender_obs`` and ``fraudster_obs``),
                          each derived from the same hidden world state.

The inference script queries two LLM agents and combines their actions into
a single ``FraudAction`` to call ``step()`` once per round.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DefenderActionType(str, Enum):
    """Actions available to the defender agent."""
    MONITOR                  = "monitor"
    CHALLENGE                = "challenge"
    FREEZE                   = "freeze"
    HOLD                     = "hold"
    BLOCK_MERCHANT           = "block_merchant"
    INVESTIGATE_NEIGHBORHOOD = "investigate_neighborhood"
    DO_NOTHING               = "do_nothing"


class FraudsterActionType(str, Enum):
    """Actions available to the fraudster agent."""
    SPLIT_PAYMENT   = "split_payment"
    ROTATE_MULE     = "rotate_mule"
    SWITCH_MERCHANT = "switch_merchant"
    ROTATE_DEVICE   = "rotate_device"
    DELAY           = "delay"
    REFUND_ABUSE    = "refund_abuse"
    CASHOUT_ATTEMPT = "cashout_attempt"
    DO_NOTHING      = "do_nothing"


class FraudFamily(str, Enum):
    """Supported fraud scenario families."""
    REFUND_ABUSE        = "refund_abuse"
    MULE_CASHOUT        = "mule_cashout"
    MERCHANT_COLLUSION  = "merchant_collusion"
    ACCOUNT_TAKEOVER    = "account_takeover"
    RANDOM              = "random"


# ---------------------------------------------------------------------------
# Unified Action (both agents in one payload)
# ---------------------------------------------------------------------------

class FraudAction(Action):
    """
    A single step action combining decisions from both agents.

    The defender submits its ``defender_action`` and optional
    ``defender_target`` (account/transaction ID).  The fraudster
    submits ``fraudster_action`` and optional ``fraudster_target``.

    Either field may be ``do_nothing`` when an agent passes its turn.

    Example — combined action::

        FraudAction(
            defender_action="freeze",
            defender_target="user_003",
            fraudster_action="rotate_mule",
            fraudster_target="mule_002",
        )
    """
    defender_action: DefenderActionType = Field(
        default=DefenderActionType.DO_NOTHING,
        description="Defender's action for this step.",
    )
    defender_target: Optional[str] = Field(
        default=None,
        description="Target ID (account/transaction/merchant) for defender.",
    )
    fraudster_action: FraudsterActionType = Field(
        default=FraudsterActionType.DO_NOTHING,
        description="Fraudster's action for this step.",
    )
    fraudster_target: Optional[str] = Field(
        default=None,
        description="Target ID (mule/merchant/device) for fraudster.",
    )

    def model_dump(self, *args, **kwargs):
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(*args, **kwargs)

    def model_dump_json(self, *args, **kwargs):
        kwargs.setdefault("exclude_none", True)
        return super().model_dump_json(*args, **kwargs)


# ---------------------------------------------------------------------------
# Unified Observation (partial views for both agents)
# ---------------------------------------------------------------------------

class FraudObservation(Observation):
    """
    Observation returned by the environment at every step.

    Contains partial views for both agents derived from the same hidden
    world state.  Neither agent receives ground-truth fraud labels directly.

    ── Shared fields ─────────────────────────────────────────────────────────
    episode_id         : current episode identifier
    step               : current step number
    step_budget        : remaining step budget
    task_name          : active fraud family
    episode_done       : True when episode has ended
    reason             : termination reason ("max_steps" | "all_fraud_done" | "all_routes_blocked")

    ── Agent observations (partial) ──────────────────────────────────────────
    defender_obs       : partial noisy observation for the defender
    fraudster_obs      : partial operational observation for the fraudster

    ── Action masks (legal actions this step) ─────────────────────────────────
    available_defender_actions   : list of legal defender actions
    available_fraudster_actions  : list of legal fraudster actions

    ── Rewards (per-step) ────────────────────────────────────────────────────
    defender_reward    : step reward for the defender
    fraudster_reward   : step reward for the fraudster

    ── Info ─────────────────────────────────────────────────────────────────
    info               : reward_breakdown + cumulative + debug
    """

    # ── Shared ───────────────────────────────────────────────────────────────
    episode_id: Optional[str] = Field(default=None)
    step: Optional[int] = Field(default=None)
    step_budget: Optional[Dict[str, int]] = Field(
        default=None, description="Keys: total, used, remaining."
    )
    task_name: Optional[str] = Field(
        default=None, description="Active fraud family."
    )
    episode_done: Optional[bool] = Field(
        default=None, description="True when episode has ended."
    )
    reason: Optional[str] = Field(
        default=None,
        description="Termination reason: max_steps | all_fraud_done | all_routes_blocked.",
    )

    # ── Partial agent observations ────────────────────────────────────────────
    defender_obs: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Defender's partial, noisy view: transaction velocities, "
            "refund ratios, device reuse counts, merchant anomaly scores, "
            "freeze/flag status, alerts."
        ),
    )
    fraudster_obs: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Fraudster's partial operational view: available mule routes, "
            "detection pressure, cashout readiness, account balances."
        ),
    )

    # ── Action masks ─────────────────────────────────────────────────────────
    available_defender_actions: Optional[List[str]] = Field(
        default=None,
        description="Legal defender actions given current state.",
    )
    available_fraudster_actions: Optional[List[str]] = Field(
        default=None,
        description="Legal fraudster actions given current state.",
    )
    defender_action_targets: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Maps each legal defender action → list of valid target IDs.",
    )
    fraudster_action_targets: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Maps each legal fraudster action → list of valid target IDs.",
    )

    # ── Per-step rewards ──────────────────────────────────────────────────────
    defender_reward: Optional[float] = Field(default=None)
    fraudster_reward: Optional[float] = Field(default=None)

    # ── Info / diagnostics ────────────────────────────────────────────────────
    info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="reward_breakdown + cumulative totals + debug.",
    )
