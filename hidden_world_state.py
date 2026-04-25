"""
Hidden world state dataclasses for the Fraud Detection RL Environment.

The full state is never exposed directly to agents — only partial, noisy
observations derived from it are sent to the defender and fraudster.
The ``state()`` endpoint returns the full ``HiddenWorldState`` for the
simulator / evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class UserNode:
    """A participant node (legitimate user or mule account)."""
    id: str
    is_mule: bool = False
    is_frozen: bool = False
    is_challenged: bool = False
    is_monitored: bool = False
    is_fraudster_controlled: bool = False   # account-takeover flag
    device_id: str = ""
    balance: float = 1000.0
    risk_score: float = 0.0                 # defender's internal risk estimate
    transaction_count: int = 0
    refund_count: int = 0
    takeover_step: Optional[int] = None     # step when account was taken over


@dataclass
class MerchantNode:
    """A merchant node (legitimate or colluding)."""
    id: str
    is_colluding: bool = False
    is_blocked: bool = False
    is_monitored: bool = False
    refund_rate: float = 0.0
    transaction_count: int = 0
    total_volume: float = 0.0
    anomaly_score: float = 0.0


@dataclass
class DeviceNode:
    """A device used for transactions."""
    id: str
    reuse_count: int = 0
    associated_users: List[str] = field(default_factory=list)
    risk_score: float = 0.0


@dataclass
class Transaction:
    """A single financial transaction in the world."""
    id: str
    from_user_id: str
    amount: float
    timestamp: int                      # step number
    to_merchant_id: Optional[str] = None
    to_user_id: Optional[str] = None    # mule-to-mule or cashout
    is_fraud: bool = False
    is_held: bool = False
    is_completed: bool = False
    is_refunded: bool = False
    device_id: str = ""
    fraud_family: Optional[str] = None


@dataclass
class FraudRoute:
    """One active fraud cashout route managed by the fraudster."""
    id: str
    family: str
    mule_ids: List[str]                         # chain of mule accounts
    merchant_id: Optional[str] = None           # colluding merchant (if any)
    target_user_ids: List[str] = field(default_factory=list)  # victims
    total_laundered: float = 0.0
    is_active: bool = True
    detection_pressure: float = 0.0             # how hard the defender watches
    steps_active: int = 0
    current_mule_index: int = 0                 # which mule is currently used


@dataclass
class HiddenWorldState:
    """
    Complete hidden state of the fraud simulation episode.

    This is the ground truth — only the simulator sees it.
    Agents receive partial views via ObservationGenerator.
    """
    episode_id: str
    step: int
    fraud_family: str
    seed: int

    # ── Graph nodes ──────────────────────────────────────────────────────────
    users: Dict[str, UserNode] = field(default_factory=dict)
    merchants: Dict[str, MerchantNode] = field(default_factory=dict)
    devices: Dict[str, DeviceNode] = field(default_factory=dict)

    # ── Transaction log ───────────────────────────────────────────────────────
    transactions: List[Transaction] = field(default_factory=list)

    # ── Fraud topology ────────────────────────────────────────────────────────
    fraud_routes: List[FraudRoute] = field(default_factory=list)

    # ── Episode progress ──────────────────────────────────────────────────────
    total_fraud_amount: float = 0.0
    total_laundered: float = 0.0
    total_prevented: float = 0.0
    fraud_attempts: int = 0
    successful_fraud: int = 0
    prevented_fraud: int = 0
    false_positives: int = 0           # legit accounts frozen
    legitimate_users_frozen: int = 0

    # ── Fraudster internal state ──────────────────────────────────────────────
    fraudster_alert_level: float = 0.0  # how detected the fraudster feels
    available_routes: List[str] = field(default_factory=list)  # active route IDs
    fraudster_delayed_steps: int = 0    # remaining delay steps

    # ── Defender internal state ───────────────────────────────────────────────
    defender_alerts: List[Dict[str, Any]] = field(default_factory=list)
    flagged_accounts: Set[str] = field(default_factory=set)
    monitored_accounts: Set[str] = field(default_factory=set)

    # ── World config ──────────────────────────────────────────────────────────
    fraud_aggressiveness: float = 0.5   # 0=cautious → 1=aggressive
    max_steps: int = 20

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict for the state() endpoint."""
        return {
            "episode_id": self.episode_id,
            "step": self.step,
            "fraud_family": self.fraud_family,
            "total_fraud_amount": round(self.total_fraud_amount, 2),
            "total_laundered": round(self.total_laundered, 2),
            "total_prevented": round(self.total_prevented, 2),
            "fraud_attempts": self.fraud_attempts,
            "successful_fraud": self.successful_fraud,
            "prevented_fraud": self.prevented_fraud,
            "false_positives": self.false_positives,
            "legitimate_users_frozen": self.legitimate_users_frozen,
            "user_count": len(self.users),
            "mule_count": sum(1 for u in self.users.values() if u.is_mule),
            "frozen_accounts": sum(1 for u in self.users.values() if u.is_frozen),
            "blocked_merchants": sum(1 for m in self.merchants.values() if m.is_blocked),
            "active_fraud_routes": sum(1 for r in self.fraud_routes if r.is_active),
            "fraudster_alert_level": round(self.fraudster_alert_level, 4),
            "fraud_aggressiveness": self.fraud_aggressiveness,
            "flagged_accounts": list(self.flagged_accounts),
            "transactions_this_episode": len(self.transactions),
        }
