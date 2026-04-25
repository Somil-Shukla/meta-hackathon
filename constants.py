"""
Task-specific constants for the Fraud Detection RL Environment.

Each fraud family has its own configuration.
Adding a new fraud family:
1. Add its name to VALID_TASK_NAMES.
2. Add world-gen tuning parameters to FRAUD_FAMILY_CONFIG.
3. Wire into ScenarioGenerator.
"""
from __future__ import annotations
from typing import Dict, List

# ---------------------------------------------------------------------------
# Registry of valid task / fraud-family names
# ---------------------------------------------------------------------------
VALID_TASK_NAMES: List[str] = [
    "refund_abuse",
    "mule_cashout",
    "merchant_collusion",
    "account_takeover",
    "random",
]

# ---------------------------------------------------------------------------
# Episode settings
# ---------------------------------------------------------------------------
MAX_STEPS: int = 20
DEFAULT_MAX_STEPS: int = 20

# ---------------------------------------------------------------------------
# World generation defaults
# ---------------------------------------------------------------------------
MIN_USERS: int = 10
MAX_USERS: int = 20
MIN_MERCHANTS: int = 3
MAX_MERCHANTS: int = 8
MIN_MULES: int = 2
MAX_MULES: int = 5
MIN_DEVICES: int = 5
MAX_DEVICES: int = 15
MIN_TRANSACTIONS_PER_STEP: int = 1
MAX_TRANSACTIONS_PER_STEP: int = 4

# ---------------------------------------------------------------------------
# Per-fraud-family world-gen tuning
# ---------------------------------------------------------------------------
FRAUD_FAMILY_CONFIG: Dict[str, Dict] = {
    "refund_abuse": {
        "mule_ratio": 0.2,
        "colluding_merchant_ratio": 0.0,
        "aggressiveness_range": (0.3, 0.7),
        "transaction_size_range": (50.0, 500.0),
        "device_reuse_probability": 0.7,
        "fraud_timing": "burst",        # burst | spread | delayed
    },
    "mule_cashout": {
        "mule_ratio": 0.35,
        "colluding_merchant_ratio": 0.0,
        "aggressiveness_range": (0.5, 0.9),
        "transaction_size_range": (100.0, 2000.0),
        "device_reuse_probability": 0.4,
        "fraud_timing": "spread",
    },
    "merchant_collusion": {
        "mule_ratio": 0.1,
        "colluding_merchant_ratio": 0.5,
        "aggressiveness_range": (0.4, 0.8),
        "transaction_size_range": (200.0, 3000.0),
        "device_reuse_probability": 0.3,
        "fraud_timing": "spread",
    },
    "account_takeover": {
        "mule_ratio": 0.15,
        "colluding_merchant_ratio": 0.1,
        "aggressiveness_range": (0.6, 1.0),
        "transaction_size_range": (500.0, 5000.0),
        "device_reuse_probability": 0.2,
        "fraud_timing": "burst",
    },
}

# ---------------------------------------------------------------------------
# Defender reward constants
# ---------------------------------------------------------------------------
DEFENDER_FRAUD_PREVENTED_REWARD: float = 1.0
DEFENDER_EARLY_DETECTION_BONUS: float = 0.3
DEFENDER_FALSE_POSITIVE_PENALTY: float = -0.5
DEFENDER_UNNECESSARY_FREEZE_PENALTY: float = -0.3
DEFENDER_MISSED_FRAUD_PENALTY: float = -1.0
DEFENDER_CUSTOMER_FRICTION_PENALTY: float = -0.15
DEFENDER_INVESTIGATION_COST: float = -0.05

# ---------------------------------------------------------------------------
# Fraudster reward constants
# ---------------------------------------------------------------------------
FRAUDSTER_SUCCESSFUL_CASHOUT_REWARD: float = 1.0
FRAUDSTER_UNDETECTED_ACTIVITY_REWARD: float = 0.05
FRAUDSTER_EVASION_REWARD: float = 0.2
FRAUDSTER_FREEZE_PENALTY: float = -0.5
FRAUDSTER_BLOCK_PENALTY: float = -0.4
FRAUDSTER_FAILED_CASHOUT_PENALTY: float = -0.3
FRAUDSTER_ROUTE_LOSS_PENALTY: float = -0.2

# ---------------------------------------------------------------------------
# Observation noise
# ---------------------------------------------------------------------------
OBSERVATION_NOISE_LEVEL: float = 0.15

# ---------------------------------------------------------------------------
# Action → internal field mappings (for action validation)
# ---------------------------------------------------------------------------
DEFENDER_ACTION_TYPES: List[str] = [
    "monitor",
    "challenge",
    "freeze",
    "hold",
    "block_merchant",
    "investigate_neighborhood",
    "do_nothing",
]

FRAUDSTER_ACTION_TYPES: List[str] = [
    "split_payment",
    "rotate_mule",
    "switch_merchant",
    "rotate_device",
    "delay",
    "refund_abuse",
    "cashout_attempt",
    "do_nothing",
]

# ---------------------------------------------------------------------------
# Grading / evaluation CSV output path
# ---------------------------------------------------------------------------
GRADING_CSV_PATH: str = "episode_grades.csv"
ROLLOUT_CSV_PATH: str = "rollout_history.csv"
