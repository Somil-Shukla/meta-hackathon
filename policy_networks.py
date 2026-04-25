"""
Policy networks for the Fraud Detection RL Environment.

Both policies share a common actor-critic architecture compatible with PPO:
  - Actor: outputs action logits over the discrete action space
  - Critic: outputs a scalar value estimate V(s)

Observation encoding:
  The raw observations (dicts) are flattened into fixed-size feature vectors
  before passing through the network.  We use a simple MLP encoder.

DefenderPolicy action space:
  7 actions × up to N targets → fixed 7-dim logit output (target selection
  is handled by sampling uniformly from the legal target list).

FraudsterPolicy action space:
  8 actions × up to M targets → fixed 8-dim logit output.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from scam_detection.constants import DEFENDER_ACTION_TYPES, FRAUDSTER_ACTION_TYPES

# Observation feature sizes (heuristic fixed-length encodings)
DEFENDER_OBS_DIM: int = 64
FRAUDSTER_OBS_DIM: int = 32
HIDDEN_DIM: int = 128


# ---------------------------------------------------------------------------
# Feature encoders (dict → flat tensor)
# ---------------------------------------------------------------------------

def encode_defender_obs(obs: Optional[Dict[str, Any]]) -> torch.Tensor:
    """Encode a defender observation dict into a fixed-length float tensor."""
    if obs is None:
        return torch.zeros(DEFENDER_OBS_DIM)

    features: List[float] = []

    # Aggregate stats
    agg = obs.get("aggregate", {})
    features += [
        float(agg.get("total_frozen", 0)) / 20.0,
        float(agg.get("total_monitored", 0)) / 20.0,
        float(agg.get("total_flagged", 0)) / 20.0,
        float(agg.get("blocked_merchants", 0)) / 8.0,
        float(agg.get("recent_transaction_count", 0)) / 20.0,
    ]

    # Per-account features (up to 10 accounts, zero-padded)
    accounts = obs.get("accounts", [])[:10]
    for acc in accounts:
        features += [
            float(acc.get("transaction_velocity", 0)) / 10.0,
            float(acc.get("refund_ratio", 0)),
            float(acc.get("device_reuse_count", 0)) / 10.0,
            float(acc.get("risk_score", 0)),
            float(acc.get("is_frozen", False)),
            float(acc.get("is_monitored", False)),
        ]
    # Pad to 10 accounts × 6 features = 60
    pad = 10 - len(accounts)
    features += [0.0] * (pad * 6)

    # Alert count normalised
    features.append(min(1.0, len(obs.get("alerts", [])) / 10.0))

    # Pad / truncate to DEFENDER_OBS_DIM
    features = features[:DEFENDER_OBS_DIM]
    features += [0.0] * (DEFENDER_OBS_DIM - len(features))
    return torch.tensor(features, dtype=torch.float32)


def encode_fraudster_obs(obs: Optional[Dict[str, Any]]) -> torch.Tensor:
    """Encode a fraudster observation dict into a fixed-length float tensor."""
    if obs is None:
        return torch.zeros(FRAUDSTER_OBS_DIM)

    features: List[float] = [
        float(obs.get("alert_level", 0)),
        float(obs.get("delayed_steps_remaining", 0)) / 5.0,
        float(obs.get("any_cashout_ready", False)),
        float(obs.get("total_laundered_so_far", 0)) / 10000.0,
    ]

    routes = obs.get("active_routes", [])[:4]
    for route in routes:
        features += [
            float(route.get("detection_pressure", 0)),
            float(route.get("cashout_ready", False)),
            float(route.get("total_laundered", 0)) / 10000.0,
            float(route.get("merchant_blocked", False)),
        ]
    pad = 4 - len(routes)
    features += [0.0] * (pad * 4)

    available_mules = len(obs.get("available_mule_ids", []))
    features.append(min(1.0, available_mules / 10.0))

    features = features[:FRAUDSTER_OBS_DIM]
    features += [0.0] * (FRAUDSTER_OBS_DIM - len(features))
    return torch.tensor(features, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Shared Actor-Critic Base
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """Shared MLP actor-critic backbone."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = HIDDEN_DIM) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor_head  = nn.Linear(hidden, n_actions)
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_logits, value_estimate)."""
        h = self.shared(x)
        return self.actor_head(h), self.critic_head(h).squeeze(-1)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action (or use provided), return (action, log_prob, entropy, value)."""
        logits, value = self.forward(x)
        dist   = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, entropy, value


# ---------------------------------------------------------------------------
# DefenderPolicy
# ---------------------------------------------------------------------------

class DefenderPolicy(ActorCritic):
    """
    PPO-compatible policy for the defender agent.

    Input : encoded defender observation (DEFENDER_OBS_DIM float vector)
    Output: distribution over DEFENDER_ACTION_TYPES (7 actions)
    """
    ACTION_TYPES: List[str] = DEFENDER_ACTION_TYPES

    def __init__(self, hidden: int = HIDDEN_DIM) -> None:
        super().__init__(
            obs_dim=DEFENDER_OBS_DIM,
            n_actions=len(self.ACTION_TYPES),
            hidden=hidden,
        )

    def encode_obs(self, obs: Optional[Dict[str, Any]]) -> torch.Tensor:
        return encode_defender_obs(obs)

    def select_action(
        self,
        obs: Optional[Dict[str, Any]],
        legal_actions: Optional[List[str]] = None,
        deterministic: bool = False,
    ) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select an action from the defender's observation.

        Returns (action_type_str, log_prob, entropy, value).
        """
        x      = self.encode_obs(obs).unsqueeze(0)
        logits, value = self.forward(x)

        # Apply action mask
        if legal_actions is not None:
            mask = torch.full((len(self.ACTION_TYPES),), float("-inf"))
            for i, a in enumerate(self.ACTION_TYPES):
                if a in legal_actions:
                    mask[i] = 0.0
            logits = logits + mask

        if deterministic:
            action_idx = logits.argmax(dim=-1)
        else:
            dist       = torch.distributions.Categorical(logits=logits)
            action_idx = dist.sample()

        dist     = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(action_idx)
        entropy  = dist.entropy()
        action_str = self.ACTION_TYPES[action_idx.item()]
        return action_str, log_prob.squeeze(), entropy.squeeze(), value.squeeze()


# ---------------------------------------------------------------------------
# FraudsterPolicy
# ---------------------------------------------------------------------------

class FraudsterPolicy(ActorCritic):
    """
    PPO-compatible policy for the fraudster agent.

    Input : encoded fraudster observation (FRAUDSTER_OBS_DIM float vector)
    Output: distribution over FRAUDSTER_ACTION_TYPES (8 actions)
    """
    ACTION_TYPES: List[str] = FRAUDSTER_ACTION_TYPES

    def __init__(self, hidden: int = HIDDEN_DIM) -> None:
        super().__init__(
            obs_dim=FRAUDSTER_OBS_DIM,
            n_actions=len(self.ACTION_TYPES),
            hidden=hidden,
        )

    def encode_obs(self, obs: Optional[Dict[str, Any]]) -> torch.Tensor:
        return encode_fraudster_obs(obs)

    def select_action(
        self,
        obs: Optional[Dict[str, Any]],
        legal_actions: Optional[List[str]] = None,
        deterministic: bool = False,
    ) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select an action from the fraudster's observation.

        Returns (action_type_str, log_prob, entropy, value).
        """
        x      = self.encode_obs(obs).unsqueeze(0)
        logits, value = self.forward(x)

        if legal_actions is not None:
            mask = torch.full((len(self.ACTION_TYPES),), float("-inf"))
            for i, a in enumerate(self.ACTION_TYPES):
                if a in legal_actions:
                    mask[i] = 0.0
            logits = logits + mask

        if deterministic:
            action_idx = logits.argmax(dim=-1)
        else:
            dist       = torch.distributions.Categorical(logits=logits)
            action_idx = dist.sample()

        dist     = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(action_idx)
        entropy  = dist.entropy()
        action_str = self.ACTION_TYPES[action_idx.item()]
        return action_str, log_prob.squeeze(), entropy.squeeze(), value.squeeze()
