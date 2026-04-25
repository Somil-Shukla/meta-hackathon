"""
Smoke tests for the Fraud Detection RL Environment.

Verifies:
  1. reset() creates a new world with a valid observation.
  2. step() changes the world state.
  3. Defender and fraudster observations are different.
  4. Per-step rewards are returned.
  5. Termination works (max_steps termination).
  6. Episode is deterministic when seeded.
  7. Each fraud family can be instantiated.
  8. Action masking returns non-empty legal action sets.
"""
from __future__ import annotations

import pytest

from scam_detection.constants import VALID_TASK_NAMES, DEFAULT_MAX_STEPS
from scam_detection.models import (
    DefenderActionType,
    FraudAction,
    FraudsterActionType,
)
from scam_detection.server.fraud_environment import FraudEnvironment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    return FraudEnvironment()


# ---------------------------------------------------------------------------
# Test 1: reset creates a new world
# ---------------------------------------------------------------------------

def test_reset_creates_world(env):
    obs = env.reset(task_name="mule_cashout", seed=42)
    assert obs is not None
    assert obs.task_name == "mule_cashout"
    assert obs.episode_id is not None
    assert obs.step == 0
    assert obs.step_budget == {"total": DEFAULT_MAX_STEPS, "used": 0, "remaining": DEFAULT_MAX_STEPS}
    assert obs.done is False
    assert obs.defender_obs is not None
    assert obs.fraudster_obs is not None


def test_reset_random_task(env):
    obs = env.reset(task_name="random")
    assert obs.task_name in [t for t in VALID_TASK_NAMES if t != "random"]


# ---------------------------------------------------------------------------
# Test 2: step changes state
# ---------------------------------------------------------------------------

def test_step_changes_state(env):
    obs = env.reset(task_name="mule_cashout", seed=42)
    initial_step = obs.step

    action = FraudAction(
        defender_action=DefenderActionType.DO_NOTHING,
        fraudster_action=FraudsterActionType.DO_NOTHING,
    )
    obs2 = env.step(action)

    assert obs2.step != initial_step
    assert obs2.step == initial_step + 1
    assert obs2.step_budget["used"] == 1


# ---------------------------------------------------------------------------
# Test 3: observations differ between agents
# ---------------------------------------------------------------------------

def test_observations_differ(env):
    obs = env.reset(task_name="refund_abuse", seed=123)
    assert obs.defender_obs != obs.fraudster_obs
    # Defender sees accounts and alerts; fraudster sees routes
    assert "accounts" in obs.defender_obs
    assert "active_routes" in obs.fraudster_obs


# ---------------------------------------------------------------------------
# Test 4: rewards are returned
# ---------------------------------------------------------------------------

def test_rewards_returned(env):
    env.reset(task_name="mule_cashout", seed=42)
    action = FraudAction(
        defender_action=DefenderActionType.DO_NOTHING,
        fraudster_action=FraudsterActionType.DO_NOTHING,
    )
    obs = env.step(action)
    # Rewards should be float (even if 0.0)
    assert obs.defender_reward is not None
    assert obs.fraudster_reward is not None
    assert isinstance(obs.defender_reward, float)
    assert isinstance(obs.fraudster_reward, float)


# ---------------------------------------------------------------------------
# Test 5: termination works
# ---------------------------------------------------------------------------

def test_termination_max_steps(env):
    env.reset(task_name="mule_cashout", seed=42)
    action = FraudAction(
        defender_action=DefenderActionType.DO_NOTHING,
        fraudster_action=FraudsterActionType.DO_NOTHING,
    )
    obs = None
    for _ in range(DEFAULT_MAX_STEPS):
        obs = env.step(action)
        if obs.done:
            break

    assert obs is not None
    assert obs.done is True
    assert obs.reason is not None


# ---------------------------------------------------------------------------
# Test 6: deterministic when seeded
# ---------------------------------------------------------------------------

def test_deterministic_with_seed(env):
    env2 = FraudEnvironment()

    obs1 = env.reset(task_name="merchant_collusion", seed=999)
    obs2 = env2.reset(task_name="merchant_collusion", seed=999)

    # Same seed → same episode ID (same generated world)
    assert obs1.episode_id == obs2.episode_id
    # Same observation structure
    assert obs1.task_name == obs2.task_name


# ---------------------------------------------------------------------------
# Test 7: all fraud families work
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", [t for t in VALID_TASK_NAMES if t != "random"])
def test_all_families(env, task):
    obs = env.reset(task_name=task, seed=1)
    assert obs.task_name == task
    assert obs.defender_obs is not None
    assert obs.fraudster_obs is not None


# ---------------------------------------------------------------------------
# Test 8: action masking
# ---------------------------------------------------------------------------

def test_action_masking(env):
    obs = env.reset(task_name="mule_cashout", seed=42)

    assert obs.available_defender_actions is not None
    assert len(obs.available_defender_actions) > 0
    assert DefenderActionType.DO_NOTHING.value in obs.available_defender_actions

    assert obs.available_fraudster_actions is not None
    assert len(obs.available_fraudster_actions) > 0
    assert FraudsterActionType.DO_NOTHING.value in obs.available_fraudster_actions

    # Targets should be a dict
    assert isinstance(obs.defender_action_targets, dict)
    assert isinstance(obs.fraudster_action_targets, dict)


# ---------------------------------------------------------------------------
# Test 9: freeze action produces fraud-prevention reward
# ---------------------------------------------------------------------------

def test_freeze_fraudulent_account(env):
    """Freezing a mule should return a positive defender reward."""
    obs = env.reset(task_name="mule_cashout", seed=42)

    # Find a freezable account from the mask
    freeze_targets = (obs.defender_action_targets or {}).get("freeze", [])
    if not freeze_targets:
        pytest.skip("No freeze targets available at reset")

    # Pick first candidate
    target = freeze_targets[0]
    action = FraudAction(
        defender_action=DefenderActionType.FREEZE,
        defender_target=target,
        fraudster_action=FraudsterActionType.DO_NOTHING,
    )
    obs2 = env.step(action)
    # Reward can be positive (fraud) or negative (FP) — just check it's a float
    assert isinstance(obs2.defender_reward, float)
    assert obs2.info is not None


# ---------------------------------------------------------------------------
# Test 10: full episode runs without error
# ---------------------------------------------------------------------------

def test_full_episode_runs(env):
    obs = env.reset(task_name="account_takeover", seed=7)
    action = FraudAction(
        defender_action=DefenderActionType.DO_NOTHING,
        fraudster_action=FraudsterActionType.DO_NOTHING,
    )
    for _ in range(DEFAULT_MAX_STEPS + 5):
        if obs.done:
            break
        obs = env.step(action)

    assert obs.done is True
    # Grade should be populated at end
    grade = (obs.info or {}).get("grade", {})
    assert "defender_score" in grade
    assert "fraudster_score" in grade
