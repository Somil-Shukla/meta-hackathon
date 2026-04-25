"""
Smoke tests for the Fraud Detection RL Environment.

Turn-based protocol (verified here):
  reset()               → fraudster obs  (current_agent="fraudster")
  step(fraudster_action)→ defender obs   (current_agent="defender", no rewards)
  step(defender_action) → fraudster obs  (current_agent="fraudster", rewards emitted)
  …repeat until done=True

Verifies:
  1. reset() creates a new world with a valid fraudster observation.
  2. step() changes the world state (step increments after full round).
  3. Defender and fraudster observations are different.
  4. Per-step rewards are returned after the defender's half-step.
  5. Termination works (max_steps termination).
  6. Episode is deterministic when seeded.
  7. Each fraud family can be instantiated.
  8. Action masking returns non-empty legal action sets.
  9. Freeze action produces a float defender reward.
 10. Full episode runs without error and grade is populated.
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


# Helper: perform one full round (fraudster then defender)
def _full_step(
    env: FraudEnvironment,
    fraudster_action: FraudsterActionType = FraudsterActionType.DO_NOTHING,
    defender_action: DefenderActionType   = DefenderActionType.DO_NOTHING,
    fraudster_target=None,
    defender_target=None,
):
    """
    Execute one complete round:
      1. Send fraudster action → returns defender obs.
      2. Send defender action  → returns next fraudster obs (or terminal).
    Returns the final observation (after defender acts).
    """
    frd_act = FraudAction(
        fraudster_action=fraudster_action,
        fraudster_target=fraudster_target,
    )
    mid_obs = env.step(frd_act)  # defender observation

    def_act = FraudAction(
        defender_action=defender_action,
        defender_target=defender_target,
    )
    return env.step(def_act)     # next fraudster obs (or terminal)


# ---------------------------------------------------------------------------
# Test 1: reset creates a new world with FRAUDSTER observation
# ---------------------------------------------------------------------------

def test_reset_creates_world(env):
    obs = env.reset(task_name="mule_cashout", seed=42)
    assert obs is not None
    assert obs.task_name == "mule_cashout"
    assert obs.episode_id is not None
    assert obs.step == 0
    assert obs.step_budget == {"total": DEFAULT_MAX_STEPS, "used": 0, "remaining": DEFAULT_MAX_STEPS}
    assert obs.done is False
    assert obs.current_agent == "fraudster"
    # Fraudster observation is populated
    assert obs.fraudster_obs is not None
    assert "active_routes" in obs.fraudster_obs
    # Defender observation is deferred
    assert obs.defender_obs is None


def test_reset_random_task(env):
    obs = env.reset(task_name="random")
    assert obs.task_name in [t for t in VALID_TASK_NAMES if t != "random"]


# ---------------------------------------------------------------------------
# Test 2: step changes state
# ---------------------------------------------------------------------------

def test_step_changes_state(env):
    obs = env.reset(task_name="mule_cashout", seed=42)
    initial_step = obs.step  # 0

    # Fraudster half-step: step counter should NOT increment yet
    mid_obs = env.step(FraudAction(fraudster_action=FraudsterActionType.DO_NOTHING))
    assert mid_obs.current_agent == "defender"
    assert mid_obs.step == initial_step  # still 0 — world.step hasn't changed

    # Defender half-step: full step completes, step counter increments
    final_obs = env.step(FraudAction(defender_action=DefenderActionType.DO_NOTHING))
    assert final_obs.step == initial_step + 1
    assert final_obs.step_budget["used"] == 1


# ---------------------------------------------------------------------------
# Test 3: observations differ between agents
# ---------------------------------------------------------------------------

def test_observations_differ(env):
    obs = env.reset(task_name="refund_abuse", seed=123)
    # reset gives fraudster obs
    assert obs.fraudster_obs is not None
    assert obs.defender_obs is None
    assert "active_routes" in obs.fraudster_obs

    # After fraudster half-step, defender obs is returned
    mid_obs = env.step(FraudAction(fraudster_action=FraudsterActionType.DO_NOTHING))
    assert mid_obs.defender_obs is not None
    assert mid_obs.fraudster_obs is None
    assert "accounts" in mid_obs.defender_obs

    # Defender and fraudster obs are structurally different
    assert obs.fraudster_obs != mid_obs.defender_obs


# ---------------------------------------------------------------------------
# Test 4: rewards are returned after defender half-step
# ---------------------------------------------------------------------------

def test_rewards_returned(env):
    env.reset(task_name="mule_cashout", seed=42)

    # Fraudster half-step: no rewards emitted
    mid_obs = env.step(FraudAction(fraudster_action=FraudsterActionType.DO_NOTHING))
    assert mid_obs.defender_reward is None
    assert mid_obs.fraudster_reward is None

    # Defender half-step: rewards emitted
    final_obs = env.step(FraudAction(defender_action=DefenderActionType.DO_NOTHING))
    assert final_obs.defender_reward is not None
    assert final_obs.fraudster_reward is not None
    assert isinstance(final_obs.defender_reward, float)
    assert isinstance(final_obs.fraudster_reward, float)


# ---------------------------------------------------------------------------
# Test 5: termination works
# ---------------------------------------------------------------------------

def test_termination_max_steps(env):
    env.reset(task_name="mule_cashout", seed=42)
    obs = None
    # Each full round = 2 step() calls
    for _ in range(DEFAULT_MAX_STEPS):
        obs = _full_step(env)
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

    assert obs1.episode_id == obs2.episode_id
    assert obs1.task_name  == obs2.task_name


# ---------------------------------------------------------------------------
# Test 7: all fraud families work
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", [t for t in VALID_TASK_NAMES if t != "random"])
def test_all_families(env, task):
    obs = env.reset(task_name=task, seed=1)
    assert obs.task_name == task
    assert obs.fraudster_obs is not None
    assert obs.current_agent == "fraudster"


# ---------------------------------------------------------------------------
# Test 8: action masking
# ---------------------------------------------------------------------------

def test_action_masking(env):
    obs = env.reset(task_name="mule_cashout", seed=42)

    # At reset, only fraudster masks are populated
    assert obs.available_fraudster_actions is not None
    assert len(obs.available_fraudster_actions) > 0
    assert FraudsterActionType.DO_NOTHING.value in obs.available_fraudster_actions
    assert isinstance(obs.fraudster_action_targets, dict)

    # Defender masks are deferred until fraudster acts
    assert obs.available_defender_actions is None
    assert obs.defender_action_targets is None

    # After fraudster half-step, defender masks are populated
    mid_obs = env.step(FraudAction(fraudster_action=FraudsterActionType.DO_NOTHING))
    assert mid_obs.available_defender_actions is not None
    assert len(mid_obs.available_defender_actions) > 0
    assert DefenderActionType.DO_NOTHING.value in mid_obs.available_defender_actions
    assert isinstance(mid_obs.defender_action_targets, dict)


# ---------------------------------------------------------------------------
# Test 9: freeze action produces a float defender reward
# ---------------------------------------------------------------------------

def test_freeze_fraudulent_account(env):
    """Freezing a mule should return a positive or negative float reward."""
    obs = env.reset(task_name="mule_cashout", seed=42)

    # Fraudster half-step to get defender obs with targets
    mid_obs = env.step(FraudAction(fraudster_action=FraudsterActionType.DO_NOTHING))
    freeze_targets = (mid_obs.defender_action_targets or {}).get("freeze", [])
    if not freeze_targets:
        pytest.skip("No freeze targets available")

    target = freeze_targets[0]
    final_obs = env.step(FraudAction(
        defender_action=DefenderActionType.FREEZE,
        defender_target=target,
    ))
    assert isinstance(final_obs.defender_reward, float)
    assert final_obs.info is not None


# ---------------------------------------------------------------------------
# Test 10: full episode runs without error and grade is populated
# ---------------------------------------------------------------------------

def test_full_episode_runs(env):
    obs = env.reset(task_name="account_takeover", seed=7)
    for _ in range(DEFAULT_MAX_STEPS + 5):
        if obs.done:
            break
        obs = _full_step(env)

    assert obs.done is True
    grade = (obs.info or {}).get("grade", {})
    assert "defender_score"  in grade
    assert "fraudster_score" in grade
