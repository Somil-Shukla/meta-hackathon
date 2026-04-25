"""
FraudEnvironment — core OpenEnv-compatible environment implementation.

Implements the OpenEnv ``Environment`` interface for the multi-agent
fraud simulation.  All world logic lives in the ``scam_detection.*``
modules; this class wires them together and exposes the standard lifecycle.

Episode lifecycle
-----------------
1. ``reset(task_name=...)`` → new randomised world generated; both agents
   receive initial partial observations.
2. Steps 1–N → both agents submit actions in a single ``step()`` call;
   fraudster action applied first, then defender; world transitions;
   rewards and next observations returned.
3. Terminal step → episode ends when max_steps reached, all routes
   deactivated, or fraud threshold exceeded.

Multi-agent design
------------------
The ``FraudAction`` carries both ``defender_action`` and ``fraudster_action``
fields.  The ``FraudObservation`` carries separate partial views
(``defender_obs``, ``fraudster_obs``) for each agent.  The base ``reward``
field is set to the defender's reward (primary training signal); the
fraudster's reward is available in ``info["fraudster_reward"]``.
"""
from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        DefenderActionType,
        FraudAction,
        FraudObservation,
        FraudsterActionType,
    )
    from ..constants import VALID_TASK_NAMES, MAX_STEPS, DEFAULT_MAX_STEPS
    from ..scenario_generator import ScenarioGenerator
    from ..observation_generator import ObservationGenerator
    from ..action_processor import ActionMasker, ActionProcessor, ActionResult
    from ..transition_engine import TransitionEngine
    from ..reward_engine import RewardEngine
    from ..termination_engine import TerminationEngine
    from ..grading_engine import GradingEngine, StepRecord
except ImportError:
    from models import (
        DefenderActionType,
        FraudAction,
        FraudObservation,
        FraudsterActionType,
    )
    from constants import VALID_TASK_NAMES, MAX_STEPS, DEFAULT_MAX_STEPS
    from scenario_generator import ScenarioGenerator
    from observation_generator import ObservationGenerator
    from action_processor import ActionMasker, ActionProcessor, ActionResult
    from transition_engine import TransitionEngine
    from reward_engine import RewardEngine
    from termination_engine import TerminationEngine
    from grading_engine import GradingEngine, StepRecord


class FraudEnvironment(Environment):
    """
    Multi-agent fraud simulation environment.

    Supports four fraud families: refund_abuse, mule_cashout,
    merchant_collusion, account_takeover.  Pass ``task_name="random"``
    to pick a fraud family at random each episode.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._world = None
        self._task_name: str = "random"

        # Cumulative rewards for grading
        self._cumulative_defender: float = 0.0
        self._cumulative_fraudster: float = 0.0
        self._step_records: List[StepRecord] = []

        # Engine singletons
        self._scenario_gen   = ScenarioGenerator()
        self._obs_gen        = ObservationGenerator()
        self._action_masker  = ActionMasker()
        self._action_proc    = ActionProcessor()
        self._transition     = TransitionEngine()
        self._reward_engine  = RewardEngine()
        self._termination    = TerminationEngine()
        self._grader         = GradingEngine()

    # ----------------------------------------------------------------- reset

    def reset(
        self,
        task_name: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> FraudObservation:
        """
        Start a new fraud episode.

        Parameters
        ----------
        task_name:
            Fraud family to simulate.  One of VALID_TASK_NAMES or "random".
        seed:
            Integer seed for deterministic world generation.
        """
        if not task_name:
            task_name = "random"
        if task_name not in VALID_TASK_NAMES:
            raise ValueError(
                f"Unknown task_name '{task_name}'. Must be one of: {VALID_TASK_NAMES}."
            )

        self._task_name          = task_name
        self._cumulative_defender = 0.0
        self._cumulative_fraudster = 0.0
        self._step_records       = []

        # Generate fresh world
        self._world = self._scenario_gen.generate(
            fraud_family=task_name,
            seed=seed,
            max_steps=DEFAULT_MAX_STEPS,
        )

        self._state = State(
            episode_id=self._world.episode_id,
            step_count=0,
        )

        # Build initial observations and action masks
        def_obs  = self._obs_gen.defender_observation(self._world)
        frd_obs  = self._obs_gen.fraudster_observation(self._world)
        def_mask = self._action_masker.defender_legal_actions(self._world)
        frd_mask = self._action_masker.fraudster_legal_actions(self._world)

        return FraudObservation(
            episode_id=self._world.episode_id,
            step=0,
            step_budget=self._budget_dict(),
            task_name=self._world.fraud_family,
            episode_done=False,
            defender_obs=def_obs,
            fraudster_obs=frd_obs,
            available_defender_actions=list(def_mask.keys()),
            available_fraudster_actions=list(frd_mask.keys()),
            defender_action_targets=def_mask,
            fraudster_action_targets=frd_mask,
            defender_reward=0.0,
            fraudster_reward=0.0,
            reward=0.0,
            done=False,
            info={
                "world_summary": self._world.to_dict(),
                "cumulative": {
                    "defender_reward": 0.0,
                    "fraudster_reward": 0.0,
                },
            },
        )

    # ------------------------------------------------------------------ step

    def step(self, action: FraudAction) -> FraudObservation:  # type: ignore[override]
        """
        Execute one combined step (both agents act simultaneously).

        Processing order:
          1. Validate action
          2. Apply fraudster action
          3. Apply defender action
          4. Transition world (background transactions, risk propagation, …)
          5. Compute rewards
          6. Check termination
          7. Build and return observations
        """
        if self._world is None:
            raise RuntimeError("Call reset() before step().")
        if self._world.step >= self._world.max_steps:
            raise RuntimeError("Episode is done. Call reset().")

        world = self._world

        # ── Apply fraudster action first ─────────────────────────────────────
        fraudster_result = self._action_proc.apply_fraudster_action(
            action_type=action.fraudster_action.value,
            target=action.fraudster_target,
            world=world,
        )

        # ── Apply defender action second ──────────────────────────────────────
        defender_result = self._action_proc.apply_defender_action(
            action_type=action.defender_action.value,
            target=action.defender_target,
            world=world,
        )

        # ── World transition ──────────────────────────────────────────────────
        self._transition.advance(world, fraudster_result, defender_result)

        # ── Compute rewards ───────────────────────────────────────────────────
        step_reward = self._reward_engine.compute(world, fraudster_result, defender_result)
        self._cumulative_defender  += step_reward.defender_reward
        self._cumulative_fraudster += step_reward.fraudster_reward

        # ── Check termination ─────────────────────────────────────────────────
        term = self._termination.check(world)

        # ── Build observations ────────────────────────────────────────────────
        def_obs  = self._obs_gen.defender_observation(world)
        frd_obs  = self._obs_gen.fraudster_observation(world)
        def_mask = self._action_masker.defender_legal_actions(world)
        frd_mask = self._action_masker.fraudster_legal_actions(world)

        self._state.step_count += 1

        # ── Record for grading CSV ────────────────────────────────────────────
        step_rec = StepRecord(
            episode_id=world.episode_id,
            fraud_family=world.fraud_family,
            step=world.step,
            defender_action=action.defender_action.value,
            defender_target=action.defender_target,
            fraudster_action=action.fraudster_action.value,
            fraudster_target=action.fraudster_target,
            defender_reward=step_reward.defender_reward,
            fraudster_reward=step_reward.fraudster_reward,
            cumulative_defender_reward=self._cumulative_defender,
            cumulative_fraudster_reward=self._cumulative_fraudster,
            total_frozen=sum(1 for u in world.users.values() if u.is_frozen),
            total_laundered=world.total_laundered,
            active_routes=len(world.available_routes),
            fraudster_alert_level=world.fraudster_alert_level,
            defender_obs_summary=json.dumps({
                "alerts": len(world.defender_alerts),
                "flagged": len(world.flagged_accounts),
            }),
            fraudster_obs_summary=json.dumps({
                "active_routes": len(world.available_routes),
                "alert_level": round(world.fraudster_alert_level, 3),
            }),
            episode_done=term.done,
            termination_reason=term.reason,
        )
        self._step_records.append(step_rec)

        # ── Grade and save if done ────────────────────────────────────────────
        grade_info: Dict[str, Any] = {}
        if term.done:
            grade = self._grader.grade(
                world,
                self._step_records,
                self._cumulative_defender,
                self._cumulative_fraudster,
            )
            grade_info = {
                "defender_score":          grade.defender_score,
                "fraudster_score":         grade.fraudster_score,
                "total_fraud_prevented":   grade.total_fraud_prevented,
                "total_fraud_escaped":     grade.total_fraud_escaped,
                "false_positive_count":    grade.false_positive_count,
                "false_positive_rate":     grade.false_positive_rate,
                "detection_delay":         grade.detection_delay,
                "customer_friction_score": grade.customer_friction_score,
                "merchant_disruption":     grade.merchant_disruption_score,
            }
            try:
                self._grader.save_grade(grade)
                self._grader.save_rollout(self._step_records, grade)
            except Exception:
                pass  # CSV write failure must not break the environment

        return FraudObservation(
            episode_id=world.episode_id,
            step=world.step,
            step_budget=self._budget_dict(),
            task_name=world.fraud_family,
            episode_done=term.done,
            reason=term.reason,
            defender_obs=def_obs,
            fraudster_obs=frd_obs,
            available_defender_actions=list(def_mask.keys()),
            available_fraudster_actions=list(frd_mask.keys()),
            defender_action_targets=def_mask,
            fraudster_action_targets=frd_mask,
            defender_reward=step_reward.defender_reward,
            fraudster_reward=step_reward.fraudster_reward,
            # Base reward = defender reward (primary training signal)
            reward=step_reward.defender_reward,
            done=term.done,
            info={
                "fraudster_reward": step_reward.fraudster_reward,
                "defender_reward_breakdown":  step_reward.defender_breakdown,
                "fraudster_reward_breakdown": step_reward.fraudster_breakdown,
                "fraudster_effect": fraudster_result.effect,
                "defender_effect":  defender_result.effect,
                "cumulative": {
                    "defender_reward":  round(self._cumulative_defender, 4),
                    "fraudster_reward": round(self._cumulative_fraudster, 4),
                },
                "world_summary": world.to_dict(),
                **({"grade": grade_info} if grade_info else {}),
            },
        )

    # ----------------------------------------------------------------- state

    @property
    def state(self) -> State:
        return self._state

    # ----------------------------------------------------------------- helpers

    def _budget_dict(self) -> Dict[str, int]:
        if self._world is None:
            return {"total": DEFAULT_MAX_STEPS, "used": 0, "remaining": DEFAULT_MAX_STEPS}
        used = self._world.step
        total = self._world.max_steps
        return {"total": total, "used": used, "remaining": total - used}
