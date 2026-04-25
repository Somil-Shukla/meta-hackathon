"""
FraudEnvironment — core OpenEnv-compatible environment implementation.

Turn-based interaction protocol
---------------------------------
The episode proceeds in alternating half-steps:

  1. ``reset(task_name=...)``
       Generates a fresh world.  Returns a **fraudster** observation.
       (current_agent="fraudster")

  2. Fraudster half-step — ``step(FraudAction(fraudster_action=..., ...))``
       Only ``fraudster_action`` and ``fraudster_target`` are consumed.
       The fraudster's action is applied to the world, but the world
       transition, rewards and termination check are deferred.
       Returns a **defender** observation. (current_agent="defender")

  3. Defender half-step — ``step(FraudAction(defender_action=..., ...))``
       Only ``defender_action`` and ``defender_target`` are consumed.
       After applying the defender's action, the engine runs the full
       world transition, computes rewards for both agents, and checks
       termination.  Returns the **next fraudster** observation
       (current_agent="fraudster") or a terminal observation (done=True).

  4. Repeat from step 2 until done=True.

Termination is *only* checked after the defender's half-step, so only the
defender can end the episode.  The fraudster always tries to extend play.

Reward / grading compatibility
---------------------------------
All reward logic (RewardEngine) and grading (GradingEngine) operate on
both agents' results simultaneously at the end of each defender half-step,
exactly as before.  No reward semantics have changed.
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
    Turn-based multi-agent fraud simulation environment.

    Supports four fraud families: refund_abuse, mule_cashout,
    merchant_collusion, account_takeover.  Pass ``task_name="random"``
    to pick a fraud family at random each episode.

    The fraudster always acts first; the defender responds.  Only the
    defender's half-step can trigger episode termination.
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

        # Turn-based phase tracking
        self._phase: str = "fraudster"          # "fraudster" | "defender"
        self._pending_fraudster_result: Optional[ActionResult] = None

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

        Returns the initial **fraudster** observation so the fraudster can
        act first.

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

        self._task_name           = task_name
        self._cumulative_defender = 0.0
        self._cumulative_fraudster = 0.0
        self._step_records        = []
        self._phase               = "fraudster"
        self._pending_fraudster_result = None

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

        # Build initial fraudster observation and action mask
        frd_obs  = self._obs_gen.fraudster_observation(self._world)
        frd_mask = self._action_masker.fraudster_legal_actions(self._world)

        return FraudObservation(
            episode_id=self._world.episode_id,
            step=0,
            step_budget=self._budget_dict(),
            task_name=self._world.fraud_family,
            episode_done=False,
            current_agent="fraudster",
            # Fraudster fields populated
            fraudster_obs=frd_obs,
            available_fraudster_actions=list(frd_mask.keys()),
            fraudster_action_targets=frd_mask,
            # Defender fields deferred until fraudster acts
            defender_obs=None,
            available_defender_actions=None,
            defender_action_targets=None,
            defender_reward=0.0,
            fraudster_reward=0.0,
            reward=0.0,
            done=False,
            info={
                "phase": "fraudster",
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
        Execute one half-step for the current agent.

        Fraudster half-step
        ~~~~~~~~~~~~~~~~~~~
        Consumes ``fraudster_action`` / ``fraudster_target`` from the action.
        Applies the fraudster's action to the world (no transition yet).
        Returns the **defender** observation (current_agent="defender").
        No rewards are emitted; termination is not checked.

        Defender half-step
        ~~~~~~~~~~~~~~~~~~
        Consumes ``defender_action`` / ``defender_target`` from the action.
        After applying the defender's action the engine runs:
          1. Full world transition (background transactions, risk propagation…)
          2. Reward computation for both agents
          3. Termination check (only the defender can end the episode)
        Returns the **next fraudster** observation (current_agent="fraudster")
        or a terminal observation (done=True).
        """
        if self._world is None:
            raise RuntimeError("Call reset() before step().")

        world = self._world

        # ══════════════════════════════════════════════════════════════════════
        # FRAUDSTER HALF-STEP
        # ══════════════════════════════════════════════════════════════════════
        if self._phase == "fraudster":
            if world.step >= world.max_steps:
                raise RuntimeError("Episode is done. Call reset().")

            # Apply fraudster action; no transition or termination yet
            fraudster_result = self._action_proc.apply_fraudster_action(
                action_type=action.fraudster_action.value,
                target=action.fraudster_target,
                world=world,
            )
            self._pending_fraudster_result = fraudster_result
            self._phase = "defender"

            # Build defender observation now that the fraudster has moved
            def_obs  = self._obs_gen.defender_observation(world)
            def_mask = self._action_masker.defender_legal_actions(world)

            return FraudObservation(
                episode_id=world.episode_id,
                step=world.step,
                step_budget=self._budget_dict(),
                task_name=world.fraud_family,
                episode_done=False,
                current_agent="defender",
                # Defender fields populated
                defender_obs=def_obs,
                available_defender_actions=list(def_mask.keys()),
                defender_action_targets=def_mask,
                # Fraudster fields deferred until full step completes
                fraudster_obs=None,
                available_fraudster_actions=None,
                fraudster_action_targets=None,
                defender_reward=None,
                fraudster_reward=None,
                reward=0.0,
                done=False,
                info={
                    "phase": "defender",
                    "fraudster_effect": fraudster_result.effect,
                    "cumulative": {
                        "defender_reward":  round(self._cumulative_defender, 4),
                        "fraudster_reward": round(self._cumulative_fraudster, 4),
                    },
                },
            )

        # ══════════════════════════════════════════════════════════════════════
        # DEFENDER HALF-STEP  (full step completion)
        # ══════════════════════════════════════════════════════════════════════
        fraudster_result = self._pending_fraudster_result  # stored from prior half-step

        # ── Apply defender action ─────────────────────────────────────────────
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

        # ── Termination (only checked after defender acts) ────────────────────
        term = self._termination.check(world)

        # ── Record for grading CSV ────────────────────────────────────────────
        # world.step was already incremented inside transition.advance()
        step_rec = StepRecord(
            episode_id=world.episode_id,
            fraud_family=world.fraud_family,
            step=world.step,
            defender_action=action.defender_action.value,
            defender_target=action.defender_target,
            fraudster_action=fraudster_result.action_type,
            fraudster_target=fraudster_result.target,
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

        # ── Reset phase state ─────────────────────────────────────────────────
        self._pending_fraudster_result = None
        self._phase = "fraudster"
        self._state.step_count += 1

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

        # Common info payload
        step_info: Dict[str, Any] = {
            "fraudster_reward":           step_reward.fraudster_reward,
            "defender_reward_breakdown":  step_reward.defender_breakdown,
            "fraudster_reward_breakdown": step_reward.fraudster_breakdown,
            "fraudster_effect":           fraudster_result.effect,
            "defender_effect":            defender_result.effect,
            "cumulative": {
                "defender_reward":  round(self._cumulative_defender, 4),
                "fraudster_reward": round(self._cumulative_fraudster, 4),
            },
            "world_summary": world.to_dict(),
            **({"grade": grade_info} if grade_info else {}),
        }

        if term.done:
            # Terminal: return both observations so agents can see final state
            def_obs  = self._obs_gen.defender_observation(world)
            frd_obs  = self._obs_gen.fraudster_observation(world)
            def_mask = self._action_masker.defender_legal_actions(world)
            frd_mask = self._action_masker.fraudster_legal_actions(world)

            return FraudObservation(
                episode_id=world.episode_id,
                step=world.step,
                step_budget=self._budget_dict(),
                task_name=world.fraud_family,
                episode_done=True,
                reason=term.reason,
                current_agent=None,
                defender_obs=def_obs,
                fraudster_obs=frd_obs,
                available_defender_actions=list(def_mask.keys()),
                available_fraudster_actions=list(frd_mask.keys()),
                defender_action_targets=def_mask,
                fraudster_action_targets=frd_mask,
                defender_reward=step_reward.defender_reward,
                fraudster_reward=step_reward.fraudster_reward,
                reward=step_reward.defender_reward,
                done=True,
                info=step_info,
            )

        # Non-terminal: return next fraudster observation
        frd_obs  = self._obs_gen.fraudster_observation(world)
        frd_mask = self._action_masker.fraudster_legal_actions(world)

        return FraudObservation(
            episode_id=world.episode_id,
            step=world.step,
            step_budget=self._budget_dict(),
            task_name=world.fraud_family,
            episode_done=False,
            current_agent="fraudster",
            fraudster_obs=frd_obs,
            available_fraudster_actions=list(frd_mask.keys()),
            fraudster_action_targets=frd_mask,
            # Defender fields deferred until next fraudster action
            defender_obs=None,
            available_defender_actions=None,
            defender_action_targets=None,
            defender_reward=step_reward.defender_reward,
            fraudster_reward=step_reward.fraudster_reward,
            reward=step_reward.defender_reward,
            done=False,
            info=step_info,
        )

    # ----------------------------------------------------------------- state

    @property
    def state(self) -> State:
        return self._state

    # ----------------------------------------------------------------- helpers

    def _budget_dict(self) -> Dict[str, int]:
        if self._world is None:
            return {"total": DEFAULT_MAX_STEPS, "used": 0, "remaining": DEFAULT_MAX_STEPS}
        used  = self._world.step
        total = self._world.max_steps
        return {"total": total, "used": used, "remaining": total - used}
