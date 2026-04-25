"""
Typed async client for the Fraud Detection Environment.

Wraps the OpenEnv ``EnvClient`` base class with concrete serialisation
and deserialisation for ``FraudAction`` and ``FraudObservation``.

Usage::

    from scam_detection.client import FraudEnvClient
    from scam_detection.models import DefenderActionType, FraudsterActionType, FraudAction

    async with FraudEnvClient(base_url="http://localhost:8000") as env:
        result = await env.reset(task_name="mule_cashout")
        obs = result.observation
        print(obs.task_name, obs.step_budget)

        action = FraudAction(
            defender_action=DefenderActionType.MONITOR,
            defender_target="user_003",
            fraudster_action=FraudsterActionType.CASHOUT_ATTEMPT,
            fraudster_target="route_000",
        )
        result = await env.step(action)
        print("Defender reward:", result.observation.defender_reward)
        print("Fraudster reward:", result.observation.fraudster_reward)
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import (
        DefenderActionType,
        FraudAction,
        FraudObservation,
        FraudsterActionType,
    )
    from .constants import VALID_TASK_NAMES
except ImportError:
    from models import (
        DefenderActionType,
        FraudAction,
        FraudObservation,
        FraudsterActionType,
    )
    from constants import VALID_TASK_NAMES


class FraudEnvClient(EnvClient[FraudAction, FraudObservation, State]):
    """
    Typed async client for the Fraud Detection Environment.

    Maintains a persistent WebSocket connection to the server so that
    each ``step()`` call incurs minimal latency.

    The client forwards ``task_name`` (and optional ``seed``) to the
    server's reset endpoint.
    """

    # ----------------------------------------------------------------- reset

    async def reset(
        self,
        task_name: str = "random",
        seed: Optional[int] = None,
    ) -> StepResult[FraudObservation]:
        """
        Start a new episode on the server.

        Parameters
        ----------
        task_name:
            Fraud family.  Must be one of VALID_TASK_NAMES or "random".
        seed:
            Optional integer seed for reproducible world generation.
        """
        if task_name not in VALID_TASK_NAMES:
            raise ValueError(
                f"Unknown task_name '{task_name}'. Must be one of: {VALID_TASK_NAMES}."
            )
        kwargs: Dict[str, Any] = {"task_name": task_name}
        if seed is not None:
            kwargs["seed"] = seed
        return await super().reset(**kwargs)

    # ----------------------------------------------------------------- wire

    def _step_payload(self, action: FraudAction) -> Dict[str, Any]:
        """Serialise a FraudAction to a JSON-safe dict."""
        payload: Dict[str, Any] = {
            "defender_action": action.defender_action.value,
            "fraudster_action": action.fraudster_action.value,
        }
        if action.defender_target is not None:
            payload["defender_target"] = action.defender_target
        if action.fraudster_target is not None:
            payload["fraudster_target"] = action.fraudster_target
        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[FraudObservation]:
        """Deserialise the server response into a typed StepResult."""
        obs_raw = payload.get("observation", {})

        # Infer current_agent from the server field; fall back to heuristic
        # based on which partial observation is populated, so older server
        # builds remain compatible.
        current_agent = obs_raw.get("current_agent")
        if current_agent is None and not obs_raw.get("episode_done"):
            if obs_raw.get("fraudster_obs") is not None and obs_raw.get("defender_obs") is None:
                current_agent = "fraudster"
            elif obs_raw.get("defender_obs") is not None and obs_raw.get("fraudster_obs") is None:
                current_agent = "defender"

        observation = FraudObservation(
            episode_id=obs_raw.get("episode_id"),
            step=obs_raw.get("step"),
            step_budget=obs_raw.get("step_budget"),
            task_name=obs_raw.get("task_name"),
            episode_done=obs_raw.get("episode_done"),
            reason=obs_raw.get("reason"),
            current_agent=current_agent,
            defender_obs=obs_raw.get("defender_obs"),
            fraudster_obs=obs_raw.get("fraudster_obs"),
            available_defender_actions=obs_raw.get("available_defender_actions"),
            available_fraudster_actions=obs_raw.get("available_fraudster_actions"),
            defender_action_targets=obs_raw.get("defender_action_targets"),
            fraudster_action_targets=obs_raw.get("fraudster_action_targets"),
            defender_reward=obs_raw.get("defender_reward"),
            fraudster_reward=obs_raw.get("fraudster_reward"),
            info=obs_raw.get("info", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """Deserialise /state response."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
