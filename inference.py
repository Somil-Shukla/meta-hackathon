"""
Inference Script — Fraud Detection Environment
===============================================

An LLM-driven agent runner that plays multi-agent episodes of the
FraudEnvironment using a **turn-based** protocol:

  1. ``reset()`` → fraudster receives its observation and acts first.
  2. Fraudster ``step()`` → environment processes fraudster action, returns
     defender observation.
  3. Defender ``step()`` → environment processes defender action, runs world
     transition, emits rewards.  Returns next fraudster observation or done.
  4. Repeat from step 2 until done=True.

Each full round (fraudster half + defender half) corresponds to one [STEP] log.

Each agent (defender, fraudster) is backed by its **own** LLM, configured
independently via environment variables.  This lets you use different models,
API keys, or endpoints for each role — e.g. a fine-tuned defender checkpoint
against a stronger adversarial fraudster.

Required environment variables
-------------------------------
Per-agent (preferred):
  DEFENDER_MODEL_NAME    Model for the defender (e.g. fine-tuned checkpoint).
  DEFENDER_API_BASE_URL  API endpoint for the defender model.
  DEFENDER_API_KEY       API key for the defender model.

  FRAUDSTER_MODEL_NAME   Model for the fraudster.
  FRAUDSTER_API_BASE_URL API endpoint for the fraudster model.
  FRAUDSTER_API_KEY      API key for the fraudster model.

Shared fallbacks (used when per-agent vars are not set):
  API_BASE_URL           Shared API endpoint.
  MODEL_NAME             Shared model identifier.
  HF_TOKEN / API_KEY     Shared authentication key.

Other:
  ENV_URL                Environment server URL (default: http://localhost:8000).

Structured log format (validator-compatible):
  [START] task=<task> env=fraud_detection defender_model=<m> fraudster_model=<m>
  [STEP]  step=<n> action=<defender_action>|<fraudster_action> done=<bool> error=<null|msg>
  [END]   task=<task> success=<bool> steps=<n> score=<float>
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import textwrap
import random
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

try:
    from scam_detection.client import FraudEnvClient
    from scam_detection.models import (
        DefenderActionType,
        FraudAction,
        FraudObservation,
        FraudsterActionType,
    )
    from scam_detection.constants import VALID_TASK_NAMES, DEFAULT_MAX_STEPS
except ImportError:
    from client import FraudEnvClient
    from models import (
        DefenderActionType,
        FraudAction,
        FraudObservation,
        FraudsterActionType,
    )
    from constants import VALID_TASK_NAMES, DEFAULT_MAX_STEPS

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Shared fallbacks — used when per-agent vars are absent
_SHARED_API_KEY:      Optional[str] = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
_SHARED_API_BASE_URL: str           = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
_SHARED_MODEL_NAME:   str           = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# Per-agent configuration — each falls back to the shared value if not set
DEFENDER_MODEL_NAME:    str           = os.getenv("DEFENDER_MODEL_NAME")    or _SHARED_MODEL_NAME
DEFENDER_API_BASE_URL:  str           = os.getenv("DEFENDER_API_BASE_URL")  or _SHARED_API_BASE_URL
DEFENDER_API_KEY:       Optional[str] = os.getenv("DEFENDER_API_KEY")       or _SHARED_API_KEY

FRAUDSTER_MODEL_NAME:   str           = os.getenv("FRAUDSTER_MODEL_NAME")   or _SHARED_MODEL_NAME
FRAUDSTER_API_BASE_URL: str           = os.getenv("FRAUDSTER_API_BASE_URL") or _SHARED_API_BASE_URL
FRAUDSTER_API_KEY:      Optional[str] = os.getenv("FRAUDSTER_API_KEY")      or _SHARED_API_KEY

ENV_URL:     str   = os.getenv("ENV_URL", "http://localhost:8000")
BENCHMARK:   str   = "fraud_detection"
TEMPERATURE: float = 0.2
MAX_TOKENS:  int   = 256

_VALID_DEFENDER_ACTIONS  = [a.value for a in DefenderActionType]
_VALID_FRAUDSTER_ACTIONS = [a.value for a in FraudsterActionType]
_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)

# ---------------------------------------------------------------------------
# Structured log helpers
# ---------------------------------------------------------------------------

def log_start(task: str, defender_model: str, fraudster_model: str) -> None:
    print(
        f"[START] task={task} env={BENCHMARK} "
        f"defender_model={defender_model} fraudster_model={fraudster_model}",
        flush=True,
    )

def log_step(
    step: int,
    def_action: str,
    frd_action: str,
    done: bool,
    error: Optional[str] = None,
) -> None:
    done_str  = "true" if done else "false"
    error_str = error or "null"
    print(
        f"[STEP] step={step} action={def_action}|{frd_action} "
        f"done={done_str} error={error_str}",
        flush=True,
    )

def log_end(task: str, success: bool, steps: int, score: float) -> None:
    print(
        f"[END] task={task} success={'true' if success else 'false'} "
        f"steps={steps} score={score}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_DEFENDER_SYSTEM = textwrap.dedent("""
    You are an expert fraud detection analyst (DEFENDER).

    Your goal is to prevent financial fraud with minimum false positives.
    Each step, you receive a partial observation of the financial system
    (accounts, merchants, alerts) and must choose ONE defensive action.

    AVAILABLE ACTIONS:
      monitor                  — place account under surveillance
      challenge                — send step-up authentication
      freeze                   — freeze account (use carefully — false positives hurt)
      hold                     — hold a suspicious pending transaction
      block_merchant           — block a suspicious merchant
      investigate_neighborhood — flag all accounts connected to a target
      do_nothing               — take no action this step

    OUTPUT: Exactly one JSON object, nothing else.
    Format:  {"action": "<action_type>", "target": "<id_or_null>"}

    STRATEGY:
    - Prefer monitor/challenge before freeze to reduce false positives.
    - Freeze only when risk_score > 0.75 AND device_reuse_count >= 3.
    - Block merchants with refund_rate > 0.4 AND anomaly_score > 0.6.
    - Use investigate_neighborhood when you see suspicious clusters.
    - do_nothing is better than a wrong freeze.
""").strip()

_FRAUDSTER_SYSTEM = textwrap.dedent("""
    You are an adaptive financial fraudster (FRAUDSTER).

    Your goal is to launder as much money as possible while avoiding detection.
    Each step you receive information about your available mule routes,
    detection pressure, and cashout readiness.

    AVAILABLE ACTIONS:
      split_payment    — make small split payments to avoid detection thresholds
      rotate_mule      — switch to a different mule account
      switch_merchant  — use a different merchant to process transactions
      rotate_device    — use a different device to reduce device-reuse risk
      delay            — lie low for 2 steps to let detection pressure drop
      refund_abuse     — exploit refund workflows for value extraction
      cashout_attempt  — attempt to cash out via a fraud route
      do_nothing       — take no action this step

    OUTPUT: Exactly one JSON object, nothing else.
    Format:  {"action": "<action_type>", "target": "<id_or_null>"}

    STRATEGY:
    - If alert_level > 0.7, prefer delay or rotate actions before cashout.
    - Attempt cashout only when cashout_ready is true.
    - Rotate mule/device when detection_pressure > 0.6 on your route.
    - Use split_payment to build up laundered value incrementally.
""").strip()

# ---------------------------------------------------------------------------
# LLM action parsers
# ---------------------------------------------------------------------------

def _parse_action_json(text: str) -> Optional[Dict[str, Any]]:
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

def _parse_defender_action(
    text: str,
    legal: List[str],
    targets: Dict[str, List[str]],
) -> Tuple[DefenderActionType, Optional[str]]:
    data = _parse_action_json(text)
    action_str = (data or {}).get("action", "")
    target_str = (data or {}).get("target")

    if action_str not in _VALID_DEFENDER_ACTIONS or action_str not in legal:
        action_str = "do_nothing"

    # Validate target
    valid_targets = targets.get(action_str, [])
    if target_str not in valid_targets:
        target_str = valid_targets[0] if valid_targets and valid_targets != ["self"] else None

    return DefenderActionType(action_str), target_str

def _parse_fraudster_action(
    text: str,
    legal: List[str],
    targets: Dict[str, List[str]],
) -> Tuple[FraudsterActionType, Optional[str]]:
    data = _parse_action_json(text)
    action_str = (data or {}).get("action", "")
    target_str = (data or {}).get("target")

    if action_str not in _VALID_FRAUDSTER_ACTIONS or action_str not in legal:
        action_str = "do_nothing"

    valid_targets = targets.get(action_str, [])
    if target_str not in valid_targets:
        target_str = valid_targets[0] if valid_targets and valid_targets != ["self"] else None

    return FraudsterActionType(action_str), target_str

# ---------------------------------------------------------------------------
# User message builders
# ---------------------------------------------------------------------------

def _build_defender_message(step: int, obs: FraudObservation) -> str:
    def_obs = obs.defender_obs or {}
    budget  = obs.step_budget or {}
    accounts = def_obs.get("accounts", [])
    alerts   = def_obs.get("alerts", [])
    agg      = def_obs.get("aggregate", {})

    parts = [
        f"STEP: {step}  |  STEPS REMAINING: {budget.get('remaining', '?')}",
        f"FRAUD FAMILY HINT: {def_obs.get('fraud_family_hint', 'unknown')}",
        "",
        f"AGGREGATE: frozen={agg.get('total_frozen',0)} "
        f"monitored={agg.get('total_monitored',0)} "
        f"flagged={agg.get('total_flagged',0)} "
        f"blocked_merchants={agg.get('blocked_merchants',0)}",
        "",
        "TOP RISKY ACCOUNTS:",
    ]
    risky = sorted(accounts, key=lambda a: a.get("risk_score", 0), reverse=True)[:5]
    for acc in risky:
        parts.append(
            f"  {acc['id']}: risk={acc.get('risk_score',0):.2f} "
            f"velocity={acc.get('transaction_velocity',0)} "
            f"device_reuse={acc.get('device_reuse_count',0)} "
            f"frozen={acc.get('is_frozen',False)}"
        )

    if alerts:
        parts.append("\nRECENT ALERTS:")
        for a in alerts[-3:]:
            parts.append(f"  {a}")

    legal = obs.available_defender_actions or []
    targets = obs.defender_action_targets or {}
    parts.append("\nLEGAL ACTIONS:")
    for act in legal:
        t = targets.get(act, [])
        parts.append(f"  {act}: targets={t[:3]}")

    parts.append("\nOutput exactly one JSON: {\"action\": \"...\", \"target\": \"...\"}")
    return "\n".join(parts)

def _build_fraudster_message(step: int, obs: FraudObservation) -> str:
    frd_obs = obs.fraudster_obs or {}
    budget  = obs.step_budget or {}

    parts = [
        f"STEP: {step}  |  STEPS REMAINING: {budget.get('remaining', '?')}",
        f"ALERT LEVEL: {frd_obs.get('alert_level', 0):.3f}",
        f"DELAY REMAINING: {frd_obs.get('delayed_steps_remaining', 0)}",
        f"ANY CASHOUT READY: {frd_obs.get('any_cashout_ready', False)}",
        f"TOTAL LAUNDERED: {frd_obs.get('total_laundered_so_far', 0):.2f}",
        "",
        "ACTIVE ROUTES:",
    ]
    for route in frd_obs.get("active_routes", []):
        parts.append(
            f"  {route['id']}: pressure={route.get('detection_pressure',0):.2f} "
            f"cashout_ready={route.get('cashout_ready',False)} "
            f"laundered={route.get('total_laundered',0):.2f}"
        )

    legal = obs.available_fraudster_actions or []
    targets = obs.fraudster_action_targets or {}
    parts.append("\nLEGAL ACTIONS:")
    for act in legal:
        t = targets.get(act, [])
        parts.append(f"  {act}: targets={t[:3]}")

    parts.append("\nOutput exactly one JSON: {\"action\": \"...\", \"target\": \"...\"}")
    return "\n".join(parts)

# ---------------------------------------------------------------------------
# Single episode runner (turn-based)
# ---------------------------------------------------------------------------

async def run_episode(
    env: FraudEnvClient,
    defender_llm: OpenAI,
    fraudster_llm: OpenAI,
    task_name: str,
) -> Tuple[bool, int, float]:
    """
    Run one episode using the turn-based protocol.

    Each agent (defender, fraudster) uses its own OpenAI client so that
    different models, endpoints, or API keys can be configured per-role.

    Round structure:
      1. Fraudster receives its obs → sends fraudster action.
      2. Defender receives its obs  → sends defender action.
      3. Rewards emitted; loop repeats until done.
    """
    result = await env.reset(task_name=task_name)
    obs    = result.observation   # fraudster observation (current_agent="fraudster")

    print(f"\n{'='*60}")
    print(f"EPISODE  task={task_name}  family={obs.task_name}")
    print(f"{'='*60}")

    total_defender_reward = 0.0
    full_steps_taken = 0

    # Track per-round actions for logging
    frd_action_type: FraudsterActionType = FraudsterActionType.DO_NOTHING
    def_action_type: DefenderActionType  = DefenderActionType.DO_NOTHING
    frd_target: Optional[str] = None
    def_target: Optional[str] = None
    step_error: Optional[str] = None

    for _half_step in range(DEFAULT_MAX_STEPS * 2 + 4):
        if obs.done:
            break

        current_agent = obs.current_agent

        # ── FRAUDSTER TURN ───────────────────────────────────────────────────
        if current_agent == "fraudster":
            full_steps_taken += 1
            step_error = None
            legal_frd   = obs.available_fraudster_actions or []
            frd_targets = obs.fraudster_action_targets or {}

            print(f"\n--- Step {full_steps_taken} ---", flush=True)

            frd_response = ""
            try:
                completion = fraudster_llm.chat.completions.create(
                    model=FRAUDSTER_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": _FRAUDSTER_SYSTEM},
                        {"role": "user",   "content": _build_fraudster_message(
                            full_steps_taken, obs
                        )},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                frd_response = completion.choices[0].message.content or ""
            except Exception as e:
                step_error = str(e)

            frd_action_type, frd_target = _parse_fraudster_action(
                frd_response, legal_frd, frd_targets
            )

            print(
                f"  [FRAUDSTER] {frd_action_type.value}({frd_target})",
                flush=True,
            )

            action = FraudAction(
                fraudster_action=frd_action_type,
                fraudster_target=frd_target,
            )
            result = await env.step(action)
            obs    = result.observation
            # obs.current_agent is now "defender"

        # ── DEFENDER TURN ────────────────────────────────────────────────────
        elif current_agent == "defender":
            legal_def   = obs.available_defender_actions or []
            def_targets = obs.defender_action_targets or {}

            def_response = ""
            try:
                completion = defender_llm.chat.completions.create(
                    model=DEFENDER_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": _DEFENDER_SYSTEM},
                        {"role": "user",   "content": _build_defender_message(
                            full_steps_taken, obs
                        )},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                def_response = completion.choices[0].message.content or ""
            except Exception as e:
                step_error = (step_error or "") + " | " + str(e)

            def_action_type, def_target = _parse_defender_action(
                def_response, legal_def, def_targets
            )

            print(
                f"  [DEFENDER]  {def_action_type.value}({def_target})",
                flush=True,
            )

            action = FraudAction(
                defender_action=def_action_type,
                defender_target=def_target,
            )
            result = await env.step(action)
            obs    = result.observation
            done   = result.done or False

            # Rewards are emitted after the defender's half-step
            d_reward = obs.defender_reward or 0.0
            f_reward = obs.fraudster_reward or 0.0
            total_defender_reward += d_reward

            print(
                f"  [REWARDS]   F-reward={f_reward:+.3f}  D-reward={d_reward:+.3f}  done={done}",
                flush=True,
            )
            log_step(
                full_steps_taken,
                def_action_type.value,
                frd_action_type.value,
                done,
                step_error,
            )

            if done:
                info  = obs.info or {}
                grade = info.get("grade", {})
                print(f"\n  EPISODE DONE — reason={obs.reason}")
                if grade:
                    print(f"  Defender score  : {grade.get('defender_score', '?')}")
                    print(f"  Fraudster score : {grade.get('fraudster_score', '?')}")
                    print(f"  Fraud prevented : {grade.get('total_fraud_prevented', '?')}")
                    print(f"  Fraud escaped   : {grade.get('total_fraud_escaped', '?')}")
                    print(f"  False positives : {grade.get('false_positive_count', '?')}")
                break

        else:
            # current_agent is None and done is False — guard against
            # stale/incompatible server responses. Emit a visible warning
            # so the user knows to restart the server.
            print(
                f"  [WARN] current_agent=None and done=False at half_step={_half_step}. "
                "Ensure the server is restarted with the latest code.",
                flush=True,
            )
            break

    score   = round(total_defender_reward / max(1, full_steps_taken), 4)
    success = (obs.info or {}).get("grade", {}).get("defender_score", 0) > 0.5
    return bool(success), full_steps_taken, score

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    if not DEFENDER_API_KEY:
        raise EnvironmentError(
            "No API key found for the defender. "
            "Set DEFENDER_API_KEY or the shared HF_TOKEN / API_KEY."
        )
    if not FRAUDSTER_API_KEY:
        raise EnvironmentError(
            "No API key found for the fraudster. "
            "Set FRAUDSTER_API_KEY or the shared HF_TOKEN / API_KEY."
        )

    # Each agent gets its own OpenAI-compatible client so models, endpoints,
    # and keys can differ independently.
    defender_llm  = OpenAI(base_url=DEFENDER_API_BASE_URL,  api_key=DEFENDER_API_KEY)
    fraudster_llm = OpenAI(base_url=FRAUDSTER_API_BASE_URL, api_key=FRAUDSTER_API_KEY)
    env           = FraudEnvClient(base_url=ENV_URL)

    print(f"Defender  model : {DEFENDER_MODEL_NAME}  ({DEFENDER_API_BASE_URL})")
    print(f"Fraudster model : {FRAUDSTER_MODEL_NAME}  ({FRAUDSTER_API_BASE_URL})")

    families = [t for t in VALID_TASK_NAMES if t != "random"]

    async with env:
        # for task_name in families:
        #select a random task from families
        task_name = random.choice(families)
        log_start(
            task=task_name,
            defender_model=DEFENDER_MODEL_NAME,
            fraudster_model=FRAUDSTER_MODEL_NAME,
        )
        success, steps, score = False, 0, 0.0
        try:
            success, steps, score = await run_episode(
                env, defender_llm, fraudster_llm, task_name
            )
        except Exception as exc:
            print(f"  [FATAL] task={task_name} crashed: {exc}", flush=True)
        finally:
            log_end(task=task_name, success=success, steps=steps, score=score)
            print()

if __name__ == "__main__":
    asyncio.run(main())
