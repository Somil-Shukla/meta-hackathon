"""
GRPO Training — Fraud Detection Environment
============================================

Trains the Defender and Fraudster LLMs using Group Relative Policy
Optimization (GRPO) via TRL's GRPOTrainer, following the OpenEnv pattern.

How GRPO works here
--------------------
The standard GRPO loop:

  1. GRPOTrainer calls rollout_func(trainer, env, tokenizer, prompt, ...).
  2. rollout_func plays ``num_generations`` complete independent episodes of
     FraudEnvironment — one per generation slot.  Different seeds give each
     generation a distinct starting world, so rewards naturally differ within
     the group.
  3. At every step inside each episode, generate_rollout_completions()
     produces one model completion for the current observation.
  4. The completion is parsed into a FraudAction.  A fixed rule-based
     opponent fills in the other agent's half of the action.
  5. Each generation's full trajectory (prompt_ids + completion_ids + logprobs)
     and its episode-level reward are returned.
  6. Reward functions receive the completions and per-generation reward data;
     they score each generation independently.
  7. GRPO uses relative reward within the group (not absolute values) to
     compute advantages and update the policy — no value model needed.

Opponent policies during training
----------------------------------
  Defender training  →  Fraudster uses ``BaselineFraudster`` (rule engine).
  Fraudster training →  Defender uses ``BaselineRuleDetector`` (rule engine).

Keeping the opponent fixed during training gives a stable signal.
After the defender is trained, swap its checkpoint in as the fraudster's
opponent for the next training pass.

Reward functions
-----------------
  Shared (both agents):
    reward_format_valid   — +1.0 if the model output is parseable JSON, else 0.
    reward_action_legal   — +1.0 if the parsed action is in the legal set, else 0.

  Defender-specific:
    reward_def_episode    — normalised cumulative defender reward for the episode.

  Fraudster-specific:
    reward_frd_episode    — normalised cumulative fraudster reward for the episode.
    reward_frd_evasion    — bonus proportional to (1 − final_alert_level); rewards
                            staying undetected until episode end.

Usage
-----
    # Train the defender with its own model
    python grpo_train.py --agent defender --defender-model Qwen/Qwen3-1.7B

    # Train the fraudster with a different (or same) model
    python grpo_train.py --agent fraudster --fraudster-model mistralai/Mistral-7B-Instruct-v0.3

    # Use --model as a shared fallback for both agents (convenience)
    python grpo_train.py --agent defender --model Qwen/Qwen3-1.7B

    # Full custom run — pass a trained checkpoint as the agent being trained
    python grpo_train.py \\
        --agent defender \\
        --defender-model outputs/defender_v0 \\
        --epochs 2 \\
        --lr 5e-6 \\
        --num-generations 4 \\
        --task random \\
        --output-dir outputs/defender_v1

Requirements
------------
    pip install trl>=0.12.0 transformers datasets torch
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

# torch is only needed inside the rollout function (tensor ops).
# Import lazily so reward functions, parsers, and the dry-run tool
# can be imported without a GPU / torch installation.
try:
    import torch as _torch
    import torch.nn.functional as _F
    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None
    _F     = None
    _TORCH_AVAILABLE = False

from dotenv import load_dotenv

try:
    from datasets import Dataset as _HFDataset
    def _make_hf_dataset(rows: list):
        return _HFDataset.from_list(rows)
except ImportError:
    _HFDataset = None
    def _make_hf_dataset(rows: list):
        return rows

load_dotenv()

# ---------------------------------------------------------------------------
# TRL imports — guard so the file can be imported without TRL installed
# ---------------------------------------------------------------------------
try:
    from trl import GRPOConfig, GRPOTrainer
    from trl.trainer.grpo_trainer import generate_rollout_completions
    _TRL_AVAILABLE = True
except ImportError:
    _TRL_AVAILABLE = False
    print("[WARNING] trl not installed. Install with: pip install trl>=0.12.0")

# ---------------------------------------------------------------------------
# Environment imports
# ---------------------------------------------------------------------------
try:
    from scam_detection.baseline_detector import BaselineRuleDetector
    from scam_detection.constants import DEFAULT_MAX_STEPS, VALID_TASK_NAMES
    from scam_detection.models import DefenderActionType, FraudAction, FraudsterActionType
    from scam_detection.server.fraud_environment import FraudEnvironment
except ImportError:
    from baseline_detector import BaselineRuleDetector
    from constants import DEFAULT_MAX_STEPS, VALID_TASK_NAMES
    from models import DefenderActionType, FraudAction, FraudsterActionType
    from server.fraud_environment import FraudEnvironment

# ---------------------------------------------------------------------------
# System prompts (mirrors inference.py prompts exactly)
# ---------------------------------------------------------------------------

_DEFENDER_SYSTEM: str = textwrap.dedent("""
    You are an expert fraud detection analyst (DEFENDER).

    Your goal is to prevent financial fraud with minimum false positives.
    Each step, you receive a partial observation of the financial system
    (accounts, merchants, alerts) and must choose ONE defensive action.

    AVAILABLE ACTIONS:
      monitor                  — place account under surveillance
      challenge                — send step-up authentication challenge
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

_FRAUDSTER_SYSTEM: str = textwrap.dedent("""
    You are an adaptive financial fraudster (FRAUDSTER).

    Your goal is to launder as much money as possible while avoiding detection.
    Each step you receive information about your available mule routes,
    detection pressure, and cashout readiness.

    AVAILABLE ACTIONS:
      split_payment    — make small split payments to avoid detection thresholds
      rotate_mule      — switch to a different mule account
      switch_merchant  — use a different merchant to process transactions
      rotate_device    — change device to reduce device-reuse risk
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
# Observation message builders (same logic as inference.py)
# ---------------------------------------------------------------------------
_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)

_VALID_DEFENDER_ACTIONS  = [a.value for a in DefenderActionType]
_VALID_FRAUDSTER_ACTIONS = [a.value for a in FraudsterActionType]


def _build_defender_message(step: int, obs) -> str:
    def_obs = obs.defender_obs or {}
    budget  = obs.step_budget or {}
    agg     = def_obs.get("aggregate", {})
    accounts = def_obs.get("accounts", [])
    alerts   = def_obs.get("alerts", [])

    parts = [
        f"STEP: {step}  |  STEPS REMAINING: {budget.get('remaining', '?')}",
        f"FRAUD FAMILY HINT: {def_obs.get('fraud_family_hint', 'unknown')}",
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
        parts.append("RECENT ALERTS:")
        for a in alerts[-3:]:
            parts.append(f"  {a}")

    legal   = obs.available_defender_actions or []
    targets = obs.defender_action_targets or {}
    parts.append("LEGAL ACTIONS:")
    for act in legal:
        t = targets.get(act, [])
        parts.append(f"  {act}: targets={t[:3]}")

    parts.append('\nOutput exactly one JSON: {"action": "...", "target": "..."}')
    return "\n".join(parts)


def _build_fraudster_message(step: int, obs) -> str:
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
    legal   = obs.available_fraudster_actions or []
    targets = obs.fraudster_action_targets or {}
    parts.append("LEGAL ACTIONS:")
    for act in legal:
        t = targets.get(act, [])
        parts.append(f"  {act}: targets={t[:3]}")

    parts.append('\nOutput exactly one JSON: {"action": "...", "target": "..."}')
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Action parsers
# ---------------------------------------------------------------------------

def _parse_json_action(text: str) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Extract {"action": "...", "target": "..."} from model output.

    Returns (action_str, target_str, is_valid_json).
    """
    match = _JSON_RE.search(text or "")
    if not match:
        return None, None, False
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, None, False
    return data.get("action"), data.get("target"), True


def _parse_defender_action(
    text: str,
    obs,
) -> Tuple[str, Optional[str], bool, bool]:
    """Parse defender output → (action_str, target, is_valid_json, is_legal)."""
    action_str, target_str, is_json = _parse_json_action(text)
    legal   = obs.available_defender_actions or []
    targets = obs.defender_action_targets or {}

    if not action_str or action_str not in _VALID_DEFENDER_ACTIONS:
        return DefenderActionType.DO_NOTHING.value, None, is_json, False

    is_legal = action_str in legal
    if not is_legal:
        action_str = DefenderActionType.DO_NOTHING.value
        target_str = None

    valid_targets = targets.get(action_str, [])
    if target_str not in valid_targets:
        target_str = valid_targets[0] if valid_targets and valid_targets != ["self"] else None

    return action_str, target_str, is_json, is_legal


def _parse_fraudster_action(
    text: str,
    obs,
) -> Tuple[str, Optional[str], bool, bool]:
    """Parse fraudster output → (action_str, target, is_valid_json, is_legal)."""
    action_str, target_str, is_json = _parse_json_action(text)
    legal   = obs.available_fraudster_actions or []
    targets = obs.fraudster_action_targets or {}

    if not action_str or action_str not in _VALID_FRAUDSTER_ACTIONS:
        return FraudsterActionType.DO_NOTHING.value, None, is_json, False

    is_legal = action_str in legal
    if not is_legal:
        action_str = FraudsterActionType.DO_NOTHING.value
        target_str = None

    valid_targets = targets.get(action_str, [])
    if target_str not in valid_targets:
        target_str = valid_targets[0] if valid_targets and valid_targets != ["self"] else None

    return action_str, target_str, is_json, is_legal


# ---------------------------------------------------------------------------
# Fixed opponent policies
# ---------------------------------------------------------------------------

class BaselineFraudster:
    """
    Rule-based fraudster used as the fixed opponent when training the defender.

    Priority: cashout_attempt → split_payment → do_nothing.
    """

    def select_action(
        self,
        frd_obs: Optional[Dict[str, Any]],
        legal: Optional[List[str]] = None,
        targets: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[str, Optional[str]]:
        if not frd_obs:
            return FraudsterActionType.DO_NOTHING.value, None

        legal   = legal or []
        targets = targets or {}

        if FraudsterActionType.CASHOUT_ATTEMPT.value in legal:
            opts = targets.get(FraudsterActionType.CASHOUT_ATTEMPT.value, [])
            if opts:
                return FraudsterActionType.CASHOUT_ATTEMPT.value, opts[0]

        if FraudsterActionType.SPLIT_PAYMENT.value in legal:
            opts = targets.get(FraudsterActionType.SPLIT_PAYMENT.value, [])
            if opts:
                return FraudsterActionType.SPLIT_PAYMENT.value, opts[0]

        return FraudsterActionType.DO_NOTHING.value, None


# ---------------------------------------------------------------------------
# Single episode runner (one generation)
# ---------------------------------------------------------------------------

def _run_single_episode(
    trainer,
    env: FraudEnvironment,
    agent: str,
    system_prompt: str,
    task_name: str,
    seed: int,
    max_turns: int,
    defender_baseline: BaselineRuleDetector,
    fraudster_baseline: BaselineFraudster,
) -> Dict[str, Any]:
    """
    Play one complete episode and collect the generation trajectory.

    Returns a dict with prompt_ids, completion_ids, logprobs (as lists of
    tensors from each step), plus per-episode reward metadata.
    """
    obs = env.reset(task_name=task_name, seed=seed)

    step_prompt_ids:     List = []
    step_completion_ids: List = []
    step_logprobs:       List = []

    ep_reward       = 0.0
    ep_format_valid = 0.0
    ep_action_legal = 0.0
    n_steps         = 0

    for step in range(1, max_turns + 1):
        if obs.done:
            break

        # ── Build the prompt for the model-being-trained ─────────────────────
        if agent == "defender":
            user_msg = _build_defender_message(step, obs)
        else:
            user_msg = _build_fraudster_message(step, obs)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg},
        ]

        # ── Generate ONE completion with the current policy ───────────────────
        # generate_rollout_completions generates num_generations completions;
        # for the sequential episode we use the first (greedy) one to advance
        # the environment.  All completions are returned to TRL for scoring.
        rollout = generate_rollout_completions(trainer, [messages])

        # Use the first completion to step the environment
        text = (
            rollout["text"][0]
            if isinstance(rollout.get("text"), list)
            else rollout.get("text", "")
        )

        # ── Parse the model's action ──────────────────────────────────────────
        if agent == "defender":
            action_str, target, is_json, is_legal = _parse_defender_action(text, obs)
        else:
            action_str, target, is_json, is_legal = _parse_fraudster_action(text, obs)

        ep_format_valid += float(is_json)
        ep_action_legal += float(is_legal)

        # ── Build combined FraudAction with the fixed opponent ────────────────
        if agent == "defender":
            frd_str, frd_target = fraudster_baseline.select_action(
                obs.fraudster_obs,
                obs.available_fraudster_actions,
                obs.fraudster_action_targets,
            )
            action = FraudAction(
                defender_action=DefenderActionType(action_str),
                defender_target=target,
                fraudster_action=FraudsterActionType(frd_str),
                fraudster_target=frd_target,
            )
        else:
            def_str, def_target = defender_baseline.select_action(
                obs.defender_obs,
                obs.available_defender_actions,
                obs.defender_action_targets,
            )
            action = FraudAction(
                defender_action=DefenderActionType(def_str),
                defender_target=def_target,
                fraudster_action=FraudsterActionType(action_str),
                fraudster_target=target,
            )

        # ── Step the environment ──────────────────────────────────────────────
        obs = env.step(action)

        step_reward = (
            obs.defender_reward if agent == "defender" else obs.fraudster_reward
        ) or 0.0
        ep_reward += step_reward

        # ── Accumulate trajectory tensors ─────────────────────────────────────
        # rollout tensors shape: (num_gen, seq_len) — we keep all gens
        if "prompt_ids" in rollout and rollout["prompt_ids"] is not None:
            step_prompt_ids.append(rollout["prompt_ids"])
        if "completion_ids" in rollout and rollout["completion_ids"] is not None:
            step_completion_ids.append(rollout["completion_ids"])
        if "logprobs" in rollout and rollout["logprobs"] is not None:
            step_logprobs.append(rollout["logprobs"])

        n_steps += 1

    # Final state info (used for evasion reward)
    final_alert = 0.0
    if obs.fraudster_obs:
        final_alert = float(obs.fraudster_obs.get("alert_level", 0.0))

    return {
        "prompt_ids":     step_prompt_ids,       # list of tensors per step
        "completion_ids": step_completion_ids,
        "logprobs":       step_logprobs,
        "episode_reward": ep_reward,
        "format_valid":   ep_format_valid / max(1, n_steps),  # 0-1 fraction
        "action_legal":   ep_action_legal / max(1, n_steps),  # 0-1 fraction
        "final_alert":    final_alert,
    }


# ---------------------------------------------------------------------------
# Rollout function factory
# ---------------------------------------------------------------------------

def make_rollout_func(
    env: FraudEnvironment,
    agent: str,
    system_prompt: str,
    max_turns: int = DEFAULT_MAX_STEPS,
):
    """
    Returns a rollout_func compatible with TRL's GRPOTrainer.

    The returned function plays ``num_generations`` independent episodes —
    each with a different random seed — giving GRPO a group of trajectories
    with naturally varying rewards to compute relative advantages from.

    Parameters
    ----------
    env:
        A ``FraudEnvironment`` instance (used directly, no HTTP server).
    agent:
        ``"defender"`` or ``"fraudster"``.
    system_prompt:
        System-role message fed to the model at every step.
    max_turns:
        Maximum episode length.
    """
    defender_baseline = BaselineRuleDetector()
    fraudster_baseline = BaselineFraudster()

    def rollout_func(trainer, _env, tokenizer, prompt, _system_prompt, _max_turns):
        """
        Play num_generations episodes and return concatenated trajectories.

        Parameters (injected by GRPOTrainer):
            trainer       : GRPOTrainer instance (model + config).
            _env          : environment passed by GRPOTrainer (we use our own).
            tokenizer     : HF tokenizer (available if needed).
            prompt        : one dataset row dict (contains task_name, seed).
            _system_prompt: system prompt from GRPOTrainer config (we override).
            _max_turns    : max turns from GRPOTrainer config (we override).

        Returns a dict with the required keys for GRPOTrainer:
            prompt_ids     : (num_gen, total_prompt_len)
            completion_ids : (num_gen, total_completion_len)
            logprobs       : (num_gen, total_completion_len)
            + extra reward-metadata keys (one value per generation)
        """
        num_gens  = trainer.args.num_generations
        task_name = prompt.get("task_name", "random")
        base_seed = prompt.get("seed", random.randint(0, 2**20))

        # ── Run num_generations independent episodes ──────────────────────────
        episodes = []
        for gen_idx in range(num_gens):
            ep = _run_single_episode(
                trainer=trainer,
                env=env,
                agent=agent,
                system_prompt=system_prompt,
                task_name=task_name,
                seed=base_seed + gen_idx,
                max_turns=max_turns,
                defender_baseline=defender_baseline,
                fraudster_baseline=fraudster_baseline,
            )
            episodes.append(ep)

        # ── Stack tensors from all generations ────────────────────────────────
        # Each episode has a list of per-step tensors; concatenate steps then
        # stack generations.  Handle empty trajectories gracefully.
        def _cat_steps(tensor_lists):
            """Concatenate step tensors for each gen, then stack across gens."""
            if not _TORCH_AVAILABLE:
                return None
            rows = []
            for step_tensors in tensor_lists:
                if step_tensors:
                    rows.append(_torch.cat(step_tensors, dim=-1))
                else:
                    rows.append(_torch.zeros(1, 1, dtype=_torch.long))
            if not rows:
                return _torch.zeros(num_gens, 1, dtype=_torch.long)
            max_len = max(r.shape[-1] for r in rows)
            padded  = [
                _F.pad(r, (0, max_len - r.shape[-1]))
                for r in rows
            ]
            return _torch.cat(padded, dim=0)   # (num_gens, max_len)

        prompt_ids_tensor     = _cat_steps([ep["prompt_ids"]     for ep in episodes])
        completion_ids_tensor = _cat_steps([ep["completion_ids"] for ep in episodes])
        logprobs_tensor       = _cat_steps([ep["logprobs"]       for ep in episodes])

        return {
            "prompt_ids":     prompt_ids_tensor,
            "completion_ids": completion_ids_tensor,
            "logprobs":       logprobs_tensor,
            # ── Per-generation reward metadata (one float per gen) ────────────
            "episode_rewards": [ep["episode_reward"] for ep in episodes],
            "format_valids":   [ep["format_valid"]   for ep in episodes],
            "action_legals":   [ep["action_legal"]   for ep in episodes],
            "alert_levels":    [ep["final_alert"]    for ep in episodes],
        }

    return rollout_func


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------
# Each reward function receives:
#   completions : List[str]  — model-generated text for each generation
#   **kwargs    : data returned by rollout_func (lists indexed by generation)
#
# Each must return List[float] of length len(completions).
# ---------------------------------------------------------------------------

def reward_format_valid(completions: List[str], format_valids: List[float], **kwargs) -> List[float]:
    """
    +1.0 if the model's average output across the episode was valid JSON.
    Rewards consistent, structured output even without correct actions.
    """
    return [float(v) for v in format_valids]


def reward_action_legal(completions: List[str], action_legals: List[float], **kwargs) -> List[float]:
    """
    +1.0 if the average fraction of steps where the model chose a legal action
    was 1.0 (all steps legal).  Partial credit for partially-legal trajectories.
    """
    return [float(v) for v in action_legals]


def reward_def_episode(completions: List[str], episode_rewards: List[float], **kwargs) -> List[float]:
    """
    Normalised cumulative defender episode reward in [-1, 1].

    The normalisation is relative to the group max absolute reward so that
    GRPO's group-relative advantage computation has consistent scale.
    """
    if not episode_rewards:
        return [0.0] * len(completions)
    max_abs = max(abs(r) for r in episode_rewards) or 1.0
    return [r / max_abs for r in episode_rewards]


def reward_frd_episode(completions: List[str], episode_rewards: List[float], **kwargs) -> List[float]:
    """
    Normalised cumulative fraudster episode reward in [-1, 1].
    """
    if not episode_rewards:
        return [0.0] * len(completions)
    max_abs = max(abs(r) for r in episode_rewards) or 1.0
    return [r / max_abs for r in episode_rewards]


def reward_frd_evasion(completions: List[str], alert_levels: List[float], **kwargs) -> List[float]:
    """
    Evasion bonus: +1.0 when the fraudster ended the episode completely
    undetected (alert_level = 0), linearly decreasing to 0 at alert_level = 1.

    Encourages the fraudster to evade even when cashout was unsuccessful.
    """
    return [max(0.0, 1.0 - float(a)) for a in alert_levels]


# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------

def build_training_dataset(task: str = "random", n_samples: int = 200) -> Dataset:
    """
    Build a simple Hugging Face Dataset for GRPO training.

    Each row is one "prompt" — a fraud family + seed that the rollout function
    uses to initialise a fresh episode.  Varying seeds ensures the model sees
    a wide distribution of worlds across training steps.
    """
    families = [t for t in VALID_TASK_NAMES if t != "random"]
    rows = []
    for i in range(n_samples):
        family = task if task != "random" else families[i % len(families)]
        rows.append({
            "task_name": family,
            "seed":      i * 137 + 42,
            "prompt":    f"Fraud episode — family: {family}",
        })
    return _make_hf_dataset(rows)


# ---------------------------------------------------------------------------
# Main training functions
# ---------------------------------------------------------------------------

def train(
    agent:            str   = "defender",
    defender_model:   str   = "Qwen/Qwen3-1.7B",
    fraudster_model:  str   = "Qwen/Qwen3-1.7B",
    task:             str   = "random",
    n_samples:        int   = 200,
    epochs:           int   = 1,
    lr:               float = 5e-6,
    num_generations:  int   = 4,
    batch_size:       int   = 1,
    grad_accum:       int   = 16,
    max_prompt_len:   int   = 2048,
    max_comp_len:     int   = 128,
    use_vllm:         bool  = False,
    output_dir:       str   = "outputs",
) -> None:
    """
    Run GRPO training for the specified agent.

    Parameters
    ----------
    agent:
        ``"defender"`` or ``"fraudster"``.
    defender_model:
        HuggingFace model ID (or local path) for the **defender** LLM.
        Used when ``agent="defender"``.
    fraudster_model:
        HuggingFace model ID (or local path) for the **fraudster** LLM.
        Used when ``agent="fraudster"``.
    task:
        Fraud family to train on, or ``"random"``.
    n_samples:
        Number of training episodes in the dataset.
    epochs:
        Number of passes over the dataset.
    lr:
        Learning rate.
    num_generations:
        Number of independent episode rollouts per training example (the
        GRPO group size).  Larger = more stable gradient but slower.
    batch_size:
        Per-device batch size (keep at 1 for long sequences).
    grad_accum:
        Gradient accumulation steps (effective batch = batch_size × grad_accum).
    max_prompt_len:
        Maximum token length of the observation prompt.
    max_comp_len:
        Maximum token length of the model's action completion.
    use_vllm:
        Enable vLLM colocate mode for faster generation (requires GPU + vllm).
    output_dir:
        Base directory for saved checkpoints.  Final path is
        ``<output_dir>/defender`` or ``<output_dir>/fraudster``.
    """
    if not _TRL_AVAILABLE:
        raise ImportError(
            "trl is required for GRPO training.\n"
            "Install with: pip install trl>=0.12.0 transformers datasets"
        )

    assert agent in ("defender", "fraudster"), \
        f"agent must be 'defender' or 'fraudster', got '{agent}'"

    # Select the model for the agent being trained
    model_name    = defender_model if agent == "defender" else fraudster_model
    system_prompt = _DEFENDER_SYSTEM if agent == "defender" else _FRAUDSTER_SYSTEM
    reward_funcs  = _get_reward_funcs(agent)
    agent_out_dir = os.path.join(output_dir, agent)

    print(f"\n{'='*60}")
    print(f"GRPO Training — agent={agent}")
    print(f"  defender_model  : {defender_model}")
    print(f"  fraudster_model : {fraudster_model}")
    print(f"  training model  : {model_name}")
    print(f"Task: {task}  |  Samples: {n_samples}  |  Epochs: {epochs}")
    print(f"num_generations={num_generations}  lr={lr}  grad_accum={grad_accum}")
    print(f"Output dir: {agent_out_dir}")
    print(f"{'='*60}\n")

    # ── Environment (single instance, reused across all rollouts) ─────────────
    env = FraudEnvironment()

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = build_training_dataset(task=task, n_samples=n_samples)

    # ── Rollout function ──────────────────────────────────────────────────────
    rollout_func = make_rollout_func(
        env=env,
        agent=agent,
        system_prompt=system_prompt,
        max_turns=DEFAULT_MAX_STEPS,
    )

    # ── GRPOConfig ────────────────────────────────────────────────────────────
    # Mirrors the Wordle example from the course; adapted for our environment.
    grpo_config = GRPOConfig(
        output_dir=agent_out_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_generations=num_generations,
        max_prompt_length=max_prompt_len,
        max_completion_length=max_comp_len,
        gradient_checkpointing=True,
        # vLLM colocate mode — shares GPU for generation + training (A100+)
        use_vllm=use_vllm,
        vllm_mode="colocate" if use_vllm else None,
        vllm_gpu_memory_utilization=0.3 if use_vllm else None,
        # Logging
        logging_steps=5,
        save_steps=50,
        report_to="none",       # swap to "wandb" or "trackio" if needed
    )

    # ── GRPOTrainer ───────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=reward_funcs,
        rollout_func=rollout_func,
        train_dataset=dataset,
        args=grpo_config,
    )

    print(f"Starting GRPO training for {agent}...")
    trainer.train()

    print(f"\nTraining complete. Saving to {agent_out_dir}/")
    trainer.save_model(agent_out_dir)
    print("Done.")


def _get_reward_funcs(agent: str) -> list:
    """Return the list of reward functions for the given agent."""
    if agent == "defender":
        return [
            reward_format_valid,   # structured output quality
            reward_action_legal,   # chose a legal action
            reward_def_episode,    # cumulative defender reward (fraud prevented - FP)
        ]
    else:
        return [
            reward_format_valid,   # structured output quality
            reward_action_legal,   # chose a legal action
            reward_frd_episode,    # cumulative fraudster reward (cashout success)
            reward_frd_evasion,    # stayed undetected (low final alert_level)
        ]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Env-var defaults for per-agent models (mirror inference.py conventions)
    _shared_default = os.getenv("MODEL_NAME", "Qwen/Qwen3-1.7B")
    _def_default    = os.getenv("DEFENDER_MODEL_NAME")  or _shared_default
    _frd_default    = os.getenv("FRAUDSTER_MODEL_NAME") or _shared_default

    parser = argparse.ArgumentParser(
        description="GRPO training for fraud detection agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Model resolution order
            ----------------------
            1. --defender-model / --fraudster-model  (explicit CLI flag)
            2. DEFENDER_MODEL_NAME / FRAUDSTER_MODEL_NAME  (env var)
            3. --model  (shared fallback CLI flag)
            4. MODEL_NAME env var
            5. Built-in default (Qwen/Qwen3-1.7B)
        """),
    )
    parser.add_argument(
        "--agent", type=str, required=True, choices=["defender", "fraudster"],
        help="Which agent to train"
    )

    # Model arguments — per-agent takes precedence over the shared --model fallback
    model_group = parser.add_argument_group("model selection")
    model_group.add_argument(
        "--defender-model", type=str, default=_def_default,
        metavar="MODEL",
        help=f"HF model ID or local path for the DEFENDER (default: {_def_default})"
    )
    model_group.add_argument(
        "--fraudster-model", type=str, default=_frd_default,
        metavar="MODEL",
        help=f"HF model ID or local path for the FRAUDSTER (default: {_frd_default})"
    )
    model_group.add_argument(
        "--model", type=str, default=None,
        metavar="MODEL",
        help="Shared fallback model — sets both --defender-model and --fraudster-model "
             "when their per-agent values are not explicitly provided"
    )

    parser.add_argument(
        "--task", type=str, default="random",
        choices=VALID_TASK_NAMES,
        help="Fraud family to train on (default: random — rotates all families)"
    )
    parser.add_argument(
        "--samples", type=int, default=200,
        help="Number of training episodes / dataset size (default: 200)"
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Number of training epochs (default: 1)"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-6,
        help="Learning rate (default: 5e-6)"
    )
    parser.add_argument(
        "--num-generations", type=int, default=4,
        help="GRPO group size — independent episodes per training example (default: 4)"
    )
    parser.add_argument(
        "--grad-accum", type=int, default=16,
        help="Gradient accumulation steps (default: 16)"
    )
    parser.add_argument(
        "--max-prompt-len", type=int, default=2048,
        help="Max observation prompt tokens (default: 2048)"
    )
    parser.add_argument(
        "--max-comp-len", type=int, default=128,
        help="Max action completion tokens (default: 128)"
    )
    parser.add_argument(
        "--use-vllm", action="store_true",
        help="Enable vLLM colocate mode (requires GPU + vllm package)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Base directory for saved checkpoints (default: outputs/)"
    )
    args = parser.parse_args()

    # Apply shared --model fallback: overrides per-agent defaults only when
    # --model is explicitly passed and the per-agent flag was not changed from
    # its env-var-derived default.
    defender_model  = args.model if args.model else args.defender_model
    fraudster_model = args.model if args.model else args.fraudster_model

    train(
        agent=args.agent,
        defender_model=defender_model,
        fraudster_model=fraudster_model,
        task=args.task,
        n_samples=args.samples,
        epochs=args.epochs,
        lr=args.lr,
        num_generations=args.num_generations,
        grad_accum=args.grad_accum,
        max_prompt_len=args.max_prompt_len,
        max_comp_len=args.max_comp_len,
        use_vllm=args.use_vllm,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
