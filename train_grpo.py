"""
train_grpo.py — GRPO Training via HTTP Environment Server
==========================================================

Bridges the two previously disconnected systems:

  SYSTEM 1  server/app.py    FraudEnvironment exposed over HTTP (/reset, /step)
  SYSTEM 2  grpo_train.py    TRL GRPOTrainer logic (reward fns, parsers, prompts)

  THIS FILE connects them:

    TRL GRPOTrainer  (LoRA on Qwen2.5-1.5B-Instruct by default)
      └─ make_rollout_func_http()   ← custom rollout function
           └─ SyncFraudEnvClient    ← blocking HTTP POST /reset, /step
                └─ server/app.py    ← FraudEnvironment simulation

Key improvements over grpo_train.py:
  1. Server-backed environment — every rollout goes through the HTTP API so
     the same server used for inference/eval is also used during training.
  2. Correct turn-based protocol — fraudster ALWAYS acts first (half-step),
     then the defender (half-step with rewards).  grpo_train.py's
     _run_single_episode built a defender prompt from the post-reset obs
     (current_agent="fraudster") before the fraudster had moved; fixed here.
  3. LoRA adapters via ``peft`` — only the adapter weights are trained and
     saved, not the full model (~1 % of parameters for rank-16 LoRA).

Prerequisites:
    pip install trl>=0.12.0 transformers peft datasets requests

Quick-start:
    # Terminal 1 — start the environment server
    uv run server
    # (or)  python -m scam_detection.server.app

    # Terminal 2 — train the defender
    python train_grpo.py --agent defender

    # Terminal 2 — train the fraudster on a specific task
    python train_grpo.py --agent fraudster --task mule_cashout \\
        --model Qwen/Qwen2.5-1.5B-Instruct --env-url http://localhost:8000

    # Load the saved adapter after training
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    model = PeftModel.from_pretrained(model, "outputs/defender")
"""
from __future__ import annotations

import argparse
import os
import random
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

# ── Optional heavy deps ──────────────────────────────────────────────────────
try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None   # type: ignore
    F     = None   # type: ignore
    _TORCH_AVAILABLE = False

try:
    from trl import GRPOConfig, GRPOTrainer
    from trl.trainer.grpo_trainer import generate_rollout_completions
    _TRL_AVAILABLE = True
except ImportError:
    _TRL_AVAILABLE = False
    print("[WARNING] trl not installed.  Install: pip install trl>=0.12.0 transformers datasets")

try:
    from peft import LoraConfig, TaskType
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False
    print("[WARNING] peft not installed.  Install: pip install peft  (LoRA will be skipped)")

try:
    from datasets import Dataset as _HFDataset
    def _make_hf_dataset(rows: list):
        return _HFDataset.from_list(rows)
except ImportError:
    def _make_hf_dataset(rows: list):  # type: ignore[misc]
        return rows

# ── Environment types + reusable helpers from grpo_train.py ─────────────────
try:
    from scam_detection.baseline_detector import BaselineRuleDetector
    from scam_detection.constants import DEFAULT_MAX_STEPS, VALID_TASK_NAMES
    from scam_detection.models import (
        DefenderActionType, FraudAction, FraudObservation, FraudsterActionType,
    )
    from scam_detection.grpo_train import (
        BaselineFraudster,
        _DEFENDER_SYSTEM,
        _FRAUDSTER_SYSTEM,
        _build_defender_message,
        _build_fraudster_message,
        _parse_defender_action,
        _parse_fraudster_action,
        _get_reward_funcs,
        build_training_dataset,
    )
except ImportError:
    from baseline_detector import BaselineRuleDetector
    from constants import DEFAULT_MAX_STEPS, VALID_TASK_NAMES
    from models import (
        DefenderActionType, FraudAction, FraudObservation, FraudsterActionType,
    )
    from grpo_train import (
        BaselineFraudster,
        _DEFENDER_SYSTEM,
        _FRAUDSTER_SYSTEM,
        _build_defender_message,
        _build_fraudster_message,
        _parse_defender_action,
        _parse_fraudster_action,
        _get_reward_funcs,
        build_training_dataset,
    )

ENV_URL       = os.getenv("ENV_URL",    "http://localhost:8000")
DEFAULT_MODEL = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")


# ---------------------------------------------------------------------------
# Synchronous HTTP client for the FraudEnvironment server
# ---------------------------------------------------------------------------

class SyncFraudEnvClient:
    """
    Blocking HTTP wrapper around ``server/app.py``'s /reset and /step endpoints.

    Uses ``requests`` (not asyncio) so it can be called directly from TRL's
    rollout_func without event-loop conflicts.  Each worker creates its own
    instance; the underlying ``requests.Session`` handles connection pooling.

    Protocol (mirrors FraudEnvironment's internal logic):
      POST /reset  →  fraudster observation   (current_agent="fraudster")
      POST /step   →  defender observation    (current_agent="defender",  no reward)
      POST /step   →  next fraudster obs      (current_agent="fraudster", rewards set)
      … repeat until done=True
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self._session = requests.Session()

    # ---------------------------------------------------------------- reset --

    def reset(self, task_name: str = "random", seed: Optional[int] = None) -> FraudObservation:
        payload: Dict[str, Any] = {"task_name": task_name}
        if seed is not None:
            payload["seed"] = seed
        r = self._session.post(f"{self.base_url}/reset", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return _parse_obs(r.json())

    # ---------------------------------------------------------------- step ---

    def step(self, action: FraudAction) -> Tuple[FraudObservation, bool]:
        """
        Send one half-step action.  The server consumes only the field that
        matches its current phase (fraudster or defender); the other is ignored.
        """
        payload: Dict[str, Any] = {
            "defender_action":  action.defender_action.value,
            "fraudster_action": action.fraudster_action.value,
        }
        if action.defender_target  is not None:
            payload["defender_target"]  = action.defender_target
        if action.fraudster_target is not None:
            payload["fraudster_target"] = action.fraudster_target
        r = self._session.post(f"{self.base_url}/step", json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        obs  = _parse_obs(data)
        done = bool(data.get("done", obs.done or False))
        return obs, done

    # ---------------------------------------------------------------- utils --

    def ping(self) -> bool:
        """Return True if the server is reachable and healthy."""
        try:
            return self._session.get(f"{self.base_url}/schema", timeout=5).status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def close(self) -> None:
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _parse_obs(data: Dict[str, Any]) -> FraudObservation:
    """
    Deserialise a /reset or /step JSON response into a FraudObservation.

    /reset returns the observation at the top level.
    /step  wraps it under an "observation" key.
    """
    obs_raw = data.get("observation", data)
    current_agent = obs_raw.get("current_agent")
    # Heuristic fallback: infer current_agent from which partial obs is populated
    if current_agent is None and not obs_raw.get("episode_done"):
        if obs_raw.get("fraudster_obs") is not None and obs_raw.get("defender_obs") is None:
            current_agent = "fraudster"
        elif obs_raw.get("defender_obs") is not None and obs_raw.get("fraudster_obs") is None:
            current_agent = "defender"
    return FraudObservation(
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
        done=data.get("done", obs_raw.get("done", False)),
        reward=data.get("reward"),
    )


# ---------------------------------------------------------------------------
# Single-episode runner — correct turn-based protocol + HTTP server
# ---------------------------------------------------------------------------

def _run_single_episode_http(
    trainer,
    env: SyncFraudEnvClient,
    agent: str,
    system_prompt: str,
    task_name: str,
    seed: int,
    max_turns: int,
    defender_baseline: BaselineRuleDetector,
    fraudster_baseline: BaselineFraudster,
) -> Dict[str, Any]:
    """
    Play one complete episode via the HTTP server.

    The server enforces the alternating half-step protocol automatically:
      reset()             → current_agent="fraudster"  (fraudster obs)
      step(frd_action)    → current_agent="defender"   (defender obs, no reward)
      step(def_action)    → current_agent="fraudster"  (fraudster obs + rewards)
      …

    For DEFENDER training:
      half-step 1: baseline fraudster fills the fraudster slot.
      half-step 2: model fills the defender slot → reward = obs.defender_reward.

    For FRAUDSTER training:
      half-step 1: model fills the fraudster slot.
      half-step 2: baseline defender fills the defender slot → reward = obs.fraudster_reward.

    Returns a dict with per-step tensor lists and per-episode scalars for
    GRPO's reward functions.
    """
    obs = env.reset(task_name=task_name, seed=seed)
    done = False

    step_prompt_ids:     List = []
    step_completion_ids: List = []
    step_logprobs:       List = []

    ep_reward       = 0.0
    ep_format_valid = 0.0
    ep_action_legal = 0.0
    n_steps         = 0

    for turn in range(1, max_turns + 1):
        if done or obs.done:
            break

        # ── Defender training ─────────────────────────────────────────────
        if agent == "defender":
            # Half-step 1: baseline fraudster advances the world
            if obs.current_agent == "fraudster":
                frd_str, frd_tgt = fraudster_baseline.select_action(
                    obs.fraudster_obs,
                    obs.available_fraudster_actions,
                    obs.fraudster_action_targets,
                )
                obs, done = env.step(FraudAction(
                    fraudster_action=FraudsterActionType(frd_str),
                    fraudster_target=frd_tgt,
                ))
                if done or obs.done:
                    break

            # Half-step 2: model picks the defender action
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": _build_defender_message(turn, obs)},
            ]
            rollout = generate_rollout_completions(trainer, [messages])
            text    = _first_text(rollout)
            action_str, target, is_json, is_legal = _parse_defender_action(text, obs)
            obs, done = env.step(FraudAction(
                defender_action=DefenderActionType(action_str),
                defender_target=target,
            ))
            step_reward = obs.defender_reward or 0.0

        # ── Fraudster training ────────────────────────────────────────────
        else:
            # Half-step 1: model picks the fraudster action
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": _build_fraudster_message(turn, obs)},
            ]
            rollout = generate_rollout_completions(trainer, [messages])
            text    = _first_text(rollout)
            action_str, target, is_json, is_legal = _parse_fraudster_action(text, obs)
            obs, done = env.step(FraudAction(
                fraudster_action=FraudsterActionType(action_str),
                fraudster_target=target,
            ))
            if done or obs.done:
                # Episode ended during fraudster half-step (rare; counts this step)
                step_reward = obs.fraudster_reward or 0.0
                ep_reward       += step_reward
                ep_format_valid += float(is_json)
                ep_action_legal += float(is_legal)
                n_steps         += 1
                _accumulate(rollout, step_prompt_ids, step_completion_ids, step_logprobs)
                break

            # Half-step 2: baseline defender responds and triggers reward
            def_str, def_tgt = defender_baseline.select_action(
                obs.defender_obs,
                obs.available_defender_actions,
                obs.defender_action_targets,
            )
            obs, done = env.step(FraudAction(
                defender_action=DefenderActionType(def_str),
                defender_target=def_tgt,
            ))
            step_reward = obs.fraudster_reward or 0.0

        # ── Accumulate (both agent branches reach here unless they broke early)
        ep_reward       += step_reward
        ep_format_valid += float(is_json)
        ep_action_legal += float(is_legal)
        n_steps         += 1
        _accumulate(rollout, step_prompt_ids, step_completion_ids, step_logprobs)

    final_alert = 0.0
    if obs.fraudster_obs:
        final_alert = float(obs.fraudster_obs.get("alert_level", 0.0))

    return {
        "prompt_ids":     step_prompt_ids,
        "completion_ids": step_completion_ids,
        "logprobs":       step_logprobs,
        "episode_reward": ep_reward,
        "format_valid":   ep_format_valid / max(1, n_steps),
        "action_legal":   ep_action_legal / max(1, n_steps),
        "final_alert":    final_alert,
    }


def _first_text(rollout: Dict[str, Any]) -> str:
    """Extract the first completion string from generate_rollout_completions output."""
    t = rollout.get("text", "")
    return t[0] if isinstance(t, list) else str(t)


def _accumulate(rollout, prompt_ids_list, completion_ids_list, logprobs_list) -> None:
    """Append per-step tensors from a generate_rollout_completions result."""
    if rollout.get("prompt_ids")     is not None:
        prompt_ids_list.append(rollout["prompt_ids"])
    if rollout.get("completion_ids") is not None:
        completion_ids_list.append(rollout["completion_ids"])
    if rollout.get("logprobs")       is not None:
        logprobs_list.append(rollout["logprobs"])


# ---------------------------------------------------------------------------
# Rollout function factory — HTTP edition
# ---------------------------------------------------------------------------

def make_rollout_func_http(
    env_url: str,
    agent: str,
    system_prompt: str,
    max_turns: int = DEFAULT_MAX_STEPS,
):
    """
    Return a ``rollout_func`` for TRL's GRPOTrainer that drives the HTTP server.

    Each call plays ``num_generations`` independent episodes — each with a
    different random seed — so GRPO sees a group of trajectories with naturally
    varying rewards to compute group-relative advantages from.

    Parameters
    ----------
    env_url:
        URL of the running FraudEnvironment server (e.g. "http://localhost:8000").
    agent:
        "defender" or "fraudster".
    system_prompt:
        System-role message injected at every step of every episode.
    max_turns:
        Maximum full rounds per episode.
    """
    defender_baseline  = BaselineRuleDetector()
    fraudster_baseline = BaselineFraudster()

    def rollout_func(trainer, _env, tokenizer, prompt, _system_prompt, _max_turns):
        """
        Play num_generations episodes and return stacked trajectory tensors.

        Signature matches TRL's GRPOTrainer rollout_func contract.
        """
        num_gens  = trainer.args.num_generations
        task_name = prompt.get("task_name", "random")
        base_seed = prompt.get("seed", random.randint(0, 2 ** 20))

        # Each call gets a fresh session (thread-safe, connection-pooled)
        with SyncFraudEnvClient(base_url=env_url) as env:
            episodes = [
                _run_single_episode_http(
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
                for gen_idx in range(num_gens)
            ]

        return {
            "prompt_ids":     _stack_steps([ep["prompt_ids"]     for ep in episodes]),
            "completion_ids": _stack_steps([ep["completion_ids"] for ep in episodes]),
            "logprobs":       _stack_steps([ep["logprobs"]       for ep in episodes]),
            # Per-generation scalars consumed by the reward functions
            "episode_rewards": [ep["episode_reward"] for ep in episodes],
            "format_valids":   [ep["format_valid"]   for ep in episodes],
            "action_legals":   [ep["action_legal"]   for ep in episodes],
            "alert_levels":    [ep["final_alert"]    for ep in episodes],
        }

    return rollout_func


def _stack_steps(tensor_lists_per_gen: List[List]) -> Optional[Any]:
    """
    Concatenate per-step tensors within each generation, then stack across gens.

    Each generation produces a list of tensors (one per step the model acted).
    We cat those into a single sequence per generation, then stack all gens
    into one (num_gens, total_len) tensor that TRL expects.
    """
    if not _TORCH_AVAILABLE:
        return None
    rows = []
    for step_tensors in tensor_lists_per_gen:
        if step_tensors:
            rows.append(torch.cat(step_tensors, dim=-1))
        else:
            rows.append(torch.zeros(1, 1, dtype=torch.long))
    if not rows:
        return torch.zeros(len(tensor_lists_per_gen), 1, dtype=torch.long)
    max_len = max(r.shape[-1] for r in rows)
    padded  = [F.pad(r, (0, max_len - r.shape[-1])) for r in rows]
    return torch.cat(padded, dim=0)


# ---------------------------------------------------------------------------
# Training entry point — LoRA + GRPOTrainer + HTTP server
# ---------------------------------------------------------------------------

def train(
    agent:           str   = "defender",
    model_name:      str   = DEFAULT_MODEL,
    env_url:         str   = ENV_URL,
    task:            str   = "random",
    n_samples:       int   = 200,
    epochs:          int   = 1,
    lr:              float = 5e-6,
    num_generations: int   = 4,
    batch_size:      int   = 1,
    grad_accum:      int   = 16,
    max_prompt_len:  int   = 2048,
    max_comp_len:    int   = 128,
    lora_rank:       int   = 16,
    lora_alpha:      int   = 32,
    lora_dropout:    float = 0.05,
    use_vllm:        bool  = False,
    output_dir:      str   = "outputs",
) -> None:
    """
    Train a fraud detection LLM agent using GRPO, driven by the HTTP server.

    The LoRA adapter is saved to ``<output_dir>/<agent>/``.  Load it later::

        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(model_name)
        model = PeftModel.from_pretrained(base, f"{output_dir}/{agent}")
    """
    if not _TRL_AVAILABLE:
        raise ImportError(
            "trl is required.\n"
            "Install with: pip install trl>=0.12.0 transformers datasets"
        )
    if agent not in ("defender", "fraudster"):
        raise ValueError(f"agent must be 'defender' or 'fraudster', got '{agent}'")

    # ── Server health check ────────────────────────────────────────────────
    print(f"Checking environment server at {env_url} … ", end="", flush=True)
    with SyncFraudEnvClient(base_url=env_url) as probe:
        if not probe.ping():
            print("UNREACHABLE")
            raise RuntimeError(
                f"\nCannot reach the environment server at {env_url}.\n"
                "Start it first:\n"
                "  uv run server\n"
                "  # or:  python -m scam_detection.server.app"
            )
    print("OK")

    system_prompt = _DEFENDER_SYSTEM if agent == "defender" else _FRAUDSTER_SYSTEM
    reward_funcs  = _get_reward_funcs(agent)
    agent_out_dir = os.path.join(output_dir, agent)

    print(f"\n{'='*65}")
    print(f"GRPO Training (HTTP server mode)  —  agent={agent}")
    print(f"  model        : {model_name}")
    print(f"  server       : {env_url}")
    print(f"  task         : {task}  |  samples={n_samples}  |  epochs={epochs}")
    print(f"  LoRA         : r={lora_rank}  alpha={lora_alpha}  dropout={lora_dropout}")
    print(f"  num_gen      : {num_generations}  |  lr={lr}  |  grad_accum={grad_accum}")
    print(f"  output dir   : {agent_out_dir}")
    print(f"{'='*65}\n")

    # ── LoRA config ────────────────────────────────────────────────────────
    peft_config = None
    if _PEFT_AVAILABLE:
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            # Standard attention + MLP modules for Qwen2.5 / LLaMA family
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        print(f"LoRA  r={lora_rank}  alpha={lora_alpha}  "
              f"modules=q/k/v/o/gate/up/down_proj\n")
    else:
        print("[WARNING] peft not installed — full fine-tune (no LoRA).\n"
              "          Install: pip install peft\n")

    # ── Dataset — one row per (task_name, seed) pair ───────────────────────
    dataset = build_training_dataset(task=task, n_samples=n_samples)

    # ── Rollout function — HTTP edition ────────────────────────────────────
    rollout_func = make_rollout_func_http(
        env_url=env_url,
        agent=agent,
        system_prompt=system_prompt,
        max_turns=DEFAULT_MAX_STEPS,
    )

    # ── GRPOConfig ─────────────────────────────────────────────────────────
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
        use_vllm=use_vllm,
        vllm_mode="colocate" if use_vllm else None,
        vllm_gpu_memory_utilization=0.3 if use_vllm else None,
        logging_steps=5,
        save_steps=50,
        report_to="none",       # swap to "wandb" or "trackio" if desired
    )

    # ── GRPOTrainer ────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=reward_funcs,
        rollout_func=rollout_func,
        train_dataset=dataset,
        args=grpo_config,
        peft_config=peft_config,
    )

    print(f"Starting GRPO training for {agent} …")
    trainer.train()

    print(f"\nSaving LoRA adapter to {agent_out_dir}/")
    trainer.save_model(agent_out_dir)

    print("\nDone.")
    print(f"\nTo load the trained adapter later:")
    print(f"  from peft import PeftModel")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  base  = AutoModelForCausalLM.from_pretrained('{model_name}')")
    print(f"  model = PeftModel.from_pretrained(base, '{agent_out_dir}')")
    print()
    print("To use it in inference.py, set:")
    print(f"  DEFENDER_MODEL_NAME='{agent_out_dir}'  (if you trained the defender)")
    print(f"  DEFENDER_API_BASE_URL='http://localhost:<vllm-port>/v1'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    _shared_default = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    _env_url        = os.getenv("ENV_URL",    "http://localhost:8000")

    parser = argparse.ArgumentParser(
        description="GRPO training — drives server/app.py via HTTP, trains LoRA on LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Quick start
            -----------
            # Terminal 1 — start the environment server
            uv run server

            # Terminal 2 — train the defender (Qwen2.5-1.5B by default)
            python train_grpo.py --agent defender

            # Terminal 2 — train the fraudster with a specific task
            python train_grpo.py --agent fraudster --task mule_cashout

            # After training, run inference with the fine-tuned adapter:
            # Set DEFENDER_MODEL_NAME=outputs/defender in .env
        """),
    )
    parser.add_argument(
        "--agent", required=True, choices=["defender", "fraudster"],
        help="Which agent to train",
    )
    parser.add_argument(
        "--model", default=_shared_default, metavar="MODEL",
        help=f"HF model ID or local path (default: {_shared_default})",
    )
    parser.add_argument(
        "--env-url", default=_env_url,
        help=f"Environment server URL (default: {_env_url})",
    )
    parser.add_argument(
        "--task", default="random", choices=VALID_TASK_NAMES,
        help="Fraud family to train on (default: random — rotates all families)",
    )
    parser.add_argument("--samples",         type=int,   default=200,  help="Dataset size / training episodes (default: 200)")
    parser.add_argument("--epochs",          type=int,   default=1,    help="Training epochs (default: 1)")
    parser.add_argument("--lr",              type=float, default=5e-6, help="Learning rate (default: 5e-6)")
    parser.add_argument("--num-generations", type=int,   default=4,    help="GRPO group size — independent episodes per sample (default: 4)")
    parser.add_argument("--grad-accum",      type=int,   default=16,   help="Gradient accumulation steps (default: 16)")
    parser.add_argument("--max-prompt-len",  type=int,   default=2048, help="Max observation prompt tokens (default: 2048)")
    parser.add_argument("--max-comp-len",    type=int,   default=128,  help="Max action completion tokens (default: 128)")
    parser.add_argument("--lora-rank",       type=int,   default=16,   help="LoRA rank r (default: 16)")
    parser.add_argument("--lora-alpha",      type=int,   default=32,   help="LoRA alpha (default: 32)")
    parser.add_argument("--lora-dropout",    type=float, default=0.05, help="LoRA dropout (default: 0.05)")
    parser.add_argument("--use-vllm",        action="store_true",      help="Enable vLLM colocate mode (requires GPU + vllm)")
    parser.add_argument("--output-dir",      default="outputs",        help="Base directory for saved adapters (default: outputs/)")
    args = parser.parse_args()

    train(
        agent=args.agent,
        model_name=args.model,
        env_url=args.env_url,
        task=args.task,
        n_samples=args.samples,
        epochs=args.epochs,
        lr=args.lr,
        num_generations=args.num_generations,
        grad_accum=args.grad_accum,
        max_prompt_len=args.max_prompt_len,
        max_comp_len=args.max_comp_len,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_vllm=args.use_vllm,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
