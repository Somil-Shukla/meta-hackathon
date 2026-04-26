"""
train_grpo.py — GRPO Training via HTTP Environment Server  (TRL 1.2.0+)
========================================================================

Connects the two previously disconnected systems:

  SYSTEM 1  server/app.py   FraudEnvironment over HTTP (/reset, /step)
  SYSTEM 2  grpo_train.py   reward functions, parsers, system prompts

  THIS FILE bridges them:

    TRL GRPOTrainer  (LoRA on Qwen2.5-1.5B-Instruct by default)
      └─ rollout_func(prompts, trainer)     ← TRL 1.2.0 signature
           └─ SyncFraudEnvClient            ← blocking HTTP POST /reset, /step
                └─ server/app.py            ← FraudEnvironment simulation

Key design decisions
--------------------
* rollout_func uses the TRL 1.2.0 signature ``(prompts, trainer)`` — not the
  old 6-argument form and does NOT use the removed
  ``generate_rollout_completions`` private API.

* model.generate() is called directly for each model action, with
  output_scores=True so per-token logprobs can be computed inline.

* For multi-step episodes the "completion" returned to TRL is the
  concatenation of ALL action token sequences from every step of the episode.
  The "prompt" is fixed to the first step's observation message.
  This gives TRL's GRPO objective a signal from all actions, not just the
  first one, while avoiding the complexity of interleaved obs/action masking.

* LoRA adapters are applied via ``peft`` — only ~1 % of parameters are
  trained and saved.  Full fine-tune falls back automatically if peft is
  absent.

Prerequisites:
    pip install trl>=1.2.0 transformers peft accelerate datasets requests

Quick-start:
    # Terminal 1 — start the environment server
    uv run server                             # or: python -m scam_detection.server.app

    # Terminal 2 — train the defender (Qwen2.5-1.5B-Instruct default)
    python train_grpo.py --agent defender

    # Terminal 2 — train fraudster on a specific fraud family
    python train_grpo.py --agent fraudster --task mule_cashout

    # Load the saved LoRA adapter after training
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    base  = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    model = PeftModel.from_pretrained(base, "outputs/defender")
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
    _TRL_AVAILABLE = True
except ImportError:
    _TRL_AVAILABLE = False
    print("[WARNING] trl not installed.  Install: pip install trl>=1.2.0 transformers datasets")

try:
    from peft import LoraConfig, TaskType
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False
    print("[WARNING] peft not installed.  LoRA will be skipped.  Install: pip install peft")

try:
    from datasets import Dataset as _HFDataset
    def _make_hf_dataset(rows: list):
        return _HFDataset.from_list(rows)
except ImportError:
    def _make_hf_dataset(rows: list):  # type: ignore[misc]
        return rows

# ── Environment types + reusable helpers ─────────────────────────────────────
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
# Synchronous HTTP client
# ---------------------------------------------------------------------------

class SyncFraudEnvClient:
    """
    Blocking HTTP wrapper around server/app.py's /reset and /step endpoints.

    Uses ``requests`` (not asyncio) so it can be called directly from TRL's
    synchronous rollout_func without event-loop conflicts.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self._session = requests.Session()

    def reset(self, task_name: str = "random", seed: Optional[int] = None) -> FraudObservation:
        payload: Dict[str, Any] = {"task_name": task_name}
        if seed is not None:
            payload["seed"] = seed
        r = self._session.post(f"{self.base_url}/reset", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return _parse_obs(r.json())

    def step(self, action: FraudAction) -> Tuple[FraudObservation, bool]:
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

    def ping(self) -> bool:
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
    obs_raw = data.get("observation", data)
    current_agent = obs_raw.get("current_agent")
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
# Model generation helper  (replaces removed generate_rollout_completions)
# ---------------------------------------------------------------------------

def _generate_action(model, tokenizer, messages: List[Dict], max_new_tokens: int = 128) -> Dict:
    """
    Run ``model.generate()`` for one observation → action step.

    Returns a dict with:
      prompt_ids     List[int]   tokenized prompt (system + observation)
      completion_ids List[int]   generated action tokens
      logprobs       List[float] per-token sampling log-probabilities
      text           str         decoded action text
    """
    device = next(model.parameters()).device

    # Apply chat template to format the conversation
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,     # Qwen3 / thinking models
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(
        prompt_text, return_tensors="pt", truncation=True, max_length=2048
    ).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    # Temporarily switch to eval mode for generation
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id or 0,
                return_dict_in_generate=True,
                output_scores=True,
            )
    finally:
        if was_training:
            model.train()

    full_ids        = output.sequences[0].tolist()
    prompt_ids      = full_ids[:prompt_len]
    completion_ids  = full_ids[prompt_len:]

    # Per-token log-probabilities from the generation scores
    logprobs: List[float] = []
    for step_idx, step_scores in enumerate(output.scores or []):
        if step_idx < len(completion_ids):
            lp = torch.log_softmax(step_scores[0], dim=-1)
            logprobs.append(lp[completion_ids[step_idx]].item())
    # Safety pad
    if len(logprobs) < len(completion_ids):
        logprobs += [0.0] * (len(completion_ids) - len(logprobs))

    text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return {
        "prompt_ids":     prompt_ids,
        "completion_ids": completion_ids,
        "logprobs":       logprobs,
        "text":           text,
    }


# ---------------------------------------------------------------------------
# Episode runner — correct turn-based protocol + HTTP server
# ---------------------------------------------------------------------------

def _run_episode_http(
    env:                SyncFraudEnvClient,
    model,
    tokenizer,
    agent:              str,
    system_prompt:      str,
    task_name:          str,
    seed:               int,
    max_turns:          int,
    max_new_tokens:     int,
    defender_baseline:  BaselineRuleDetector,
    fraudster_baseline: BaselineFraudster,
) -> Dict:
    """
    Play one complete episode via HTTP, generating model actions at each
    step the agent-under-training must act.

    Turn-based protocol (enforced server-side):
      reset()           → current_agent="fraudster"  (obs, no reward)
      step(frd_action)  → current_agent="defender"   (obs, no reward)
      step(def_action)  → current_agent="fraudster"  (obs + rewards)
      … repeat until done=True

    Multi-step trajectory encoding
    --------------------------------
    The "prompt" returned to TRL is the FIRST step's observation message.
    The "completion" is the concatenation of ALL action token sequences
    from every step where the model acted.
    The "logprobs" are the corresponding per-token sampling log-probs.

    TRL's GRPO objective thus sees the whole episode as one long action
    sequence and updates all action tokens proportionally to the total
    episode reward.
    """
    obs = env.reset(task_name=task_name, seed=seed)
    done = False

    first_prompt_ids: Optional[List[int]] = None
    all_completion_ids: List[int]  = []
    all_logprobs:       List[float] = []

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

            # Half-step 2: model generates the defender action
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": _build_defender_message(turn, obs)},
            ]
            gen = _generate_action(model, tokenizer, messages, max_new_tokens)
            if first_prompt_ids is None:
                first_prompt_ids = gen["prompt_ids"]
            action_str, target, is_json, is_legal = _parse_defender_action(gen["text"], obs)
            obs, done = env.step(FraudAction(
                defender_action=DefenderActionType(action_str),
                defender_target=target,
            ))
            step_reward = obs.defender_reward or 0.0

        # ── Fraudster training ────────────────────────────────────────────
        else:
            # Half-step 1: model generates the fraudster action
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": _build_fraudster_message(turn, obs)},
            ]
            gen = _generate_action(model, tokenizer, messages, max_new_tokens)
            if first_prompt_ids is None:
                first_prompt_ids = gen["prompt_ids"]
            action_str, target, is_json, is_legal = _parse_fraudster_action(gen["text"], obs)
            obs, done = env.step(FraudAction(
                fraudster_action=FraudsterActionType(action_str),
                fraudster_target=target,
            ))
            if done or obs.done:
                step_reward = obs.fraudster_reward or 0.0
                all_completion_ids.extend(gen["completion_ids"])
                all_logprobs.extend(gen["logprobs"])
                ep_reward       += step_reward
                ep_format_valid += float(is_json)
                ep_action_legal += float(is_legal)
                n_steps         += 1
                break

            # Half-step 2: baseline defender responds + triggers rewards
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

        # Accumulate (both branches reach here unless they broke early)
        all_completion_ids.extend(gen["completion_ids"])
        all_logprobs.extend(gen["logprobs"])
        ep_reward       += step_reward
        ep_format_valid += float(is_json)
        ep_action_legal += float(is_legal)
        n_steps         += 1

    # Edge-case: episode ended before the model ever acted
    if not all_completion_ids:
        dummy = '{"action": "do_nothing", "target": null}'
        dummy_ids = tokenizer.encode(dummy, add_special_tokens=False)
        first_prompt_ids   = first_prompt_ids or []
        all_completion_ids = dummy_ids
        all_logprobs       = [0.0] * len(dummy_ids)

    final_alert = 0.0
    if obs.fraudster_obs:
        final_alert = float(obs.fraudster_obs.get("alert_level", 0.0))

    return {
        "prompt_ids":     first_prompt_ids or [],
        "completion_ids": all_completion_ids,
        "logprobs":       all_logprobs,
        "episode_reward": ep_reward,
        "format_valid":   ep_format_valid / max(1, n_steps),
        "action_legal":   ep_action_legal / max(1, n_steps),
        "final_alert":    final_alert,
    }


# ---------------------------------------------------------------------------
# Rollout function factory — TRL 1.2.0 API: (prompts, trainer)
# ---------------------------------------------------------------------------

def make_rollout_func_http(
    env_url:       str,
    agent:         str,
    system_prompt: str,
    max_turns:     int = DEFAULT_MAX_STEPS,
    max_new_tokens: int = 128,
):
    """
    Return a ``rollout_func`` compatible with TRL 1.2.0's GRPOTrainer.

    The returned function signature is ``(prompts, trainer)`` as required by
    TRL 1.2.0.  It plays ``num_generations`` independent episodes per prompt
    against the HTTP server, each with a different seed so GRPO sees varied
    rewards within the group to compute relative advantages from.

    Parameters
    ----------
    env_url:
        URL of the running FraudEnvironment server.
    agent:
        "defender" or "fraudster".
    system_prompt:
        System-role message injected at every episode step.
    max_turns:
        Maximum full rounds per episode.
    max_new_tokens:
        Maximum new tokens the model generates per action step.
    """
    defender_baseline  = BaselineRuleDetector()
    fraudster_baseline = BaselineFraudster()

    def rollout_func(prompts: List[Dict], trainer) -> Dict:
        """
        TRL 1.2.0 rollout_func signature: (prompts, trainer).

        ``prompts`` is the per-process slice of the training batch (may contain
        multiple rows).  For each row we play ``num_generations`` independent
        episodes and return the concatenated results.

        Required return keys: "prompt_ids", "completion_ids", "logprobs"
        Extra keys are forwarded to the reward functions.
        """
        num_gens   = trainer.args.num_generations
        model      = trainer.model
        tokenizer  = trainer.processing_class

        all_prompt_ids:     List[List[int]]   = []
        all_completion_ids: List[List[int]]   = []
        all_logprobs:       List[List[float]] = []
        all_ep_rewards:     List[float]       = []
        all_format_valids:  List[float]       = []
        all_action_legals:  List[float]       = []
        all_alert_levels:   List[float]       = []

        with SyncFraudEnvClient(base_url=env_url) as env:
            for prompt in prompts:
                task_name = prompt.get("task_name", "random")
                base_seed = prompt.get("seed", random.randint(0, 2 ** 20))

                for gen_idx in range(num_gens):
                    ep = _run_episode_http(
                        env=env,
                        model=model,
                        tokenizer=tokenizer,
                        agent=agent,
                        system_prompt=system_prompt,
                        task_name=task_name,
                        seed=base_seed + gen_idx,
                        max_turns=max_turns,
                        max_new_tokens=max_new_tokens,
                        defender_baseline=defender_baseline,
                        fraudster_baseline=fraudster_baseline,
                    )
                    all_prompt_ids.append(ep["prompt_ids"])
                    all_completion_ids.append(ep["completion_ids"])
                    all_logprobs.append(ep["logprobs"])
                    all_ep_rewards.append(ep["episode_reward"])
                    all_format_valids.append(ep["format_valid"])
                    all_action_legals.append(ep["action_legal"])
                    all_alert_levels.append(ep["final_alert"])

        return {
            # Required by TRL 1.2.0 rollout_func contract
            "prompt_ids":     all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs":       all_logprobs,
            # Extra fields forwarded to reward functions via **kwargs
            "episode_rewards": all_ep_rewards,
            "format_valids":   all_format_valids,
            "action_legals":   all_action_legals,
            "alert_levels":    all_alert_levels,
        }

    return rollout_func


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
    max_comp_len:    int   = 256,
    max_new_tokens:  int   = 128,
    lora_rank:       int   = 16,
    lora_alpha:      int   = 32,
    lora_dropout:    float = 0.05,
    use_vllm:        bool  = False,
    output_dir:      str   = "outputs",
) -> None:
    """
    Train a fraud detection LLM using GRPO, driven by the HTTP server.

    The LoRA adapter is saved to ``<output_dir>/<agent>/``.  Load it later::

        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        base  = AutoModelForCausalLM.from_pretrained(model_name)
        model = PeftModel.from_pretrained(base, f"outputs/{agent}")

    GPU requirement
    ---------------
    Training a 1.5B+ parameter model requires a GPU with at least 16 GB VRAM
    (Qwen2.5-1.5B with LoRA rank-16 needs ~8 GB).  On CPU only, this will be
    extremely slow.  For inference/debugging on CPU, use inference.py instead.
    """
    if not _TRL_AVAILABLE:
        raise ImportError(
            "trl is required.\n"
            "Install with: pip install trl>=1.2.0 transformers datasets"
        )
    if agent not in ("defender", "fraudster"):
        raise ValueError(f"agent must be 'defender' or 'fraudster', got '{agent}'")
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for training.  Install: pip install torch")

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

    # ── Warn if no GPU ─────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print(
            "\n[WARNING] No CUDA GPU detected — training will be very slow on CPU.\n"
            "          For a 1.5B model with LoRA, a 16 GB GPU is recommended.\n"
        )

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
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        print(f"LoRA  r={lora_rank}  alpha={lora_alpha}  modules=q/k/v/o/gate/up/down_proj\n")
    else:
        print("[WARNING] peft not installed — full fine-tune (no LoRA).\n")

    # ── Dataset ────────────────────────────────────────────────────────────
    dataset = build_training_dataset(task=task, n_samples=n_samples)

    # ── Rollout function ───────────────────────────────────────────────────
    rollout_func = make_rollout_func_http(
        env_url=env_url,
        agent=agent,
        system_prompt=system_prompt,
        max_turns=DEFAULT_MAX_STEPS,
        max_new_tokens=max_new_tokens,
    )

    # ── GRPOConfig ─────────────────────────────────────────────────────────
    _no_gpu = not torch.cuda.is_available()
    grpo_config = GRPOConfig(
        output_dir=agent_out_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_generations=num_generations,
        max_completion_length=max_comp_len,
        gradient_checkpointing=not _no_gpu,  # not supported on CPU
        use_cpu=_no_gpu,
        bf16=False,
        fp16=False,
        use_vllm=use_vllm and not _no_gpu,
        vllm_gpu_memory_utilization=0.3 if (use_vllm and not _no_gpu) else None,
        logging_steps=1,
        save_steps=50,
        report_to="none",
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
    print(f"\nTo load the trained adapter:")
    print(f"  from peft import PeftModel")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  base  = AutoModelForCausalLM.from_pretrained('{model_name}')")
    print(f"  model = PeftModel.from_pretrained(base, '{agent_out_dir}')")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    _shared_default = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    _env_url        = os.getenv("ENV_URL",    "http://localhost:8000")

    parser = argparse.ArgumentParser(
        description="GRPO training (TRL 1.2.0) — drives server/app.py via HTTP, trains LoRA on LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Quick start
            -----------
            # Terminal 1 — start the environment server
            uv run server

            # Terminal 2 — train the defender
            python train_grpo.py --agent defender

            # After training, use the adapter in inference.py:
            # Set DEFENDER_MODEL_NAME=outputs/defender in .env

            GPU note
            --------
            A 16 GB GPU is recommended for Qwen2.5-1.5B + LoRA rank-16.
            For CPU-only testing, use a very small --samples value and expect
            very slow progress.  Use inference.py for CPU-based evaluation.
        """),
    )
    parser.add_argument("--agent",    required=True, choices=["defender", "fraudster"])
    parser.add_argument("--model",    default=_shared_default, metavar="MODEL",
                        help=f"HF model ID or local path (default: {_shared_default})")
    parser.add_argument("--env-url",  default=_env_url,
                        help=f"Environment server URL (default: {_env_url})")
    parser.add_argument("--task",     default="random", choices=VALID_TASK_NAMES,
                        help="Fraud family (default: random — rotates all families)")
    parser.add_argument("--samples",          type=int,   default=200)
    parser.add_argument("--epochs",           type=int,   default=1)
    parser.add_argument("--lr",               type=float, default=5e-6)
    parser.add_argument("--num-generations",  type=int,   default=4,
                        help="GRPO group size — episodes per training sample (default: 4)")
    parser.add_argument("--grad-accum",       type=int,   default=16)
    parser.add_argument("--max-comp-len",     type=int,   default=256)
    parser.add_argument("--max-new-tokens",   type=int,   default=128,
                        help="Max tokens generated per action step (default: 128)")
    parser.add_argument("--lora-rank",        type=int,   default=16)
    parser.add_argument("--lora-alpha",       type=int,   default=32)
    parser.add_argument("--lora-dropout",     type=float, default=0.05)
    parser.add_argument("--use-vllm",         action="store_true",
                        help="Enable vLLM colocate mode (requires GPU + vllm)")
    parser.add_argument("--output-dir",       default="outputs")
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
        max_comp_len=args.max_comp_len,
        max_new_tokens=args.max_new_tokens,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_vllm=args.use_vllm,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
