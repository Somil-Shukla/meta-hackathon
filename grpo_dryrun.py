"""
GRPO Dry-Run Diagnostic
=======================

Verifies that every layer of the GRPO training pipeline is working correctly
WITHOUT requiring a GPU, TRL, or a real language model.

A synthetic "mock model" that returns valid JSON actions is used in place of
generate_rollout_completions().  This tests all environment logic, reward
functions, rollout mechanics, and GRPO readiness checks in < 30 seconds.

What is checked
---------------
  [1] Environment health        — reset/step produce valid observations
  [2] Reward variation          — rewards differ across seeds (GRPO needs this)
  [3] Action parser roundtrip   — JSON → action → env.step() works cleanly
  [4] Legal action compliance   — mock model stays within legal action set
  [5] Reward function outputs   — all reward functions return valid floats
  [6] GRPO group reward spread  — std-dev > 0 across num_generations episodes
  [7] Full rollout simulation   — simulates exactly what grpo_train.py will do
  [8] Reward signal summary     — prints per-agent statistics

Usage
-----
    python grpo_dryrun.py                   # defender + fraudster, 4 gens, mule_cashout
    python grpo_dryrun.py --agent defender
    python grpo_dryrun.py --num-gens 8 --task account_takeover
    python grpo_dryrun.py --steps 5         # short episodes for speed
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
import time
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Environment imports
# ---------------------------------------------------------------------------
try:
    from scam_detection.constants import DEFAULT_MAX_STEPS, VALID_TASK_NAMES
    from scam_detection.grpo_train import (
        BaselineFraudster,
        _build_defender_message,
        _build_fraudster_message,
        _parse_defender_action,
        _parse_fraudster_action,
        reward_action_legal,
        reward_def_episode,
        reward_format_valid,
        reward_frd_episode,
        reward_frd_evasion,
        build_training_dataset,
    )
    from scam_detection.models import DefenderActionType, FraudAction, FraudsterActionType
    from scam_detection.server.fraud_environment import FraudEnvironment
    from scam_detection.baseline_detector import BaselineRuleDetector
except ImportError:
    from constants import DEFAULT_MAX_STEPS, VALID_TASK_NAMES
    from grpo_train import (
        BaselineFraudster,
        _build_defender_message,
        _build_fraudster_message,
        _parse_defender_action,
        _parse_fraudster_action,
        reward_action_legal,
        reward_def_episode,
        reward_format_valid,
        reward_frd_episode,
        reward_frd_evasion,
        build_training_dataset,
    )
    from models import DefenderActionType, FraudAction, FraudsterActionType
    from server.fraud_environment import FraudEnvironment
    from baseline_detector import BaselineRuleDetector

# ---------------------------------------------------------------------------
# ANSI colours (disabled on Windows if needed)
# ---------------------------------------------------------------------------
_USE_COLOUR = sys.platform != "win32" or "TERM" in __import__("os").environ

def _green(s: str) -> str:  return f"\033[92m{s}\033[0m" if _USE_COLOUR else s
def _red(s: str)   -> str:  return f"\033[91m{s}\033[0m" if _USE_COLOUR else s
def _yellow(s: str)-> str:  return f"\033[93m{s}\033[0m" if _USE_COLOUR else s
def _bold(s: str)  -> str:  return f"\033[1m{s}\033[0m"  if _USE_COLOUR else s

PASS = _green("PASS")
FAIL = _red("FAIL")
WARN = _yellow("WARN")

# ---------------------------------------------------------------------------
# Mock model: returns a syntactically valid JSON action drawn from the legal
# set, with configurable error injection for testing parser robustness.
# ---------------------------------------------------------------------------

class MockModel:
    """
    Synthetic stand-in for generate_rollout_completions().

    Modes
    -----
    "legal"    — always returns a valid action in the legal set.
    "random"   — sometimes picks an illegal/random action (tests parser fallback).
    "invalid"  — returns malformed JSON (tests parser robustness).
    """

    def __init__(self, mode: str = "legal", rng_seed: int = 0):
        self._rng  = random.Random(rng_seed)
        self.mode  = mode
        self.calls = 0

    def generate(self, obs, agent: str) -> str:
        """Return a completion string for the given observation and agent."""
        self.calls += 1

        if self.mode == "invalid":
            return "I am not sure what to do here."

        if agent == "defender":
            legal   = obs.available_defender_actions or []
            targets = obs.defender_action_targets or {}
        else:
            legal   = obs.available_fraudster_actions or []
            targets = obs.fraudster_action_targets or {}

        if not legal or self.mode == "random" and self._rng.random() < 0.3:
            # Occasionally return an illegal action to test fallback
            action = self._rng.choice(
                [a.value for a in DefenderActionType]
                if agent == "defender"
                else [a.value for a in FraudsterActionType]
            )
            target = None
        else:
            action = self._rng.choice(legal)
            opts   = targets.get(action, [])
            target = opts[0] if opts and opts != ["self"] else None

        return json.dumps({"action": action, "target": target})


# ---------------------------------------------------------------------------
# Single-episode rollout (mirrors _run_single_episode in grpo_train.py)
# ---------------------------------------------------------------------------

def _run_episode(
    env: FraudEnvironment,
    agent: str,
    task_name: str,
    seed: int,
    max_turns: int,
    mock: MockModel,
    defender_baseline: BaselineRuleDetector,
    fraudster_baseline: BaselineFraudster,
) -> Dict[str, Any]:
    """
    Run one episode following the correct turn-based protocol:
      reset() → current_agent="fraudster"
      step(fraudster_action) → current_agent="defender"  [no rewards]
      step(defender_action)  → current_agent="fraudster" [rewards emitted]
      ...

    When training the DEFENDER:
      - Fraudster half-step uses the fixed baseline (not the model).
      - The model generates its action from the resulting defender obs.

    When training the FRAUDSTER:
      - Model generates fraudster action from current obs.
      - Defender half-step uses the fixed baseline.
    """
    obs = env.reset(task_name=task_name, seed=seed)
    # obs.current_agent == "fraudster" after reset

    ep_reward    = 0.0
    ep_fmt_valid = 0.0
    ep_act_legal = 0.0
    n_steps      = 0
    final_alert  = 0.0

    for _ in range(max_turns):
        if obs.done:
            break

        if agent == "defender":
            # 1. Fraudster half-step with fixed baseline
            frd_str, frd_tgt = fraudster_baseline.select_action(
                obs.fraudster_obs,
                obs.available_fraudster_actions,
                obs.fraudster_action_targets,
            )
            obs = env.step(FraudAction(
                fraudster_action=FraudsterActionType(frd_str),
                fraudster_target=frd_tgt,
            ))
            if obs.done:
                break
            # obs now has available_defender_actions populated
            # 2. Model generates defender action
            text = mock.generate(obs, "defender")
            action_str, target, is_json, is_legal = _parse_defender_action(text, obs)
            ep_fmt_valid += float(is_json)
            ep_act_legal += float(is_legal)
            n_steps += 1
            obs = env.step(FraudAction(
                defender_action=DefenderActionType(action_str),
                defender_target=target,
            ))

        else:  # fraudster
            # 1. Model generates fraudster action from current obs
            text = mock.generate(obs, "fraudster")
            action_str, target, is_json, is_legal = _parse_fraudster_action(text, obs)
            ep_fmt_valid += float(is_json)
            ep_act_legal += float(is_legal)
            n_steps += 1
            obs = env.step(FraudAction(
                fraudster_action=FraudsterActionType(action_str),
                fraudster_target=target,
            ))
            if obs.done:
                break
            # 2. Defender half-step with fixed baseline
            def_str, def_tgt = defender_baseline.select_action(
                obs.defender_obs,
                obs.available_defender_actions,
                obs.defender_action_targets,
            )
            obs = env.step(FraudAction(
                defender_action=DefenderActionType(def_str),
                defender_target=def_tgt,
            ))

        step_reward = (
            obs.defender_reward if agent == "defender" else obs.fraudster_reward
        ) or 0.0
        ep_reward += step_reward

    if obs.fraudster_obs:
        final_alert = float(obs.fraudster_obs.get("alert_level", 0.0))

    return {
        "episode_reward": ep_reward,
        "format_valid":   ep_fmt_valid / max(1, n_steps),
        "action_legal":   ep_act_legal / max(1, n_steps),
        "final_alert":    final_alert,
        "n_steps":        n_steps,
        "done":           obs.done,
    }


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------

def _check(label: str, ok: bool, detail: str = "") -> bool:
    status = PASS if ok else FAIL
    line   = f"  {status}  {label}"
    if detail:
        line += f"  — {detail}"
    print(line)
    return ok


def _section(title: str) -> None:
    print(f"\n{_bold(title)}")
    print("  " + "-" * (len(title) + 2))


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_env_health(env: FraudEnvironment, task: str) -> bool:
    _section("[1] Environment health")
    ok = True

    obs = env.reset(task_name=task, seed=42)
    ok &= _check("reset() returns obs",           obs is not None)
    ok &= _check("obs has episode_id",             bool(obs.episode_id))
    ok &= _check("obs.done is False after reset",  obs.done is False)
    ok &= _check("current_agent='fraudster'",      obs.current_agent == "fraudster")
    ok &= _check("fraudster_obs populated",        bool(obs.fraudster_obs))
    ok &= _check("defender_obs deferred",          obs.defender_obs is None)

    # Fraudster half-step
    mid = env.step(FraudAction(fraudster_action=FraudsterActionType.DO_NOTHING))
    ok &= _check("fraudster step → current_agent='defender'", mid.current_agent == "defender")
    ok &= _check("defender_obs populated after fraudster step", bool(mid.defender_obs))
    ok &= _check("no rewards on fraudster half-step", mid.defender_reward is None)

    # Defender half-step
    fin = env.step(FraudAction(defender_action=DefenderActionType.DO_NOTHING))
    ok &= _check("defender step returns float rewards", isinstance(fin.defender_reward, float))
    ok &= _check("fraudster reward returned",           isinstance(fin.fraudster_reward, float))
    ok &= _check("step counter increments",             fin.step == 1)
    return ok


def check_reward_variation(env: FraudEnvironment, task: str, n_seeds: int = 6) -> bool:
    _section("[2] Reward variation across seeds")
    def_base  = BaselineRuleDetector()
    frd_base  = BaselineFraudster()
    def_rewards, frd_rewards = [], []

    for seed in range(n_seeds):
        mock = MockModel(mode="legal", rng_seed=seed)
        # Run 3 full rounds to collect meaningful reward signal
        obs = env.reset(task_name=task, seed=seed * 100)
        ep_def = ep_frd = 0.0
        for _ in range(3):
            if obs.done:
                break
            frd_str, frd_tgt = frd_base.select_action(
                obs.fraudster_obs,
                obs.available_fraudster_actions,
                obs.fraudster_action_targets,
            )
            obs = env.step(FraudAction(fraudster_action=FraudsterActionType(frd_str), fraudster_target=frd_tgt))
            if obs.done:
                break
            text = mock.generate(obs, "defender")
            a, t, _, _ = _parse_defender_action(text, obs)
            obs = env.step(FraudAction(defender_action=DefenderActionType(a), defender_target=t))
            ep_def += obs.defender_reward or 0.0
            ep_frd += obs.fraudster_reward or 0.0
        def_rewards.append(ep_def)
        frd_rewards.append(ep_frd)

    def_var = len(set(round(r, 6) for r in def_rewards)) > 1
    frd_var = len(set(round(r, 6) for r in frd_rewards)) > 1

    # Reward variation is REQUIRED for GRPO gradient signal.
    # A 3-step snippet may show constant rewards when the baseline opponent
    # always picks the same action.  Full episodes (DEFAULT_MAX_STEPS) tested
    # in [6] will reveal actual variation.  Emit WARN here, not FAIL.
    ok = True
    _check(
        "Defender rewards vary across seeds (3-step check)",
        def_var,
        f"values={[round(r,3) for r in def_rewards]}",
    )
    if not def_var:
        print(f"    {WARN}  Constant defender reward over short horizon — expected; "
              f"full-episode check [6] is definitive.")

    _check(
        "Fraudster rewards vary across seeds (3-step check)",
        frd_var,
        f"values={[round(r,3) for r in frd_rewards]}",
    )
    if not frd_var:
        print(f"    {WARN}  Constant fraudster reward over short horizon — expected; "
              f"full-episode check [6] is definitive.")

    return ok  # never hard-fail here; [6] is the authoritative variation check


def check_action_parser(env: FraudEnvironment, task: str) -> bool:
    _section("[3] Action parser roundtrip")
    ok = True

    obs = env.reset(task_name=task, seed=7)

    # Valid JSON with legal action
    legal_frd = (obs.available_fraudster_actions or ["do_nothing"])[0]
    targets   = obs.fraudster_action_targets or {}
    tgt       = (targets.get(legal_frd) or [None])[0]
    text_ok   = json.dumps({"action": legal_frd, "target": tgt})

    a, t, is_json, is_legal = _parse_fraudster_action(text_ok, obs)
    ok &= _check("valid JSON + legal action parsed",       is_json and is_legal, f"action={a}")

    # Invalid JSON falls back gracefully
    a2, _, is_json2, is_legal2 = _parse_fraudster_action("not json at all", obs)
    ok &= _check("invalid JSON → do_nothing fallback",     not is_json2 and not is_legal2, f"action={a2}")

    # Unknown action falls back
    a3, _, is_json3, is_legal3 = _parse_fraudster_action('{"action": "hack_mainframe"}', obs)
    ok &= _check("unknown action → do_nothing fallback",   is_json3 and not is_legal3, f"action={a3}")

    # Defender parser
    mid = env.step(FraudAction(fraudster_action=FraudsterActionType.DO_NOTHING))
    legal_def = (mid.available_defender_actions or ["do_nothing"])[0]
    def_text  = json.dumps({"action": legal_def, "target": None})
    ad, _, is_jd, is_ld = _parse_defender_action(def_text, mid)
    ok &= _check("defender parser: valid JSON + legal action", is_jd and is_ld, f"action={ad}")

    return ok


def check_legal_compliance(env: FraudEnvironment, task: str, n_steps: int = 10) -> bool:
    _section("[4] Legal action compliance (mock model)")
    mock     = MockModel(mode="legal", rng_seed=0)
    def_base = BaselineRuleDetector()
    frd_base = BaselineFraudster()
    violations = 0

    obs = env.reset(task_name=task, seed=42)
    for _ in range(n_steps):
        if obs.done:
            break
        # Fraudster turn
        text = mock.generate(obs, "fraudster")
        a, _, _, is_legal = _parse_fraudster_action(text, obs)
        if not is_legal:
            violations += 1
        obs = env.step(FraudAction(fraudster_action=FraudsterActionType(a)))

        # Defender turn
        text = mock.generate(obs, "defender")
        a, _, _, is_legal = _parse_defender_action(text, obs)
        if not is_legal:
            violations += 1
        obs = env.step(FraudAction(defender_action=DefenderActionType(a)))

    ok = violations == 0
    _check(
        f"Mock model legal action compliance ({n_steps} steps)",
        ok,
        f"violations={violations}",
    )
    return ok


def check_reward_functions(num_gens: int = 4) -> bool:
    _section("[5] Reward function outputs")
    ok   = True
    comp = ["a"] * num_gens

    fmt_vals  = [float(i % 2) for i in range(num_gens)]
    act_vals  = [float(i % 2) for i in range(num_gens)]
    ep_rews   = [float(i - num_gens / 2) for i in range(num_gens)]
    alerts    = [i / num_gens for i in range(num_gens)]

    r = reward_format_valid(comp, format_valids=fmt_vals)
    ok &= _check("reward_format_valid returns list[float]",
                 isinstance(r, list) and all(isinstance(v, float) for v in r),
                 f"output={r}")

    r = reward_action_legal(comp, action_legals=act_vals)
    ok &= _check("reward_action_legal returns list[float]",
                 isinstance(r, list) and all(isinstance(v, float) for v in r),
                 f"output={r}")

    r = reward_def_episode(comp, episode_rewards=ep_rews)
    ok &= _check("reward_def_episode returns normalised list",
                 isinstance(r, list) and all(-1.0 <= v <= 1.0 for v in r),
                 f"output={[round(v,2) for v in r]}")

    r = reward_frd_episode(comp, episode_rewards=ep_rews)
    ok &= _check("reward_frd_episode returns normalised list",
                 isinstance(r, list) and all(-1.0 <= v <= 1.0 for v in r),
                 f"output={[round(v,2) for v in r]}")

    # Use explicit [0.0, 0.5, 1.0] so first=1.0 and last=0.0 regardless of num_gens
    r_ev = reward_frd_evasion(["x","y","z"], alert_levels=[0.0, 0.5, 1.0])
    ok &= _check("reward_frd_evasion: 0 alert → 1.0, max alert → 0.0",
                 abs(r_ev[0] - 1.0) < 1e-6 and abs(r_ev[-1]) < 1e-6,
                 f"output={[round(v,2) for v in r_ev]}")
    return ok


def check_grpo_group_spread(
    env: FraudEnvironment,
    agent: str,
    task: str,
    num_gens: int,
    max_steps: int,
) -> bool:
    _section(f"[6] GRPO group reward spread (agent={agent}, num_gens={num_gens})")
    def_base  = BaselineRuleDetector()
    frd_base  = BaselineFraudster()
    mock      = MockModel(mode="random", rng_seed=99)  # inject some noise
    rewards   = []

    for gen_idx in range(num_gens):
        ep = _run_episode(
            env=env,
            agent=agent,
            task_name=task,
            seed=gen_idx * 137 + 42,
            max_turns=max_steps,
            mock=mock,
            defender_baseline=def_base,
            fraudster_baseline=frd_base,
        )
        rewards.append(ep["episode_reward"])

    spread  = stdev(rewards) if len(rewards) > 1 else 0.0
    non_dup = len(set(round(r, 4) for r in rewards))
    ok = True

    ok &= _check(
        f"Episode rewards across {num_gens} gens",
        True,
        f"values={[round(r,3) for r in rewards]}",
    )
    ok &= _check(
        "At least 2 distinct reward values (GRPO needs variation)",
        non_dup >= 2,
        f"distinct_values={non_dup}  std={spread:.4f}",
    )
    if spread < 1e-6:
        print(f"    {WARN}  Reward std-dev is nearly zero — GRPO gradient will vanish.")
        print(        "         Consider: more diverse seeds, longer episodes, or injected noise.")

    return ok


def check_full_rollout(
    env: FraudEnvironment,
    agent: str,
    task: str,
    num_gens: int,
    max_steps: int,
) -> bool:
    _section(f"[7] Full rollout simulation (agent={agent})")
    def_base = BaselineRuleDetector()
    frd_base = BaselineFraudster()
    mock     = MockModel(mode="legal", rng_seed=0)
    episodes = []

    t0 = time.time()
    for gen_idx in range(num_gens):
        ep = _run_episode(
            env=env,
            agent=agent,
            task_name=task,
            seed=gen_idx * 137 + 42,
            max_turns=max_steps,
            mock=mock,
            defender_baseline=def_base,
            fraudster_baseline=frd_base,
        )
        episodes.append(ep)
    elapsed = time.time() - t0

    ok = True
    # "completed" = episode ended (done=True) OR ran all max_turns steps
    all_ran  = all(ep["done"] or ep["n_steps"] >= max_steps for ep in episodes)
    avg_fmt  = mean(ep["format_valid"]   for ep in episodes)
    avg_legal= mean(ep["action_legal"]   for ep in episodes)
    avg_steps= mean(ep["n_steps"]        for ep in episodes)
    avg_reward=mean(ep["episode_reward"] for ep in episodes)

    ok &= _check(
        f"All {num_gens} episodes ran (done or reached max_steps={max_steps})",
        all_ran, f"elapsed={elapsed:.1f}s"
    )
    ok &= _check("Avg format-valid rate = 1.0",          abs(avg_fmt - 1.0) < 1e-6,
                 f"avg={avg_fmt:.3f}")
    ok &= _check("Avg legal-action rate = 1.0",          abs(avg_legal - 1.0) < 1e-6,
                 f"avg={avg_legal:.3f}")
    ok &= _check(f"Avg steps per episode > 0",           avg_steps > 0,
                 f"avg_steps={avg_steps:.1f}")
    _check("Avg episode reward (informational)",          True,
           f"avg={avg_reward:.4f}  (positive is good for {agent})")

    mock_calls_per_ep = mock.calls / max(1, num_gens)
    _check("Mock model calls / episode (= approx steps)", True,
           f"{mock_calls_per_ep:.1f}")
    return ok


def check_dataset(task: str) -> bool:
    _section("[8] Training dataset")
    ds = build_training_dataset(task=task, n_samples=16)
    ok = True
    ok &= _check("Dataset has 16 rows",        len(ds) == 16)
    ok &= _check("Row has 'task_name' field",   "task_name" in ds[0])
    ok &= _check("Row has 'seed' field",        "seed"      in ds[0])
    ok &= _check("Row has 'prompt' field",      "prompt"    in ds[0])
    ok &= _check("Seeds are unique",            len({r["seed"] for r in ds}) == 16)
    return ok


# ---------------------------------------------------------------------------
# Reward signal summary
# ---------------------------------------------------------------------------

def print_reward_summary(
    env: FraudEnvironment,
    task: str,
    max_steps: int,
    n_episodes: int = 8,
) -> None:
    _section("[9] Reward signal summary (both agents)")
    def_base = BaselineRuleDetector()
    frd_base = BaselineFraudster()

    def_rewards, frd_rewards = [], []
    for seed in range(n_episodes):
        mock = MockModel(mode="legal", rng_seed=seed)
        for agent, lst in [("defender", def_rewards), ("fraudster", frd_rewards)]:
            ep = _run_episode(
                env=env,
                agent=agent,
                task_name=task,
                seed=seed * 137 + 42,
                max_turns=max_steps,
                mock=mock,
                defender_baseline=def_base,
                fraudster_baseline=frd_base,
            )
            lst.append(ep["episode_reward"])

    for agent, rewards in [("defender", def_rewards), ("fraudster", frd_rewards)]:
        mn = mean(rewards)
        sd = stdev(rewards) if len(rewards) > 1 else 0.0
        mn_r, mx_r = min(rewards), max(rewards)
        colour = _green if mn > 0 else (_yellow if mn == 0 else _red)
        print(
            f"  {_bold(agent):20s}  "
            f"mean={colour(f'{mn:+.4f}'):12s}  "
            f"std={sd:.4f}  "
            f"min={mn_r:+.4f}  max={mx_r:+.4f}"
        )

    print()
    print("  Guidance:")
    print("  • mean > 0  → agent is already receiving positive reinforcement.")
    print("  • std > 0   → reward varies across episodes → GRPO has gradient signal.")
    print("  • mean ≈ 0  → actions are neutral; reward shaping may need tuning.")
    print("  • std ≈ 0   → all episodes identical; increase seed diversity or episode length.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GRPO dry-run diagnostic — no GPU or TRL required",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Exit codes
            ----------
            0 — all checks passed
            1 — one or more checks failed
        """),
    )
    parser.add_argument("--agent",    default="both",  choices=["defender","fraudster","both"])
    parser.add_argument("--task",     default="mule_cashout", choices=VALID_TASK_NAMES)
    parser.add_argument("--num-gens", type=int, default=4,
                        help="GRPO group size to simulate (default: 4)")
    parser.add_argument("--steps",    type=int, default=DEFAULT_MAX_STEPS,
                        help=f"Max steps per episode (default: {DEFAULT_MAX_STEPS})")
    args = parser.parse_args()

    agents     = ["defender", "fraudster"] if args.agent == "both" else [args.agent]
    env        = FraudEnvironment()
    all_passed = True

    print(_bold(f"\nGRPO Dry-Run Diagnostic"))
    print(f"  task={args.task}  num_gens={args.num_gens}  max_steps={args.steps}  agents={agents}\n")

    # ── Checks shared across agents ────────────────────────────────────────
    all_passed &= check_env_health(env, args.task)
    all_passed &= check_reward_variation(env, args.task)
    all_passed &= check_action_parser(env, args.task)
    all_passed &= check_legal_compliance(env, args.task)
    all_passed &= check_reward_functions(args.num_gens)
    all_passed &= check_dataset(args.task)

    # ── Per-agent checks ───────────────────────────────────────────────────
    for agent in agents:
        all_passed &= check_grpo_group_spread(
            env, agent, args.task, args.num_gens, args.steps
        )
        all_passed &= check_full_rollout(
            env, agent, args.task, args.num_gens, args.steps
        )

    # ── Summary ────────────────────────────────────────────────────────────
    print_reward_summary(env, args.task, args.steps)

    print("\n" + "=" * 55)
    if all_passed:
        print(_green(_bold("  ALL CHECKS PASSED — ready to run grpo_train.py")))
        print()
        print("  Next step:")
        for agent in agents:
            print(f"    python grpo_train.py --agent {agent} --model <HF_MODEL_ID>")
    else:
        print(_red(_bold("  SOME CHECKS FAILED — fix the issues above before training")))
    print("=" * 55 + "\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
