"""
Evaluation script — runs PPO policies and the rule-based baseline
against the FraudEnvironment and prints comparison metrics.

Compares:
  1. PPO Defender (trained) vs PPO Fraudster (trained)
  2. Baseline Defender (rule-based) vs Random Fraudster

Usage:
    python evaluate.py                       # evaluate with random task
    python evaluate.py --task mule_cashout   # specific fraud family
    python evaluate.py --episodes 50 --checkpoint-path checkpoints
"""
from __future__ import annotations

import argparse
import random
from typing import Dict, List

from scam_detection.baseline_detector import BaselineRuleDetector
from scam_detection.constants import DEFAULT_MAX_STEPS, VALID_TASK_NAMES
from scam_detection.models import DefenderActionType, FraudAction, FraudsterActionType
from scam_detection.server.fraud_environment import FraudEnvironment


def run_ppo_episode(env: FraudEnvironment, task_name: str) -> Dict:
    """Run one episode using trained PPO policies."""
    try:
        import torch
        from scam_detection.policy_networks import DefenderPolicy, FraudsterPolicy
        defender_policy  = DefenderPolicy()
        fraudster_policy = FraudsterPolicy()
    except ImportError:
        return run_baseline_episode(env, task_name)

    obs = env.reset(task_name=task_name)
    total_def  = 0.0
    total_frd  = 0.0

    for _ in range(DEFAULT_MAX_STEPS):
        legal_def = obs.available_defender_actions or []
        legal_frd = obs.available_fraudster_actions or []
        def_targets = obs.defender_action_targets or {}
        frd_targets = obs.fraudster_action_targets or {}

        def_str, _, _, _ = defender_policy.select_action(
            obs.defender_obs, legal_def, deterministic=True
        )
        frd_str, _, _, _ = fraudster_policy.select_action(
            obs.fraudster_obs, legal_frd, deterministic=True
        )
        def_target = _pick(def_str, def_targets)
        frd_target = _pick(frd_str, frd_targets)

        action = FraudAction(
            defender_action=DefenderActionType(def_str),
            defender_target=def_target,
            fraudster_action=FraudsterActionType(frd_str),
            fraudster_target=frd_target,
        )
        obs = env.step(action)
        total_def += obs.defender_reward or 0.0
        total_frd += obs.fraudster_reward or 0.0

        if obs.done:
            break

    grade = (obs.info or {}).get("grade", {})
    return {
        "method": "PPO",
        "task": task_name,
        "cumulative_defender_reward": round(total_def, 4),
        "cumulative_fraudster_reward": round(total_frd, 4),
        "defender_score":   grade.get("defender_score",   0.0),
        "fraudster_score":  grade.get("fraudster_score",  0.0),
        "fraud_prevented":  grade.get("total_fraud_prevented", 0.0),
        "fraud_escaped":    grade.get("total_fraud_escaped",   0.0),
        "false_positives":  grade.get("false_positive_count",  0),
    }


def run_baseline_episode(env: FraudEnvironment, task_name: str) -> Dict:
    """Run one episode using the rule-based baseline defender + random fraudster."""
    baseline = BaselineRuleDetector()
    obs = env.reset(task_name=task_name)
    total_def = 0.0
    total_frd = 0.0

    for _ in range(DEFAULT_MAX_STEPS):
        legal_def = obs.available_defender_actions or []
        legal_frd = obs.available_fraudster_actions or []
        def_targets = obs.defender_action_targets or {}
        frd_targets = obs.fraudster_action_targets or {}

        def_str, def_target = baseline.select_action(
            obs.defender_obs, legal_def, def_targets
        )
        frd_str = random.choice(legal_frd) if legal_frd else "do_nothing"
        frd_target = _pick(frd_str, frd_targets)

        action = FraudAction(
            defender_action=DefenderActionType(def_str),
            defender_target=def_target,
            fraudster_action=FraudsterActionType(frd_str),
            fraudster_target=frd_target,
        )
        obs = env.step(action)
        total_def += obs.defender_reward or 0.0
        total_frd += obs.fraudster_reward or 0.0

        if obs.done:
            break

    grade = (obs.info or {}).get("grade", {})
    return {
        "method": "Baseline",
        "task": task_name,
        "cumulative_defender_reward": round(total_def, 4),
        "cumulative_fraudster_reward": round(total_frd, 4),
        "defender_score":   grade.get("defender_score",   0.0),
        "fraudster_score":  grade.get("fraudster_score",  0.0),
        "fraud_prevented":  grade.get("total_fraud_prevented", 0.0),
        "fraud_escaped":    grade.get("total_fraud_escaped",   0.0),
        "false_positives":  grade.get("false_positive_count",  0),
    }


def _pick(action_str: str, targets: Dict) -> None:
    options = targets.get(action_str, [])
    if not options or options == ["self"]:
        return None
    return random.choice(options)


def print_summary(results: List[Dict]) -> None:
    ppo_r = [r for r in results if r["method"] == "PPO"]
    base_r = [r for r in results if r["method"] == "Baseline"]

    def avg(records, key):
        vals = [r[key] for r in records if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    header = f"{'Metric':<35} {'PPO':>10} {'Baseline':>10}"
    print(header)
    print("-" * 60)
    metrics = [
        ("Defender Score",            "defender_score"),
        ("Fraudster Score",           "fraudster_score"),
        ("Cumulative Defender Reward","cumulative_defender_reward"),
        ("Fraud Prevented",           "fraud_prevented"),
        ("Fraud Escaped",             "fraud_escaped"),
        ("False Positives",           "false_positives"),
    ]
    for label, key in metrics:
        ppo_v  = avg(ppo_r,  key) if ppo_r  else "N/A"
        base_v = avg(base_r, key)
        print(f"{label:<35} {str(ppo_v):>10} {str(base_v):>10}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fraud detection policies")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--task",     type=str, default="random")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints")
    args = parser.parse_args()

    tasks = [t for t in VALID_TASK_NAMES if t != "random"] if args.task == "all" \
        else [args.task]

    env = FraudEnvironment()
    all_results: List[Dict] = []

    for ep in range(args.episodes):
        task = random.choice(tasks) if args.task in ("random", "all") else args.task
        ppo_result  = run_ppo_episode(env, task)
        base_result = run_baseline_episode(env, task)
        all_results.extend([ppo_result, base_result])
        if (ep + 1) % 5 == 0:
            print(f"Episode {ep+1}/{args.episodes}: "
                  f"PPO defender={ppo_result['defender_score']:.3f}  "
                  f"Baseline defender={base_result['defender_score']:.3f}")

    print_summary(all_results)


if __name__ == "__main__":
    main()
