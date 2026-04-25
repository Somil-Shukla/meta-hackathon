"""
Training script — runs PPO training for defender and fraudster policies
directly against the internal FraudEnvironment (no HTTP server required).

Usage:
    python train.py                          # 200 episodes, random fraud family
    python train.py --episodes 500 --task mule_cashout
    python train.py --episodes 1000 --save-path checkpoints/run1
"""
from __future__ import annotations

import argparse

from scam_detection.ppo_trainer import PPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fraud detection RL policies")
    parser.add_argument(
        "--episodes", type=int, default=200,
        help="Number of training episodes (default: 200)"
    )
    parser.add_argument(
        "--task", type=str, default="random",
        help="Fraud family to train on (default: random)"
    )
    parser.add_argument(
        "--rollout-steps", type=int, default=256,
        help="Steps per PPO update cycle (default: 256)"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate (default: 3e-4)"
    )
    parser.add_argument(
        "--save-path", type=str, default="checkpoints",
        help="Directory to save policy checkpoints (default: checkpoints)"
    )
    parser.add_argument(
        "--no-verbose", action="store_true",
        help="Suppress per-episode output"
    )
    args = parser.parse_args()

    print(f"Training fraud detection policies")
    print(f"  Episodes      : {args.episodes}")
    print(f"  Task          : {args.task}")
    print(f"  Rollout steps : {args.rollout_steps}")
    print(f"  Learning rate : {args.lr}")
    print(f"  Save path     : {args.save_path}")
    print()

    trainer = PPOTrainer(
        n_rollout_steps=args.rollout_steps,
        lr=args.lr,
        save_path=args.save_path,
    )

    history = trainer.train(
        n_episodes=args.episodes,
        task_name=args.task,
        verbose=not args.no_verbose,
    )

    # Print training summary
    if history["defender_reward"]:
        last_20_def = history["defender_reward"][-20:]
        last_20_frd = history["fraudster_reward"][-20:]
        print(f"\nTraining complete.")
        print(f"  Last 20 ep avg defender  reward: {sum(last_20_def)/len(last_20_def):.4f}")
        print(f"  Last 20 ep avg fraudster reward: {sum(last_20_frd)/len(last_20_frd):.4f}")
        print(f"  Checkpoints saved to: {args.save_path}/")


if __name__ == "__main__":
    main()
