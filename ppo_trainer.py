"""
PPOTrainer — online PPO training loop for both defender and fraudster agents.

Architecture:
  - Two independent policies: DefenderPolicy and FraudsterPolicy.
  - A shared rollout buffer collects (obs, action, log_prob, reward, done, value).
  - After ``rollout_length`` steps, advantages are computed (GAE) and both
    policies are updated with separate gradient steps.
  - Defender and fraudster losses are computed independently; they do not
    share gradients.

Turn-based interaction
-----------------------
The environment now uses a turn-based protocol: the fraudster acts first,
then the defender.  Each pair of half-steps constitutes one full step.
Rewards are emitted only after the defender's half-step (full-step boundary).
The trainer follows this protocol:
  1. obs = env.reset()  → fraudster obs
  2. Apply fraudster policy → step() → defender obs
  3. Apply defender policy  → step() → next fraudster obs (+ rewards)
  4. Store one Transition per agent (rewards assigned at full-step boundary)
  5. Repeat from 2.

Usage::

    from scam_detection.ppo_trainer import PPOTrainer
    trainer = PPOTrainer()
    trainer.train(n_episodes=500)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from scam_detection.constants import DEFAULT_MAX_STEPS, VALID_TASK_NAMES
from scam_detection.models import (
    DefenderActionType,
    FraudAction,
    FraudsterActionType,
)
from scam_detection.policy_networks import DefenderPolicy, FraudsterPolicy
from scam_detection.server.fraud_environment import FraudEnvironment


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    obs: Any
    action_idx: int
    log_prob: float
    reward: float
    done: bool
    value: float
    agent: str   # "defender" | "fraudster"


@dataclass
class RolloutBuffer:
    transitions: List[Transition] = field(default_factory=list)

    def add(self, t: Transition) -> None:
        self.transitions.append(t)

    def clear(self) -> None:
        self.transitions.clear()

    def defender_transitions(self) -> List[Transition]:
        return [t for t in self.transitions if t.agent == "defender"]

    def fraudster_transitions(self) -> List[Transition]:
        return [t for t in self.transitions if t.agent == "fraudster"]


# ---------------------------------------------------------------------------
# GAE helper
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[List[float], List[float]]:
    """Compute GAE advantages and returns."""
    advantages = [0.0] * len(rewards)
    returns    = [0.0] * len(rewards)
    last_gae   = 0.0
    last_value = 0.0

    for t in reversed(range(len(rewards))):
        delta     = rewards[t] + gamma * last_value * (1 - dones[t]) - values[t]
        last_gae  = delta + gamma * lam * (1 - dones[t]) * last_gae
        last_value = values[t]
        advantages[t] = last_gae
        returns[t]    = last_gae + values[t]

    return advantages, returns


# ---------------------------------------------------------------------------
# PPOTrainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """
    Trains defender and fraudster policies using PPO.

    The trainer runs an internal ``FraudEnvironment`` (no HTTP server needed)
    and collects rollouts directly.

    Parameters
    ----------
    n_rollout_steps:
        Number of environment steps to collect per update cycle.
    n_epochs:
        Number of PPO epochs per update.
    batch_size:
        Mini-batch size for each PPO epoch.
    lr:
        Learning rate for both policies.
    gamma / lam:
        Discount factor and GAE lambda.
    clip_eps:
        PPO clip epsilon.
    vf_coef:
        Value function loss coefficient.
    ent_coef:
        Entropy bonus coefficient.
    save_path:
        Directory to save policy checkpoints.
    """

    def __init__(
        self,
        n_rollout_steps: int = 256,
        n_epochs: int = 4,
        batch_size: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        save_path: str = "checkpoints",
    ) -> None:
        self.n_rollout_steps = n_rollout_steps
        self.n_epochs        = n_epochs
        self.batch_size      = batch_size
        self.gamma           = gamma
        self.lam             = lam
        self.clip_eps        = clip_eps
        self.vf_coef         = vf_coef
        self.ent_coef        = ent_coef
        self.save_path       = save_path

        self.defender_policy  = DefenderPolicy()
        self.fraudster_policy = FraudsterPolicy()

        self.defender_opt  = optim.Adam(self.defender_policy.parameters(), lr=lr)
        self.fraudster_opt = optim.Adam(self.fraudster_policy.parameters(), lr=lr)

        self._env    = FraudEnvironment()
        self._buffer = RolloutBuffer()

    # ---------------------------------------------------------------- public

    def train(
        self,
        n_episodes: int = 200,
        task_name: str = "random",
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train both policies for ``n_episodes`` episodes.

        Returns a dict of per-episode metrics for analysis.
        """
        import os
        os.makedirs(self.save_path, exist_ok=True)

        history: Dict[str, List[float]] = {
            "defender_reward": [],
            "fraudster_reward": [],
            "defender_loss": [],
            "fraudster_loss": [],
        }

        # reset() returns fraudster obs (current_agent="fraudster")
        obs = self._env.reset(task_name=task_name)
        episode_def_reward = 0.0
        episode_frd_reward = 0.0
        episode_count      = 0
        global_full_step   = 0  # counts completed full steps (fraudster + defender pairs)

        # Temporary holders across half-steps
        _frd_obs     = None
        _frd_idx     = None
        _frd_lp      = None
        _frd_val     = None

        for _half in range(n_episodes * DEFAULT_MAX_STEPS * 2 + 8):
            if obs.done:
                # Episode ended on the previous defender half-step
                episode_count += 1
                history["defender_reward"].append(episode_def_reward)
                history["fraudster_reward"].append(episode_frd_reward)

                if verbose and episode_count % 10 == 0:
                    print(
                        f"[Episode {episode_count}] "
                        f"D-reward={episode_def_reward:.3f}  "
                        f"F-reward={episode_frd_reward:.3f}"
                    )

                episode_def_reward = 0.0
                episode_frd_reward = 0.0

                if episode_count >= n_episodes:
                    break
                obs = self._env.reset(task_name=task_name)
                _frd_obs = _frd_idx = _frd_lp = _frd_val = None

            current_agent = obs.current_agent

            # ── FRAUDSTER HALF-STEP ──────────────────────────────────────────
            if current_agent == "fraudster":
                frd_obs = obs.fraudster_obs
                legal_frd   = obs.available_fraudster_actions or []
                frd_targets = obs.fraudster_action_targets or {}

                frd_action_str, frd_lp, frd_ent, frd_val = (
                    self.fraudster_policy.select_action(frd_obs, legal_frd)
                )
                frd_target = self._pick_target(frd_action_str, frd_targets)

                # Store fraudster half-step info for later buffer insertion
                _frd_obs = frd_obs
                _frd_idx = self.fraudster_policy.ACTION_TYPES.index(frd_action_str)
                _frd_lp  = frd_lp
                _frd_val = frd_val

                action = FraudAction(
                    fraudster_action=FraudsterActionType(frd_action_str),
                    fraudster_target=frd_target,
                )
                obs = self._env.step(action)
                # obs.current_agent is now "defender"

            # ── DEFENDER HALF-STEP ───────────────────────────────────────────
            elif current_agent == "defender":
                def_obs = obs.defender_obs
                legal_def   = obs.available_defender_actions or []
                def_targets = obs.defender_action_targets or {}

                def_action_str, def_lp, def_ent, def_val = (
                    self.defender_policy.select_action(def_obs, legal_def)
                )
                def_target = self._pick_target(def_action_str, def_targets)

                action = FraudAction(
                    defender_action=DefenderActionType(def_action_str),
                    defender_target=def_target,
                )
                obs = self._env.step(action)
                # Rewards are emitted now (after defender's half-step)
                d_reward = obs.defender_reward or 0.0
                f_reward = obs.fraudster_reward or 0.0
                done     = obs.done or False

                episode_def_reward += d_reward
                episode_frd_reward += f_reward
                global_full_step   += 1

                # Store one Transition per agent for the completed full step
                def_idx = self.defender_policy.ACTION_TYPES.index(def_action_str)
                self._buffer.add(Transition(
                    obs=def_obs, action_idx=def_idx,
                    log_prob=def_lp.item(), reward=d_reward,
                    done=done, value=def_val.item(), agent="defender",
                ))
                if _frd_obs is not None:
                    self._buffer.add(Transition(
                        obs=_frd_obs, action_idx=_frd_idx,
                        log_prob=_frd_lp.item(), reward=f_reward,
                        done=done, value=_frd_val.item(), agent="fraudster",
                    ))
                    _frd_obs = _frd_idx = _frd_lp = _frd_val = None

                # ── Update every n_rollout_steps full steps ──────────────────
                if global_full_step > 0 and global_full_step % self.n_rollout_steps == 0:
                    d_loss = self._update_policy(
                        self.defender_policy,
                        self.defender_opt,
                        self._buffer.defender_transitions(),
                        self.defender_policy.encode_obs,
                    )
                    f_loss = self._update_policy(
                        self.fraudster_policy,
                        self.fraudster_opt,
                        self._buffer.fraudster_transitions(),
                        self.fraudster_policy.encode_obs,
                    )
                    history["defender_loss"].append(d_loss)
                    history["fraudster_loss"].append(f_loss)
                    self._buffer.clear()

            else:
                # current_agent is None → done=True handled at top of next iter
                pass

        # Save final checkpoints
        torch.save(
            self.defender_policy.state_dict(),
            f"{self.save_path}/defender_policy.pt",
        )
        torch.save(
            self.fraudster_policy.state_dict(),
            f"{self.save_path}/fraudster_policy.pt",
        )
        return history

    # ---------------------------------------------------------------- private

    def _update_policy(
        self,
        policy: nn.Module,
        optimizer: optim.Optimizer,
        transitions: List[Transition],
        encoder,
    ) -> float:
        """Run PPO update for one policy; return mean loss."""
        if not transitions:
            return 0.0

        rewards  = [t.reward    for t in transitions]
        values   = [t.value     for t in transitions]
        dones    = [t.done      for t in transitions]
        old_lps  = torch.tensor([t.log_prob   for t in transitions])
        actions  = torch.tensor([t.action_idx for t in transitions])

        advantages, returns = compute_gae(rewards, values, dones, self.gamma, self.lam)
        adv_t   = torch.tensor(advantages, dtype=torch.float32)
        ret_t   = torch.tensor(returns,    dtype=torch.float32)
        adv_t   = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        obs_tensors = torch.stack([encoder(t.obs) for t in transitions])

        total_loss = 0.0
        n_updates  = 0

        for _ in range(self.n_epochs):
            idxs = torch.randperm(len(transitions))
            for start in range(0, len(transitions), self.batch_size):
                b = idxs[start : start + self.batch_size]

                _, new_lp, entropy, new_val = policy.get_action_and_value(
                    obs_tensors[b], actions[b]
                )
                ratio      = (new_lp - old_lps[b]).exp()
                surr1      = ratio * adv_t[b]
                surr2      = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_t[b]
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(new_val, ret_t[b])
                loss        = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

                total_loss += loss.item()
                n_updates  += 1

        return total_loss / max(1, n_updates)

    def _pick_target(
        self, action_str: str, targets: Dict[str, List[str]]
    ) -> Optional[str]:
        """Pick a random legal target for the chosen action."""
        options = targets.get(action_str, [])
        if not options or options == ["self"]:
            return None
        return random.choice(options)

    def load_checkpoints(self, path: str = "checkpoints") -> None:
        """Load saved policy weights."""
        import os
        d_path = os.path.join(path, "defender_policy.pt")
        f_path = os.path.join(path, "fraudster_policy.pt")
        if os.path.exists(d_path):
            self.defender_policy.load_state_dict(torch.load(d_path))
        if os.path.exists(f_path):
            self.fraudster_policy.load_state_dict(torch.load(f_path))
