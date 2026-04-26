---
title: Fraud Detection Env
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# Fraud Detection RL Environment

An OpenEnv-compatible **multi-agent** RL environment for training and evaluating
fraud detection policies.

Two agents interact every episode:
- **Defender** (LLM / PPO) — learns to detect and block fraudulent activity.
- **Fraudster** (LLM / PPO) — acts as an adaptive adversary trying to launder money.

The world is **partially observable** — agents see different, noisy views derived
from the same hidden ground truth.

---

## Quick start

```bash
# Install dependencies
pip install -e ".[dev]"
# or with uv:
uv sync

# Start the environment server
uv run server
# or: python -m scam_detection.server.app

# In a second terminal, run the LLM agent (inference)
export HF_TOKEN=<your_token>
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
python inference.py

# Or run offline PPO training (no server needed)
python train.py --episodes 200 --task mule_cashout

# Evaluate trained vs baseline
python evaluate.py --episodes 20
```

With Docker:

```bash
docker build -t fraud-detection-env:latest .
docker run -p 8000:8000 fraud-detection-env:latest
python inference.py  # connects via ENV_URL
```

---

## Fraud families (task variants)

| Task name            | Description |
|----------------------|-------------|
| `refund_abuse`       | Mule users repeatedly buy and refund to extract value |
| `mule_cashout`       | Stolen funds layered through a chain of mule accounts |
| `merchant_collusion` | Colluding merchant processes fake high-value transactions |
| `account_takeover`   | Compromised legitimate accounts make rapid large transfers |
| `random`             | Random family chosen each episode |

---

## Episode structure

```
reset(task_name=...)
  └─ FraudObservation:
       task_name, step_budget, fraud_family_hint
       defender_obs  (partial, noisy)
       fraudster_obs (partial, operational)
       available_defender_actions + targets
       available_fraudster_actions + targets

step(FraudAction(defender_action, fraudster_action, targets))   × N
  └─ FraudObservation:
       defender_obs / fraudster_obs (updated)
       defender_reward / fraudster_reward
       available_* actions
       info.grade (at terminal step)
       done, reason
```

Maximum steps per episode: **20** (configurable via `DEFAULT_MAX_STEPS`).

---

## Action space

### Defender actions

| Action | Description |
|--------|-------------|
| `monitor` | Place account under surveillance |
| `challenge` | Send step-up authentication challenge |
| `freeze` | Freeze account (high precision required) |
| `hold` | Hold a pending transaction |
| `block_merchant` | Block a suspicious merchant |
| `investigate_neighborhood` | Flag all connected accounts |
| `do_nothing` | No action |

### Fraudster actions

| Action | Description |
|--------|-------------|
| `split_payment` | Make small split payments to avoid thresholds |
| `rotate_mule` | Switch to a different mule account |
| `switch_merchant` | Use a different merchant |
| `rotate_device` | Change device to reduce device-reuse signal |
| `delay` | Lie low for 2 steps to lower detection pressure |
| `refund_abuse` | Exploit refund workflows for value extraction |
| `cashout_attempt` | Attempt to cash out via a fraud route |
| `do_nothing` | No action |

---

## Reward structure

### Defender per-step rewards

| Condition | Reward |
|-----------|--------|
| Fraud account frozen (TP) | +1.0 |
| Early detection bonus | up to +0.3 |
| Fraudulent transaction held | +0.8 |
| Colluding merchant blocked | +0.9 |
| Legit account frozen (FP) | -0.5 + -0.3 + -0.15 |
| Fraudster cashes out (missed) | -0.5 |
| Investigation cost | -0.05 |

### Fraudster per-step rewards

| Condition | Reward |
|-----------|--------|
| Successful cashout | +1.0 |
| Undetected activity | +0.05 |
| Evasion (rotate when pressure high) | +0.2 |
| Account frozen by defender | -0.5 |
| Merchant blocked | -0.4 |
| Failed cashout | -0.3 |
| Route deactivated | -0.2 |

---

## Observation contract

Agents never receive: ground-truth fraud labels, opponent's internal state, or raw reward equations.

```python
# Reset observation
obs.task_name             # str — active fraud family
obs.step_budget           # {"total": 20, "used": 0, "remaining": 20}
obs.defender_obs          # dict — partial, noisy defender view
obs.fraudster_obs         # dict — partial, operational fraudster view
obs.available_defender_actions   # List[str]
obs.available_fraudster_actions  # List[str]
obs.defender_action_targets      # Dict[str, List[str]]
obs.fraudster_action_targets     # Dict[str, List[str]]

# Step observation (adds)
obs.defender_reward       # float
obs.fraudster_reward      # float
obs.info["grade"]         # dict — episode grade at terminal step
obs.done                  # bool
obs.reason                # "max_steps" | "all_routes_blocked" | "fraud_threshold_exceeded"
```

### Defender observation fields

```python
defender_obs = {
  "step": int,
  "fraud_family_hint": str,       # noisy hint (80% accurate)
  "accounts": [{
    "id", "transaction_velocity", "refund_ratio",
    "device_reuse_count", "risk_score",  # noisy
    "is_frozen", "is_challenged", "is_monitored",
    "balance_range",
  }],
  "merchants": [{
    "id", "transaction_count", "refund_rate",
    "anomaly_score",   # noisy
    "is_blocked", "is_monitored",
  }],
  "alerts": [{"type", "account_id"/"merchant_id", "step", ...}],
  "aggregate": {
    "total_frozen", "total_monitored", "total_flagged", "blocked_merchants",
    "recent_transaction_count",
  },
}
```

### Fraudster observation fields

```python
fraudster_obs = {
  "step": int,
  "alert_level": float,           # 0=safe, 1=detected
  "delayed_steps_remaining": int,
  "active_routes": [{
    "id", "family", "detection_pressure",
    "mule_status": [{"id", "is_frozen", "balance", "device_id"}],
    "merchant_blocked", "total_laundered", "cashout_ready", "steps_active",
  }],
  "available_mule_ids": List[str],
  "any_cashout_ready": bool,
  "total_laundered_so_far": float,
}
```

---

## Episode grading metrics (terminal only)

| Metric | Description |
|--------|-------------|
| `total_fraud_prevented` | Amount of fraud stopped |
| `total_fraud_escaped` | Amount laundered |
| `false_positive_count` | Legit accounts/merchants wrongly actioned |
| `false_positive_rate` | FP / (FP + TP) |
| `detection_delay` | Average step of first true-positive action |
| `customer_friction_score` | Proportion of legit users disrupted |
| `merchant_disruption_score` | Proportion of legit merchants blocked |
| `defender_score` | Composite 0–1 |
| `fraudster_score` | Composite 0–1 |

Grading is saved to `episode_grades.csv`.  Full rollout history (per-step
observations, actions, rewards) is saved to `rollout_history.csv` for
fine-tuning.

---

## Project structure

```
scam_detection/
├── server/
│   ├── app.py                   # FastAPI server
│   └── fraud_environment.py     # FraudEnvironment (OpenEnv interface)
├── models.py                    # FraudAction, FraudObservation
├── constants.py                 # All constants
├── hidden_world_state.py        # Dataclasses for hidden world
├── scenario_generator.py        # ScenarioGenerator (world gen per family)
├── observation_generator.py     # ObservationGenerator (partial views)
├── action_processor.py          # ActionMasker + ActionProcessor
├── transition_engine.py         # TransitionEngine (world evolution)
├── reward_engine.py             # RewardEngine (per-step rewards)
├── termination_engine.py        # TerminationEngine (episode end)
├── grading_engine.py            # GradingEngine (episode metrics + CSV)
├── policy_networks.py           # DefenderPolicy + FraudsterPolicy (PPO)
├── ppo_trainer.py               # PPOTrainer (online training loop)
├── baseline_detector.py         # BaselineRuleDetector (non-RL baseline)
├── client.py                    # FraudEnvClient (typed async client)
├── inference.py                 # LLM agent runner (both agents)
├── train.py                     # PPO training script
├── evaluate.py                  # Evaluation + baseline comparison
├── tests/
│   └── test_smoke.py            # Smoke tests
├── pyproject.toml
├── Dockerfile
└── README.md
```

---

## OpenEnv lifecycle

```python
from scam_detection.client import FraudEnvClient
from scam_detection.models import FraudAction, DefenderActionType, FraudsterActionType

async with FraudEnvClient(base_url="http://localhost:8000") as env:
    # 1. Reset
    result = await env.reset(task_name="mule_cashout", seed=42)
    obs    = result.observation

    # 2. Steps
    for _ in range(20):
        action = FraudAction(
            defender_action=DefenderActionType.MONITOR,
            defender_target=obs.defender_action_targets.get("monitor", [None])[0],
            fraudster_action=FraudsterActionType.CASHOUT_ATTEMPT,
            fraudster_target=(obs.fraudster_action_targets.get("cashout_attempt") or [None])[0],
        )
        result = await env.step(action)
        obs    = result.observation
        if obs.done:
            print("Episode done:", obs.reason)
            print("Grade:", obs.info.get("grade"))
            break
```

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | OpenAI-compatible LLM endpoint |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` / `API_KEY` | Authentication key |
| `ENV_URL` | Environment server URL (default: http://localhost:8000) |
| `PORT` | Server port (default: 8000) |
