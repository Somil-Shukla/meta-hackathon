"""Quick verification of grpo_train.py without needing TRL or GPU."""
import sys
import types

# Stub torch
torch_mod = types.ModuleType("torch")
torch_mod.Tensor = object
torch_mod.zeros = lambda *a, **k: None
torch_mod.cat = lambda *a, **k: None
torch_nn_mod = types.ModuleType("torch.nn")
torch_nn_func = types.ModuleType("torch.nn.functional")
torch_nn_func.pad = lambda *a, **k: None
torch_nn_mod.functional = torch_nn_func
torch_mod.nn = torch_nn_mod
sys.modules["torch"] = torch_mod
sys.modules["torch.nn"] = torch_nn_mod
sys.modules["torch.nn.functional"] = torch_nn_func

# Stub datasets
datasets_mod = types.ModuleType("datasets")
class _FakeDataset(list):
    @classmethod
    def from_list(cls, rows):
        obj = cls(rows)
        return obj
    def __getitem__(self, idx):
        return list.__getitem__(self, idx)
datasets_mod.Dataset = _FakeDataset
sys.modules["datasets"] = datasets_mod

# Stub trl so imports work without GPU/TRL installed
trl_mod = types.ModuleType("trl")
trl_trainer_mod = types.ModuleType("trl.trainer")
trl_grpo_mod = types.ModuleType("trl.trainer.grpo_trainer")
trl_grpo_mod.generate_rollout_completions = lambda *a, **k: {}
trl_mod.GRPOConfig = object
trl_mod.GRPOTrainer = object
sys.modules["trl"] = trl_mod
sys.modules["trl.trainer"] = trl_trainer_mod
sys.modules["trl.trainer.grpo_trainer"] = trl_grpo_mod

sys.path.insert(0, "c:\\Users\\somil\\hackathon_filan\\meta-hackathon")

from scam_detection.grpo_train import (
    _DEFENDER_SYSTEM,
    _FRAUDSTER_SYSTEM,
    _parse_defender_action,
    _parse_fraudster_action,
    reward_format_valid,
    reward_action_legal,
    reward_def_episode,
    reward_frd_episode,
    reward_frd_evasion,
    build_training_dataset,
    BaselineFraudster,
)

print("All grpo_train imports OK")

# --- Dataset ---
ds = build_training_dataset(task="random", n_samples=8)
print(f"Dataset: {len(ds)} rows, first task={ds[0]['task_name']} seed={ds[0]['seed']}")

# --- Reward functions ---
comps = ["a", "b", "c", "d"]
r = reward_format_valid(comps, format_valids=[1.0, 0.5, 0.0, 1.0])
assert r == [1.0, 0.5, 0.0, 1.0], r
print("reward_format_valid:", r)

r = reward_action_legal(comps, action_legals=[1.0, 1.0, 0.0, 0.5])
assert r == [1.0, 1.0, 0.0, 0.5], r
print("reward_action_legal:", r)

r = reward_def_episode(comps, episode_rewards=[2.0, -1.0, 0.5, 1.5])
assert all(-1.0 <= v <= 1.0 for v in r), r
print("reward_def_episode (normalised):", r)

r = reward_frd_evasion(comps, alert_levels=[0.0, 0.5, 1.0, 0.2])
assert r[0] == 1.0 and r[2] == 0.0, r
print("reward_frd_evasion:", r)

# --- BaselineFraudster ---
bf = BaselineFraudster()
act, tgt = bf.select_action(
    {"alert_level": 0.3},
    ["cashout_attempt", "do_nothing"],
    {"cashout_attempt": ["route_000"]},
)
assert act == "cashout_attempt" and tgt == "route_000"
print(f"BaselineFraudster: {act} -> {tgt}")

# --- System prompts exist ---
assert "JSON" in _DEFENDER_SYSTEM
assert "JSON" in _FRAUDSTER_SYSTEM
print("System prompts: OK")

# --- Parse defender action ---
class FakeObs:
    available_defender_actions = ["freeze", "monitor", "do_nothing"]
    defender_action_targets = {"freeze": ["user_001"], "monitor": ["user_002"], "do_nothing": ["self"]}

obs = FakeObs()
a, t, is_json, is_legal = _parse_defender_action('{"action": "freeze", "target": "user_001"}', obs)
assert a == "freeze" and t == "user_001" and is_json and is_legal
print(f"parse_defender_action: {a} -> {t}  json={is_json} legal={is_legal}")

a, t, is_json, is_legal = _parse_defender_action("invalid text", obs)
assert not is_json and not is_legal
print(f"parse_defender_action fallback: {a}  json={is_json} legal={is_legal}")

# --- Parse fraudster action ---
class FakeObs2:
    available_fraudster_actions = ["cashout_attempt", "do_nothing"]
    fraudster_action_targets = {"cashout_attempt": ["route_000"], "do_nothing": ["self"]}

obs2 = FakeObs2()
a, t, is_json, is_legal = _parse_fraudster_action('{"action": "cashout_attempt", "target": "route_000"}', obs2)
assert a == "cashout_attempt" and t == "route_000" and is_json and is_legal
print(f"parse_fraudster_action: {a} -> {t}  json={is_json} legal={is_legal}")

print("\nALL grpo_train CHECKS PASSED")
