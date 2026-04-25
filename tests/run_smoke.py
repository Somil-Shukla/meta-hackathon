"""Quick smoke test script (no pytest dependency)."""
import sys
sys.path.insert(0, 'c:\\Users\\somil\\hackathon_filan\\meta-hackathon')

from scam_detection.server.fraud_environment import FraudEnvironment
from scam_detection.models import FraudAction, DefenderActionType, FraudsterActionType
from scam_detection.constants import VALID_TASK_NAMES, DEFAULT_MAX_STEPS

env = FraudEnvironment()

# Test 1: All fraud families
families = [t for t in VALID_TASK_NAMES if t != 'random']
for family in families:
    obs = env.reset(task_name=family, seed=7)
    assert obs.task_name == family
    routes = len((obs.fraudster_obs or {}).get('active_routes', []))
    print(f"  Family {family}: OK - routes={routes}")

# Test 2: Full episode with cashout
obs = env.reset(task_name='mule_cashout', seed=42)
for i in range(DEFAULT_MAX_STEPS + 5):
    if obs.done:
        break
    frd_targets = (obs.fraudster_action_targets or {}).get('cashout_attempt', [])
    frd_target = frd_targets[0] if frd_targets else None
    fa = FraudsterActionType.CASHOUT_ATTEMPT if frd_target else FraudsterActionType.DO_NOTHING
    action = FraudAction(
        defender_action=DefenderActionType.DO_NOTHING,
        fraudster_action=fa,
        fraudster_target=frd_target,
    )
    obs = env.step(action)

assert obs.done, "Episode must end"
grade = (obs.info or {}).get('grade', {})
print(f"  Episode ended: reason={obs.reason}, step={obs.step}")
print(f"  Defender score={grade.get('defender_score')} fraudster={grade.get('fraudster_score')}")

# Test 3: Freeze action produces reward
obs = env.reset(task_name='mule_cashout', seed=42)
freeze_targets = (obs.defender_action_targets or {}).get('freeze', [])
if freeze_targets:
    action = FraudAction(
        defender_action=DefenderActionType.FREEZE,
        defender_target=freeze_targets[0],
        fraudster_action=FraudsterActionType.DO_NOTHING,
    )
    obs2 = env.step(action)
    assert isinstance(obs2.defender_reward, float)
    print(f"  Freeze action reward: {obs2.defender_reward}")

# Test 4: Deterministic seed
env2 = FraudEnvironment()
obs_a = env.reset(task_name='refund_abuse', seed=999)
obs_b = env2.reset(task_name='refund_abuse', seed=999)
assert obs_a.episode_id == obs_b.episode_id, "Same seed must give same episode"
print("  Deterministic seed: OK")

# Test 5: Observations differ between agents
assert obs_a.defender_obs != obs_a.fraudster_obs
print("  Agent observations differ: OK")

print("\nALL SMOKE TESTS PASSED")
