"""
TransitionEngine — updates the hidden world state after each step.

Processing order (per step):
  1. Fraudster action is applied first.
  2. Defender action is applied second.
  3. World evolves: new transactions generated, risk scores updated,
     fraud routes age, delayed consequences take effect.

The engine also handles:
  - Delayed consequences (held transactions, challenged accounts)
  - Fraudster adaptation after heavy defender pressure
  - Risk score propagation (connected accounts inherit elevated risk)
"""
from __future__ import annotations

import random
import uuid
from typing import List

from scam_detection.constants import (
    MAX_TRANSACTIONS_PER_STEP,
    MIN_TRANSACTIONS_PER_STEP,
)
from scam_detection.hidden_world_state import (
    FraudRoute,
    HiddenWorldState,
    Transaction,
    UserNode,
)
from scam_detection.action_processor import ActionResult


class TransitionEngine:
    """
    Applies post-action world state updates for one step.

    Call ``advance(world, fraudster_result, defender_result)`` after both
    agents' actions have been processed by ``ActionProcessor``.
    """

    def advance(
        self,
        world: HiddenWorldState,
        fraudster_result: ActionResult,
        defender_result: ActionResult,
    ) -> None:
        """
        Advance the world state by one step.

        Steps:
          1. Decrement fraudster delay counter.
          2. Generate organic background transactions.
          3. Age fraud routes and escalate aggressiveness.
          4. Propagate risk scores.
          5. Generate defender alerts.
          6. Apply adaptation (fraudster reacts to high detection pressure).
          7. Check whether fraud routes are exhausted.
          8. Increment step counter.
        """
        rng = random.Random(world.seed ^ world.step ^ 0xABC)

        self._tick_delay(world)
        self._generate_transactions(world, rng)
        self._age_routes(world, rng)
        self._propagate_risk(world)
        self._generate_alerts(world, fraudster_result, defender_result)
        self._fraudster_adaptation(world, rng)
        self._deactivate_exhausted_routes(world)

        world.step += 1

    # =========================================================================
    # Sub-steps
    # =========================================================================

    def _tick_delay(self, world: HiddenWorldState) -> None:
        if world.fraudster_delayed_steps > 0:
            world.fraudster_delayed_steps -= 1

    def _generate_transactions(
        self, world: HiddenWorldState, rng: random.Random
    ) -> None:
        """Generate organic (background) transactions this step."""
        n = rng.randint(MIN_TRANSACTIONS_PER_STEP, MAX_TRANSACTIONS_PER_STEP)
        user_ids  = list(world.users.keys())
        merch_ids = list(world.merchants.keys())
        if not user_ids or not merch_ids:
            return

        for _ in range(n):
            user_id    = rng.choice(user_ids)
            merch_id   = rng.choice(merch_ids)
            user       = world.users[user_id]
            merchant   = world.merchants[merch_id]

            amount = rng.uniform(10.0, 200.0)
            if user.balance < amount:
                continue  # insufficient funds

            tx_id = f"tx_{world.step:03d}_{uuid.uuid4().hex[:6]}"
            is_fraud = user.is_fraudster_controlled and not user.is_frozen

            # Fraudster-controlled accounts make larger, more frequent transactions
            if is_fraud and world.fraudster_delayed_steps == 0:
                amount *= rng.uniform(2.0, 5.0)
                amount = min(amount, user.balance * 0.8)

            tx = Transaction(
                id=tx_id,
                from_user_id=user_id,
                to_merchant_id=merch_id,
                amount=round(amount, 2),
                timestamp=world.step,
                is_fraud=is_fraud,
                device_id=user.device_id,
                fraud_family=world.fraud_family if is_fraud else None,
            )
            user.balance -= amount
            user.transaction_count += 1
            merchant.transaction_count += 1
            merchant.total_volume += amount

            world.transactions.append(tx)

    def _age_routes(
        self, world: HiddenWorldState, rng: random.Random
    ) -> None:
        """Age active fraud routes and optionally escalate aggressiveness."""
        for route in world.fraud_routes:
            if not route.is_active:
                continue
            route.steps_active += 1

            # Gradually increase detection pressure from organic monitoring
            route.detection_pressure = min(
                1.0, route.detection_pressure + rng.uniform(0.01, 0.05)
            )

            # Escalate aggressiveness over time if alert level is low
            if world.fraudster_alert_level < 0.3 and route.steps_active % 5 == 0:
                world.fraud_aggressiveness = min(1.0, world.fraud_aggressiveness + 0.05)

    def _propagate_risk(self, world: HiddenWorldState) -> None:
        """
        Propagate elevated risk scores to accounts sharing a device or
        appearing in the same fraud route as known bad actors.
        """
        # Device-based propagation
        device_risk: dict = {}
        for uid, user in world.users.items():
            dev = user.device_id
            if user.risk_score > 0.5:
                device_risk[dev] = max(device_risk.get(dev, 0.0), user.risk_score * 0.5)

        for uid, user in world.users.items():
            inherited = device_risk.get(user.device_id, 0.0)
            if inherited > user.risk_score:
                user.risk_score = min(1.0, user.risk_score * 0.8 + inherited * 0.2)

        # Mule-network propagation
        for route in world.fraud_routes:
            if not route.is_active:
                continue
            for mid in route.mule_ids:
                if mid in world.users:
                    world.users[mid].risk_score = min(
                        1.0, world.users[mid].risk_score + 0.05
                    )

    def _generate_alerts(
        self,
        world: HiddenWorldState,
        fraudster_result: ActionResult,
        defender_result: ActionResult,
    ) -> None:
        """Generate alerts for the defender based on recent activity."""
        # Alert on high-velocity accounts
        step_txs = [tx for tx in world.transactions if tx.timestamp == world.step]
        velocity_map: dict = {}
        for tx in step_txs:
            velocity_map[tx.from_user_id] = velocity_map.get(tx.from_user_id, 0) + 1

        for uid, count in velocity_map.items():
            if count >= 3:
                world.defender_alerts.append({
                    "type": "high_velocity",
                    "account_id": uid,
                    "count": count,
                    "step": world.step,
                })

        # Alert on high device-reuse
        for uid, user in world.users.items():
            dev = world.devices.get(user.device_id)
            if dev and dev.reuse_count >= 4:
                world.defender_alerts.append({
                    "type": "device_reuse",
                    "account_id": uid,
                    "device_id": user.device_id,
                    "reuse_count": dev.reuse_count,
                    "step": world.step,
                })

        # Alert on high merchant refund rate
        for mid, merchant in world.merchants.items():
            if merchant.refund_rate > 0.4 and not merchant.is_blocked:
                world.defender_alerts.append({
                    "type": "high_refund_rate",
                    "merchant_id": mid,
                    "refund_rate": round(merchant.refund_rate, 3),
                    "step": world.step,
                })

        # Trim alerts to last 50
        if len(world.defender_alerts) > 50:
            world.defender_alerts = world.defender_alerts[-50:]

    def _fraudster_adaptation(
        self, world: HiddenWorldState, rng: random.Random
    ) -> None:
        """
        Model adaptive fraudster behaviour.

        If detection pressure is high on a route, the fraudster may
        spontaneously rotate mules or delay as a built-in adaptation
        (separate from the chosen action).
        """
        for route in world.fraud_routes:
            if not route.is_active:
                continue
            if route.detection_pressure > 0.7 and rng.random() < 0.3:
                # Adaptive: lower pressure by slightly rotating strategy
                route.detection_pressure *= 0.85
                world.fraudster_alert_level = min(
                    1.0, world.fraudster_alert_level + 0.05
                )

    def _deactivate_exhausted_routes(self, world: HiddenWorldState) -> None:
        """Deactivate fraud routes where all mules are frozen."""
        for route in world.fraud_routes:
            if not route.is_active:
                continue
            all_frozen = all(
                world.users.get(mid, UserNode(id="x", is_frozen=True)).is_frozen
                for mid in route.mule_ids
            )
            merchant_blocked = False
            if route.merchant_id:
                m = world.merchants.get(route.merchant_id)
                if m and m.is_blocked:
                    merchant_blocked = True
            if all_frozen or merchant_blocked:
                route.is_active = False

        world.available_routes = [r.id for r in world.fraud_routes if r.is_active]
