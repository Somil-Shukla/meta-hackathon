"""
ActionProcessor and ActionMasker for the Fraud Detection RL Environment.

ActionMasker generates legal action sets for each agent at each step.
ActionProcessor applies validated actions to the hidden world state and
returns a structured result indicating what happened.

Design:
  - Action types are fixed (enum values).
  - Action targets are dynamic, generated from the current world state.
  - Only legal actions (non-empty target lists) are reported as available.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from scam_detection.hidden_world_state import HiddenWorldState
from scam_detection.models import DefenderActionType, FraudsterActionType


@dataclass
class ActionResult:
    """Result of applying one agent's action to the world state."""
    success: bool
    action_type: str
    target: Optional[str]
    effect: str                             # human-readable description
    reward_signal: float = 0.0             # immediate raw signal (for reward engine)
    details: Dict[str, Any] = field(default_factory=dict)


class ActionMasker:
    """
    Computes which actions are legal and what targets are available.

    Called at the start of each step so agents can select from a valid set.
    """

    def defender_legal_actions(
        self, world: HiddenWorldState
    ) -> Dict[str, List[str]]:
        """
        Returns ``{action_type: [target_id, ...]}`` for all legal defender
        actions.  An action is legal only when at least one valid target exists.
        ``do_nothing`` is always legal (empty target list = self).
        """
        legal: Dict[str, List[str]] = {}

        # Accounts that can be monitored (not already monitored)
        monitorable = [
            uid for uid, u in world.users.items()
            if not u.is_monitored and not u.is_frozen
        ]
        if monitorable:
            legal[DefenderActionType.MONITOR.value] = monitorable

        # Accounts that can be challenged (not already challenged, not frozen)
        challengeable = [
            uid for uid, u in world.users.items()
            if not u.is_challenged and not u.is_frozen
        ]
        if challengeable:
            legal[DefenderActionType.CHALLENGE.value] = challengeable

        # Accounts that can be frozen (not already frozen)
        freezable = [
            uid for uid, u in world.users.items()
            if not u.is_frozen
        ]
        if freezable:
            legal[DefenderActionType.FREEZE.value] = freezable

        # Pending transactions that can be held
        holdable = [
            tx.id for tx in world.transactions
            if not tx.is_held and not tx.is_completed and tx.timestamp == world.step
        ]
        if holdable:
            legal[DefenderActionType.HOLD.value] = holdable

        # Merchants that can be blocked
        blockable = [
            mid for mid, m in world.merchants.items()
            if not m.is_blocked
        ]
        if blockable:
            legal[DefenderActionType.BLOCK_MERCHANT.value] = blockable

        # Accounts with connections to investigate (monitored or flagged accounts)
        investigatable = [
            uid for uid in world.flagged_accounts
            if uid in world.users
        ] + [
            uid for uid in world.monitored_accounts
            if uid in world.users
        ]
        investigatable = list(set(investigatable))
        if investigatable:
            legal[DefenderActionType.INVESTIGATE_NEIGHBORHOOD.value] = investigatable

        # do_nothing is always legal
        legal[DefenderActionType.DO_NOTHING.value] = ["self"]

        return legal

    def fraudster_legal_actions(
        self, world: HiddenWorldState
    ) -> Dict[str, List[str]]:
        """
        Returns ``{action_type: [target_id, ...]}`` for all legal fraudster
        actions.
        """
        legal: Dict[str, List[str]] = {}

        active_mules = [
            uid for uid, u in world.users.items()
            if u.is_mule and u.is_fraudster_controlled and not u.is_frozen
        ]

        active_routes = [r for r in world.fraud_routes if r.is_active]
        available_merchants = [
            mid for mid, m in world.merchants.items()
            if not m.is_blocked
        ]
        all_devices = list(world.devices.keys())

        # Split payment — any active mule
        if active_mules:
            legal[FraudsterActionType.SPLIT_PAYMENT.value] = active_mules

        # Rotate mule — only if multiple mules available
        alt_mules = [
            uid for uid, u in world.users.items()
            if u.is_mule and u.is_fraudster_controlled
            and not u.is_frozen
        ]
        if len(alt_mules) >= 2:
            legal[FraudsterActionType.ROTATE_MULE.value] = alt_mules

        # Switch merchant
        if available_merchants and active_routes:
            legal[FraudsterActionType.SWITCH_MERCHANT.value] = available_merchants

        # Rotate device
        if all_devices and active_mules:
            legal[FraudsterActionType.ROTATE_DEVICE.value] = all_devices

        # Delay — always legal if not already delayed
        if world.fraudster_delayed_steps == 0:
            legal[FraudsterActionType.DELAY.value] = ["self"]

        # Refund abuse — relevant for refund_abuse and mule_cashout families
        if world.fraud_family in ("refund_abuse", "merchant_collusion") and active_mules:
            legal[FraudsterActionType.REFUND_ABUSE.value] = active_mules

        # Cashout attempt — requires at least one ready route
        cashout_targets = []
        for route in active_routes:
            any_unfrozen = any(
                not world.users[mid].is_frozen
                for mid in route.mule_ids
                if mid in world.users
            )
            merch_blocked = False
            if route.merchant_id:
                m = world.merchants.get(route.merchant_id)
                if m:
                    merch_blocked = m.is_blocked
            merchant_ok = not merch_blocked
            if any_unfrozen and merchant_ok:
                cashout_targets.append(route.id)

        if cashout_targets and world.fraudster_delayed_steps == 0:
            legal[FraudsterActionType.CASHOUT_ATTEMPT.value] = cashout_targets

        # do_nothing always legal
        legal[FraudsterActionType.DO_NOTHING.value] = ["self"]

        return legal


class ActionProcessor:
    """
    Applies a single agent's action to the hidden world state.

    Returns an ``ActionResult`` describing the outcome and providing a
    raw reward signal that the ``RewardEngine`` uses to compute final rewards.
    """

    def apply_fraudster_action(
        self,
        action_type: str,
        target: Optional[str],
        world: HiddenWorldState,
    ) -> ActionResult:
        """Apply fraudster action; returns structured result."""
        dispatch = {
            FraudsterActionType.SPLIT_PAYMENT.value:   self._fraudster_split_payment,
            FraudsterActionType.ROTATE_MULE.value:     self._fraudster_rotate_mule,
            FraudsterActionType.SWITCH_MERCHANT.value: self._fraudster_switch_merchant,
            FraudsterActionType.ROTATE_DEVICE.value:   self._fraudster_rotate_device,
            FraudsterActionType.DELAY.value:           self._fraudster_delay,
            FraudsterActionType.REFUND_ABUSE.value:    self._fraudster_refund_abuse,
            FraudsterActionType.CASHOUT_ATTEMPT.value: self._fraudster_cashout_attempt,
            FraudsterActionType.DO_NOTHING.value:      self._do_nothing,
        }
        handler = dispatch.get(action_type, self._do_nothing)
        return handler(target, world)

    def apply_defender_action(
        self,
        action_type: str,
        target: Optional[str],
        world: HiddenWorldState,
    ) -> ActionResult:
        """Apply defender action; returns structured result."""
        dispatch = {
            DefenderActionType.MONITOR.value:                  self._defender_monitor,
            DefenderActionType.CHALLENGE.value:                self._defender_challenge,
            DefenderActionType.FREEZE.value:                   self._defender_freeze,
            DefenderActionType.HOLD.value:                     self._defender_hold,
            DefenderActionType.BLOCK_MERCHANT.value:           self._defender_block_merchant,
            DefenderActionType.INVESTIGATE_NEIGHBORHOOD.value: self._defender_investigate,
            DefenderActionType.DO_NOTHING.value:               self._do_nothing,
        }
        handler = dispatch.get(action_type, self._do_nothing)
        return handler(target, world)

    # =========================================================================
    # Fraudster action handlers
    # =========================================================================

    def _fraudster_split_payment(
        self, target: Optional[str], world: HiddenWorldState
    ) -> ActionResult:
        user = world.users.get(target or "")
        if not user or user.is_frozen:
            return ActionResult(
                success=False, action_type="split_payment", target=target,
                effect="target frozen or not found", reward_signal=-0.1,
            )
        # Splitting reduces per-transaction anomaly score
        amount = min(user.balance * 0.1, 50.0)
        user.balance -= amount
        user.transaction_count += 1
        world.fraud_attempts += 1
        world.total_fraud_amount += amount
        return ActionResult(
            success=True, action_type="split_payment", target=target,
            effect=f"split payment of {amount:.2f} from {target}",
            reward_signal=0.05,
            details={"amount": amount},
        )

    def _fraudster_rotate_mule(
        self, target: Optional[str], world: HiddenWorldState
    ) -> ActionResult:
        if not target:
            return ActionResult(
                success=False, action_type="rotate_mule", target=None,
                effect="no target specified", reward_signal=0.0,
            )
        for route in world.fraud_routes:
            if route.is_active and target in route.mule_ids:
                route.detection_pressure *= 0.7  # evasion reduces pressure
                return ActionResult(
                    success=True, action_type="rotate_mule", target=target,
                    effect=f"rotated to mule {target}, reduced detection pressure",
                    reward_signal=0.1,
                )
        return ActionResult(
            success=False, action_type="rotate_mule", target=target,
            effect="mule not found in active routes", reward_signal=0.0,
        )

    def _fraudster_switch_merchant(
        self, target: Optional[str], world: HiddenWorldState
    ) -> ActionResult:
        merchant = world.merchants.get(target or "")
        if not merchant or merchant.is_blocked:
            return ActionResult(
                success=False, action_type="switch_merchant", target=target,
                effect="merchant blocked or not found", reward_signal=-0.1,
            )
        for route in world.fraud_routes:
            if route.is_active:
                route.merchant_id = target
                route.detection_pressure *= 0.8
        return ActionResult(
            success=True, action_type="switch_merchant", target=target,
            effect=f"switched to merchant {target}",
            reward_signal=0.1,
        )

    def _fraudster_rotate_device(
        self, target: Optional[str], world: HiddenWorldState
    ) -> ActionResult:
        device = world.devices.get(target or "")
        if not device:
            return ActionResult(
                success=False, action_type="rotate_device", target=target,
                effect="device not found", reward_signal=0.0,
            )
        # Assign new device to a random controlled mule to reduce risk
        mules = [
            uid for uid, u in world.users.items()
            if u.is_fraudster_controlled and u.is_mule and not u.is_frozen
        ]
        if mules:
            import random
            chosen = random.choice(mules)
            world.users[chosen].device_id = target
            world.fraudster_alert_level = max(0.0, world.fraudster_alert_level - 0.1)
        return ActionResult(
            success=True, action_type="rotate_device", target=target,
            effect=f"rotated to device {target}",
            reward_signal=0.08,
        )

    def _fraudster_delay(
        self, target: Optional[str], world: HiddenWorldState
    ) -> ActionResult:
        world.fraudster_delayed_steps = 2  # pause activity for 2 steps
        world.fraudster_alert_level = max(0.0, world.fraudster_alert_level - 0.15)
        return ActionResult(
            success=True, action_type="delay", target=None,
            effect="fraudster lying low for 2 steps",
            reward_signal=0.05,
        )

    def _fraudster_refund_abuse(
        self, target: Optional[str], world: HiddenWorldState
    ) -> ActionResult:
        user = world.users.get(target or "")
        if not user or user.is_frozen:
            return ActionResult(
                success=False, action_type="refund_abuse", target=target,
                effect="target frozen or not found", reward_signal=-0.15,
            )
        amount = min(user.balance * 0.15, 100.0)
        user.refund_count += 1
        user.transaction_count += 1
        world.fraud_attempts += 1
        world.total_fraud_amount += amount * 0.5  # partial value from refund
        return ActionResult(
            success=True, action_type="refund_abuse", target=target,
            effect=f"refund abuse of {amount:.2f} via {target}",
            reward_signal=0.15,
            details={"amount": amount},
        )

    def _fraudster_cashout_attempt(
        self, target: Optional[str], world: HiddenWorldState
    ) -> ActionResult:
        # target is a route ID
        route = next((r for r in world.fraud_routes if r.id == target and r.is_active), None)
        if not route:
            return ActionResult(
                success=False, action_type="cashout_attempt", target=target,
                effect="route not found or inactive", reward_signal=-0.2,
            )
        # Check if any mule in route is unfrozen
        usable_mules = [
            mid for mid in route.mule_ids
            if mid in world.users and not world.users[mid].is_frozen
        ]
        merchant_ok = (
            route.merchant_id is None
            or not world.merchants.get(route.merchant_id, type("", (), {"is_blocked": True})()).is_blocked
        )
        if not usable_mules or not merchant_ok:
            world.fraud_attempts += 1
            return ActionResult(
                success=False, action_type="cashout_attempt", target=target,
                effect="cashout failed — mules frozen or merchant blocked",
                reward_signal=-0.3,
            )
        # Successful cashout
        amount = sum(
            world.users[mid].balance * 0.5
            for mid in usable_mules
            if mid in world.users
        )
        for mid in usable_mules:
            if mid in world.users:
                world.users[mid].balance *= 0.5
        route.total_laundered += amount
        world.total_laundered += amount
        world.total_fraud_amount += amount
        world.successful_fraud += 1
        world.fraud_attempts += 1
        world.fraudster_alert_level = min(1.0, world.fraudster_alert_level + 0.1)
        return ActionResult(
            success=True, action_type="cashout_attempt", target=target,
            effect=f"cashout of {amount:.2f} via route {target}",
            reward_signal=1.0,
            details={"amount": amount, "route_id": target},
        )

    # =========================================================================
    # Defender action handlers
    # =========================================================================

    def _defender_monitor(
        self, target: Optional[str], world: HiddenWorldState
    ) -> ActionResult:
        user = world.users.get(target or "")
        if not user:
            return ActionResult(
                success=False, action_type="monitor", target=target,
                effect="account not found", reward_signal=0.0,
            )
        user.is_monitored = True
        world.monitored_accounts.add(target)
        # Increase detection pressure on any route using this account
        for route in world.fraud_routes:
            if target in route.mule_ids:
                route.detection_pressure = min(1.0, route.detection_pressure + 0.2)
        world.fraudster_alert_level = min(1.0, world.fraudster_alert_level + 0.05)
        return ActionResult(
            success=True, action_type="monitor", target=target,
            effect=f"monitoring {target}",
            reward_signal=0.05,
        )

    def _defender_challenge(
        self, target: Optional[str], world: HiddenWorldState
    ) -> ActionResult:
        user = world.users.get(target or "")
        if not user or user.is_frozen:
            return ActionResult(
                success=False, action_type="challenge", target=target,
                effect="account not found or already frozen", reward_signal=0.0,
            )
        user.is_challenged = True
        # Challenged accounts cannot transact until resolved (delayed effect)
        for route in world.fraud_routes:
            if target in route.mule_ids:
                route.detection_pressure = min(1.0, route.detection_pressure + 0.3)
        world.flagged_accounts.add(target)
        return ActionResult(
            success=True, action_type="challenge", target=target,
            effect=f"challenge sent to {target}",
            reward_signal=0.1,
            details={"is_mule": user.is_mule},
        )

    def _defender_freeze(
        self, target: Optional[str], world: HiddenWorldState
    ) -> ActionResult:
        user = world.users.get(target or "")
        if not user:
            return ActionResult(
                success=False, action_type="freeze", target=target,
                effect="account not found", reward_signal=0.0,
            )
        if user.is_frozen:
            return ActionResult(
                success=False, action_type="freeze", target=target,
                effect="account already frozen", reward_signal=0.0,
            )
        user.is_frozen = True
        world.flagged_accounts.add(target)

        is_fraud = user.is_mule or user.is_fraudster_controlled
        if is_fraud:
            world.prevented_fraud += 1
            # Disable routes using this mule
            for route in world.fraud_routes:
                if target in route.mule_ids:
                    route.detection_pressure = min(1.0, route.detection_pressure + 0.5)
            world.fraudster_alert_level = min(1.0, world.fraudster_alert_level + 0.2)
            return ActionResult(
                success=True, action_type="freeze", target=target,
                effect=f"froze fraudulent account {target}",
                reward_signal=1.0,  # fraud prevented
                details={"is_fraud": True},
            )
        else:
            world.false_positives += 1
            world.legitimate_users_frozen += 1
            return ActionResult(
                success=True, action_type="freeze", target=target,
                effect=f"froze legitimate account {target} (false positive)",
                reward_signal=-0.5,  # false positive penalty
                details={"is_fraud": False},
            )

    def _defender_hold(
        self, target: Optional[str], world: HiddenWorldState
    ) -> ActionResult:
        tx = next(
            (t for t in world.transactions if t.id == target and not t.is_held),
            None,
        )
        if not tx:
            return ActionResult(
                success=False, action_type="hold", target=target,
                effect="transaction not found or already held", reward_signal=0.0,
            )
        tx.is_held = True
        if tx.is_fraud:
            world.prevented_fraud += 1
            world.total_prevented += tx.amount
            return ActionResult(
                success=True, action_type="hold", target=target,
                effect=f"held fraudulent transaction {target}",
                reward_signal=0.8,
                details={"is_fraud": True, "amount": tx.amount},
            )
        else:
            world.false_positives += 1
            return ActionResult(
                success=True, action_type="hold", target=target,
                effect=f"held legitimate transaction {target} (false positive)",
                reward_signal=-0.3,
                details={"is_fraud": False, "amount": tx.amount},
            )

    def _defender_block_merchant(
        self, target: Optional[str], world: HiddenWorldState
    ) -> ActionResult:
        merchant = world.merchants.get(target or "")
        if not merchant or merchant.is_blocked:
            return ActionResult(
                success=False, action_type="block_merchant", target=target,
                effect="merchant not found or already blocked", reward_signal=0.0,
            )
        merchant.is_blocked = True
        is_colluding = merchant.is_colluding
        if is_colluding:
            world.prevented_fraud += 1
            for route in world.fraud_routes:
                if route.merchant_id == target:
                    route.detection_pressure = 1.0
            return ActionResult(
                success=True, action_type="block_merchant", target=target,
                effect=f"blocked colluding merchant {target}",
                reward_signal=0.8,
                details={"is_colluding": True},
            )
        else:
            world.false_positives += 1
            return ActionResult(
                success=True, action_type="block_merchant", target=target,
                effect=f"blocked legitimate merchant {target} (false positive)",
                reward_signal=-0.4,
                details={"is_colluding": False},
            )

    def _defender_investigate(
        self, target: Optional[str], world: HiddenWorldState
    ) -> ActionResult:
        user = world.users.get(target or "")
        if not user:
            return ActionResult(
                success=False, action_type="investigate_neighborhood", target=target,
                effect="account not found", reward_signal=0.0,
            )
        # Find connected accounts (same device or mule chain)
        connected: List[str] = []
        for uid, u in world.users.items():
            if uid != target and u.device_id == user.device_id:
                connected.append(uid)
                u.is_monitored = True
                world.monitored_accounts.add(uid)
                world.flagged_accounts.add(uid)

        for route in world.fraud_routes:
            if target in route.mule_ids:
                for mid in route.mule_ids:
                    if mid != target:
                        connected.append(mid)
                        if mid in world.users:
                            world.users[mid].is_monitored = True
                            world.monitored_accounts.add(mid)

        for route in world.fraud_routes:
            if target in route.mule_ids:
                route.detection_pressure = min(1.0, route.detection_pressure + 0.4)

        return ActionResult(
            success=True, action_type="investigate_neighborhood", target=target,
            effect=f"investigated neighborhood of {target}, flagged {len(connected)} connected accounts",
            reward_signal=0.15,
            details={"connected_accounts": connected},
        )

    def _do_nothing(
        self, target: Optional[str], world: HiddenWorldState
    ) -> ActionResult:
        return ActionResult(
            success=True, action_type="do_nothing", target=None,
            effect="no action taken", reward_signal=0.0,
        )
