"""
ScenarioGenerator — creates a randomised hidden world for each episode reset.

Supports four fraud families plus a random picker:
  - refund_abuse        : users exploit refund workflows with device reuse
  - mule_cashout        : money flows through layered mule accounts
  - merchant_collusion  : fake/colluding merchant drains card data
  - account_takeover    : compromised legitimate accounts make large transfers

Each ``generate()`` call produces a fresh ``HiddenWorldState`` that is
deterministic when the same seed is provided.
"""
from __future__ import annotations

import random
import uuid
from typing import List, Optional

from scam_detection.constants import (
    FRAUD_FAMILY_CONFIG,
    MAX_DEVICES,
    MAX_MERCHANTS,
    MAX_MULES,
    MAX_USERS,
    MIN_DEVICES,
    MIN_MERCHANTS,
    MIN_MULES,
    MIN_USERS,
    VALID_TASK_NAMES,
)
from scam_detection.hidden_world_state import (
    DeviceNode,
    FraudRoute,
    HiddenWorldState,
    MerchantNode,
    UserNode,
)


class ScenarioGenerator:
    """
    Generates a complete randomised fraud world for each episode.

    The generator is configurable: new fraud families can be added by
    creating a method ``_build_<family>_topology`` and registering it in
    ``_FAMILY_BUILDERS``.
    """

    def generate(
        self,
        fraud_family: str,
        seed: Optional[int] = None,
        max_steps: int = 20,
    ) -> HiddenWorldState:
        """
        Generate a fresh hidden world.

        Parameters
        ----------
        fraud_family:
            One of VALID_TASK_NAMES.  Pass "random" to pick one at random.
        seed:
            RNG seed for reproducibility.  ``None`` uses a random seed.
        max_steps:
            Episode length cap.
        """
        if fraud_family == "random":
            families = [f for f in VALID_TASK_NAMES if f != "random"]
            fraud_family = random.choice(families)

        if fraud_family not in FRAUD_FAMILY_CONFIG:
            raise ValueError(
                f"Unknown fraud family '{fraud_family}'. "
                f"Valid: {list(FRAUD_FAMILY_CONFIG.keys())}"
            )

        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        rng = random.Random(seed)
        cfg = FRAUD_FAMILY_CONFIG[fraud_family]

        episode_id = str(uuid.UUID(int=rng.getrandbits(128), version=4))

        # ── Generate base nodes ───────────────────────────────────────────────
        n_users     = rng.randint(MIN_USERS, MAX_USERS)
        n_merchants = rng.randint(MIN_MERCHANTS, MAX_MERCHANTS)
        n_devices   = rng.randint(MIN_DEVICES, MAX_DEVICES)
        n_mules     = max(MIN_MULES, int(n_users * cfg["mule_ratio"]))
        n_mules     = min(n_mules, MAX_MULES)

        devices   = self._make_devices(n_devices, rng)
        merchants = self._make_merchants(n_merchants, cfg, rng)
        users     = self._make_users(n_users, n_mules, devices, rng)

        aggressiveness = rng.uniform(*cfg["aggressiveness_range"])

        world = HiddenWorldState(
            episode_id=episode_id,
            step=0,
            fraud_family=fraud_family,
            seed=seed,
            users=users,
            merchants=merchants,
            devices=devices,
            fraud_aggressiveness=aggressiveness,
            max_steps=max_steps,
        )

        # ── Build fraud topology ──────────────────────────────────────────────
        builder = {
            "refund_abuse":        self._build_refund_abuse,
            "mule_cashout":        self._build_mule_cashout,
            "merchant_collusion":  self._build_merchant_collusion,
            "account_takeover":    self._build_account_takeover,
        }[fraud_family]

        builder(world, rng, cfg)

        # ── Set available routes ──────────────────────────────────────────────
        world.available_routes = [r.id for r in world.fraud_routes if r.is_active]

        return world

    # =========================================================================
    # Node factories
    # =========================================================================

    def _make_devices(self, count: int, rng: random.Random) -> dict:
        devices = {}
        for i in range(count):
            did = f"device_{i:03d}"
            devices[did] = DeviceNode(id=did)
        return devices

    def _make_merchants(self, count: int, cfg: dict, rng: random.Random) -> dict:
        merchants = {}
        n_colluding = max(0, int(count * cfg["colluding_merchant_ratio"]))
        for i in range(count):
            mid = f"merchant_{i:03d}"
            is_colluding = i < n_colluding
            merchants[mid] = MerchantNode(
                id=mid,
                is_colluding=is_colluding,
                refund_rate=rng.uniform(0.4, 0.9) if is_colluding else rng.uniform(0.0, 0.1),
                anomaly_score=rng.uniform(0.5, 1.0) if is_colluding else rng.uniform(0.0, 0.3),
            )
        return merchants

    def _make_users(
        self, count: int, n_mules: int, devices: dict, rng: random.Random
    ) -> dict:
        device_ids = list(devices.keys())
        users = {}
        for i in range(count):
            uid = f"user_{i:03d}"
            is_mule = i < n_mules
            # mules are more likely to share devices
            device_id = rng.choice(device_ids[:max(1, len(device_ids) // 2)]) \
                if is_mule else rng.choice(device_ids)
            balance = rng.uniform(100.0, 500.0) if is_mule else rng.uniform(200.0, 5000.0)
            users[uid] = UserNode(
                id=uid,
                is_mule=is_mule,
                device_id=device_id,
                balance=balance,
                risk_score=rng.uniform(0.4, 0.9) if is_mule else rng.uniform(0.0, 0.2),
            )
            devices[device_id].associated_users.append(uid)
            devices[device_id].reuse_count = len(devices[device_id].associated_users)
        return users

    # =========================================================================
    # Fraud topology builders
    # =========================================================================

    def _build_refund_abuse(
        self, world: HiddenWorldState, rng: random.Random, cfg: dict
    ) -> None:
        """
        Refund-abuse: mule users make purchases then immediately refund them,
        slowly draining merchant refund budgets and extracting value.
        """
        mule_ids  = [uid for uid, u in world.users.items() if u.is_mule]
        legit_ids = [uid for uid, u in world.users.items() if not u.is_mule]
        merchants = list(world.merchants.keys())

        # Mark mules as fraudster-controlled
        for uid in mule_ids:
            world.users[uid].is_fraudster_controlled = True

        n_routes = rng.randint(1, max(1, len(mule_ids) // 2))
        for i in range(n_routes):
            route_mules = rng.sample(mule_ids, min(2, len(mule_ids)))
            target_users = rng.sample(legit_ids, min(3, len(legit_ids)))
            route = FraudRoute(
                id=f"route_{i:03d}",
                family="refund_abuse",
                mule_ids=route_mules,
                merchant_id=rng.choice(merchants),
                target_user_ids=target_users,
            )
            world.fraud_routes.append(route)

    def _build_mule_cashout(
        self, world: HiddenWorldState, rng: random.Random, cfg: dict
    ) -> None:
        """
        Mule cashout: stolen funds are layered through multiple mule accounts
        before being withdrawn.
        """
        mule_ids  = [uid for uid, u in world.users.items() if u.is_mule]
        legit_ids = [uid for uid, u in world.users.items() if not u.is_mule]
        merchants = list(world.merchants.keys())

        for uid in mule_ids:
            world.users[uid].is_fraudster_controlled = True

        n_routes = rng.randint(1, max(1, len(mule_ids) // 2))
        for i in range(n_routes):
            chain_len = rng.randint(2, min(4, len(mule_ids)))
            chain = rng.sample(mule_ids, chain_len)
            targets = rng.sample(legit_ids, min(2, len(legit_ids)))
            route = FraudRoute(
                id=f"route_{i:03d}",
                family="mule_cashout",
                mule_ids=chain,
                merchant_id=rng.choice(merchants) if merchants else None,
                target_user_ids=targets,
            )
            world.fraud_routes.append(route)

    def _build_merchant_collusion(
        self, world: HiddenWorldState, rng: random.Random, cfg: dict
    ) -> None:
        """
        Merchant collusion: a colluding merchant processes fake transactions.
        Victim accounts are drained via inflated charges.
        """
        mule_ids       = [uid for uid, u in world.users.items() if u.is_mule]
        legit_ids      = [uid for uid, u in world.users.items() if not u.is_mule]
        colluding_mids = [mid for mid, m in world.merchants.items() if m.is_colluding]

        if not colluding_mids:
            # Ensure at least one colluding merchant
            mid = list(world.merchants.keys())[0]
            world.merchants[mid].is_colluding = True
            world.merchants[mid].refund_rate  = rng.uniform(0.4, 0.9)
            world.merchants[mid].anomaly_score = rng.uniform(0.6, 1.0)
            colluding_mids = [mid]

        for uid in mule_ids:
            world.users[uid].is_fraudster_controlled = True

        n_routes = len(colluding_mids)
        for i, merchant_id in enumerate(colluding_mids):
            targets = rng.sample(legit_ids, min(4, len(legit_ids)))
            route = FraudRoute(
                id=f"route_{i:03d}",
                family="merchant_collusion",
                mule_ids=mule_ids[:2] if mule_ids else [],
                merchant_id=merchant_id,
                target_user_ids=targets,
            )
            world.fraud_routes.append(route)

    def _build_account_takeover(
        self, world: HiddenWorldState, rng: random.Random, cfg: dict
    ) -> None:
        """
        Account takeover: the fraudster compromises legitimate accounts and
        makes large rapid transfers to mule accounts.
        """
        mule_ids  = [uid for uid, u in world.users.items() if u.is_mule]
        legit_ids = [uid for uid, u in world.users.items() if not u.is_mule]
        merchants = list(world.merchants.keys())

        n_compromised = rng.randint(1, min(3, len(legit_ids)))
        compromised   = rng.sample(legit_ids, n_compromised)
        for uid in compromised:
            world.users[uid].is_fraudster_controlled = True
            world.users[uid].takeover_step = 0  # taken over at episode start
            # Switch device to signal unusual device
            all_devs = list(world.devices.keys())
            new_dev  = rng.choice(all_devs)
            world.users[uid].device_id = new_dev

        for uid in mule_ids:
            world.users[uid].is_fraudster_controlled = True

        n_routes = rng.randint(1, max(1, n_compromised))
        for i in range(n_routes):
            victim   = compromised[i % len(compromised)]
            cashout_mules = rng.sample(mule_ids, min(2, len(mule_ids)))
            route = FraudRoute(
                id=f"route_{i:03d}",
                family="account_takeover",
                mule_ids=cashout_mules,
                merchant_id=rng.choice(merchants) if merchants else None,
                target_user_ids=[victim],
            )
            world.fraud_routes.append(route)
