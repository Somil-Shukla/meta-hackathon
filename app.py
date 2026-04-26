"""
Fraud Battle Arena — Streamlit UI
===================================
Visual simulation of the Fraud Detection multi-agent environment.
Runs a full episode automatically once started, step by step.
"""
from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Battle Arena",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0c1018; }
[data-testid="stSidebar"]          { display: none; }
[data-testid="collapsedControl"]   { display: none; }
section.main > div { padding-top: 0; }
body, p, div { color: #d4dff5; }

.arena-title {
  text-align: center; font-size: clamp(22px,3.5vw,38px); font-weight: 900;
  letter-spacing: 5px; text-transform: uppercase;
  background: linear-gradient(90deg,#ff3366 0%,#aa44ff 50%,#00b4ff 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; margin: 0;
}
.arena-sub {
  text-align: center; color: #6677aa; font-size: 12px; letter-spacing: 2px;
  text-transform: uppercase; margin: 8px 0 6px;
}
.step-indicator {
  text-align: center; font-size: 13px; color: #8899cc; font-weight: 600;
  letter-spacing: 1px; margin: 4px 0 4px;
}
.ep-bar-wrap {
  background: #151c2e; border-radius: 20px; height: 7px;
  overflow: hidden; margin: 6px 0 2px; border: 1px solid #1e2a44;
}
.ep-bar-fill {
  height: 100%; border-radius: 20px;
  background: linear-gradient(90deg,#ff3366,#aa44ff,#00b4ff);
  transition: width 0.5s ease;
}

/* Agent cards */
.agent-card {
  border-radius: 16px; padding: 16px 16px 14px;
  border: 2px solid; transition: box-shadow 0.3s ease;
}
.frd-card { background: linear-gradient(145deg,#1e0a0e,#130c10); border-color: #8c1c28; box-shadow: 0 0 14px rgba(210,30,50,0.16); }
.def-card { background: linear-gradient(145deg,#061828,#0a1022); border-color: #006080; box-shadow: 0 0 14px rgba(0,180,255,0.14); }
.frd-card.active {
  border-color: #e02040 !important;
  box-shadow: 0 0 28px rgba(210,30,50,0.55), 0 0 6px rgba(210,30,50,0.28) inset;
  animation: glow-red 2s ease-in-out infinite;
}
.def-card.active {
  border-color: #00b4ff !important;
  box-shadow: 0 0 28px rgba(0,180,255,0.5), 0 0 6px rgba(0,180,255,0.25) inset;
  animation: glow-blue 2s ease-in-out infinite;
}
.agent-icon      { font-size: 44px; text-align: center; margin-bottom: 2px; }
.agent-icon-done { font-size: 56px; text-align: center; margin-bottom: 2px; }
.agent-name  { font-size: 16px; font-weight: 800; letter-spacing: 1px; text-align: center; margin-bottom: 6px; }
.frd-name    { color: #ff6670; }
.def-name    { color: #55ddff; }

.turn-badge {
  display: block; text-align: center; padding: 3px 0; border-radius: 20px;
  font-size: 11px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase;
  margin-bottom: 0;
}
.badge-active-red  { background: rgba(210,30,50,0.22); color:#e83050; border:1px solid #e02040; animation: pulse 1.4s infinite; }
.badge-active-blue { background: rgba(0,180,255,0.2); color:#22ccff; border:1px solid #00b4ff; animation: pulse 1.4s infinite; }
.badge-wait        { background: #121828; color:#3a4a6a; border:1px solid #1e2840; }
.badge-done        { background: rgba(85,0,136,0.3); color:#bb77ff; border:1px solid #7700bb; }

/* Agent card stat rows (stacked inside col2 of row1) */
.stat-stack { display:flex; flex-direction:column; gap:6px; }
.stat-row-item {
  background:#0e1428; border-radius:8px; border:1px solid #192038;
  padding:7px 12px; display:flex; justify-content:space-between; align-items:center;
}
.stat-val { font-size:16px; font-weight:800; line-height:1.1; }
.stat-lbl { font-size:11px; color:#5a6e90; text-transform:uppercase; letter-spacing:0.8px; }
.frd-col  { color:#ff3366; }
.def-col  { color:#00b4ff; }
.pos-col  { color:#00ff88; }
.neg-col  { color:#ff5555; }
.neu-col  { color:#445566; }

/* Action block */
.action-block {
  border-radius: 8px; padding: 10px 14px; margin-top: 10px; font-size: 14px;
}
.action-frd { background:rgba(210,30,50,0.12); border-left:4px solid #e02040; }
.action-def { background:rgba(0,180,255,0.12); border-left:4px solid #00b4ff; }
.action-name { font-weight:700; font-size:16px; color:#e8eeff; }
.action-tgt  { color:#7a8caa; font-size:13px; margin-top:3px; }
.action-desc { font-size:13px; color:#6677aa; margin-top:3px; }

/* Observation panel */
.obs-section-label {
  font-size:12px; color:#6688aa; text-transform:uppercase;
  letter-spacing:1.5px; margin-bottom:7px; font-weight:700;
}
.obs-card {
  background:#0c1220; border:1px solid #182035; border-radius:10px;
  padding:10px 14px; margin-bottom:7px;
}
.obs-row {
  display:flex; justify-content:space-between; align-items:center;
  padding:5px 0; border-bottom:1px solid #111828; font-size:14px;
}
.obs-row:last-child { border-bottom:none; }
.obs-key { color:#6699ff; }
.obs-val-str  { color:#88ffcc; }
.obs-val-num  { color:#ffcc44; font-weight:600; }
.obs-val-t    { color:#44ff99; }
.obs-val-f    { color:#ff6666; }
.obs-route {
  background:#0c1220; border:1px solid #182035; border-radius:6px;
  padding:7px 12px; margin:4px 0; font-size:13px;
}
.obs-route-id { font-weight:700; color:#ccdeff; font-size:14px; }

/* Reward display */
.reward-block {
  border-radius:10px; padding:10px 14px; text-align:center; border:2px solid;
}
.reward-pos { background:linear-gradient(135deg,#001f0f,#002a18); border-color:#00cc66; animation:pop 0.5s; }
.reward-neg { background:linear-gradient(135deg,#200000,#300000); border-color:#ff4444; animation:shake 0.4s; }
.reward-nil { background:#0a0d1f; border-color:#1a2040; }
.reward-val { font-size:26px; font-weight:900; line-height:1; }
.reward-lbl { font-size:10px; color:#556677; text-transform:uppercase; letter-spacing:1px; margin-top:2px; }

/* History */
.hist-wrap { max-height:280px; overflow-y:auto; }
.hist-row {
  display:flex; align-items:center; gap:8px;
  padding:6px 10px; border-radius:6px; margin:2px 0;
  font-size:13px; border-left:3px solid;
}
.hist-frd { background:rgba(210,30,50,0.08); border-color:#e02040; }
.hist-def { background:rgba(0,180,255,0.07); border-color:#00b4ff; }
.hist-sn  { color:#3a4a60; font-size:11px; min-width:32px; }
.hist-ag  { font-weight:700; min-width:78px; }
.hist-ac  { flex:1; }
.hist-rw  { font-weight:700; min-width:52px; text-align:right; font-size:13px; }

/* VS */
.vs-col { display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; }
.vs-txt { font-size:24px; font-weight:900; color:#222840; margin:4px 0; }
.vs-ln  { width:2px; flex:1; background:linear-gradient(180deg,transparent,#222840,transparent); }

/* Episode done */
.done-banner {
  border-radius:14px; padding:16px 20px; text-align:center;
  background:linear-gradient(135deg,#060c1e,#0c0722);
  border:1px solid #301f4a; margin-bottom:10px;
}
.done-title { font-size:17px; font-weight:900; letter-spacing:2px; color:#bb77ff; text-transform:uppercase; }
.done-grid  { display:flex; gap:12px; justify-content:center; flex-wrap:wrap; margin-top:12px; }
.done-cell  { background:#0e1428; border-radius:8px; padding:10px 18px; min-width:100px; text-align:center; border:1px solid #1e2840; }
.done-val   { font-size:22px; font-weight:800; }
.done-lbl   { font-size:10px; color:#5a6e90; text-transform:uppercase; letter-spacing:1px; margin-top:3px; }

/* Divider */
.sec-title {
  font-size:12px; font-weight:700; color:#4a5e80; text-transform:uppercase;
  letter-spacing:2px; margin:12px 0 7px; padding-bottom:5px;
  border-bottom:1px solid #182035;
}

/* Scrollbar */
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:#090d16; }
::-webkit-scrollbar-thumb { background:#1e2840; border-radius:2px; }

/* Animations */
@keyframes glow-red  { 0%,100%{box-shadow:0 0 18px rgba(210,30,50,0.38)} 50%{box-shadow:0 0 36px rgba(210,30,50,0.68)} }
@keyframes glow-blue { 0%,100%{box-shadow:0 0 18px rgba(0,180,255,0.35)}  50%{box-shadow:0 0 36px rgba(0,180,255,0.65)}  }
@keyframes pulse     { 0%,100%{opacity:1} 50%{opacity:0.55} }
@keyframes pop       { 0%{transform:scale(0.7);opacity:0} 60%{transform:scale(1.05)} 100%{transform:scale(1);opacity:1} }
@keyframes shake     { 0%,100%{transform:translateX(0)} 25%{transform:translateX(-4px)} 75%{transform:translateX(4px)} }
@keyframes fadeUp    { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility:hidden; }
div[data-testid="stDecoration"] { display:none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
TASK_NAMES = ["mule_cashout", "account_takeover", "merchant_collusion", "refund_abuse"]
TASK_EMOJI = {"mule_cashout": "💸", "account_takeover": "🕵️", "merchant_collusion": "🏪", "refund_abuse": "🔁"}

ACTION_EMOJI = {
    "split_payment": "💰", "rotate_mule": "🔄", "switch_merchant": "🏪",
    "rotate_device": "📱", "delay": "⏳", "refund_abuse": "🔁",
    "cashout_attempt": "💸", "do_nothing": "😶",
    "monitor": "👁️", "challenge": "🔐", "freeze": "❄️",
    "hold": "⏸️", "block_merchant": "🚫",
    "investigate_neighborhood": "🔍",
}
ACTION_DESC = {
    "split_payment": "Split payments to evade thresholds",
    "rotate_mule": "Switch to a different mule account",
    "switch_merchant": "Use a different merchant",
    "rotate_device": "Change device to reduce reuse risk",
    "delay": "Lay low — let detection pressure drop",
    "refund_abuse": "Exploit refund workflows",
    "cashout_attempt": "Attempt cash-out via fraud route",
    "do_nothing": "Pass this turn",
    "monitor": "Place account under surveillance",
    "challenge": "Send step-up authentication",
    "freeze": "Freeze account immediately",
    "hold": "Hold a suspicious transaction",
    "block_merchant": "Block a suspicious merchant",
    "investigate_neighborhood": "Flag all connected accounts",
}

# ─────────────────────────────────────────────────────────────────────────────
# Mock Simulator (offline demo)
# ─────────────────────────────────────────────────────────────────────────────
class MockSim:
    def __init__(self, task_name: str, max_steps: int = 20):
        self.task_name  = task_name
        self.max_steps  = max_steps
        self.step       = 0
        self.alert_level = random.uniform(0.25, 0.45)
        self.laundered  = 0.0
        self.frozen_count   = 0
        self.blocked_merchants = 0
        self.current_agent: Optional[str] = "fraudster"
        self.done = False
        n_acc   = random.randint(10, 16)
        n_mule  = random.randint(3, 5)
        n_mer   = random.randint(4, 7)
        n_route = random.randint(2, 4)
        n_dev   = random.randint(5, 8)
        self.accounts  = [f"user_{i:03d}" for i in range(n_acc)]
        self.mules     = [f"mule_{i:03d}" for i in range(n_mule)]
        self.merchants = [f"merchant_{i:03d}" for i in range(n_mer)]
        self.routes    = [f"route_{i:03d}" for i in range(n_route)]
        self.devices   = [f"device_{i:03d}" for i in range(n_dev)]
        self._pending_frd_action: str = "do_nothing"
        self._pending_frd_target: Optional[str] = None

    def _fraudster_obs(self) -> Dict[str, Any]:
        routes = []
        for r in self.routes:
            dp = round(min(1.0, max(0.0, self.alert_level + random.gauss(0, 0.12))), 3)
            routes.append({
                "id": r,
                "detection_pressure": dp,
                "cashout_ready": dp < 0.5 and random.random() > 0.4,
                "total_laundered": round(self.laundered / max(1, len(self.routes)) + random.uniform(0, 200), 2),
            })
        return {
            "alert_level": round(self.alert_level, 3),
            "delayed_steps_remaining": 0,
            "any_cashout_ready": any(r["cashout_ready"] for r in routes),
            "total_laundered_so_far": round(self.laundered, 2),
            "active_routes": routes,
        }

    def _defender_obs(self) -> Dict[str, Any]:
        sample = random.sample(self.accounts, min(8, len(self.accounts)))
        accs = []
        for a in sample:
            risk = round(min(1.0, max(0.0, self.alert_level * 0.75 + random.gauss(0, 0.22))), 3)
            accs.append({
                "id": a,
                "risk_score": risk,
                "transaction_velocity": random.randint(1, 24),
                "device_reuse_count": random.randint(0, 6),
                "is_frozen": random.random() < (self.frozen_count / max(1, len(self.accounts))),
            })
        alerts = []
        if self.alert_level > 0.45:
            for _ in range(random.randint(1, 3)):
                a = random.choice(self.accounts)
                kind = random.choice(["High-velocity txn", "Suspicious device reuse", "Large transfer"])
                alerts.append(f"{kind} on {a}")
        return {
            "fraud_family_hint": self.task_name,
            "accounts": sorted(accs, key=lambda x: x["risk_score"], reverse=True),
            "alerts": alerts,
            "aggregate": {
                "total_frozen": self.frozen_count,
                "total_monitored": random.randint(0, 5),
                "total_flagged": random.randint(0, 3),
                "blocked_merchants": self.blocked_merchants,
            },
        }

    def _choose_frd(self, obs: Dict) -> Tuple[str, Optional[str]]:
        al = obs["alert_level"]
        if obs["any_cashout_ready"] and al < 0.55:
            ready = [r["id"] for r in obs["active_routes"] if r["cashout_ready"]]
            return "cashout_attempt", (ready[0] if ready else self.routes[0])
        if al > 0.72:
            action = random.choice(["delay", "rotate_mule", "rotate_device"])
            target = (random.choice(self.mules)  if action == "rotate_mule"  else
                      random.choice(self.devices) if action == "rotate_device" else None)
            return action, target
        action = random.choice(["split_payment", "rotate_mule", "refund_abuse", "switch_merchant"])
        target = (random.choice(self.mules)     if action == "rotate_mule"     else
                  random.choice(self.merchants) if action == "switch_merchant" else
                  random.choice(self.accounts))
        return action, target

    def _choose_def(self, obs: Dict) -> Tuple[str, Optional[str]]:
        unfrozen = [a for a in obs["accounts"] if not a.get("is_frozen")]
        if not unfrozen:
            return "do_nothing", None
        top = unfrozen[0]
        risk, target = top["risk_score"], top["id"]
        if risk > 0.75 and top["device_reuse_count"] >= 3:
            return "freeze", target
        if risk > 0.6:
            return "challenge", target
        if risk > 0.42:
            return "monitor", target
        if self.blocked_merchants == 0 and len(obs["alerts"]) > 1 and random.random() > 0.5:
            return "block_merchant", random.choice(self.merchants)
        if len(obs["alerts"]) >= 2 and random.random() > 0.6:
            return "investigate_neighborhood", target
        return "do_nothing", None

    def _compute_rewards(self, frd_action: str, def_action: str) -> Tuple[float, float]:
        frd_r = def_r = 0.0
        if frd_action == "cashout_attempt":
            success = self.alert_level < 0.5 and random.random() > 0.38
            if success:
                frd_r += random.uniform(0.6, 1.0); self.laundered += random.uniform(400, 2500)
                def_r -= random.uniform(0.3, 0.6)
            else:
                frd_r -= random.uniform(0.2, 0.4)
        elif frd_action in ("split_payment", "refund_abuse"):
            if random.random() > 0.28:
                frd_r += 0.05; self.laundered += random.uniform(40, 350)
            else:
                frd_r -= 0.15
        elif frd_action in ("rotate_mule", "switch_merchant", "rotate_device", "delay"):
            if self.alert_level > 0.5:
                frd_r += 0.2; self.alert_level = max(0, self.alert_level - random.uniform(0.05, 0.14))
            else:
                frd_r -= 0.02
        if def_action == "freeze":
            if random.random() > 0.32:
                def_r += random.uniform(0.65, 1.0); frd_r -= random.uniform(0.35, 0.55)
                self.frozen_count += 1; self.alert_level = min(1.0, self.alert_level + 0.08)
            else:
                def_r -= 0.95
        elif def_action == "hold":
            def_r += (random.uniform(0.5, 0.8) if random.random() > 0.38 else -0.25)
        elif def_action == "block_merchant":
            if random.random() > 0.38:
                def_r += random.uniform(0.7, 0.95); frd_r -= 0.4; self.blocked_merchants += 1
            else:
                def_r -= 0.5
        elif def_action in ("monitor", "challenge", "investigate_neighborhood"):
            def_r -= 0.05
        self.alert_level = min(1.0, self.alert_level + random.uniform(-0.04, 0.13))
        return (round(max(-1.0, min(1.0, frd_r)), 3),
                round(max(-1.0, min(1.0, def_r)), 3))

    def half_step(self) -> Dict[str, Any]:
        agent = self.current_agent
        if agent == "fraudster":
            obs = self._fraudster_obs()
            action, target = self._choose_frd(obs)
            self._pending_frd_action = action
            self._pending_frd_target = target
            self.current_agent = "defender"
            return {"agent": "fraudster", "obs": obs, "action": action, "target": target,
                    "frd_reward": None, "def_reward": None, "step": self.step, "done": False, "grade": None}
        else:
            self.step += 1
            obs = self._defender_obs()
            action, target = self._choose_def(obs)
            frd_r, def_r = self._compute_rewards(self._pending_frd_action, action)
            done = (self.step >= self.max_steps or
                    (self.frozen_count >= len(self.routes) + 1 and random.random() > 0.7))
            self.done = done
            self.current_agent = None if done else "fraudster"
            return {"agent": "defender", "obs": obs, "action": action, "target": target,
                    "frd_reward": frd_r, "def_reward": def_r, "step": self.step,
                    "done": done, "grade": self._make_grade() if done else None}

    def _make_grade(self) -> Dict[str, Any]:
        fraud_prev = self.frozen_count * 1200 + self.blocked_merchants * 600
        fraud_esc  = max(0.0, self.laundered - fraud_prev)
        def_score  = round(min(1.0, self.frozen_count / max(1, len(self.routes)) * 0.8 + self.blocked_merchants * 0.1), 3)
        frd_score  = round(min(1.0, self.laundered / 12000), 3)
        return {"defender_score": def_score, "fraudster_score": frd_score,
                "total_fraud_prevented": round(fraud_prev, 2), "total_fraud_escaped": round(fraud_esc, 2),
                "false_positive_count": random.randint(0, 3)}


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
def init_state() -> None:
    defaults = dict(
        started=False, done=False, running=False,
        task_name="mule_cashout", max_steps=20,
        step=0, current_agent=None, sim=None,
        # per-turn
        last_agent=None,
        frd_obs=None, def_obs=None,
        frd_action=None, frd_target=None,
        def_action=None, def_target=None,
        frd_reward=None, def_reward=None,
        # accumulators
        total_frd=0.0, total_def=0.0,
        history=[],
        grade=None,
        # pending fraudster action (waiting for defender half-step)
        pending_frd_action=None, pending_frd_target=None,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def do_start() -> None:
    task  = st.session_state.task_name
    max_s = st.session_state.max_steps
    sim   = MockSim(task, max_s)
    st.session_state.update(
        started=True, done=False, running=True,
        step=0, sim=sim, current_agent="fraudster",
        last_agent=None,
        frd_obs=None, def_obs=None,
        frd_action=None, frd_target=None,
        def_action=None, def_target=None,
        frd_reward=None, def_reward=None,
        total_frd=0.0, total_def=0.0,
        history=[], grade=None,
        pending_frd_action=None, pending_frd_target=None,
    )


def do_half_step() -> None:
    sim = st.session_state.sim
    if not sim or sim.done:
        st.session_state.running = False
        return
    r = sim.half_step()
    agent = r["agent"]
    st.session_state.last_agent = agent
    st.session_state.step = r["step"]
    st.session_state.done = r["done"]
    st.session_state.current_agent = sim.current_agent

    if agent == "fraudster":
        st.session_state.frd_obs    = r["obs"]
        st.session_state.frd_action = r["action"]
        st.session_state.frd_target = r["target"]
        st.session_state.pending_frd_action = r["action"]
        st.session_state.pending_frd_target = r["target"]
        # Keep previous round's rewards visible during fraudster's turn
    else:
        st.session_state.def_obs    = r["obs"]
        st.session_state.def_action = r["action"]
        st.session_state.def_target = r["target"]
        fr, dr = r["frd_reward"], r["def_reward"]
        st.session_state.frd_reward = fr
        st.session_state.def_reward = dr
        if fr is not None:
            st.session_state.total_frd += fr
        if dr is not None:
            st.session_state.total_def += dr
        st.session_state.history.append({
            "step": r["step"], "agent": agent,
            "frd_action": st.session_state.pending_frd_action,
            "frd_target": st.session_state.pending_frd_target,
            "def_action": r["action"], "def_target": r["target"],
            "frd_reward": fr, "def_reward": dr,
        })
        if r.get("grade"):
            st.session_state.grade = r["grade"]
    if r["done"]:
        st.session_state.running = False


# ─────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _reward_color(v: Optional[float]) -> str:
    if v is None: return "#445566"
    return "#00ff88" if v > 0 else ("#ff5555" if v < 0 else "#445566")


def _reward_str(v: Optional[float]) -> str:
    if v is None: return "—"
    return f"+{v:.3f}" if v > 0 else f"{v:.3f}"


def render_agent_card(
    agent: str,
    is_active: bool,
    action: Optional[str],
    target: Optional[str],
    step_reward: Optional[float],
    total_reward: float,
    step_num: int,
    extra_stat_label: str,
    extra_stat_value: int,
) -> None:
    is_frd  = agent == "fraudster"
    card_cls = ("frd-card" if is_frd else "def-card") + (" active" if is_active else "")
    name_cls = "frd-name" if is_frd else "def-name"
    icon     = "🦹" if is_frd else "🛡️"
    name     = "FRAUDSTER" if is_frd else "DEFENDER"

    done = st.session_state.done
    if done:
        badge = ""
        icon_cls = "agent-icon-done"
    elif is_active:
        badge_cls = "badge-active-red" if is_frd else "badge-active-blue"
        badge = f'<span class="turn-badge {badge_cls}">● ACTIVE</span>'
        icon_cls = "agent-icon"
    else:
        badge = '<span class="turn-badge badge-wait">Waiting</span>'
        icon_cls = "agent-icon"

    tr_color    = _reward_color(total_reward if total_reward != 0.0 else None)
    sr_color    = _reward_color(step_reward)
    tr_str      = f"+{total_reward:.2f}" if total_reward > 0 else f"{total_reward:.2f}"
    sr_str      = _reward_str(step_reward)
    agent_color = "#e02040" if is_frd else "#00b4ff"

    # ── Row 1: icon+name+badge  |  stacked stats ────────────────────────────
    row1 = (
        f'<div style="display:flex;gap:14px;align-items:stretch;margin-bottom:10px;">'
        # col1: identity
        f'<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;'
        f'min-width:90px;max-width:110px;">'
        f'<div class="{icon_cls}">{icon}</div>'
        f'<div class="agent-name {name_cls}">{name}</div>'
        f'{badge}'
        f'</div>'
        # col2: stats stacked
        f'<div class="stat-stack" style="flex:1;">'
        f'<div class="stat-row-item">'
        f'<span class="stat-lbl">Total Reward</span>'
        f'<span class="stat-val" style="color:{tr_color};">{tr_str}</span>'
        f'</div>'
        f'<div class="stat-row-item">'
        f'<span class="stat-lbl">Last Step</span>'
        f'<span class="stat-val" style="color:{sr_color};">{sr_str}</span>'
        f'</div>'
        f'<div class="stat-row-item">'
        f'<span class="stat-lbl">{extra_stat_label}</span>'
        f'<span class="stat-val" style="color:{agent_color};">{extra_stat_value}</span>'
        f'</div>'
        f'</div>'
        f'</div>'
    )

    # ── Row 2: action block (bigger font via CSS classes) ────────────────────
    action_part = ""
    if action:
        ac_css   = "action-frd" if is_frd else "action-def"
        emoji    = ACTION_EMOJI.get(action, "❓")
        desc     = ACTION_DESC.get(action, "")
        tgt_part = f'<div class="action-tgt">&#127919; {target}</div>' if target else ""
        action_part = (
            f'<div class="action-block {ac_css}">'
            f'<div class="action-name">{emoji} {action.replace("_"," ").title()}</div>'
            f'{tgt_part}'
            f'<div class="action-desc">{desc}</div>'
            f'</div>'
        )

    # ── Row 3: step reward ───────────────────────────────────────────────────
    reward_part = ""
    if step_reward is not None:
        rv  = _reward_str(step_reward)
        rc  = _reward_color(step_reward)
        rbg = "rgba(210,30,50,0.1)" if is_frd else "rgba(0,180,255,0.1)"
        rbd = "#e0204055" if is_frd else "#00b4ff55"
        reward_part = (
            f'<div style="margin-top:10px;padding:11px 14px;border-radius:8px;'
            f'text-align:center;background:{rbg};border:1px solid {rbd};">'
            f'<div style="font-size:26px;font-weight:900;color:{rc};">{rv}</div>'
            f'<div style="font-size:11px;color:#6677aa;text-transform:uppercase;letter-spacing:1px;margin-top:3px;">'
            f'{"&#129399;" if is_frd else "&#128737;"} Step Reward</div>'
            f'</div>'
        )

    html = (
        f'<div class="agent-card {card_cls}">'
        f'{row1}'
        f'{action_part}'
        f'{reward_part}'
        f'</div>'
    )
    st.html(html)


def render_fraudster_obs(obs: Optional[Dict]) -> None:
    if not obs:
        st.html('<div style="color:#2a3050;font-size:12px;padding:10px;">Waiting for first observation&#8230;</div>')
        return
    al      = obs.get("alert_level", 0)
    al_col  = "#ff5555" if al > 0.7 else ("#ffcc44" if al > 0.4 else "#00ff88")
    launch  = obs.get("total_laundered_so_far", 0)
    cashout = obs.get("any_cashout_ready", False)

    rows = f"""
    <div class="obs-card">
      <div class="obs-row"><span class="obs-key">Alert Level</span>
        <span style="color:{al_col};font-weight:700;">{al:.3f}</span></div>
      <div class="obs-row"><span class="obs-key">Total Laundered</span>
        <span class="obs-val-num">${launch:,.2f}</span></div>
      <div class="obs-row"><span class="obs-key">Any Cashout Ready</span>
        <span class="{'obs-val-t' if cashout else 'obs-val-f'}">{'YES ⚡' if cashout else 'No'}</span></div>
    </div>"""

    st.html(
        f'<div class="obs-section-label">&#128308; Fraudster Observation</div>'
        f'{rows}'
    )


def render_defender_obs(obs: Optional[Dict]) -> None:
    if not obs:
        st.html('<div style="color:#2a3050;font-size:12px;padding:10px;">Waiting for first observation&#8230;</div>')
        return
    agg    = obs.get("aggregate", {})

    agg_html = f"""
    <div class="obs-card">
      <div class="obs-row"><span class="obs-key">Fraud Family</span>
        <span class="obs-val-str">{obs.get("fraud_family_hint","?")}</span></div>
      <div class="obs-row"><span class="obs-key">Frozen Accounts</span>
        <span class="obs-val-num">{agg.get("total_frozen",0)}</span></div>
      <div class="obs-row"><span class="obs-key">Monitored</span>
        <span class="obs-val-num">{agg.get("total_monitored",0)}</span></div>
      <div class="obs-row"><span class="obs-key">Blocked Merchants</span>
        <span class="obs-val-num">{agg.get("blocked_merchants",0)}</span></div>
    </div>"""

    st.html(
        f'<div class="obs-section-label">&#128309; Defender Observation</div>'
        f'{agg_html}'
    )


def render_history() -> None:
    hist = st.session_state.history
    if not hist:
        st.html('<div style="color:#1a2040;font-size:12px;padding:6px 0;">History will appear here as the simulation runs&#8230;</div>')
        return
    rows = ""
    for e in reversed(hist[-30:]):
        sn   = e["step"]
        fa   = e.get("frd_action","?")
        da   = e.get("def_action","?")
        ft   = e.get("frd_target","")
        dt   = e.get("def_target","")
        fr   = e.get("frd_reward")
        dr   = e.get("def_reward")
        fe   = ACTION_EMOJI.get(fa,"&#10067;")
        de   = ACTION_EMOJI.get(da,"&#10067;")
        fr_s = _reward_str(fr); fr_c = _reward_color(fr)
        dr_s = _reward_str(dr); dr_c = _reward_color(dr)
        ft_s = f" &#8594; {ft}" if ft else ""
        dt_s = f" &#8594; {dt}" if dt else ""
        rows += (
            f'<div style="display:flex;gap:8px;padding:6px 10px;border-radius:6px;'
            f'background:#0c1220;margin:2px 0;border:1px solid #182035;font-size:13px;align-items:center;">'
            f'<span style="color:#3a4a60;min-width:28px;">S{sn}</span>'
            f'<span style="color:#ff6688;min-width:130px;">{fe} {fa.replace("_"," ")}{ft_s}</span>'
            f'<span style="color:#44aad4;flex:1;">{de} {da.replace("_"," ")}{dt_s}</span>'
            f'<span style="color:{fr_c};min-width:52px;text-align:right;font-weight:700;">{fr_s}</span>'
            f'<span style="color:{dr_c};min-width:52px;text-align:right;font-weight:700;">{dr_s}</span>'
            f'</div>'
        )
    header = (
        '<div style="display:flex;gap:8px;padding:4px 10px;font-size:11px;color:#3a4a60;'
        'text-transform:uppercase;letter-spacing:1px;">'
        '<span style="min-width:28px;">Step</span>'
        '<span style="min-width:130px;">Fraudster Action</span>'
        '<span style="flex:1;">Defender Action</span>'
        '<span style="min-width:52px;text-align:right;">F-Rwd</span>'
        '<span style="min-width:52px;text-align:right;">D-Rwd</span>'
        '</div>'
    )
    st.html(f'<div style="max-height:260px;overflow-y:auto;">{header}{rows}</div>')


def render_episode_done() -> None:
    grade = st.session_state.grade
    def_s = (grade or {}).get("defender_score", round(st.session_state.total_def / max(1, st.session_state.step), 3))
    frd_s = (grade or {}).get("fraudster_score", round(st.session_state.total_frd / max(1, st.session_state.step), 3))
    prev  = (grade or {}).get("total_fraud_prevented", 0)
    esc   = (grade or {}).get("total_fraud_escaped", 0)
    fp    = (grade or {}).get("false_positive_count", 0)

    st.html(
        f'<div class="done-banner">'
        f'<div class="done-title">&#9989; Episode Complete &#8212; {st.session_state.step} Steps</div>'
        f'<div class="done-grid">'
        f'<div class="done-cell"><div class="done-val" style="color:#00b4ff;">{def_s:.3f}</div><div class="done-lbl">Defender Score</div></div>'
        f'<div class="done-cell"><div class="done-val" style="color:#ff3366;">{frd_s:.3f}</div><div class="done-lbl">Fraudster Score</div></div>'
        f'<div class="done-cell"><div class="done-val" style="color:#00ff88;">${prev:,.0f}</div><div class="done-lbl">Fraud Prevented</div></div>'
        f'<div class="done-cell"><div class="done-val" style="color:#ff5555;">${esc:,.0f}</div><div class="done-lbl">Fraud Escaped</div></div>'
        f'<div class="done-cell"><div class="done-val" style="color:#ffcc44;">{fp}</div><div class="done-lbl">False Positives</div></div>'
        f'<div class="done-cell"><div class="done-val" style="color:#aa66ff;">{st.session_state.total_def:+.2f}</div><div class="done-lbl">Total D-Reward</div></div>'
        f'</div></div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    init_state()

    ss    = st.session_state
    step  = ss.step
    max_s = ss.max_steps
    pct   = int(step / max_s * 100) if max_s else 0

    # ── Header ────────────────────────────────────────────────────────────────
    step_indicator = ""
    if ss.running and not ss.done:
        step_indicator = f'<div class="step-indicator">Step {step} / {max_s}</div>'

    st.markdown(f"""
    <div style="text-align:center;padding:0 0 6px;">
      <p class="arena-title" style="padding:0 0 2px;">⚔️ Fraud Battle Environment</p>
      <p class="arena-sub" style="padding:0 0 14px;">Multi-Agent Financial Crime Simulation</p>
      {step_indicator}
      <div class="ep-bar-wrap"><div class="ep-bar-fill" style="width:{pct}%;"></div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Episode done banner ───────────────────────────────────────────────────
    if ss.done:
        render_episode_done()

    # ── Agent cards ───────────────────────────────────────────────────────────
    current = ss.current_agent
    col_frd, col_vs, col_def = st.columns([10, 1, 10])

    with col_frd:
        frd_ac = ss.frd_action if ss.last_agent == "fraudster" else (
            ss.pending_frd_action if ss.last_agent == "defender" else None)
        frd_tg = ss.frd_target if ss.last_agent == "fraudster" else (
            ss.pending_frd_target if ss.last_agent == "defender" else None)
        cashout_count = sum(1 for h in ss.history if h.get("frd_action") == "cashout_attempt")
        render_agent_card(
            agent="fraudster",
            is_active=(current == "fraudster" and ss.started and not ss.done),
            action=frd_ac, target=frd_tg,
            step_reward=ss.frd_reward, total_reward=ss.total_frd,
            step_num=step,
            extra_stat_label="Cashouts",
            extra_stat_value=cashout_count,
        )

    with col_vs:
        st.markdown("""
        <div class="vs-col" style="min-height:200px;">
          <div class="vs-ln"></div>
          <div class="vs-txt">VS</div>
          <div class="vs-ln"></div>
        </div>""", unsafe_allow_html=True)

    with col_def:
        freeze_count = sum(1 for h in ss.history if h.get("def_action") == "freeze")
        render_agent_card(
            agent="defender",
            is_active=(current == "defender" and ss.started and not ss.done),
            action=ss.def_action,
            target=ss.def_target,
            step_reward=ss.def_reward, total_reward=ss.total_def,
            step_num=step,
            extra_stat_label="Freezes",
            extra_stat_value=freeze_count,
        )

    # ── Observations ──────────────────────────────────────────────────────────
    st.html('<div class="sec-title">Current Observations</div>')
    obs_col_frd, obs_col_def = st.columns(2)
    with obs_col_frd:
        render_fraudster_obs(ss.frd_obs)
    with obs_col_def:
        render_defender_obs(ss.def_obs)

    # ── History ───────────────────────────────────────────────────────────────
    st.html('<div class="sec-title">Step History</div>')
    render_history()

    # ── Controls ──────────────────────────────────────────────────────────────

    if not ss.started and not ss.done:
        # Task + max_steps selectors inline (no sidebar)
        cfg1, cfg2, cfg3 = st.columns([4, 3, 2])
        with cfg1:
            st.selectbox(
                "Fraud Scenario",
                TASK_NAMES,
                format_func=lambda x: f"{TASK_EMOJI.get(x,'🎲')} {x.replace('_',' ').title()}",
                key="task_name",
            )
        with cfg2:
            st.slider("Max Steps", min_value=5, max_value=40, step=5, key="max_steps")
        with cfg3:
            st.html("<div style='height:28px'></div>")
            if st.button("▶ Start Episode", type="primary", use_container_width=True):
                do_start()
                st.rerun()

    elif ss.done:
        if st.button("🔄 Run New Episode", type="primary"):
            # Fully reset
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    # ── Auto-run loop ─────────────────────────────────────────────────────────
    if ss.running and not ss.done:
        time.sleep(0.7)
        do_half_step()
        st.rerun()


if __name__ == "__main__":
    main()
