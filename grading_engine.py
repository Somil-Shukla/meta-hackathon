"""
GradingEngine — computes episode-level evaluation metrics after rollout.

This is separate from per-step RL rewards.  Grading is for reporting,
fine-tuning, and comparing policies.

Episode metrics:
  - total_fraud_prevented
  - total_fraud_escaped
  - false_positive_count
  - false_positive_rate
  - detection_delay (average steps to first freeze of a fraud account)
  - customer_friction_score
  - merchant_disruption_score
  - defender_score  (composite 0-1)
  - fraudster_score (composite 0-1)

Rollout CSV columns (per episode, for fine-tuning):
  episode_id, fraud_family, step, defender_action, fraudster_action,
  defender_reward, fraudster_reward, cumulative_defender_reward,
  cumulative_fraudster_reward, observation_summary, grade metrics ...
"""
from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from scam_detection.constants import GRADING_CSV_PATH, ROLLOUT_CSV_PATH
from scam_detection.hidden_world_state import HiddenWorldState


@dataclass
class EpisodeGrade:
    """All grading metrics for one completed episode."""
    episode_id: str
    fraud_family: str
    total_steps: int

    total_fraud_prevented: float
    total_fraud_escaped: float
    false_positive_count: int
    false_positive_rate: float
    detection_delay: float          # average steps to first true-positive freeze
    customer_friction_score: float  # 0=no friction, 1=maximum friction
    merchant_disruption_score: float

    defender_score: float           # composite 0-1
    fraudster_score: float          # composite 0-1

    cumulative_defender_reward: float
    cumulative_fraudster_reward: float


@dataclass
class StepRecord:
    """One row in the rollout CSV (one step of one episode)."""
    episode_id: str
    fraud_family: str
    step: int
    defender_action: str
    defender_target: Optional[str]
    fraudster_action: str
    fraudster_target: Optional[str]
    defender_reward: float
    fraudster_reward: float
    cumulative_defender_reward: float
    cumulative_fraudster_reward: float
    total_frozen: int
    total_laundered: float
    active_routes: int
    fraudster_alert_level: float
    # Observation summaries (JSON-encoded strings for CSV)
    defender_obs_summary: str
    fraudster_obs_summary: str
    # Grade at end of episode (filled in after rollout)
    defender_score: Optional[float] = None
    fraudster_score: Optional[float] = None
    episode_done: bool = False
    termination_reason: Optional[str] = None


class GradingEngine:
    """
    Grades a completed episode and optionally saves results to CSV.

    Usage::

        grader = GradingEngine()
        grade = grader.grade(world, step_records)
        grader.save_grade(grade)
        grader.save_rollout(step_records)
    """

    def grade(
        self,
        world: HiddenWorldState,
        step_records: List[StepRecord],
        cumulative_defender: float,
        cumulative_fraudster: float,
    ) -> EpisodeGrade:
        """Compute all episode-level metrics from the final world state."""
        total_frozen_fraud = world.prevented_fraud
        total_fraud_escaped = world.successful_fraud
        false_positives = world.false_positives
        total_legit_frozen = world.legitimate_users_frozen

        # False positive rate = FP / (FP + TP freezes)
        total_interventions = total_frozen_fraud + false_positives
        fp_rate = (
            round(false_positives / total_interventions, 4)
            if total_interventions > 0 else 0.0
        )

        # Detection delay — average step of first true-positive action
        # We approximate from step_records: look for defender freeze on fraud accounts
        detection_delays = []
        for rec in step_records:
            if rec.defender_action == "freeze" and rec.defender_reward > 0.5:
                detection_delays.append(rec.step)
        avg_delay = (
            round(sum(detection_delays) / len(detection_delays), 2)
            if detection_delays else float(world.max_steps)
        )

        # Customer friction = proportion of legit users frozen
        total_legit = sum(1 for u in world.users.values() if not u.is_mule)
        friction = round(total_legit_frozen / max(1, total_legit), 4)

        # Merchant disruption = proportion of legit merchants blocked
        total_legit_merchants = sum(1 for m in world.merchants.values() if not m.is_colluding)
        disrupted_legit = sum(
            1 for m in world.merchants.values()
            if not m.is_colluding and m.is_blocked
        )
        disruption = round(disrupted_legit / max(1, total_legit_merchants), 4)

        # Defender score: combination of precision (low FP), recall (fraud caught), and speed
        precision = 1.0 - fp_rate
        total_fraud_accounts = sum(
            1 for u in world.users.values() if u.is_mule or u.is_fraudster_controlled
        )
        recall = (
            round(total_frozen_fraud / max(1, total_fraud_accounts), 4)
            if total_fraud_accounts > 0 else 0.0
        )
        speed_bonus = max(0.0, 1.0 - avg_delay / world.max_steps)
        defender_score = round(
            0.4 * precision + 0.4 * recall + 0.2 * speed_bonus - 0.1 * disruption, 4
        )
        defender_score = max(0.0, min(1.0, defender_score))

        # Fraudster score: how much was laundered vs total possible
        total_possible = world.total_fraud_amount if world.total_fraud_amount > 0 else 1.0
        launder_rate = round(min(1.0, world.total_laundered / total_possible), 4)
        evasion_bonus = max(0.0, 1.0 - world.fraudster_alert_level)
        fraudster_score = round(0.7 * launder_rate + 0.3 * evasion_bonus, 4)

        return EpisodeGrade(
            episode_id=world.episode_id,
            fraud_family=world.fraud_family,
            total_steps=world.step,
            total_fraud_prevented=round(world.total_prevented, 2),
            total_fraud_escaped=round(world.total_laundered, 2),
            false_positive_count=false_positives,
            false_positive_rate=fp_rate,
            detection_delay=avg_delay,
            customer_friction_score=friction,
            merchant_disruption_score=disruption,
            defender_score=defender_score,
            fraudster_score=fraudster_score,
            cumulative_defender_reward=round(cumulative_defender, 4),
            cumulative_fraudster_reward=round(cumulative_fraudster, 4),
        )

    def save_grade(
        self, grade: EpisodeGrade, path: str = GRADING_CSV_PATH
    ) -> None:
        """Append episode grade to the grading CSV."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        file_exists = os.path.isfile(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(grade).keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(asdict(grade))

    def save_rollout(
        self,
        step_records: List[StepRecord],
        grade: EpisodeGrade,
        path: str = ROLLOUT_CSV_PATH,
    ) -> None:
        """Append all step records for the episode to the rollout CSV.

        Also fills in the episode-level grade fields on each row.
        """
        if not step_records:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        file_exists = os.path.isfile(path)

        for rec in step_records:
            rec.defender_score = grade.defender_score
            rec.fraudster_score = grade.fraudster_score

        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(step_records[0]).keys()))
            if not file_exists:
                writer.writeheader()
            for rec in step_records:
                writer.writerow(asdict(rec))
