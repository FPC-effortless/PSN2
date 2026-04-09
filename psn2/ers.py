"""Experience Replay Substrate (ERS) — four-tier memory with VSA retrieval.

PRD spec:
  Tiers: Working (128), Episodic (100k), Semantic (10k), Procedural (5k)
  Working eviction: FIFO
  Episodic eviction: decay_state < 0.05 AND reuse_count < 2
  Semantic eviction: contradiction audit or explicit retirement
  Procedural: offline 8-gate pipeline only; strict versioning
  Promotion: Working->Episodic if utility>0.40 at session end OR >0.70 mid-session
             Episodic->Semantic if cross-family transfer evidence >= 3 AND utility > 0.65
  Utility: 0.50*verifier_weight + 0.25*efficiency_gain + 0.15*motif_bonus - 0.05*veto - 0.10*(revision>3)
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import torch
import torch.nn.functional as F

from .vsa import cosine

TAU_ERS = 0.45
DECAY_PER_EPISODE = 0.001
WORKING_CAPACITY = 128
EPISODIC_CAPACITY = 10000
SEMANTIC_CAPACITY = 10000
PROCEDURAL_CAPACITY = 5000


@dataclass
class ERSTuple:
    input_vsa: torch.Tensor
    trace_vsa: torch.Tensor
    output_vsa: torch.Tensor
    source: str                          # "own"|"social_observation"|"inner_speech"|"curiosity_resolution"
    utility_score: float
    decay_state: float = 1.0
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_family: str = "default"
    pulse_count: int = 0
    regime_trace: str = ""
    veto_count: int = 0
    revision_count: int = 0
    motif_ids: List[str] = field(default_factory=list)
    tier: str = "Working"
    verifier_verdict: str = "pass"       # "pass"|"fail"|"partial"
    reuse_count: int = 0
    content_hash: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.content_hash:
            raw = self.input_vsa.detach().cpu().numpy().tobytes()
            self.content_hash = hashlib.sha256(raw).hexdigest()[:16]

    def decay(self):
        self.decay_state = max(0.0, self.decay_state - DECAY_PER_EPISODE)


def compute_utility(verifier_verdict: str, pulse_count: int, baseline_pulses: int,
                    motif_ids: List[str], veto_count: int, revision_count: int) -> float:
    """PRD utility formula."""
    verifier_weight = {"pass": 1.0, "partial": 0.3, "fail": -1.0}.get(verifier_verdict, 0.0)
    efficiency_gain = max(0.0, (baseline_pulses - pulse_count) / max(baseline_pulses, 1))
    motif_bonus = len(motif_ids) * 0.05
    penalty_veto = 0.05 * veto_count
    penalty_revision = 0.10 if revision_count > 3 else 0.0
    return (0.50 * verifier_weight
            + 0.25 * efficiency_gain
            + 0.15 * motif_bonus
            - penalty_veto
            - penalty_revision)


class ExperienceReplaySubstrate:
    """Four-tier ERS: Working, Episodic, Semantic, Procedural."""

    def __init__(self, dim: int = 512,
                 max_working: int = WORKING_CAPACITY,
                 max_episodic: int = EPISODIC_CAPACITY,
                 max_semantic: int = SEMANTIC_CAPACITY,
                 max_procedural: int = PROCEDURAL_CAPACITY):
        self.dim = dim
        self.max_working = max_working
        self.max_episodic = max_episodic
        self.max_semantic = max_semantic
        self.max_procedural = max_procedural

        self.tau_ers = TAU_ERS
        self.promotion_threshold_mid = 0.70   # mid-session promotion
        self.promotion_threshold_end = 0.40   # session-end promotion

        self.working: List[ERSTuple] = []
        self.episodic: List[ERSTuple] = []
        self.semantic: List[ERSTuple] = []
        self.procedural: List[ERSTuple] = []

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def write(self, tier: str, tup: ERSTuple):
        tier_l = tier.lower()
        tup.tier = tier.capitalize()
        if tier_l == "working":
            self.working.append(tup)
            self._fifo_evict(self.working, self.max_working)
        elif tier_l == "episodic":
            self.episodic.append(tup)
            self._decay_evict(self.episodic, self.max_episodic)
        elif tier_l == "semantic":
            self.semantic.append(tup)
            self._utility_evict(self.semantic, self.max_semantic)
        elif tier_l == "procedural":
            self.procedural.append(tup)
            self._utility_evict(self.procedural, self.max_procedural)
        else:
            raise ValueError(f"Unknown tier: {tier}")

    def _fifo_evict(self, lst: List[ERSTuple], limit: int):
        """Working tier: FIFO eviction."""
        while len(lst) > limit:
            lst.pop(0)

    def _decay_evict(self, lst: List[ERSTuple], limit: int):
        """Episodic tier: evict decay_state < 0.05 AND reuse_count < 2 first, then FIFO."""
        if len(lst) > limit:
            lst[:] = [t for t in lst if not (t.decay_state < 0.05 and t.reuse_count < 2)]
        while len(lst) > limit:
            lst.pop(0)

    def _utility_evict(self, lst: List[ERSTuple], limit: int):
        """Semantic/Procedural: evict lowest utility."""
        if len(lst) > limit:
            lst.sort(key=lambda t: t.utility_score, reverse=True)
            del lst[limit:]

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------
    def retrieve(self, query_vsa: torch.Tensor, k: int = 8,
                 goal_type: Optional[str] = None) -> List[ERSTuple]:
        """VSA cosine retrieval across Working + Episodic + Semantic."""
        all_tuples = self.working + self.episodic + self.semantic
        candidates = []
        for tup in all_tuples:
            if goal_type and tup.source != goal_type:
                continue
            sim = float(cosine(query_vsa, tup.input_vsa).item())
            if sim > self.tau_ers:
                tup.reuse_count += 1
                candidates.append((sim, tup))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in candidates[:k]]

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------
    def attempt_promotions(self, session_end: bool = False):
        """Promote Working->Episodic and Episodic->Semantic per PRD rules.
        Each call does one tier transition only (no cascading within a single call).
        """
        threshold = self.promotion_threshold_end if session_end else self.promotion_threshold_mid

        # Working -> Episodic
        promoted, remaining = [], []
        for tup in self.working:
            if tup.utility_score > threshold:
                promoted.append(tup)
            else:
                remaining.append(tup)
        self.working = remaining
        for tup in promoted:
            tup.tier = "Episodic"
            self.episodic.append(tup)
            self._decay_evict(self.episodic, self.max_episodic)

        # Episodic -> Semantic (cross-family transfer: simplified as utility > 0.65)
        # Only promote items that were already in Episodic before this call
        # (i.e., not the ones just moved from Working)
        promoted_ep, remaining_ep = [], []
        for tup in self.episodic:
            if tup in promoted:
                # Just arrived from Working this call — don't cascade
                remaining_ep.append(tup)
            elif tup.utility_score > 0.65:
                promoted_ep.append(tup)
            else:
                remaining_ep.append(tup)
        self.episodic = remaining_ep
        for tup in promoted_ep:
            self.write("semantic", tup)

    def consolidate(self):
        """Nightly consolidation: decay all episodic, evict utility < 0.10."""
        for tup in self.episodic:
            tup.decay()
        self.episodic = [t for t in self.episodic if t.utility_score >= 0.10]
        self._decay_evict(self.episodic, self.max_episodic)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        def _ser(lst):
            return [
                {
                    "input_vsa": t.input_vsa.tolist(),
                    "trace_vsa": t.trace_vsa.tolist(),
                    "output_vsa": t.output_vsa.tolist(),
                    "source": t.source,
                    "utility_score": t.utility_score,
                    "decay_state": t.decay_state,
                    "episode_id": t.episode_id,
                    "task_family": t.task_family,
                    "pulse_count": t.pulse_count,
                    "regime_trace": t.regime_trace,
                    "veto_count": t.veto_count,
                    "revision_count": t.revision_count,
                    "motif_ids": t.motif_ids,
                    "tier": t.tier,
                    "verifier_verdict": t.verifier_verdict,
                    "reuse_count": t.reuse_count,
                    "content_hash": t.content_hash,
                    "timestamp": t.timestamp,
                }
                for t in lst
            ]
        return {
            "working": _ser(self.working),
            "episodic": _ser(self.episodic),
            "semantic": _ser(self.semantic),
            "procedural": _ser(self.procedural),
        }

    def load_state_dict(self, state: dict):
        def _deser(lst):
            result = []
            for s in lst:
                tup = ERSTuple(
                    input_vsa=torch.tensor(s["input_vsa"]),
                    trace_vsa=torch.tensor(s["trace_vsa"]),
                    output_vsa=torch.tensor(s["output_vsa"]),
                    source=s["source"],
                    utility_score=s["utility_score"],
                    decay_state=s.get("decay_state", 1.0),
                    episode_id=s.get("episode_id", str(uuid.uuid4())),
                    task_family=s.get("task_family", "default"),
                    pulse_count=s.get("pulse_count", 0),
                    regime_trace=s.get("regime_trace", ""),
                    veto_count=s.get("veto_count", 0),
                    revision_count=s.get("revision_count", 0),
                    motif_ids=s.get("motif_ids", []),
                    tier=s.get("tier", "Working"),
                    verifier_verdict=s.get("verifier_verdict", "pass"),
                    reuse_count=s.get("reuse_count", 0),
                    content_hash=s.get("content_hash", ""),
                    timestamp=s.get("timestamp", ""),
                )
                result.append(tup)
            return result
        self.working = _deser(state.get("working", []))
        self.episodic = _deser(state.get("episodic", []))
        self.semantic = _deser(state.get("semantic", []))
        self.procedural = _deser(state.get("procedural", []))
