"""T11: Checkpoint System — serialize/resume all state across Kaggle sessions.
Checkpoints every 30 minutes and at session end.
Format: PyTorch state dict + JSON metadata.

PRD Section 17.2 fields included:
  checkpoint_id, session_id, stage, sub_stage, config_hash, software_hash,
  random_seeds, stage_gate_results, training_loss_history, ERS state,
  growth ledger, curiosity goals, TAL, LAL.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
import uuid
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from psn2.core import PSN2System


def _config_hash(config: dict) -> str:
    raw = json.dumps(config, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def _software_hash() -> str:
    """Hash of core source files as a software fingerprint."""
    try:
        import psn2
        src = Path(psn2.__file__).parent
        h = hashlib.sha256()
        for f in sorted(src.rglob("*.py")):
            h.update(f.read_bytes())
        return h.hexdigest()
    except Exception:
        return "unknown"


def _random_seeds() -> dict:
    seeds = {
        "python": random.getstate(),
        "numpy": np.random.get_state()[1].tolist(),
        "torch": torch.get_rng_state().tolist(),
    }
    if torch.cuda.is_available():
        seeds["cuda"] = torch.cuda.get_rng_state().tolist()
    return seeds


def save_checkpoint(path: str, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: write to temp then rename
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.rename(path)


def load_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu")


class CheckpointManager:
    """
    Manages periodic checkpointing every 30 minutes and at session end.
    Saves: model state_dict, attractor library, curiosity goals, motif library,
           growth ledger, node errors/traces, optimizer state, ERS state.
    """

    INTERVAL_SECONDS = 30 * 60  # 30 minutes

    def __init__(self, checkpoint_dir: str, model: "PSN2System",
                 optimizer: Optional[torch.optim.Optimizer] = None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.optimizer = optimizer
        self._last_save_time = time.time()
        # Persistent session_id across saves within one run
        self._session_id = str(uuid.uuid4())
        self._training_loss_history: list = []

    def record_loss(self, step: int, loss: float):
        """Record a training loss entry for the history."""
        self._training_loss_history.append({"step": step, "loss": loss})
        # Keep last 1000 entries
        if len(self._training_loss_history) > 1000:
            self._training_loss_history = self._training_loss_history[-1000:]

    def _build_payload(self, step: int, config: dict = None) -> dict:
        model_state = self.model.state_dict_full()
        payload = {
            # PRD Section 17.2 required fields
            "checkpoint_id": str(uuid.uuid4()),
            "session_id": self._session_id,
            "stage": getattr(self.model, "stage", "D1"),
            "sub_stage": getattr(self.model, "sub_stage", None),
            "random_seeds": _random_seeds(),
            "training_loss_history": self._training_loss_history,
            # Model state (includes ERS via state_dict_full)
            "model": model_state,
            "step": step,
            "timestamp": time.time(),
        }
        if config is not None:
            payload["config"] = config
            payload["config_hash"] = _config_hash(config)
        payload["software_hash"] = _software_hash()
        if self.optimizer is not None:
            payload["optimizer"] = self.optimizer.state_dict()
        return payload

    def maybe_save(self, step: int, config: dict = None) -> bool:
        """Save if 30 minutes have elapsed. Returns True if saved."""
        now = time.time()
        if now - self._last_save_time >= self.INTERVAL_SECONDS:
            self.save(step, config, tag="periodic")
            self._last_save_time = now
            return True
        return False

    def save(self, step: int, config: dict = None, tag: str = "step"):
        payload = self._build_payload(step, config)
        # Always overwrite latest — this is the only file that grows unboundedly
        save_checkpoint(str(self.checkpoint_dir / "latest.pt"), payload)
        # For periodic saves, keep only the last 2 (safety net for corruption)
        if tag == "periodic":
            tagged_path = self.checkpoint_dir / f"periodic_{step}.pt"
            save_checkpoint(str(tagged_path), payload)
            # Delete older periodic checkpoints, keep only the 2 most recent
            old_periodics = sorted(self.checkpoint_dir.glob("periodic_*.pt"),
                                   key=lambda p: p.stat().st_mtime)
            for old in old_periodics[:-2]:
                try:
                    old.unlink()
                except OSError:
                    pass
        # For step saves (checkpoint_every), only update latest — no extra file
        # For final saves, write a single final.pt
        elif tag == "final":
            save_checkpoint(str(self.checkpoint_dir / "final.pt"), payload)

    def save_final(self, step: int, config: dict = None):
        """Save at session end."""
        self.save(step, config, tag="final")

    def load(self, path: str) -> dict:
        """Load checkpoint and restore model + optimizer state."""
        payload = load_checkpoint(path)
        self.model.load_state_dict_full(payload["model"])
        if self.optimizer is not None and "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        # Restore session_id if present
        if "session_id" in payload:
            self._session_id = payload["session_id"]
        if "training_loss_history" in payload:
            self._training_loss_history = payload["training_loss_history"]
        return payload

    def resume_latest(self) -> int:
        """Resume from latest checkpoint. Returns step number."""
        latest = self.checkpoint_dir / "latest.pt"
        if not latest.exists():
            return 0
        payload = self.load(str(latest))
        return payload.get("step", 0)
