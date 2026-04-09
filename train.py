"""PSN-2 Training Loop — stage-aware curriculum with real datasets.

Stage → data mapping (PRD Section 16):
  D1: ARC-AGI-2 grids (60%) + synthetic relational graphs (40%)
  D2: ARC-AGI-2 causal tasks (60%) + synthetic causal graphs (40%)
  D3: ToM / ToMi theory-of-mind (60%) + synthetic graphs (40%)
  D4: Wikitext language (60%) + ARC-AGI-2 (40%)
  D5: ARC-AGI-2 (40%) + GSM8K math (35%) + synthetic planning (25%)
  D6: All D1-D5 data in meta-episode format (50%) + Wikitext (30%) + BBH (20%)

Falls back to synthetic data if real data files are not found.
"""
from __future__ import annotations

import argparse
import os
from itertools import cycle
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from tqdm import tqdm

from psn2.config import load_config
from psn2.utils import set_seed, ensure_dir, to_device
from psn2.datasets import (
    ARCGridDataset, RelationalGraphDataset,
    ARCAGI2Dataset, ToMDataset, ToMiDataset,
    WikitextDataset, GSM8KDataset, BBHDataset,
)
from psn2.core import PSN2System
from psn2.checkpoint import CheckpointManager


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate(batch):
    out = {"type": batch[0]["type"]}
    for k in batch[0].keys():
        if k == "type":
            continue
        out[k] = torch.stack([item[k] for item in batch], dim=0)
    return out


# ---------------------------------------------------------------------------
# Dataset builders per stage
# ---------------------------------------------------------------------------

def _exists(path: str) -> bool:
    return path and Path(path).exists()


def build_loaders(cfg: dict, stage: str, batch_size: int):
    """
    Returns (primary_loader, secondary_loader, mix_ratio) for the given stage.
    mix_ratio = fraction of batches drawn from primary_loader.
    Falls back to synthetic datasets if real data paths are missing.
    """
    data_dir = cfg.get("data_dir", "data")
    grid_size = cfg.get("grid_size", 8)
    grid_vocab = cfg.get("grid_vocab", 10)
    rel_vocab = cfg.get("rel_vocab_size", 64)
    n_syn = cfg.get("n_synthetic_samples", 5000)

    def arc_synthetic():
        return ARCGridDataset(n_samples=n_syn, grid_size=grid_size, vocab=grid_vocab)

    def graph_synthetic():
        return RelationalGraphDataset(n_samples=n_syn, vocab_size=rel_vocab, num_entities=6)

    def make_loader(ds: Dataset, shuffle: bool = True) -> DataLoader:
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=collate, drop_last=True, num_workers=0)

    arc2_train = os.path.join(data_dir, "d5_arc_agi2", "train.jsonl")
    tom_train   = os.path.join(data_dir, "d3_tom", "train.jsonl")
    tomi_train  = os.path.join(data_dir, "d3_tomi", "train.jsonl")
    wiki_train  = os.path.join(data_dir, "d4_wikitext", "train.jsonl")
    gsm_train   = os.path.join(data_dir, "d5_gsm8k", "train.jsonl")
    bbh_test    = os.path.join(data_dir, "d6_bbh", "test.jsonl")

    max_wiki = cfg.get("max_wikitext_samples", 50000)
    max_arc2 = cfg.get("max_arc2_samples", None)

    if stage == "D1":
        # 60% ARC-AGI-2 grids, 40% synthetic relational graphs
        if _exists(arc2_train):
            primary = ARCAGI2Dataset(arc2_train, max_grid_size=grid_size,
                                     max_samples=max_arc2)
            print(f"  D1 primary: ARC-AGI-2 ({len(primary)} samples)")
        else:
            primary = arc_synthetic()
            print(f"  D1 primary: synthetic ARC ({len(primary)} samples)")
        secondary = graph_synthetic()
        return make_loader(primary), make_loader(secondary), 0.60

    elif stage == "D2":
        # 60% ARC-AGI-2 (causal tasks), 40% synthetic causal graphs
        if _exists(arc2_train):
            primary = ARCAGI2Dataset(arc2_train, max_grid_size=grid_size,
                                     max_samples=max_arc2)
            print(f"  D2 primary: ARC-AGI-2 ({len(primary)} samples)")
        else:
            primary = arc_synthetic()
        secondary = graph_synthetic()
        return make_loader(primary), make_loader(secondary), 0.60

    elif stage == "D3":
        # 60% ToM/ToMi, 40% synthetic graphs
        datasets = []
        if _exists(tom_train):
            ds = ToMDataset(tom_train, vocab_size=rel_vocab)
            datasets.append(ds)
            print(f"  D3: ToM ({len(ds)} samples)")
        if _exists(tomi_train):
            ds = ToMiDataset(tomi_train, vocab_size=rel_vocab)
            datasets.append(ds)
            print(f"  D3: ToMi ({len(ds)} samples)")
        if datasets:
            primary = ConcatDataset(datasets)
        else:
            primary = graph_synthetic()
            print("  D3 primary: synthetic graphs (ToM data not found)")
        secondary = graph_synthetic()
        return make_loader(primary), make_loader(secondary), 0.60

    elif stage == "D4":
        # 60% Wikitext, 40% ARC-AGI-2
        if _exists(wiki_train):
            primary = WikitextDataset(wiki_train, vocab_size=grid_vocab,
                                      grid_size=grid_size, max_samples=max_wiki)
            print(f"  D4 primary: Wikitext ({len(primary)} samples)")
        else:
            primary = arc_synthetic()
            print("  D4 primary: synthetic ARC (Wikitext not found)")
        if _exists(arc2_train):
            secondary = ARCAGI2Dataset(arc2_train, max_grid_size=grid_size,
                                       max_samples=max_arc2)
        else:
            secondary = arc_synthetic()
        return make_loader(primary), make_loader(secondary), 0.60

    elif stage == "D5":
        # 40% ARC-AGI-2, 35% GSM8K, 25% synthetic
        datasets_arc = []
        if _exists(arc2_train):
            ds = ARCAGI2Dataset(arc2_train, max_grid_size=grid_size, max_samples=max_arc2)
            datasets_arc.append(ds)
            print(f"  D5: ARC-AGI-2 ({len(ds)} samples)")
        else:
            datasets_arc.append(arc_synthetic())

        datasets_math = []
        if _exists(gsm_train):
            ds = GSM8KDataset(gsm_train, vocab_size=rel_vocab)
            datasets_math.append(ds)
            print(f"  D5: GSM8K ({len(ds)} samples)")

        if datasets_math:
            primary = ConcatDataset(datasets_arc + datasets_math)
        else:
            primary = ConcatDataset(datasets_arc)
        secondary = graph_synthetic()
        return make_loader(primary), make_loader(secondary), 0.75

    elif stage == "D6":
        # 50% mixed D1-D5, 30% Wikitext, 20% BBH
        all_datasets = []
        if _exists(arc2_train):
            all_datasets.append(ARCAGI2Dataset(arc2_train, max_grid_size=grid_size,
                                               max_samples=max_arc2))
        if _exists(gsm_train):
            all_datasets.append(GSM8KDataset(gsm_train, vocab_size=rel_vocab))
        if _exists(tom_train):
            all_datasets.append(ToMDataset(tom_train, vocab_size=rel_vocab))
        if _exists(bbh_test):
            all_datasets.append(BBHDataset(bbh_test, vocab_size=rel_vocab))
        if not all_datasets:
            all_datasets = [arc_synthetic(), graph_synthetic()]

        primary = ConcatDataset(all_datasets)
        if _exists(wiki_train):
            secondary = WikitextDataset(wiki_train, vocab_size=grid_vocab,
                                        grid_size=grid_size, max_samples=max_wiki)
        else:
            secondary = arc_synthetic()
        print(f"  D6 primary: {len(primary)} samples")
        return make_loader(primary), make_loader(secondary), 0.80

    else:
        # Unknown stage — fall back to D1 synthetic
        print(f"  Unknown stage {stage}, using synthetic D1 data")
        return make_loader(arc_synthetic()), make_loader(graph_synthetic()), 0.60


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.json")
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(cfg["checkpoint_dir"])

    stage = cfg.get("stage", "D1")
    batch_size = cfg["batch_size"]
    steps = cfg["steps"]

    print(f"Stage: {stage} | Device: {device} | Steps: {steps} | Batch: {batch_size}")
    primary_loader, secondary_loader, mix_ratio = build_loaders(cfg, stage, batch_size)

    model = PSN2System(
        dim=cfg["vsa_dim"],
        max_nodes=cfg["max_nodes"],
        grid_vocab=cfg["grid_vocab"],
        rel_vocab=cfg["rel_vocab_size"],
        stage=stage,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr_ff", cfg.get("lr", 1e-4)))
    ckpt_mgr = CheckpointManager(cfg["checkpoint_dir"], model, opt)
    start_step = 0

    if args.resume:
        start_step = ckpt_mgr.load(args.resume).get("step", 0)
        print(f"Resumed from step {start_step}")
    elif (Path(cfg["checkpoint_dir"]) / "latest.pt").exists():
        start_step = ckpt_mgr.resume_latest()
        if start_step > 0:
            print(f"Auto-resumed from step {start_step}")

    primary_iter = cycle(primary_loader)
    secondary_iter = cycle(secondary_loader)

    for step in tqdm(range(start_step, steps), desc=f"PSN-2 {stage}"):
        use_primary = (torch.rand(1).item() < mix_ratio)
        batch = next(primary_iter) if use_primary else next(secondary_iter)
        batch = to_device(batch, device)

        # Phase progression within stage: perceptive → compositional → recursive
        frac = (step - start_step) / max(steps - start_step, 1)
        if frac < 0.33:
            phase = "perceptive"
        elif frac < 0.66:
            phase = "compositional"
        else:
            phase = "recursive"

        out = model.forward_batch(batch, phase=phase)
        loss = out["loss"]

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        model.maybe_grow(step, float(out["loss_pred"].item()))
        ckpt_mgr.record_loss(step, float(loss.item()))

        if step % cfg["log_every"] == 0:
            n_active = int(model.node_bank.active.sum().item())
            print(
                f"step={step} stage={stage} phase={phase} "
                f"loss={loss.item():.4f} pred={out['loss_pred'].item():.4f} "
                f"shape={out['loss_shape'].item():.4f} "
                f"nodes={n_active}/{model.max_nodes} "
                f"attractors={len(model.attractors)} "
                f"goals={len(model.curiosity.goals)} "
                f"motifs={len(model.motifs.motifs)}"
            )

        ckpt_mgr.maybe_save(step, cfg)
        if step % cfg["checkpoint_every"] == 0 and step > start_step:
            ckpt_mgr.save(step, cfg, tag="step")

    ckpt_mgr.save_final(steps, cfg)
    print(f"Training complete. Stage {stage} done.")


if __name__ == "__main__":
    main()
