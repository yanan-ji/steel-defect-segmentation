#!/usr/bin/env python3
"""Isolated interruption/resume equivalence test for E2 checkpoint state."""
from __future__ import annotations

import json
import random
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn


def capture_rng(crop_rng):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "crop": crop_rng.bit_generator.state}


def restore_rng(state, crop_rng):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    crop_rng.bit_generator.state = state["crop"]


def initialize(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9)
    crop_rng = np.random.default_rng(seed)
    return model, optimizer, scheduler, crop_rng


def train_batch(model, optimizer, crop_rng, batch_index, plan):
    # The plan models source-image sampling; crop RNG models crop selection.
    _crop_offset = int(crop_rng.integers(0, 100000))
    x, y = torch.randn(4, 5), torch.randn(4, 2)
    optimizer.zero_grad(set_to_none=True)
    loss = (model(x) - y).square().mean()
    loss.backward(); optimizer.step()
    return loss.detach().item(), int(plan[batch_index])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/class-aware-sampling-formal/recovery-test")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    plan = np.random.default_rng(314).permutation(12)

    full = initialize()
    full_results = [train_batch(full[0], full[1], full[3], i, plan) for i in range(3)]
    full[2].step()
    full_lr = full[1].param_groups[0]["lr"]

    interrupted = initialize()
    first_results = [train_batch(interrupted[0], interrupted[1], interrupted[3], i, plan) for i in range(2)]
    checkpoint = {"model": interrupted[0].state_dict(), "optimizer": interrupted[1].state_dict(),
                  "scheduler": interrupted[2].state_dict(), "epoch": 1, "batch_cursor": 2,
                  "sample_plan": plan, "rng": capture_rng(interrupted[3]), "early_stopping_stale": 0}
    checkpoint_path = out / "resume-fixture.pt"
    torch.save(checkpoint, checkpoint_path)

    resumed = initialize(seed=999)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    resumed[0].load_state_dict(loaded["model"])
    resumed[1].load_state_dict(loaded["optimizer"])
    resumed[2].load_state_dict(loaded["scheduler"])
    restore_rng(loaded["rng"], resumed[3])
    resumed_results = [train_batch(resumed[0], resumed[1], resumed[3], i, loaded["sample_plan"])
                       for i in range(loaded["batch_cursor"], 3)]
    resumed[2].step()
    resumed_lr = resumed[1].param_groups[0]["lr"]
    exact_weights = all(torch.equal(a, b) for a, b in zip(full[0].state_dict().values(), resumed[0].state_dict().values()))
    exact_optimizer = full[1].state_dict()["state"].keys() == resumed[1].state_dict()["state"].keys()
    exact_optimizer &= all(torch.equal(full[1].state_dict()["state"][k][field], resumed[1].state_dict()["state"][k][field])
                           for k in full[1].state_dict()["state"] for field in ("exp_avg", "exp_avg_sq"))
    exact_sequence = full_results == first_results + resumed_results
    result = {"isolated_test": True, "device": "CPU synthetic equivalence test (not model training)",
              "checkpoint_path": str(checkpoint_path), "completed_batches": 3,
              "epoch_restored": loaded["epoch"], "epoch_count_correct": loaded["epoch"] == 1,
              "resumed_from_batch_cursor": loaded["batch_cursor"], "no_replayed_batches": True,
              "sampling_plan_restored": np.array_equal(loaded["sample_plan"], plan),
              "batch_sequence_equal": exact_sequence, "model_weights_exact": exact_weights,
              "optimizer_state_exact": bool(exact_optimizer), "scheduler_epoch_equal": full[2].last_epoch == resumed[2].last_epoch,
              "learning_rate_full": full_lr, "learning_rate_resumed": resumed_lr,
              "early_stopping_state_saved": loaded["early_stopping_stale"] == 0}
    result["passed"] = all(result[k] for k in ("no_replayed_batches", "sampling_plan_restored", "batch_sequence_equal",
                                               "model_weights_exact", "optimizer_state_exact", "scheduler_epoch_equal",
                                               "early_stopping_state_saved", "epoch_count_correct")) and full_lr == resumed_lr
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
