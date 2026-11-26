"""Prepare read-only SAK inputs, fit H4 folds, and evaluate one policy at a time."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .model import ModelConfig, build_model
from .protocol import (H4_END, RobustScaler, Windows, evaluate, maturity_split,
                       paired_daily_uncertainty, regular_positions, utc_index)

QUARTERS = tuple(str(q) for q in pd.period_range("2023Q2", "2024Q4", freq="Q"))


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str)+"\n", encoding="utf-8")


def verify_source(source):
    receipt_path = Path(__file__).resolve().parents[1]/"source_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    verified = {}
    for entry in receipt["files"]:
        relative = Path(entry["path"]).relative_to("SAK_GROSS_V01_COMPLETE")
        # Verify executable modules, contracts, schema and all frozen weights.
        if (relative.parts[0] in ("contracts", "model_files") or relative.suffix == ".py"
                or relative.parts[:2] == ("evidence", "replay")):
            target = source/relative
            if not target.is_file() or digest(target) != entry["sha256"]:
                raise ValueError(f"Source identity mismatch: {relative}")
            verified[str(relative)] = entry["sha256"]
    return verified


def prepare(source, inputs, output):
    """Calls the audited feature constructor; never rebuilds or changes SAK."""
    source, inputs, output = Path(source).resolve(), Path(inputs).resolve(), Path(output).resolve()
    if output.exists() or output == source or source in output.parents or output == inputs or inputs in output.parents:
        raise ValueError("Choose a new cache folder outside source and original inputs.")
    verified = verify_source(source)
    schema = json.loads((source/"model_files/feature_schema.json").read_text())
    features = schema["direction"]
    if len(features) != 146 or len(set(features)) != 146:
        raise ValueError("Expected the original 146 distinct direction inputs.")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    try:
        module = importlib.import_module("research_data")
        if Path(module.__file__).resolve() != source/"research_data.py":
            raise ValueError("Wrong research_data module imported.")
        data = module.ResearchData(inputs)
    finally:
        sys.path.remove(str(source))
    panel_index = utc_index(data.panel.index)
    admitted = np.flatnonzero(panel_index < H4_END)
    index = panel_index[admitted]
    grid, positions = regular_positions(index)
    if len(admitted) != len(data.panel) and not np.array_equal(admitted, np.arange(len(admitted))):
        raise ValueError("H4 calendar is not a chronological prefix.")
    panel_positions = np.full(len(data.panel), -1, dtype=np.int64)
    panel_positions[admitted] = positions
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"STATUS.json", {"status": "PREPARING_NOT_TRAINED"})
    x = np.lib.format.open_memmap(output/"features.npy", mode="w+", dtype="float32",
                                 shape=(len(grid), len(features)))
    x[:] = np.nan
    for start in range(0, len(admitted), 16384):
        rows = admitted[start:start+16384]
        frame = data.prepared(rows)
        if not utc_index(frame.index).equals(panel_index[rows]):
            raise ValueError("Prepared features changed the original decision calendar.")
        values = frame[features].to_numpy("float32")
        if np.isinf(values).any():
            raise ValueError("SAK source produced infinite features.")
        x[panel_positions[rows]] = values
        print(f"Prepared {start+len(rows)}/{len(admitted)} origins", flush=True)
    x.flush()
    del x
    y = np.full(len(grid), np.nan, dtype="float64")
    y[positions] = data.panel.r.iloc[admitted].to_numpy(float)
    available = np.full(len(grid), np.iinfo(np.int64).min, dtype="int64")
    available[positions] = pd.DatetimeIndex(
        pd.to_datetime(data.panel.available_at.iloc[admitted], utc=True)).as_unit("ns").asi8
    np.save(output/"target.npy", y)
    np.save(output/"available_ns.npy", available)
    np.save(output/"timestamps_ns.npy", grid.asi8)
    folds = {}
    (output/"folds").mkdir()
    for quarter in QUARTERS:
        issue, full, train, inner, calibration, test = data.split(quarter)
        train, test = np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)
        tr, te = panel_positions[train], panel_positions[test]
        if (tr < 0).any() or (te < 0).any():
            raise ValueError("A fold accesses protected 2025 data.")
        if not np.isfinite(y[np.r_[tr, te]]).all():
            raise ValueError("Invalid target in original SAK split.")
        if not (available[tr] < pd.Timestamp(issue).value).all():
            raise ValueError("Immature training outcome.")
        reference = pd.read_parquet(source/f"evidence/replay/{quarter}.parquet").sort_index()
        if not utc_index(reference.index).equals(grid[te]):
            raise ValueError(f"Reference calendar mismatch: {quarter}")
        np.testing.assert_allclose(reference.actual_bps, y[te], rtol=0, atol=1e-10)
        if not np.isfinite(reference[["p_up", "predicted_magnitude_bps"]].to_numpy()).all():
            raise ValueError("Nonfinite original model predictions.")
        np.save(output/f"folds/{quarter}_train.npy", tr)
        np.save(output/f"folds/{quarter}_test.npy", te)
        ref = reference[["p_up", "predicted_magnitude_bps", "actual_bps"]].rename(
            columns={"p_up": "reference_p_up"})
        ref["target_available_at"] = pd.to_datetime(available[te], utc=True)
        ref["model_fold"] = quarter
        ref.to_parquet(output/f"folds/{quarter}_reference.parquet")
        folds[quarter] = {"issue": str(issue), "train_rows": len(tr), "test_rows": len(te)}
    identities = {
        str(p.relative_to(inputs)): digest(p)
        for p in (inputs/"research_compact/X.npy",
                  inputs/"research_compact/matrix_complete.json",
                  inputs/"research_week2_v2/aligned_panel.parquet")
    }
    cache_hashes = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*"))
                    if p.is_file() and p.name != "STATUS.json"}
    write_json(output/"metadata.json", {
        "schema_version": 1, "status": "PREPARED_NOT_TRAINED", "feature_names": features,
        "source": str(source), "source_hashes": verified, "input_hashes": identities,
        "cache_hashes": cache_hashes, "rows": len(grid), "folds": folds,
        "source_commit": "ca64313fbafe6b4bcc5917e3746b579974c7b19c",
        "h4a_accessed": False, "missing_minutes_remain_missing": True,
        "target": "10000*(P(t+16min)/P(t+1min)-1)",
    })
    write_json(output/"STATUS.json", {"status": "PREPARED_NOT_TRAINED"})


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def loader(x, y, rows, scaler, config, batch, shuffle, seed):
    return DataLoader(Windows(x, y, rows, scaler, config), batch_size=batch,
                      shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def epoch(model, batches, device, optimizer=None):
    model.train(optimizer is not None)
    total, count = 0., 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            loss = F.binary_cross_entropy_with_logits(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss.")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            total += float(loss.detach())*len(y)
            count += len(y)
    if not count:
        raise ValueError("Empty fit/evaluation loader.")
    return total/count


def probabilities(model, batches, device):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for x, _ in batches:
            outputs.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
    return np.concatenate(outputs)


def fit_fold(x, y, index, available, train_rows, test_rows, issue, config, kind,
             output, seed, device, epochs=20, patience=3, batch=256, validation_days=30):
    """Select epoch using prior data, then refit from scratch on original train rows."""
    if epochs < 1 or patience < 1:
        raise ValueError("Positive epochs/patience required.")
    inner, validation = maturity_split(index, available, train_rows, issue, validation_days)
    index = utc_index(index)
    if len(index) != len(x) or len(y) != len(x) or not (np.diff(index.asi8) == 60_000_000_000).all():
        raise ValueError("Fit requires a complete minute grid aligned with features and targets.")
    for rows in (train_rows, test_rows):
        if not len(rows) or len(np.unique(rows)) != len(rows) or (np.asarray(rows) < 0).any() or (np.asarray(rows) >= len(x)).any():
            raise ValueError("Invalid or duplicate fit/evaluation rows.")
    if (index[test_rows] < pd.Timestamp(issue)).any():
        raise ValueError("Test origins precede the issue.")
    if np.intersect1d(train_rows, test_rows).size:
        raise ValueError("Training/evaluation overlap.")
    if (index[np.r_[train_rows, test_rows]] >= H4_END).any():
        raise ValueError("H4a is excluded from this training runner.")
    if not np.isfinite(y[np.r_[train_rows, test_rows]]).all():
        raise ValueError("Nonfinite target.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    scaler = RobustScaler.fit(x, inner)
    seed_all(seed)
    model = build_model(config, kind, float(np.mean(y[inner]>0))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    training = loader(x, y, inner, scaler, config, batch, True, seed)
    valid = loader(x, y, validation, scaler, config, batch, False, seed)
    history, best_loss, best_epoch, stale = [], float("inf"), 1, 0
    for n in range(1, epochs+1):
        train_loss = epoch(model, training, device, optimizer)
        validation_loss = epoch(model, valid, device)
        history.append({"phase": "inner", "epoch": n, "train_log_loss": train_loss,
                        "validation_log_loss": validation_loss})
        print(json.dumps(history[-1]), flush=True)
        if validation_loss < best_loss-1e-6:
            best_loss, best_epoch, stale = validation_loss, n, 0
        else:
            stale += 1
        if stale >= patience:
            break
    del model, optimizer
    seed_all(seed)
    scaler = RobustScaler.fit(x, train_rows)
    model = build_model(config, kind, float(np.mean(y[train_rows]>0))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    training = loader(x, y, train_rows, scaler, config, batch, True, seed)
    for n in range(1, best_epoch+1):
        loss = epoch(model, training, device, optimizer)
        history.append({"phase": "outer_refit", "epoch": n, "train_log_loss": loss})
    p = probabilities(model, loader(x, y, test_rows, scaler, config, batch, False, seed), device)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, output/"weights.pt")
    np.savez(output/"scaler.npz", center=scaler.center, scale=scaler.scale)
    record = {
        "status": "FITTED_H4_RESEARCH", "kind": kind, "config": asdict(config), "seed": seed,
        "issue": str(issue), "best_epoch": best_epoch, "validation_log_loss": best_loss,
        "train_rows": len(train_rows), "inner_rows": len(inner), "validation_rows": len(validation),
        "test_rows": len(test_rows), "parameters": sum(p.numel() for p in model.parameters()),
        "history": history, "raw_probabilities_no_posthoc_calibration": True,
        "training_policy": "original SAK direction training indices; chronological early stop; full refit",
        "weights_sha256": digest(output/"weights.pt"), "scaler_sha256": digest(output/"scaler.npz"),
    }
    write_json(output/"fit.json", record)
    return p, record


def replay_fold(cache, checkpoint, quarter, device="cpu", batch=256):
    cache, checkpoint = Path(cache), Path(checkpoint)
    record = json.loads((checkpoint/"fit.json").read_text())
    if digest(checkpoint/"weights.pt") != record["weights_sha256"] or digest(checkpoint/"scaler.npz") != record["scaler_sha256"]:
        raise ValueError("Checkpoint hash mismatch.")
    expected_issue = pd.Period(quarter, freq="Q").start_time.tz_localize("UTC")
    if pd.Timestamp(record["issue"]) != expected_issue:
        raise ValueError("Checkpoint does not belong to the requested quarter.")
    config = ModelConfig(**record["config"])
    model = build_model(config, record["kind"]).to(device)
    model.load_state_dict(torch.load(checkpoint/"weights.pt", map_location=device, weights_only=True))
    with np.load(checkpoint/"scaler.npz", allow_pickle=False) as stored:
        scaler = RobustScaler(stored["center"], stored["scale"])
    x = np.load(cache/"features.npy", mmap_mode="r")
    y = np.load(cache/"target.npy", mmap_mode="r")
    rows = np.load(cache/f"folds/{quarter}_test.npy")
    return probabilities(model, loader(x, y, rows, scaler, config, batch, False, record["seed"]), device)


def train(cache, output, folds, kind, seed, device, epochs, batch, tracking_uri=None):
    cache, output = Path(cache).resolve(), Path(output).resolve()
    if output.exists() or output == cache or cache in output.parents:
        raise ValueError("Choose a new run folder outside the prepared cache.")
    metadata = json.loads((cache/"metadata.json").read_text())
    if metadata["status"] != "PREPARED_NOT_TRAINED" or len(metadata["feature_names"]) != 146:
        raise ValueError("Use a verified SAK prepared cache.")
    for name, expected in metadata["cache_hashes"].items():
        if digest(cache/name) != expected:
            raise ValueError(f"Prepared cache changed: {name}")
    folds = tuple(folds)
    if not folds or any(q not in QUARTERS for q in folds) or len(set(folds)) != len(folds):
        raise ValueError("Invalid historical quarters.")
    if tuple(sorted(folds)) != folds:
        raise ValueError("Use chronological folds.")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    x = np.load(cache/"features.npy", mmap_mode="r")
    y = np.load(cache/"target.npy", mmap_mode="r")
    index = pd.to_datetime(np.load(cache/"timestamps_ns.npy"), utc=True)
    available = pd.to_datetime(np.load(cache/"available_ns.npy"), utc=True)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"STATUS.json", {"status": "TRAINING_INCOMPLETE", "folds": folds})
    config = ModelConfig()
    write_json(output/"run_contract.json", {
        "model": kind, "config": asdict(config), "seed": seed, "epochs_max": epochs,
        "batch": batch, "folds": folds,
        "versions": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "device": device, "cache_metadata_sha256": digest(cache/"metadata.json"),
        "source_hashes": metadata["source_hashes"], "target": metadata["target"],
        "h4a_accessed": False, "first_policy": "ungated", "threshold_search": False,
    })
    frames = []
    for quarter in folds:
        train_rows = np.load(cache/f"folds/{quarter}_train.npy")
        test_rows = np.load(cache/f"folds/{quarter}_test.npy")
        issue = metadata["folds"][quarter]["issue"]
        p, record = fit_fold(x, y, index, available, train_rows, test_rows, issue, config,
                             kind, output/quarter, seed, device, epochs=epochs, batch=batch)
        frame = pd.read_parquet(cache/f"folds/{quarter}_reference.parquet")
        frame["p_up"] = p.astype(float)
        frame.to_parquet(output/quarter/"predictions.parquet")
        # Check serialization, preprocessing and no-fit replay before scoring.
        replay = replay_fold(cache, output/quarter, quarter, device, batch)
        np.testing.assert_allclose(replay, p, rtol=0, atol=1e-7)
        frames.append(frame)
        if tracking_uri:
            import mlflow
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("SAK_TCN_DIR_V01")
            with mlflow.start_run(run_name=f"{kind}_{quarter}_seed{seed}"):
                mlflow.log_params({"kind": kind, "quarter": quarter, "seed": seed,
                                   "epochs": record["best_epoch"], "parameters": record["parameters"]})
                mlflow.log_metric("inner_validation_log_loss", record["validation_log_loss"])
                mlflow.log_artifacts(str(output/quarter))
                write_json(output/quarter/"mlflow.json", {
                    "run_id": mlflow.active_run().info.run_id, "tracking_uri": tracking_uri})
    predictions = pd.concat(frames).sort_index()
    if not predictions.index.is_unique:
        raise ValueError("Duplicate outer forecasts.")
    predictions.to_parquet(output/"predictions.parquet")
    score(output, "ungated")
    write_json(output/"STATUS.json", {"status": "COMPLETE_H4_RESEARCH_NOT_DEPLOYED",
                                    "folds": folds, "rows": len(predictions), "h4a_accessed": False})


def score(run, rule):
    run = Path(run)
    frame = pd.read_parquet(run/"predictions.parquet")
    destination = run/f"evaluation_{rule}"
    destination.mkdir(exist_ok=False)
    table, monthly, ledger = evaluate(frame, rule)
    table.to_csv(destination/"comparison.csv", index=False)
    monthly.to_csv(destination/"monthly.csv", index=False)
    ledger.to_parquet(destination/"trades.parquet")
    # Aggregate ledger economics by quarter without refitting percentile cutoffs.
    quarter_rows = []
    for model in ("SAK", "challenger"):
        for quarter in frame.model_fold.unique():
            selected = ledger[(ledger.model == model) & (ledger.model_fold == quarter)]
            gross = selected.gross_bps.to_numpy(float)
            quarter_rows.append({
                "model": model, "rule": rule, "quarter": quarter,
                "forecasts": int(frame.model_fold.eq(quarter).sum()), "trades": len(gross),
                "longs": int(selected.side.gt(0).sum()), "shorts": int(selected.side.lt(0).sum()),
                "gross_bps": float(gross.mean()) if len(gross) else None,
                "net_bps_assuming_4bp_round_trip": float(gross.mean()-4) if len(gross) else None,
                "sum_entry_bps": float(gross.sum()),
            })
    pd.DataFrame(quarter_rows).to_csv(destination/"quarterly.csv", index=False)
    uncertainty = paired_daily_uncertainty(frame, rule)
    write_json(destination/"paired_uncertainty.json", uncertainty)
    # Probability/score diagnostics include rejected forecasts, crucial for H4a-style inactivity.
    diagnostics = []
    for model, col in (("SAK", "reference_p_up"), ("challenger", "p_up")):
        for month, part in frame.groupby(frame.index.strftime("%Y-%m")):
            strength = np.abs((2*part[col]-1)*part.predicted_magnitude_bps)
            record = {"model": model, "month": month, "forecasts": len(part),
                      "confidence_mean": float(np.abs(2*part[col]-1).mean()),
                      "magnitude_mean_bps": float(part.predicted_magnitude_bps.mean()),
                      "score_ge10_count": int((strength >= 10).sum())}
            record.update({f"strength_q{q}": float(strength.quantile(q)) for q in (.5, .9, .99, .999, 1.)})
            diagnostics.append(record)
    pd.DataFrame(diagnostics).to_csv(destination/"score_diagnostics.csv", index=False)
    print(table.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--source", required=True)
    p.add_argument("--inputs", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("train")
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--folds", nargs="+", choices=QUARTERS, default=list(QUARTERS))
    p.add_argument("--kind", choices=["tcn", "snapshot", "summary"], default="tcn")
    p.add_argument("--seed", type=int, default=20261127)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--tracking-uri")
    p = commands.add_parser("score")
    p.add_argument("--run", required=True)
    p.add_argument("--rule", choices=["ungated", "top1", "pm10", "both"], required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    globals()[command](**args)


if __name__ == "__main__":
    main()
