"""Replay the supplied frozen 48h v45 weights. No training is performed.

The supplied archive omits its preprocessing/network helpers. This runner
checks possible helper implementations against archived predictions, then
requires full development-set agreement before predicting another input.
It fails closed if the missing helpers cannot be reconstructed accurately.
Requires numpy, pandas, scipy, torch, and duckdb (or a pandas parquet engine).
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from torch import nn


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent.parent / 'Week 1/Dataset/TEAM_48H_PLAYGROUND_V1/TEAM_48H_PLAYGROUND_V1/data/X_development.parquet'


def read_frame(path):
    try:
        frame = pd.read_parquet(path)
    except Exception:
        import duckdb
        frame = duckdb.read_parquet(str(path)).df()
    if 'decision_time' in frame:
        frame = frame.set_index('decision_time')
    elif '__index_level_0__' in frame:
        frame = frame.set_index('__index_level_0__')
    if isinstance(frame.index, pd.RangeIndex):
        raise ValueError('Input must have a decision_time column or timestamp index.')
    frame.index = pd.to_datetime(frame.index, utc=True)
    if frame.index.has_duplicates:
        raise ValueError('Duplicate decision timestamps.')
    return frame


def columns(frame, prep, clip):
    names = list(prep['fields'])
    absent = set(names) - set(frame.columns)
    if absent:
        raise ValueError(f'Missing {len(absent)} required features: {sorted(absent)[:8]}')
    x = frame[names].to_numpy(dtype=float)
    missing = ~np.isfinite(x)
    stats = prep['fields']
    med = np.array([stats[n]['median'] for n in names])
    center = np.array([stats[n]['center'] for n in names])
    scale = np.array([stats[n]['scale'] for n in names])
    raw = np.where(missing, med, x)
    z = (raw - center) / scale
    if clip:
        z = np.clip(z, -clip, clip)
    return names, raw, z, missing.astype(float)


def design(frame, prep, clip, layout, interaction_mode):
    names, raw, z, missing = columns(frame, prep, clip)
    interaction = prep['interaction']
    a, b = [names.index(n) for n in interaction['features']]
    source = raw if interaction_mode == 'raw' else z
    product = source[:, a] * source[:, b]
    if interaction_mode != 'unscaled':
        product = (product - interaction['center']) / interaction['scale']
    if layout == 'interleaved':
        values = np.stack([z, missing], axis=2).reshape(len(frame), -1)
    else:
        values = np.column_stack([z, missing])
    return np.column_stack([values, product])


def base_probability(frame, model, recipe):
    base = model['base_model']
    a = design(frame, base['preprocessing'], recipe['clip'], recipe['base_layout'], recipe['interaction'])
    ones = np.ones(len(a))
    a = np.column_stack([ones, a]) if recipe['intercept'] == 'first' else np.column_stack([a, ones])
    return expit(a @ np.asarray(base['coefficients']))


def make_network(state, activation):
    act = {'relu': nn.ReLU, 'silu': nn.SiLU, 'tanh': nn.Tanh, 'gelu': nn.GELU}[activation]
    net = nn.Sequential(nn.Linear(769, 16), act(), nn.Dropout(.15), nn.Linear(16, 8), act(), nn.Linear(8, 1))
    net.load_state_dict(state, strict=True)
    return net.eval()


def predict(frame, models, states, recipe):
    predictions = []
    with torch.inference_mode():
        for model, state in zip(models, states):
            base = base_probability(frame, model, recipe)
            a = design(frame, model['preprocessing'], recipe['clip'], recipe['nn_layout'], recipe['interaction'])
            net = make_network(state, recipe['activation'])
            delta = (.15 * torch.tanh(net(torch.tensor(a, dtype=torch.float32)))).numpy().ravel()
            if recipe['probability_math'] == 'torch':
                p = torch.tensor(base, dtype=torch.float32)
                corrected = torch.sigmoid(torch.logit(p.clamp(1e-7, 1-1e-7)) + torch.from_numpy(delta)).numpy()
            else:
                corrected = expit(logit(np.clip(base, 1e-7, 1-1e-7)) + delta)
            predictions.append(.75 * base + .25 * corrected)
    return np.mean(predictions, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=HERE / 'neural_finetune_v45/neural_finetune_v45')
    parser.add_argument('--reference-input', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--input', type=Path, help='Feature parquet to predict; defaults to reference input.')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'experiments/direction_v45_replay')
    args = parser.parse_args()
    torch.set_num_threads(1)
    pipeline_path = args.model_dir / 'final_pipeline.json'
    pipeline = json.loads(pipeline_path.read_text(encoding='utf-8'))
    models = pipeline['models']
    if len(models) != 6:
        raise ValueError('Expected the six frozen v45 models.')
    states = []
    for model in models:
        info = model['checkpoints']['48']
        path = args.model_dir / 'models' / Path(info['file']).name
        if hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
            raise ValueError(f'Checkpoint hash mismatch: {path}')
        states.append(torch.load(path, map_location='cpu', weights_only=True))
    reference = read_frame(args.reference_input)
    golden = pd.read_csv(args.model_dir / 'predictions.csv')
    golden['decision_time'] = pd.to_datetime(golden['decision_time'], utc=True)
    golden = golden.set_index('decision_time')['p_up']
    if golden.index.has_duplicates or not golden.index.isin(reference.index).all():
        raise ValueError('Reference data does not cover the archived prediction timestamps.')
    reference = reference.loc[golden.index]
    positions = np.unique(np.linspace(0, len(reference)-1, min(64, len(reference)), dtype=int))
    sample, expected = reference.iloc[positions], golden.iloc[positions].to_numpy()
    # This is helper reconstruction against frozen outputs, not outcome-based
    # model selection. No labels, optimizer, fitting, or weight updates are used.
    best_error, best = float('inf'), None
    keys = ['clip', 'base_layout', 'nn_layout', 'interaction', 'intercept', 'activation', 'probability_math']
    choices = [[None, 8., 10., 5.], ['interleaved', 'blocked'], ['interleaved', 'blocked'],
               ['standardized', 'raw', 'unscaled'], ['first', 'last'],
               ['relu', 'silu', 'tanh', 'gelu'], ['numpy', 'torch']]
    print('Checking missing helper behavior against archived probabilities; no training.', flush=True)
    for values in itertools.product(*choices):
        recipe = dict(zip(keys, values))
        # The bounded correction cannot move the core probability by > .009375.
        base = base_probability(sample, models[0], recipe)
        if np.max(np.abs(base - expected)) > .009376:
            continue
        error = float(np.max(np.abs(predict(sample, models, states, recipe) - expected)))
        if error < best_error:
            best_error, best = error, recipe
    tolerance = 2e-7  # permits float32/library differences, not bitwise identity
    if best is None or best_error > tolerance:
        raise RuntimeError(f'Cannot verify missing helpers (best sample max error={best_error:.9g}). '
                           'No predictions written. Original shared.py helpers are required.')
    replay = predict(reference, models, states, best)
    error = float(np.max(np.abs(replay - golden.to_numpy())))
    if error > tolerance:
        raise RuntimeError(f'Full reference replay failed: max error={error:.9g}. No predictions written.')
    target = read_frame(args.input) if args.input else reference
    probability = predict(target, models, states, best) if args.input else replay
    if not np.isfinite(probability).all():
        raise ValueError('Nonfinite predictions; no output written.')
    stamp = pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    destination = args.output_dir / stamp
    destination.mkdir(parents=True, exist_ok=False)
    pd.DataFrame({'decision_time': target.index, 'p_up': probability, 'p_down': 1-probability,
                  'horizon_hours': 48}).to_csv(destination / 'predictions.csv', index=False)
    audit = {'model': 'frozen neural_finetune_v45', 'training_performed': False,
             'reference_rows': len(reference), 'reference_max_absolute_error': error,
             'tolerance': tolerance, 'bitwise_identity_claimed': False, 'helper_recipe': best,
             'pipeline_sha256': hashlib.sha256(pipeline_path.read_bytes()).hexdigest(),
             'input': str(args.input or args.reference_input), 'rows': len(target),
             'features': list(models[0]['preprocessing']['fields'])}
    (destination / 'verification.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(f'Verified {len(reference):,} archived predictions; max error {error:.9g}.')
    print(f'Output: {destination}')


if __name__ == '__main__':
    main()
