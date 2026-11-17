"""Train a new 48h direction model using the supplied v45 approach.

Run: python Code/fit_direction_v45_style.py
Requires numpy, pandas, scipy, scikit-learn, torch, joblib, duckdb.
No original shared.py or v26 weights are needed. This is a new experiment,
not an exact reproduction of the historical v45 scores. No fits on import.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.metadata
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1] / 'Week 1/Dataset/TEAM_48H_PLAYGROUND_V1/TEAM_48H_PLAYGROUND_V1'
SEEDS = [48, 148, 248]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def read_frame(path):
    try:
        frame = pd.read_parquet(path)
    except Exception:
        import duckdb
        frame = duckdb.read_parquet(str(path)).df()
    for key in ['decision_time', '__index_level_0__']:
        if key in frame:
            frame = frame.set_index(key)
            break
    if isinstance(frame.index, pd.RangeIndex):
        raise ValueError(f'{path}: missing decision-time index')
    frame.index = pd.to_datetime(frame.index, utc=True)
    if frame.index.has_duplicates or frame.index.hasnans or not frame.columns.is_unique:
        raise ValueError(f'{path}: duplicate/missing timestamps or duplicate columns')
    return frame.sort_index()


def validate(x, y):
    if not x.index.equals(y.index) or not len(x):
        raise ValueError('Feature and target timestamps must match exactly and be nonempty')
    y['label_end'] = pd.to_datetime(y.label_end, utc=True)
    if not (y.label_end == y.index + pd.Timedelta(hours=48)).all():
        raise ValueError('Expected 48-hour endpoint targets')
    returns = y['signed_return_log_bps'].to_numpy(float)
    if not np.isfinite(returns).all() or not y.direction_up.isin([0, 1]).all():
        raise ValueError('Invalid returns or binary targets; targets are never imputed')
    if not np.array_equal(y.direction_up.to_numpy(), (returns > 0).astype(int)):
        raise ValueError('direction_up must equal signed_return_log_bps > 0')


def fit_prep(x, names, interaction):
    a = x[names].replace([np.inf, -np.inf], np.nan)
    med = a.median().fillna(0.)
    filled = a.fillna(med)
    center = filled.mean()
    scale = filled.std(ddof=0).where(lambda s: s > 1e-8, 1.)
    product = filled[interaction[0]] * filled[interaction[1]]
    return dict(names=names, median=med, center=center, scale=scale,
                interaction=interaction, interaction_center=float(product.mean()),
                interaction_scale=max(float(product.std(ddof=0)), 1e-8))


def design(x, p):
    a = x[p['names']].replace([np.inf, -np.inf], np.nan)
    filled = a.fillna(p['median'])
    z = ((filled - p['center']) / p['scale']).clip(-8, 8)
    product = filled[p['interaction'][0]] * filled[p['interaction'][1]]
    product = ((product - p['interaction_center']) / p['interaction_scale']).clip(-8, 8)
    out = np.column_stack([z.to_numpy(), a.isna().to_numpy(), product.to_numpy()]).astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite design matrix')
    return out


def network(n):
    return nn.Sequential(nn.Linear(n, 16), nn.ReLU(), nn.Dropout(.15),
                         nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))


def probability(net, a, base):
    delta = .15 * torch.tanh(net(a).flatten())
    return .75 * base + .25 * torch.sigmoid(torch.logit(base.clamp(1e-7, 1-1e-7)) + delta), delta


def optimize(net, a, target, base, epochs, lr, original=None):
    # Initial training uses dropout; fine-tuning disables it, as in supplied run.py.
    net.train(original is None)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=.01)
    losses = []
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        prob, delta = probability(net, a, base)
        penalty = delta.square().mean() if original is None else (delta-original).square().mean()
        loss = nn.functional.binary_cross_entropy(prob, target) + (prob-target).square().mean() + penalty
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.)
        opt.step()
        losses.append(float(loss.detach()))
    net.eval()
    return losses


def fit(x, y, names, core, interaction, out, initial_lr):
    out.mkdir(parents=True, exist_ok=False)
    if y.direction_up.nunique() != 2:
        raise ValueError('Training needs both direction classes')
    prep = fit_prep(x, names, interaction)
    core_prep = fit_prep(x, core, interaction)
    a = torch.from_numpy(design(x, prep))
    target = torch.tensor(y.direction_up.to_numpy(), dtype=torch.float32)
    logistic = LogisticRegression(C=.003, solver='lbfgs', max_iter=3000, tol=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        logistic.fit(design(x, core_prep).astype(float), target.numpy())
    base = torch.tensor(logistic.predict_proba(design(x, core_prep))[:, 1], dtype=torch.float32)
    model = dict(prep=prep, core_prep=core_prep, logistic=logistic,
                 input_dim=a.shape[1], seeds=SEEDS, training_rows=len(x),
                 first_origin=x.index.min(), last_origin=x.index.max(),
                 max_label_end=y.label_end.max(), prevalence=float(target.mean()))
    joblib.dump(model, out / 'pipeline.joblib')
    history = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        net = network(a.shape[1])
        history[f'initial_s{seed}'] = optimize(net, a, target, base, 32, initial_lr)
        torch.save(net.state_dict(), out / f'initial_s{seed}.pt')
        with torch.inference_mode():
            _, original = probability(net, a, base)
        original = original.clone().detach()
        for branch in ['head', 'all']:
            tuned = copy.deepcopy(net)
            if branch == 'head':
                tuned.requires_grad_(False)
                tuned[-1].requires_grad_(True)
            history[f'{branch}_s{seed}'] = optimize(tuned, a, target, base, 16, .0003, original)
            if branch == 'head':
                for key, value in tuned.state_dict().items():
                    if not key.startswith('5.') and not torch.equal(value, net.state_dict()[key]):
                        raise RuntimeError('Head-only training changed hidden weights')
            torch.save(tuned.state_dict(), out / f'{branch}_s{seed}.pt')
        print(f'{out.name}: seed {seed}, initial + both fine-tuning branches complete', flush=True)
    dump(out / 'loss_history.json', history)
    return model


def predict(x, folder):
    model = joblib.load(folder / 'pipeline.joblib')
    a = torch.from_numpy(design(x, model['prep']))
    base = model['logistic'].predict_proba(design(x, model['core_prep']))[:, 1]
    base_tensor = torch.tensor(base, dtype=torch.float32)
    result = pd.DataFrame({'constant': model['prevalence'], 'logistic_core': base}, index=x.index)
    with torch.inference_mode():
        for branch in ['initial', 'head', 'all']:
            values = []
            for seed in model['seeds']:
                net = network(model['input_dim'])
                net.load_state_dict(torch.load(folder / f'{branch}_s{seed}.pt', map_location='cpu', weights_only=True))
                net.eval()
                p, _ = probability(net, a, base_tensor)
                values.append(p.numpy())
            result[branch] = np.mean(values, axis=0)
    result['p_up'] = (result['head'] + result['all']) / 2
    result['p_down'] = 1 - result.p_up
    if not np.isfinite(result.to_numpy()).all() or not result.p_up.between(0, 1).all():
        raise ValueError('Invalid output probabilities')
    result.index.name = 'decision_time'
    return result


def scores(frame, period):
    rows = []
    for name in ['constant', 'logistic_core', 'initial', 'head', 'all', 'p_up']:
        p = frame[name].to_numpy(float)
        y = frame.y.to_numpy(int)
        rows.append(dict(period=period, model=name, rows=len(y),
                         log_loss=float(log_loss(y, p, labels=[0, 1])),
                         brier=float(brier_score_loss(y, p)),
                         auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
                         accuracy=float(np.mean((p >= .5) == y)),
                         mean_p_up=float(p.mean()), actual_up_rate=float(y.mean())))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=PACKAGE / 'data')
    parser.add_argument('--template', type=Path, default=HERE / 'neural_finetune_v45/neural_finetune_v45/final_pipeline.json')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'experiments/direction_v45_style_v1')
    parser.add_argument('--initial-lr', type=float, default=.001)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--predict-input', type=Path, help='Predict a feature parquet with a completed new run; no training')
    parser.add_argument('--model-run', type=Path, help='Completed run folder for --predict-input')
    args = parser.parse_args()
    if args.threads < 1 or args.initial_lr <= 0:
        parser.error('threads and initial-lr must be positive')
    if bool(args.predict_input) != bool(args.model_run):
        parser.error('--predict-input and --model-run must be used together')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    stamp = pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    out = args.output_dir / stamp
    if args.predict_input:
        if not (args.model_run / 'COMPLETE.json').exists():
            raise ValueError('Model run is not marked complete')
        frame = predict(read_frame(args.predict_input), args.model_run / 'final')
        out.mkdir(parents=True, exist_ok=False)
        frame.to_csv(out / 'predictions.csv')
        dump(out / 'INFERENCE.json', dict(training=False, model_run=args.model_run,
                                         input=args.predict_input, input_sha256=sha(args.predict_input)))
        print(f'Predictions saved: {out}')
        return
    template = json.loads(args.template.read_text(encoding='utf-8'))['models'][0]
    names = list(template['preprocessing']['fields'])
    core = template['base_model']['config']['features']
    interaction = template['preprocessing']['interaction']['features']
    if len(names) != 384 or len(core) != 5 or not set(core + interaction).issubset(names):
        raise ValueError('Expected supplied 384-feature neural and five-feature core definitions')
    inputs = [args.data_dir / name for name in ['X_train.parquet', 'y_train.parquet', 'X_development.parquet', 'y_development.parquet']]
    xt, yt, xd, yd = [read_frame(p) for p in inputs]
    validate(xt, yt)
    validate(xd, yd)
    if not set(names).issubset(xt.columns) or not set(names).issubset(xd.columns):
        raise ValueError('Missing direction features: use original package X tables, not magnitude-only prepared features')
    if yt.label_end.max() >= xd.index.min() or len(xt.index.intersection(xd.index)):
        raise ValueError('Training labels overlap development origins')
    folds = [(pd.Timestamp(a, tz='UTC'), pd.Timestamp(b, tz='UTC')) for a, b in
             [('2023-04-01', '2023-07-01'), ('2023-07-01', '2023-10-01'), ('2023-10-01', '2024-01-01')]]
    out.mkdir(parents=True, exist_ok=False)
    (out / 'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    contract = dict(model='new v45-style direction training', exact_v45_reproduction=False,
                    target='48h signed_return_log_bps > 0; zero belongs to non-UP (p_down)',
                    feature_names=names, core_features=core, interaction=interaction,
                    architecture=[769, 16, 8, 1], seeds=SEEDS, initial_epochs=32,
                    initial_lr=args.initial_lr, fine_tune_epochs=16, fine_tune_lr=.0003,
                    optimizer='AdamW', weight_decay=.01, full_batch=True, gradient_clip=1,
                    dropout_initial=.15, dropout_fine_tune=0, correction_bound=.15,
                    probability_blend=[.75, .25], logistic_C=.003,
                    objective_initial='BCE + Brier + mean(delta squared)',
                    objective_fine_tune='BCE + Brier + mean((delta-original_delta) squared)',
                    new_implementation_choices=['ReLU activation', 'initial learning rate and residual penalty',
                        'median imputation, mean/std scaling, clipping to +/-8',
                        'blocked missing indicators, standardized raw-product interaction',
                        'sklearn logistic solver; fixed supplied 384-column list in all folds',
                        'always fit final ensemble; no historical v45 acceptance gate'],
                    validation='three purged periods; feature list historically chosen, not nested selection',
                    development='historically exposed; no tuning or branch selection on these rows',
                    input_hashes={str(p): sha(p) for p in inputs}, template_sha256=sha(args.template),
                    source_sha256=sha(Path(__file__)), versions={p: importlib.metadata.version(p)
                    for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'torch', 'joblib']})
    dump(out / 'CONTRACT.json', contract)
    metrics, oof = [], []
    with threadpool_limits(limits=args.threads):
        for i, (start, end) in enumerate(folds, 1):
            tr = xt.index[yt.label_end < start]
            va = xt.index[(xt.index >= start) & (xt.index < end)]
            if not len(tr) or not len(va) or yt.loc[tr, 'label_end'].max() >= va.min():
                raise ValueError(f'Invalid purged fold {i}')
            folder = out / f'fold_{i}'
            fit(xt.loc[tr], yt.loc[tr], names, core, interaction, folder, args.initial_lr)
            frame = predict(xt.loc[va], folder)
            frame['y'] = yt.loc[va, 'direction_up']
            frame['fold'] = i
            frame.to_csv(folder / 'predictions.csv')
            metrics.extend(scores(frame, f'fold_{i}'))
            oof.append(frame)
        pooled = pd.concat(oof).sort_index()
        if pooled.index.has_duplicates:
            raise ValueError('Duplicate validation timestamps')
        pooled.to_csv(out / 'validation_predictions.csv')
        metrics.extend(scores(pooled, 'validation_pooled'))
        fit(xt, yt, names, core, interaction, out / 'final', args.initial_lr)
        frame = predict(xd, out / 'final')
        frame['y'] = yd.direction_up
        frame.to_csv(out / 'development_predictions.csv')
        metrics.extend(scores(frame, 'development'))
        for month, group in frame.groupby(frame.index.strftime('%Y-%m')):
            metrics.extend(scores(group, f'development_{month}'))
    table = pd.DataFrame(metrics)
    table.to_csv(out / 'metrics.csv', index=False)
    dump(out / 'COMPLETE.json', dict(status='COMPLETE', exact_v45_reproduction=False,
         validation_rows=len(pooled), development_rows=len(frame), model_fits=40,
         artifacts={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}))
    print(table[table.period.isin(['validation_pooled', 'development'])].to_string(index=False))
    print(f'Completed. Results and models: {out}')


if __name__ == '__main__':
    main()
