"""Matched activation comparison; freeze prior-validation choices before outer evaluation."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '1'
import sys, json, shutil, subprocess, concurrent.futures, importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import joblib, torch
from torch import nn
import individual_4h_additions as prior
s = prior.s
e = prior.e
OUT = s.HERE / 'experiments/activation_4h_v1'
ACTIVATIONS = ['silu', 'relu', 'leaky_relu', 'gelu', 'tanh']
SEEDS = [48, 149, 250]
COLS = sorted(prior.PACKS['add_4'])
ACTIVE = 'silu'

class ActivationNet(nn.Module):
    def __init__(self, coef, intercept):
        super().__init__()
        n = len(coef)
        self.linear = nn.Linear(n, 1)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor(coef, dtype=torch.float32).reshape(1, n))
            self.linear.bias.fill_(intercept)
        self.linear.requires_grad_(False)
        layers = []
        constructors = dict(silu=nn.SiLU, relu=nn.ReLU, leaky_relu=lambda: nn.LeakyReLU(.01), gelu=nn.GELU, tanh=nn.Tanh)
        for width in e.WIDTHS:
            layers.extend([nn.Linear(n, width), constructors[ACTIVE](), nn.Dropout(e.DROP)])
            n = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(n, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    def forward(self, x):
        return (self.linear(x) + self.head(self.body(x))).squeeze(1).clamp(min=0)

def configure(activation, config):
    global ACTIVE
    ACTIVE = activation
    e.configure('mlp', config)
    s.m.Net = ActivationNet
    s.SEEDS = SEEDS

def load():
    x, y, dev, base = joblib.load(prior.OUT / 'prepared.joblib')
    return x[COLS], y, dev, base

def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'MANIFEST.json').exists():
        manifest = json.loads((OUT / 'MANIFEST.json').read_text())
        assert all(s.q.sha(p) == digest for p, digest in manifest['sources'].items())
        return
    x, y, dev, base = load()
    assert len(dev) == 18481 and x.index.equals(y.index)
    assert (y.label_end == y.index + pd.Timedelta(hours=4)).all()
    np.testing.assert_allclose(base.truth, y.loc[dev, 'magnitude_log_bps'], atol=1e-8, rtol=0)
    configs = {}
    sources = [Path(__file__), Path(prior.__file__), Path(e.__file__), Path(s.__file__), Path(s.m.__file__), Path(s.q.__file__), Path(s.q.b.__file__), prior.OUT/'prepared.joblib', prior.OUT/'predictions.csv']
    for issue in s.q.STARTS:
        f = prior.OUT/'mlp'/'add_4'/f'{issue:%Y%m%d}'
        configs[str(issue)] = json.loads((f/'SELECTION.json').read_text())['config']
        sources.extend([f/'SELECTION.json', f/'confirmation'/'bundle.joblib', f/'outer'/'bundle.joblib'])
    contract = dict(target='Absolute 4h endpoint return in log bps', features=COLS, activations=ACTIVATIONS,
        leaky_relu_negative_slope=.01, seeds=SEEDS, fixed_configs_by_window=configs,
        design='Reuse earlier prior-validation-selected widths/dropout per window for every activation. Same initial weights, optimizer, preprocessing, seeds and early-stop rule. Only hidden activation changes; best epoch remains validation-selected.',
        selection='Minimum three-seed ensemble preceding-validation MAE within each outer window; ties use listed activation order. Freeze all five trials and selection before outer refits. Report all fixed-activation strategies and the validation-selected strategy.',
        training='Frozen fitted median-regression skip plus nonlinear correction, AdamW lr .001, weight decay .01 on matrices, batch512, max48 epochs, patience8, minimum improvement .01bps; three-seed mean.',
        reuse='Existing identical SiLU inner/outer bundles replayed and copied; identical training-only linear fits copied for other activations.',
        limitation='Historical evaluation already examined; feature set chosen from prior historical results. Exploratory comparison, not independent confirmation. No test-driven retries or hyperparameter expansion.')
    s.dump(OUT/'CONTRACT.json', contract)
    fw = Path('C:/Users/Siddh/Desktop/ML/ml_framework')
    docs = [s.ROOT/'PROJECT_CONTEXT.md', s.HERE/'PROJECT_CONTEXT.md'] + [fw/n for n in ['AGENTS.md','CONTEXT.md','docs/INDEX.md','docs/methods/WORKFLOW.md','docs/methods/DATA_AND_METRICS.md','docs/findings/RECORDED_RESULTS.md','jobs.json']]
    s.dump(OUT/'MANIFEST.json', dict(sources={str(p):s.q.sha(p) for p in sources}, instructions={str(p):s.q.sha(p) for p in docs}, versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','torch','scikit-learn','joblib']}))
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    s.dump(OUT/'STATUS.json', dict(stage='prepared', new_checkpoints=96, reused_checkpoints=24))

def fit(activation, config, x, y, keys, cut, folder, old, phase, vk=None, chosen=None):
    configure(activation, config)
    folder.mkdir(parents=True, exist_ok=True)
    if activation == 'silu':
        if not (folder/'bundle.joblib').exists():
            shutil.copy2(old/phase/'bundle.joblib', folder/'bundle.joblib')
        return joblib.load(folder/'bundle.joblib')
    dest = folder/'linear'/f'{cut:%Y%m%d}.joblib'
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old/phase/'linear'/dest.name, dest)
    return s.fit_pack('mlp', x, y, keys, cut, COLS, folder, vk=vk, chosen=chosen)

def audit_bundle(a, x, y, keys, cut, vk=None):
    assert a['keys'].equals(keys) and a['cutoff'] == cut and a['cols'] == COLS
    assert a['linear']['train_keys'].equals(keys) and a['linear']['cutoff'] == cut
    assert y.loc[keys,'label_end'].max() < cut
    expected = s.q.prep(x, keys, COLS)
    for field in ['median', 'scale']:
        pd.testing.assert_series_equal(expected[field], a['linear']['prep'][field])
    for seed, neural in zip(SEEDS, a['neural']):
        assert neural['seed'] == seed and neural['train_keys'].equals(keys)
        assert neural['cutoff'] == cut
        if vk is not None: assert neural['validation_keys'].equals(vk)
    assert len(a['neural']) == 3

def worker(i):
    x,y,dev,base = load()
    issue = s.q.STARTS[i]
    inner,ik,vk,ot,te = s.q.windows(y,dev,issue)
    assert y.loc[vk,'label_end'].max() < issue
    f = OUT/f'{issue:%Y%m%d}'
    f.mkdir(parents=True, exist_ok=True)
    if (f/'DONE.json').exists(): return
    old = prior.OUT/'mlp'/'add_4'/f.name
    config = json.loads((old/'SELECTION.json').read_text())['config']
    trials = []
    valid = pd.DataFrame(dict(truth=y.loc[vk,'magnitude_log_bps']))
    original_fit = s.m.fit
    def logged_fit(*args, **kwargs):
        path = args[4]
        s.dump(f/'STATUS.json', dict(stage='fitting', activation=ACTIVE, checkpoint=str(path)))
        result = original_fit(*args, **kwargs)
        print('CHECKPOINT', path, 'epoch', result['epoch'], flush=True)
        return result
    s.m.fit = logged_fit
    for activation in ACTIVATIONS:
        ai = fit(activation,config,x,y,ik,inner,f/activation/'inner',old,'confirmation',vk=vk)
        audit_bundle(ai,x,y,ik,inner,vk)
        valid[activation] = s.predict(ai,x,vk)
        trials.append(dict(activation=activation, config=config, epochs=ai['epochs'], **s.metrics(valid[activation],valid.truth)))
        s.dump(f/'VALIDATION.json', trials)
    choice = min(trials, key=lambda row:(row['mae'],ACTIVATIONS.index(row['activation'])))
    s.dump(f/'SELECTION.json', dict(issue=str(issue), selected=choice, selection_before_outer_refits=True))
    valid.to_csv(f/'validation_predictions.csv')
    preds = pd.DataFrame(dict(truth=y.loc[te,'magnitude_log_bps']))
    for activation in ACTIVATIONS:
        ai = joblib.load(f/activation/'inner'/'bundle.joblib')
        ao = fit(activation,config,x,y,ot,issue,f/activation/'outer',old,'outer',chosen=ai)
        audit_bundle(ao,x,y,ot,issue)
        assert ao['epochs'] == ai['epochs']
        preds[activation] = s.predict(ao,x,te)
        np.testing.assert_allclose(preds[activation],s.predict(joblib.load(f/activation/'outer'/'bundle.joblib'),x,te),atol=0,rtol=0)
    old_pred = pd.read_csv(old/'predictions.csv',index_col=0,parse_dates=True)
    np.testing.assert_allclose(preds.silu, old_pred.prediction, atol=1e-8, rtol=0)
    preds['validation_selected'] = preds[choice['activation']]
    preds.to_csv(f/'predictions.csv')
    s.dump(f/'DONE.json',dict(status='EXECUTED_AND_REPLAYED', selected=choice['activation'], rows=len(te),config=config,strict_purges=True,training_only_preprocessing=True,silu_reference_replayed=True))
    s.dump(f/'STATUS.json',dict(stage='complete'))

def summarize():
    x,y,dev,base = load()
    pieces=[];validations=[];choices=[];window_rows=[]
    for issue in s.q.STARTS:
        f=OUT/f'{issue:%Y%m%d}'
        assert (f/'DONE.json').exists()
        p=pd.read_csv(f/'predictions.csv',index_col=0,parse_dates=True)
        v=pd.read_csv(f/'validation_predictions.csv',index_col=0,parse_dates=True)
        selection=json.loads((f/'SELECTION.json').read_text());choice=selection['selected']
        trials=json.loads((f/'VALIDATION.json').read_text())
        assert choice==min(trials,key=lambda row:(row['mae'],ACTIVATIONS.index(row['activation'])))
        for a in ACTIVATIONS:
            configure(a,choice['config'])
            np.testing.assert_allclose(s.predict(joblib.load(f/a/'outer'/'bundle.joblib'),x,p.index),p[a],atol=1e-8,rtol=0)
            np.testing.assert_allclose(s.predict(joblib.load(f/a/'inner'/'bundle.joblib'),x,v.index),v[a],atol=1e-8,rtol=0)
            np.testing.assert_allclose(s.metrics(v[a],v.truth)['mae'],next(r['mae'] for r in trials if r['activation']==a),atol=1e-8,rtol=0)
        np.testing.assert_allclose(p.validation_selected,p[choice['activation']],atol=0,rtol=0)
        np.testing.assert_allclose(p.truth,y.loc[p.index,'magnitude_log_bps'],atol=1e-8,rtol=0)
        for a in ACTIVATIONS+['validation_selected']:window_rows.append(dict(issue=str(issue),activation=a,**s.metrics(p[a],p.truth)))
        p['issue']=str(issue);v['issue']=str(issue)
        pieces.append(p);validations.append(v);choices.append(selection)
    p=pd.concat(pieces);v=pd.concat(validations)
    assert p.index.equals(dev) and p.index.is_unique
    old=pd.read_csv(prior.OUT/'predictions.csv',index_col=0,parse_dates=True)
    np.testing.assert_allclose(p.silu,old.mlp_add_4,atol=1e-8,rtol=0)
    baseline=s.metrics(p.silu,p.truth)
    scores=pd.DataFrame([dict(activation=a,**s.metrics(p[a],p.truth),mae_change_vs_silu=s.metrics(p[a],p.truth)['mae']-baseline['mae']) for a in ACTIVATIONS+['validation_selected']])
    scores.to_csv(OUT/'scores.csv',index=False);p.to_csv(OUT/'predictions.csv');v.to_csv(OUT/'validation_predictions.csv')
    pd.DataFrame(window_rows).to_csv(OUT/'window_scores.csv',index=False)
    s.dump(OUT/'SELECTIONS.json',choices)
    uncertainty=[]
    for a in ACTIVATIONS[1:]+['validation_selected']:
        delta=abs(p[a]-p.truth)-abs(p.silu-p.truth)
        for days in [7,14]:
            block=(dev.asi8-dev.asi8.min())//pd.Timedelta(days=days).value
            agg=pd.DataFrame(dict(delta=delta.to_numpy(),block=block)).groupby('block').delta.agg(['sum','count'])
            ids=np.random.default_rng(48).integers(0,len(agg),(3000,len(agg)))
            draws=agg['sum'].to_numpy()[ids].sum(1)/agg['count'].to_numpy()[ids].sum(1)
            lo,hi=np.quantile(draws,[.025,.975]);uncertainty.append(dict(activation=a,block_days=days,mae_change=delta.mean(),lower=lo,upper=hi))
    pd.DataFrame(uncertainty).to_csv(OUT/'uncertainty.csv',index=False)
    manifest=json.loads((OUT/'MANIFEST.json').read_text())
    assert all(s.q.sha(path)==digest for path,digest in manifest['sources'].items())
    best=scores.loc[scores.activation.isin(ACTIVATIONS)].sort_values(['mae','activation']).iloc[0].to_dict()
    report=scores.to_string(index=False)+'\n\nValidation choices: '+str([(r['issue'],r['selected']['activation']) for r in choices])+'\nBest fixed activation on historical evaluation (exploratory ranking): '+str(best)+'\nNo independent confirmation; feature set was previously selected on this historical period.\n'
    (OUT/'SUMMARY.txt').write_text(report,encoding='utf-8')
    s.dump(OUT/'AUDIT.json',dict(status='PASS',all_inner_outer_predictions_replayed=True,silu_matches_previous_result=True,sources_unchanged=True,rows=len(dev),new_checkpoints=96,reused_checkpoints=24))
    s.dump(OUT/'COMPLETE.json',dict(status='EXECUTED_AND_REPLAYED',scores=scores.to_dict('records'),best_fixed_historical=best,latest_prior_validation_choice=choices[-1]['selected']['activation']))
    s.dump(OUT/'STATUS.json',dict(stage='complete'))
    print(report,flush=True)

def run():
    prepare()
    if (OUT/'COMPLETE.json').exists():
        print((OUT/'SUMMARY.txt').read_text());return
    def launch(i):
        with (OUT/f'worker_{i}.log').open('a') as log:
            result=subprocess.run([sys.executable,__file__,'worker',str(i)],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Window {i} failed; inspect worker log')
        print('FINISHED WINDOW',i,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(launch,range(4)))
    summarize()

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if len(sys.argv)>1 and sys.argv[1]=='worker':worker(int(sys.argv[2]))
        elif len(sys.argv)>1 and sys.argv[1]=='prepare':prepare()
        elif len(sys.argv)>1 and sys.argv[1]=='summarize':summarize()
        else:run()
