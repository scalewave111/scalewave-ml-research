"""Empty-start bounded search of causal long features with simple models."""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import discover_excursion_features_72h as b
from fresh_features_72h import build

CONFIGS = {'depth1': dict(max_depth=1,max_leaf_nodes=2,min_samples_leaf=256),
           'depth2': dict(max_depth=2,max_leaf_nodes=4,min_samples_leaf=256),
           'depth3': dict(max_depth=3,max_leaf_nodes=8,min_samples_leaf=512)}
WEIGHTS = [.25,.5,.75,1.]


def fit(a,y,kind):
    if a.shape[1]==0 or kind=='median':
        return {'constant':float(np.median(y))}
    config,weight=kind.split('@');weight=float(weight)
    prep=b.transform_fit(a)
    model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=80,
        learning_rate=.05,l2_regularization=5.,early_stopping=False,
        random_state=48,**CONFIGS[config])
    model.fit(b.transform(a,prep),y)
    return dict(prep=prep,model=model,weight=weight,median=float(np.median(y)))


def predict(model,a):
    raw=b.predict(model,a)
    return model.get('weight',1.)*raw+(1-model.get('weight',1.))*model.get('median',0.)


def improves(new, old, weights, gain):
    return (np.average(new, weights=weights) < np.average(old, weights=weights)*(1-gain)
            and int((new < old).sum()) >= 2)


def accepted(new, baseline, weights, gain, max_loss):
    # Each earlier period is guarded, not just pooled error.
    return improves(new, baseline, weights, gain) and bool(np.all(new <= baseline*(1+max_loss)+1e-12))


def search(x, y, folds, args, folder):
    names = list(x)
    a = x.to_numpy(float)
    weights = [len(va) for _,va in folds]
    score = lambda v: float(np.average(v, weights=weights))
    cache, trials, quality = {}, [], []
    eligible = []
    for j, name in enumerate(names):
        count = [int(np.isfinite(a[tr,j]).sum()) for tr,_ in folds]
        coverage = [float(np.isfinite(a[tr,j]).mean()) for tr,_ in folds]
        ok = min(count)>=1000 and min(coverage)>=.8 and all(pd.Series(a[tr,j]).nunique()>1 for tr,_ in folds)
        quality.append(dict(feature=name,min_count=min(count),min_coverage=min(coverage),eligible=ok))
        if ok:
            eligible.append(j)
    pd.DataFrame(quality).to_csv(folder/'quality.csv',index=False)

    fit_cache={}
    def evaluate(cols, kind, screen=False):
        cols=tuple(sorted(cols));key=(cols,kind,screen)
        if key not in cache:
            config=kind.split('@')[0]
            weight=float(kind.split('@')[1]) if '@' in kind else 0.
            fit_key=(cols,config,screen)
            if fit_key not in fit_cache:
                values=[]
                for tr,va in folds:
                    use=tr[np.unique(np.linspace(0,len(tr)-1,min(args.screen_rows,len(tr)),dtype=int))] if screen else tr
                    model=fit(a[np.ix_(use,cols)],y[use],config+'@1.0' if cols else 'median')
                    values.append((y[va],predict(model,a[np.ix_(va,cols)]),float(np.median(y[use]))))
                fit_cache[fit_key]=values
            errors=[float(np.abs(truth-(weight*raw+(1-weight)*median)).mean()) for truth,raw,median in fit_cache[fit_key]]
            cache[key]=np.array(errors)
            trials.append(dict(stage='screen' if screen else 'search',model=kind,
                features=json.dumps([names[j] for j in cols]),pooled_mae=score(cache[key]),
                **{f'window_{i+1}_mae':v for i,v in enumerate(errors)}))
        return cache[key]

    baseline = evaluate([], 'median')
    ranked = sorted(eligible,key=lambda j:(min(score(evaluate([j],kind+'@1.0',True)) for kind in CONFIGS),names[j]))
    short = []
    for j in ranked:
        redundant = False
        for k in short:
            pair = x.iloc[folds[0][0],[j,k]].dropna()
            if len(pair)>=1000 and abs(pair.iloc[:,0].corr(pair.iloc[:,1],method='spearman'))>=.95:
                redundant = True
                break
        if not redundant:
            short.append(j)
        if len(short)>=args.shortlist:
            break
    b.dump(folder/'shortlist.json',[names[j] for j in short])
    winners = []
    path = []
    for kind in [c+'@'+str(w) for c in CONFIGS for w in WEIGHTS]:
        selected, current = [], baseline
        for step in range(args.max_features):
            options = [(j,evaluate(selected+[j],kind)) for j in short if j not in selected]
            options = [(j,v) for j,v in options if improves(v,current,weights,args.min_gain)]
            if not options:
                break
            j,current = min(options,key=lambda p:(score(p[1]),names[p[0]]))
            selected.append(j)
            allowed = accepted(current,baseline,weights,args.min_gain,args.max_window_loss)
            path.append(dict(model=kind,features=[names[k] for k in sorted(selected)],
                pooled_mae=score(current),window_mae=current.tolist(),accepted=allowed))
            if allowed:
                winners.append((kind,selected.copy(),current.copy()))
        print(f'  {kind}: explored {len(selected)} features',flush=True)
    chosen = min(winners,key=lambda v:(score(v[2]),len(v[1]),v[0])) if winners else ('median',[],baseline)
    pd.DataFrame(trials).to_csv(folder/'validation_trials.csv',index=False)
    b.dump(folder/'search_paths.json',path)
    result = dict(model=chosen[0],features=[names[j] for j in sorted(chosen[1])],
        validation_mae=score(chosen[2]),baseline_mae=score(baseline),
        validation_window_mae=chosen[2].tolist(),baseline_window_mae=baseline.tolist(),
        max_window_loss=args.max_window_loss)
    b.dump(folder/'selection.json',result)
    return result
