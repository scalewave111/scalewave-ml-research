"""Empty-start bounded search of causal long features with simple models."""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
import discover_excursion_features_300h as b
import long_features_300h

CONFIGS = {'ridge_strong': {'alpha_per_row': 1.},
           'ridge_stronger': {'alpha_per_row': 10.},
           'shallow_tree': {'max_depth': 2, 'min_samples_leaf': 256}}


def build(r, origins, hours=300):
    b.check(hours == 300, 'Only 300h supported')
    x, groups = long_features_300h.features(r, origins)
    dictionary = pd.DataFrame([dict(feature=c, group=g, definition=c.removeprefix('LONG300__'))
                               for g, cols in groups.items() for c in cols])
    return x, dictionary


def fit(a, y, kind):
    if a.shape[1] == 0 or kind == 'median':
        return {'constant': float(np.median(y))}
    prep = b.transform_fit(a)
    if kind.startswith('ridge_'):
        model = Ridge(alpha=CONFIGS[kind]['alpha_per_row']*len(y))
    else:
        b.check(kind == 'shallow_tree', 'Unknown model')
        model = DecisionTreeRegressor(criterion='absolute_error', random_state=48, **CONFIGS[kind])
    model.fit(b.transform(a, prep), y)
    return {'prep': prep, 'model': model}


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

    def evaluate(cols, kind, screen=False):
        cols = tuple(sorted(cols))
        key = (cols,kind,screen)
        if key not in cache:
            errors = []
            for tr,va in folds:
                use = tr[np.unique(np.linspace(0,len(tr)-1,min(args.screen_rows,len(tr)),dtype=int))] if screen else tr
                model = fit(a[np.ix_(use,cols)],y[use],kind)
                errors.append(float(np.abs(y[va]-b.predict(model,a[np.ix_(va,cols)])).mean()))
            cache[key] = np.array(errors)
            trials.append(dict(stage='screen' if screen else 'search',model=kind,
                features=json.dumps([names[j] for j in cols]),pooled_mae=score(cache[key]),
                **{f'window_{i+1}_mae':v for i,v in enumerate(errors)}))
        return cache[key]

    baseline = evaluate([], 'median')
    ranked = sorted(eligible,key=lambda j:(min(score(evaluate([j],kind,True)) for kind in CONFIGS),names[j]))
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
    for kind in CONFIGS:
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
