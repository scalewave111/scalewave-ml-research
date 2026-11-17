"""Fixed, training-only cheap screen and nested shortlist policy. No network."""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

METHOD={'version':'timing_staged_v1','screen':'16 training-quantile bins; training target median per supported bin',
        'bins':16,'minimum_bin_rows':64,'shortlist_per_inner_fold':3,
        'selection':'Inner binned MAE then feature name; tree inner MAE then depth then larger leaf then feature name',
        'fallback':'Training target median on every evaluation row',
        'limitations':'Proxy shortlist can miss useful tree features. Outer results are development evidence, not untouched testing.'}


def binned_fit(xtrain,ytrain,xeval,yeval,minimum=1000):
    start=time.perf_counter(); observed=np.isfinite(xtrain); ev=np.isfinite(xeval)
    center=float(np.median(ytrain)); prediction=np.full(len(xeval),center)
    eligible=int(observed.sum())>=minimum and np.unique(xtrain[observed]).size>=2
    edges=np.array([]); medians=np.array([center])
    if eligible:
        edges=np.unique(np.quantile(xtrain[observed],np.arange(1,METHOD['bins'])/METHOD['bins']))
        def groups(x):
            ok=np.isfinite(x)
            return np.where(ok,np.searchsorted(edges,np.where(ok,x,0),side='right'),len(edges)+1)
        train_groups=groups(xtrain); eval_groups=groups(xeval)
        medians=np.full(len(edges)+2,center)
        for group in range(len(medians)):
            chosen=train_groups==group
            if chosen.sum()>=METHOD['minimum_bin_rows']: medians[group]=np.median(ytrain[chosen])
        prediction=medians[eval_groups]
        independent=np.full(len(xeval),center)
        for group,value in enumerate(medians): independent[eval_groups==group]=value
        assert np.array_equal(prediction,independent)
    assert np.isfinite(prediction).all()
    error=np.abs(prediction-yeval)
    score={'status':'PASS' if eligible else 'TRAINING_COVERAGE_OR_VARIATION_FALLBACK',
           'MAE_minutes':float(error.mean()),'absolute_error_sum':float(error.sum()),
           'train_rows':len(xtrain),'train_observed':int(observed.sum()),
           'validation_rows':len(xeval),'validation_observed':int(ev.sum()),
           'missing_rows_MAE':float(error[~ev].mean()) if (~ev).any() else None,
           'observed_rows_MAE':float(error[ev].mean()) if ev.any() else None,
           'bin_edges':edges.tolist(),'bin_medians':medians.tolist(),
           'independent_bin_lookup':'PASS','seconds':time.perf_counter()-start}
    return score,prediction


def screen_feature(x,times,y,horizon,config):
    folds=[]
    for first,last in config['folds']:
        m=REF['partition_masks'](times,horizon,first,last)
        it,iv,ot,ov=[m[k] for k in ('inner_train','inner_valid','outer_train','outer_valid')]
        assert it.any() and iv.any() and ot.any() and ov.any()
        inner,_=binned_fit(x[it],y[it],x[iv],y[iv],config['minimum_observed_training'])
        outer,_=binned_fit(x[ot],y[ot],x[ov],y[ov],config['minimum_observed_training'])
        for score in (inner,outer):
            score['effective_bin_count']=len(score.pop('bin_edges'))+1
            score.pop('bin_medians')  # Keep compact scores, not losing fitted models.
        folds.append({'inner':inner,'outer':outer})
    return {'folds':folds,'pooled_MAE_minutes':sum(f['outer']['absolute_error_sum'] for f in folds)/sum(f['outer']['validation_rows'] for f in folds)}


def shortlist(screen,fold):
    eligible=[feature for feature,v in screen.items() if v['folds'][fold]['inner']['status']=='PASS']
    return sorted(eligible,key=lambda f:(screen[f]['folds'][fold]['inner']['MAE_minutes'],f))[:METHOD['shortlist_per_inner_fold']]


def refine(state,times,y,horizon,config,get_x,save,legacy=lambda feature:None,workers=2):
    """Only inner scores select features/settings. Outer scores never enter selection."""
    from threadpoolctl import threadpool_limits
    state.setdefault('tuning',{}); state.setdefault('policy_folds',{})
    for fold,(first,last) in enumerate(config['folds']):
        key=str(fold)
        if key in state['policy_folds']: continue
        names=shortlist(state['screen'],fold)
        m=REF['partition_masks'](times,horizon,first,last)
        it,iv,ot,ov=[m[k] for k in ('inner_train','inner_valid','outer_train','outer_valid')]
        candidates=[]
        with threadpool_limits(limits=1),ThreadPoolExecutor(max_workers=workers) as pool:
            for feature in names:
                x=get_x(feature); old=legacy(feature)
                prior=old.get('fits',{}) if old else {}
                prepared=None; futures={}
                for depth,leaf in config['grid']:
                    fit_key=f'{feature}|{horizon}|{fold}|inner|{depth}|{leaf}'
                    if fit_key not in state['tuning']:
                        if fit_key in prior:
                            state['tuning'][fit_key]={'score':prior[fit_key],'origin':'REUSED_VERIFIED_LEGACY'}
                        else:
                            if prepared is None: prepared=OPT['prepare'](x[it],y[it],x[iv],y[iv],config['minimum_observed_training'])
                            future=pool.submit(OPT['prepared_fit'],prepared,depth,leaf)
                            futures[future]=fit_key
                for future in as_completed(futures):
                    score,prediction=future.result(); del prediction
                    state['tuning'][futures[future]]={'score':score,'origin':'NEW_STAGED_FIT'}
                    save()
                    print(f'{horizon}m fold {fold+1}: {feature} inner setting saved',flush=True)
                for depth,leaf in config['grid']:
                    score=state['tuning'][f'{feature}|{horizon}|{fold}|inner|{depth}|{leaf}']['score']
                    if score['status']=='PASS': candidates.append((score,feature))
        chosen=min(candidates,key=lambda v:(v[0]['MAE_minutes'],v[0]['depth'],-v[0]['min_samples_leaf'],v[1])) if candidates else None
        provenance='NEW_STAGED_FIT'
        if chosen:
            setting,feature=chosen; x=get_x(feature); old=legacy(feature)
            old_fold=old.get('folds',{}).get(f'{feature}|{horizon}|{fold}') if old else None
            selected={'depth':setting['depth'],'min_samples_leaf':setting['min_samples_leaf']}
            if old_fold and old_fold['selected']==selected:
                score=old['fits'][f'{feature}|{horizon}|{fold}|outer']; provenance='REUSED_VERIFIED_LEGACY'
            else:
                with threadpool_limits(limits=1):
                    score,prediction=OPT['prepared_fit'](OPT['prepare'](x[ot],y[ot],x[ov],y[ov],config['minimum_observed_training']),setting['depth'],setting['min_samples_leaf'])
                    del prediction
        else:
            feature=None; selected=None; score={'status':'NO_ELIGIBLE_INNER_CANDIDATE'}
        if score['status']!='PASS':
            errors=np.abs(y[ov]-float(np.median(y[ot])))
            score=dict(score,fallback='training_median_time_to_high',MAE_minutes=float(errors.mean()),absolute_error_sum=float(errors.sum()))
        state['policy_folds'][key]={'shortlist':names,'selected_feature':feature,'selected_settings':selected,
            'score':score,'origin':provenance,'validation_rows':int(ov.sum()),'train_rows':int(ot.sum()),
            'inner_train_rows':int(it.sum()),'inner_validation_rows':int(iv.sum()),
            'inner_latest_training_label_end':str((times[it]+__import__('pandas').Timedelta(minutes=horizon)).max()),
            'inner_latest_validation_label_end':str((times[iv]+__import__('pandas').Timedelta(minutes=horizon)).max()),
            'outer_latest_training_label_end':str((times[ot]+__import__('pandas').Timedelta(minutes=horizon)).max())}
        save(); print(f'{horizon}m: shortlist policy fold {fold+1}/5 complete',flush=True)
    return state
