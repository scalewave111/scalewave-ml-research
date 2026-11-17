"""Synthetic full ten-horizon staged pipeline, recovery and leakage tests."""
import contextlib
import copy
import io
import json
import runpy
from pathlib import Path
import numpy as np
import pandas as pd

root=Path(__file__).parent
ref=runpy.run_path(str(root/'screen_runner.py'))
opt=runpy.run_path(str(root/'screen_throughput.py'),init_globals={'REF':ref})
core=runpy.run_path(str(root/'staged_timing_core.py'),init_globals={'REF':ref,'OPT':opt})
runner=runpy.run_path(str(root/'staged_timing_runner.py'),init_globals={'REF':ref})
report=runpy.run_path(str(root/'staged_timing_report.py'))
c=json.loads((root/'screen_config.json').read_text()); c['candidate_features']=c['candidate_features'][:4]; c['minimum_observed_training']=20
t=pd.date_range('2022-04-22','2024-06-30',freq='12h',tz='UTC')
rng=np.random.default_rng(48); x=rng.normal(size=len(t)); x[::9]=np.nan
features=dict(zip(c['candidate_features'],[x,-x,np.full(len(x),np.nan),np.ones(len(x))]))
states={}; baselines={'completed':{}}; calls=[0]
original=opt['prepared_fit']
def counted(*args): calls[0]+=1; return original(*args)
opt['prepared_fit']=counted
with contextlib.redirect_stdout(io.StringIO()):
    for h in c['horizons']:
        y=rng.integers(0,h+1,len(t)).astype(float)
        state={'signature':'synthetic','horizon_minutes':h,'screen':{f:core['screen_feature'](v,t,y,h,c) for f,v in features.items()}}
        # Outer score perturbation cannot change any shortlist.
        changed=copy.deepcopy(state['screen'])
        for value in changed.values():
            value['pooled_MAE_minutes']=-999999
            for fold in value['folds']: fold['outer']['MAE_minutes']=-999999
        assert all(core['shortlist'](state['screen'],i)==core['shortlist'](changed,i) for i in range(5))
        checkpoint=[]; inject=[h==15]
        def save():
            checkpoint.append(json.dumps({'state':state,'sha256':ref['signature'](state)}))
            if inject[0]: inject[0]=False; raise OSError('Simulated interruption after durable fit')
        try: core['refine'](state,t,y,h,c,features.__getitem__,save)
        except OSError:
            state=runner['unpack'](checkpoint[-1],'synthetic',h)
            core['refine'](state,t,y,h,c,features.__getitem__,save)
        assert len(state['policy_folds'])==5
        for i,(first,last) in enumerate(c['folds']):
            v=state['policy_folds'][str(i)]
            assert pd.Timestamp(v['outer_latest_training_label_end'])<pd.Timestamp(first,tz='UTC')
            assert pd.Timestamp(v['inner_latest_validation_label_end'])<pd.Timestamp(first,tz='UTC')
            feature=v['selected_feature']; m=ref['partition_masks'](t,h,first,last)
            if feature:
                selected=v['selected_settings']
                expected=ref['tree_fit'](features[feature][m['outer_train']],y[m['outer_train']],features[feature][m['outer_valid']],y[m['outer_valid']],selected['depth'],selected['min_samples_leaf'],20)
                assert opt['scientific'](expected)==opt['scientific'](v['score'])
        before=calls[0]; core['refine'](state,t,y,h,c,features.__getitem__,save); assert calls[0]==before
        state['status']='COMPLETE'; states[str(h)]=state
        errors=[]; counts=[]
        for first,last in c['folds']:
            m=ref['partition_masks'](t,h,first,last); err=np.abs(y[m['outer_valid']]-np.median(y[m['outer_train']]))
            errors.append(float(err.sum())); counts.append(len(err))
        baselines['completed'][str(h)]={'pooled':{'training_median_time_to_high':{'MAE_minutes':sum(errors)/sum(counts)}}}
# Reuse an entire previously verified tree shortlist without any new fit.
old=states[str(h)]; legacy_jobs={f:{'fits':{},'folds':{}} for f in features}
for fit_key,entry in old['tuning'].items(): legacy_jobs[fit_key.split('|')[0]]['fits'][fit_key]=entry['score']
for fold,v in old['policy_folds'].items():
    f=v['selected_feature']
    if f:
        key=f'{f}|{h}|{fold}'
        legacy_jobs[f]['fits'][key+'|outer']=v['score']
        legacy_jobs[f]['folds'][key]={'selected':v['selected_settings']}
fresh={'screen':copy.deepcopy(old['screen'])}
before=calls[0]
with contextlib.redirect_stdout(io.StringIO()): core['refine'](fresh,t,y,h,c,features.__getitem__,lambda:None,legacy_jobs.get)
assert calls[0]==before
assert all(v['origin']=='REUSED_VERIFIED_LEGACY' for v in fresh['policy_folds'].values())
# Bin fitting must not use evaluation values to build edges/medians.
train=np.arange(256,dtype=float); labels=np.repeat([3.,9.],128)
a,pa=core['binned_fit'](train,labels,np.array([0.,100.,np.nan]),np.array([1.,2.,3.]),20)
b,pb=core['binned_fit'](train,labels,np.array([0.,100.,np.nan,1e12]),np.array([-1e9,1e9,9.,0.]),20)
assert a['bin_edges']==b['bin_edges'] and a['bin_medians']==b['bin_medians'] and np.array_equal(pa,pb[:3])
fallback,pred=core['binned_fit'](np.full(30,np.nan),np.arange(30.),np.array([1.,np.nan]),np.array([0.,1.]),20)
assert fallback['status']!='PASS' and np.array_equal(pred,np.full(2,14.5))
try: runner['unpack'](json.dumps({'state':states['15'],'sha256':'wrong'}),'synthetic',15)
except AssertionError: pass
else: raise AssertionError('Corruption accepted')
result=report['make'](c,baselines,states)
assert result['staged_summary.pdf'].startswith(b'%PDF')
assert len(result['cheap_feature_rankings.csv'].decode('utf-8-sig').splitlines())==41
assert len(result['horizon_matrix.csv'].decode('utf-8-sig').splitlines())==11
print('PASS: synthetic all-ten-horizon workflow, 5 folds/6 settings, interruption resume, zero-fit complete resume, outer-score-independent selection, training-only bins, fallback, reference outer scores, purges and PDF/CSV generation.')
