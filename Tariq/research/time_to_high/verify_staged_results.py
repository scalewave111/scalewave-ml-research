"""Verify downloaded score receipts without loading datasets or training models."""
import csv, hashlib, json, math
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent
D=ROOT/'Results'/'staged'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sig(v): return hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False).encode()).hexdigest()
def close(a,b): assert math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-9),(a,b)
def date(s): return datetime.fromisoformat(s.replace('Z','+00:00')).replace(tzinfo=timezone.utc)
c=read(D/'CONTRACT.json'); done=read(D/'COMPLETION.json'); ident=c['identity']; config=ident['config']
assert sig(ident)==c['signature']==done['signature']
assert done['status']=='COMPLETE_ALL_TEN_HORIZONS_STAGED' and done['reserved_rows_used'] is False
assert config==read(ROOT/'screen_config.json')
assert ident['pilot_signature']==read(ROOT/'SCREEN_RESULTS.json')['signature']
setup=read(ROOT/'staged_setup_verification.json')
assert all(setup['hashes'][k]==v for k,v in ident['hashes'].items())
baseline=read(ROOT/'BASELINE_RESULTS.json')
assert baseline['signature']==config['baseline_signature']
matrix=list(csv.DictReader((D/'horizon_matrix.csv').read_text(encoding='utf-8').lstrip('\ufeff\u00ef\u00bb\u00bf').splitlines()))
assert len(matrix)==10
results=[]; score_count=0; fallback_count=0
for row,h in zip(matrix,done['horizons']):
    env=read(D/f'horizon_{h}m.json'); s=env['state']; b=baseline['completed'][str(h)]
    assert sig(s)==env['sha256'] and s['signature']==c['signature']
    assert s['status']=='COMPLETE' and s['horizon_minutes']==h
    assert set(s['screen'])==set(config['candidate_features']) and len(s['screen'])==386
    assert set(s['policy_folds'])==set(map(str,range(5)))
    for feature,v in s['screen'].items():
        assert len(v['folds'])==5
        for i,f in enumerate(v['folds']):
            p=s['policy_folds'][str(i)]
            for phase,train_n,valid_n in [('inner',p['inner_train_rows'],p['inner_validation_rows']),('outer',p['train_rows'],p['validation_rows'])]:
                x=f[phase]; score_count+=1
                assert x['train_rows']==train_n and x['validation_rows']==valid_n
                assert 0<=x['train_observed']<=train_n and 0<=x['validation_observed']<=valid_n
                assert x['independent_bin_lookup']=='PASS'
                close(x['MAE_minutes']*valid_n,x['absolute_error_sum'])
        close(v['pooled_MAE_minutes'],sum(f['outer']['absolute_error_sum'] for f in v['folds'])/sum(f['outer']['validation_rows'] for f in v['folds']))
    for i,(first,last) in enumerate(config['folds']):
        p=s['policy_folds'][str(i)]; bf=b['folds'][i]
        assert p['train_rows']==bf['train_rows'] and p['validation_rows']==bf['validation_rows']
        start=date(first); month=start.year*12+start.month-1-3
        inner=start.replace(year=month//12,month=month%12+1)
        assert date(p['inner_latest_training_label_end'])<inner
        assert date(p['inner_latest_validation_label_end'])<start
        assert date(p['outer_latest_training_label_end'])<start
        names=sorted((f for f,v in s['screen'].items() if v['folds'][i]['inner']['status']=='PASS'),key=lambda f:(s['screen'][f]['folds'][i]['inner']['MAE_minutes'],f))[:3]
        assert names==p['shortlist']
        choices=[]
        for f in names:
            for depth,leaf in config['grid']:
                x=s['tuning'][f'{f}|{h}|{i}|inner|{depth}|{leaf}']['score']
                if x['status']=='PASS':
                    assert x['independent_tree_traversal']=='PASS'
                    assert x['coverage']['train_rows']==p['inner_train_rows'] and x['coverage']['validation_rows']==p['inner_validation_rows']
                    close(x['MAE_minutes']*p['inner_validation_rows'],x['absolute_error_sum'])
                    choices.append((x['MAE_minutes'],depth,-leaf,f))
        chosen=min(choices)
        assert p['selected_feature']==chosen[3] and p['selected_settings']=={'depth':chosen[1],'min_samples_leaf':-chosen[2]}
        x=p['score']; close(x['MAE_minutes']*p['validation_rows'],x['absolute_error_sum'])
        if x['status']=='PASS': assert x['independent_tree_traversal']=='PASS' and x['coverage']['validation_rows']==p['validation_rows']
        else: fallback_count+=1
    mae=sum(p['score']['absolute_error_sum'] for p in s['policy_folds'].values())/sum(p['validation_rows'] for p in s['policy_folds'].values())
    name,best=min(b['pooled'].items(),key=lambda kv:kv[1]['MAE_minutes']); base=best['MAE_minutes']
    improvement=100*(base-mae)/base
    assert int(row['horizon_minutes'])==h and int(row['screened_features'])==386 and int(row['policy_folds'])==5 and row['status']=='COMPLETE'
    close(row['baseline_MAE_minutes'],base); close(row['policy_MAE_minutes'],mae); close(row['improvement_percent'],improvement)
    close(row['policy_MAE_percent_of_horizon'],100*mae/h)
    assert date(b['latest_label_end'])<date('2024-07-24T00:01:00Z')
    results.append(dict(horizon_minutes=h,baseline_MAE_minutes=base,policy_MAE_minutes=mae,improvement_percent=improvement,baseline=name))
receipt=dict(status='PASS_SAVED_SCORE_RECEIPTS',contract_signature=c['signature'],horizons=10,feature_horizon_combinations=3860,cheap_fold_scores_checked=score_count,selected_outer_folds=50,selected_fallback_folds=fallback_count,results=results,checks=['Saved state checksums and local frozen configuration/source hashes','All 386 features across ten horizons and five folds','Training-only shortlist and six-setting selection reconstructed from inner scores','Recorded strict label-end purges and baseline population counts','Cheap and tree score arithmetic; pooled matrix arithmetic','Reported independent prediction checks'],limitations=['Raw datasets and model inference were not replayed locally; equal row counts do not independently prove identical row identities','Reserve exclusion is supported by frozen code and saved receipts, not independent raw-data replay','Development evidence only; shortlist can miss useful features; upstream publication timing remains unresolved'])
(D/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
progress=read(ROOT/'research_progress.json')
progress.update(current_step='All-ten-horizon staged time_to_high_min screen completed; saved score receipts verified.',next='Review the small shorter-horizon gains and longer-horizon losses; prepare time_to_low_min with its own baselines and staged contract.',staged_results_folder='https://drive.google.com/drive/folders/1-Ly5SHesR7s2xd9eftlxlVCvsmOyJHlI',staged_verification='Results/staged/verification.json')
(ROOT/'research_progress.json').write_text(json.dumps(progress,indent=2)+'\n')
print(json.dumps({'status':receipt['status'],'checked_scores':score_count,'results':results},indent=2))
