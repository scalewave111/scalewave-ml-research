from pathlib import Path
import json,subprocess,sys
import numpy as np
import pandas as pd
import joblib
import lstm_walk_forward as q

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'experiments/baseline_single_feature_v6'
OUT=HERE/'experiments/vol_ratio_train_test_review_v1'
OUT.mkdir(exist_ok=True)
x,y,dev,ref=joblib.load(SOURCE/'prepared.joblib')
decisions=json.loads((SOURCE/'SELECTIONS.json').read_text())
parts=[];windows=[];hashes={}
for d in decisions:
 issue=pd.Timestamp(d['issue']);folder=SOURCE/issue.strftime('%Y%m%d')
 candidate=next(c for c in d['candidates'] if c['name']=='vol_ratio')
 keys=y.index[y.label_end<issue]
 def forecast(pair_path,rf_path,weight,columns):
  a=joblib.load(pair_path);b=joblib.load(rf_path)
  for path,model in [(pair_path,a),(rf_path,b)]:
   hashes[str(path)]=q.sha(path)
   assert model['train_keys'].equals(keys) and pd.Timestamp(model['issuance'])==issue
   assert model['prep']['columns']==columns and y.loc[keys,'label_end'].max()<issue
  return (1-weight)*q.b.predict(a,x.loc[keys])+weight*q.b.predict(b,x.loc[keys])
 cp=forecast(folder/'vol_ratio_pair_outer.joblib',folder/'vol_ratio_rf_outer.joblib',candidate['weight'],q.b.PAIR+['PRICE__vol_ratio_4h_24h'])
 bp=forecast(q.b.BASE/'models'/f'{issue:%Y-%m}__report_outer_pair__expanding.joblib',q.b.BASE/'models'/f'{issue:%Y-%m}__report_outer_rf__expanding.joblib',d['baseline_weight'],q.b.PAIR)
 truth=y.loc[keys,'magnitude_log_bps']
 parts.append(pd.DataFrame(dict(vol_ratio=cp,baseline=bp,truth=truth,issue=str(issue)),index=keys))
 for model,pred in [('vol_ratio',cp),('baseline',bp)]:windows.append(dict(issue=str(issue),model=model,split='training_in_sample',**q.b.metric(pred,truth)))
train=pd.concat(parts)
rows=[dict(split='training_in_sample',model=model,unique_rows=train.index.nunique(),**q.b.metric(train[model],train.truth)) for model in ['baseline','vol_ratio']]
saved=pd.read_csv(SOURCE/'scores.csv')
for row in saved[saved.model.isin(['baseline','vol_ratio'])].to_dict('records'):
 row['unique_rows']=row['rows'];rows.append(row)
result=pd.DataFrame(rows)
result.to_csv(OUT/'scores.csv',index=False)
pd.DataFrame(windows).to_csv(OUT/'training_window_scores.csv',index=False)
for path,digest in hashes.items():assert q.sha(path)==digest
manifest=dict(status='CALCULATED_FROM_SAVED_MODELS',no_training_performed=True,training_rows=len(train),unique_training_origins=train.index.nunique(),training_definition='Pooled in-sample predictions from the four outer fitted models; earlier origins repeat across expanding training windows. Each fold uses its own previously selected blend weight.',model_hashes=hashes,prepared_data_sha256=q.sha(SOURCE/'prepared.joblib'),existing_scores_sha256=q.sha(SOURCE/'scores.csv'),script_sha256=q.sha(__file__))
q.dump(OUT/'AUDIT.json',manifest)
text='Fixed volatility-ratio candidate: training, validation and historical walk-forward test scores.\n\n'+result.to_string(index=False)+'\n\nTraining is in-sample and pooled across four expanding model fits. Origins repeat across training windows. Different splits cover different time periods, so their MAE levels also reflect different target distributions. No model was trained or changed.\n'
(OUT/'RESULTS.txt').write_text(text,encoding='utf-8')
context=HERE/'PROJECT_CONTEXT.md';old=context.read_text(encoding='utf-8');marker='## Volatility-ratio train/test metrics — 2026-11-04'
if marker not in old:
 c=result[result.model=='vol_ratio'].set_index('split')
 note='\n\n'+marker+'\n\nUser requested training and test MAE/IC for the frozen three-feature candidate. Calculated in-sample predictions using the saved four outer models and their original validation-selected blend weights; no retraining or model changes.\n\n'
 for split in ['training_in_sample','validation','test']:note+=f"- {split}: MAE {c.loc[split,'mae']:.6f} bps; Spearman IC {c.loc[split,'rank_ic']:.6f}.\n"
 note+=f"\nPooled training has {len(train)} prediction rows over {train.index.nunique()} unique origins; dates repeat across expanding training windows. Sources: `experiments/vol_ratio_train_test_review_v1/scores.csv`, `training_window_scores.csv`, `AUDIT.json`, and `RESULTS.txt`. Runner: `report_vol_ratio_train_test.py`. Test remains the already examined historical walk-forward comparison.\n"
 context.write_text(old+note,encoding='utf-8')
print(result.to_string(index=False))
print(pd.DataFrame(windows).to_string(index=False))
subprocess.run([sys.executable,str(HERE/'refresh_file_index.py')],check=True,capture_output=True,text=True)
print('Saved train/validation/test metrics; model files unchanged; index refreshed.')
