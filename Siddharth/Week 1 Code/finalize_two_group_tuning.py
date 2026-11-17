"""Wait for requested run, independently audit predictions, and record final status."""
import time,json,hashlib,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import two_group_tuning as t
s=t.s
def run():
    while not (t.OUT/'COMPLETE.json').exists():time.sleep(30)
    x,y,dev,ref=joblib.load(t.e.OUT/'prepared.joblib');p=pd.read_csv(t.OUT/'predictions.csv',index_col=0,parse_dates=True)
    assert p.index.equals(dev);np.testing.assert_allclose(p.baseline,ref.report_hybrid,atol=1e-8,rtol=0)
    for method in t.CONFIGS:
        for group in t.GROUPS:
            for issue in s.q.STARTS:
                f=t.OUT/method/group/f'{issue:%Y%m%d}';trials=json.loads((f/'TRIALS.json').read_text());chosen=json.loads((f/'SELECTION.json').read_text())
                assert len(trials)==2 and chosen==min(trials,key=lambda r:(r['mae'],r['id']))
                t.e.configure(method,chosen['config']);inner,ik,vk,ot,te=s.q.windows(y,dev,issue)
                ai=joblib.load(f/f"c{chosen['id']}"/'inner/bundle.joblib');ao=joblib.load(f/'outer/bundle.joblib')
                assert ai['keys'].equals(ik) and ao['keys'].equals(ot)
                assert ai['cols']==ao['cols']==sorted(t.GROUPS[group])
                assert y.loc[ik,'label_end'].max()<inner and y.loc[vk,'label_end'].max()<issue and y.loc[ot,'label_end'].max()<issue
                assert abs(s.metrics(s.predict(ai,x,vk),y.loc[vk,'magnitude_log_bps'])['mae']-chosen['mae'])<1e-8
                np.testing.assert_allclose(s.predict(ao,x,te),p.loc[te,f'{method}_{group}'],atol=1e-8,rtol=0)
                if method=='mlp':
                    confirm=joblib.load(f/'confirmation/bundle.joblib');assert confirm['epochs']==ao['epochs'] and len(ao['epochs'])==3
                else:assert ai['weight']==ao['weight']
    for row in json.loads((t.OUT/'SELECTIONS.json').read_text()):
        issue=pd.Timestamp(row['issue']);method=row['method'];_,_,_,_,te=s.q.windows(y,dev,issue)
        options={g:json.loads((t.OUT/method/g/f'{issue:%Y%m%d}'/'SELECTION.json').read_text()) for g in t.GROUPS}
        assert row['group']==min(options,key=lambda g:(options[g]['mae'],len(t.GROUPS[g])))
        np.testing.assert_allclose(p.loc[te,method+'_selected'],p.loc[te,method+'_'+row['group']],atol=1e-8,rtol=0)
    manifest=json.loads((t.OUT/'MANIFEST.json').read_text());assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha for path,sha in manifest['inputs_and_dependencies'].items())
    scores=pd.read_csv(t.OUT/'scores.csv')
    for row in scores.to_dict('records'):assert abs(np.mean(abs(p[row['model']]-p.truth))-row['mae'])<1e-8
    s.dump(t.OUT/'INDEPENDENT_AUDIT.json',dict(status='PASS',rows=len(p),selection_argmin=True,purges=True,replay=True,sources_unchanged=True))
    summary=scores.to_string(index=False)+'\n\nPrimary strategies: rf_selected and mlp_selected. Fixed-group scores are exploratory secondary comparisons. Previously examined historical data; no claim of untouched confirmation. Independent audit passed.\n'
    (t.OUT/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    with (s.ROOT/'PROJECT_CONTEXT.md').open('a',encoding='utf-8') as f:f.write('\n\n## Two-group tuning finished\nExecuted and independently audited. Results: Code/experiments/two_group_tuning_v1/SUMMARY.txt, scores.csv, uncertainty.csv and SELECTIONS.json.\n'+scores.to_string(index=False)+'\n')
    subprocess.run([sys.executable,str(s.ROOT/'refresh_file_index.py')],stdout=subprocess.DEVNULL)
    print(summary,flush=True)
if __name__=='__main__':
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):run()
