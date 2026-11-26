"""Compare fresh inference with the delivered replay; test information timing."""
from pathlib import Path
import sys,json,shutil,gc,os
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'source/SAK_GROSS_V01_COMPLETE'
WORK=ROOT/'independent_inputs'
sys.path.insert(0,str(SOURCE))
from research_data import ResearchData
from engineered_features import engineered

def main():
    results=[]
    for old_path in sorted((SOURCE/'evidence/replay').glob('20*.parquet')):
        before=pd.read_parquet(old_path);after=pd.read_parquet(ROOT/'independent_replay'/old_path.name)
        assert before.index.equals(after.index)
        errors={name:float(np.max(np.abs(before[name].to_numpy()-after[name].to_numpy()))) for name in ['p_up','p_down','predicted_magnitude_bps','signed_score_bps','actual_bps','gross_bps']}
        discrete={name:bool(np.array_equal(before[name].to_numpy(),after[name].to_numpy())) for name in ['eligible','predicted_side','trade_side','model_fold','entry_timestamp_utc','exit_timestamp_utc']}
        results.append({'quarter':old_path.stem,'rows':len(before),'max_absolute_errors':errors,'discrete_exact':discrete})
        assert all(discrete.values()) and max(errors.values())<1e-10
    d=ResearchData(WORK)
    folds=[]
    for q in ['2023Q2','2023Q3','2023Q4','2024Q1','2024Q2','2024Q3','2024Q4']:
        issue,full,tr,inner,cal,te=d.split(q)
        assert (d.panel.available_at.iloc[full]<issue).all()
        assert (d.panel.available_at.iloc[inner]<issue-pd.Timedelta(days=60)).all()
        assert (d.panel.available_at.iloc[cal]<issue-pd.Timedelta(days=30)).all()
        assert not np.intersect1d(full,te).size
        frozen=json.loads((SOURCE/f'model_files/SAK_DIR_V01_{q}_calibration.json').read_text())
        assert len(tr)==frozen['training_rows']
        assert float((d.r[tr]>0).mean())==frozen['prior_up']
        assert float((d.r[inner]>0).mean())==frozen['inner_prior_up']
        folds.append({'quarter':q,'magnitude_train':len(full),'direction_train':len(tr),'inner_train':len(inner),'calibration_rows':len(cal),'outer_rows':len(te),'last_training_label':str(d.panel.available_at.iloc[full].max()),'last_calibration_label':str(d.panel.available_at.iloc[cal].max()),'maturity_and_prior_parity':True})
    cutoff=pd.Timestamp('2024-10-01T00:00:00Z')
    probe=np.flatnonzero((d.panel.index>=cutoff-pd.Timedelta(days=2))&(d.panel.index<=cutoff))
    expected_features=d.prepared(probe)
    mutation=ROOT/'causality_inputs'
    mutation.mkdir(exist_ok=False);(mutation/'research_compact').mkdir();(mutation/'research_week2_v2').mkdir()
    # Hard-link read-only X: only the independent label-copy is mutated.
    os.link(WORK/'research_compact/X.npy',mutation/'research_compact/X.npy')
    shutil.copy2(WORK/'research_compact/matrix_complete.json',mutation/'research_compact/matrix_complete.json')
    changed=d.panel.copy();immature=changed.available_at.ge(cutoff)
    changed.loc[immature,'r']=1_000_000.0
    changed.to_parquet(mutation/'research_week2_v2/aligned_panel.parquet')
    perturbed_rows=int(immature.sum())
    # Prefix check covers derived transforms/scales from the raw full minute grid.
    stop=int(np.searchsorted(pd.date_range(d.meta['start'],periods=len(d.x),freq='min').asi8,cutoff.value,side='right'))
    prefix=engineered(d.x,d.meta,stop=stop)
    ids=d.ids[probe]
    prefix_matrix=np.column_stack([np.asarray(v,dtype='float32')[ids] for v in prefix.values()])
    prefix_matrix[~np.isfinite(prefix_matrix)]=np.nan
    np.testing.assert_allclose(prefix_matrix,d.derived[ids],rtol=0,atol=0,equal_nan=True)
    # Preserve only the expected rows while releasing large full-history arrays.
    del prefix,prefix_matrix,d;gc.collect()
    altered=ResearchData(mutation)
    pd.testing.assert_frame_equal(expected_features,altered.prepared(probe),check_exact=True)
    record={'status':'PASS','replay_rows':sum(x['rows'] for x in results),'replay':results,'folds':folds,
            'future_immature_label_perturbation':{'changed_labels':perturbed_rows,'probe_origins':len(probe),'cutoff':str(cutoff),'prepared_features_exactly_unchanged':True},
            'raw_derived_prefix_check':{'probe_origins':len(probe),'exact':True},
            'limitation':'These checks verify declared temporal construction and one cutoff perturbation, not independence from earlier model/feature selection on the same historical quarters.'}
    (ROOT/'FULL_REPLAY_AND_CAUSALITY.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2),flush=True)

if __name__=='__main__':main()
