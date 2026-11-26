"""Bounded direct native-release check of all 86 raw Sakthi inputs."""
from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
import pandas as pd
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
H4=Path(r'D:\Team\November\HANDOFF_4_ML_RESEARCH_V1')
CACHE=Path(r'D:\APEX_ARENA_V2\BTCUSDT_1m_eligible')
fields=[x['field_id'] for x in json.loads((ROOT/'source/SAK_GROSS_V01_COMPLETE/contracts/feature_config.json').read_text())]
sys.path.insert(0,str(H4))
from ml.dataset import Dataset
dataset=Dataset(H4);checks=[]
for day in ['2022-04-22','2023-04-01','2023-10-01','2024-01-01','2024-07-01','2024-10-01']:
    native,mask,meta=dataset.x_day(day,'BTCUSDT',fields)
    cache=pq.ParquetFile(CACHE/f'date={day}/part.parquet').read(columns=['decision_time']+fields).to_pandas().set_index('decision_time')
    expected=native[fields].astype('float32');expected.index.name='decision_time'
    pd.testing.assert_frame_equal(cache,expected,check_exact=True,check_freq=False)
    assert native.notna().equals(mask)
    known=meta.x_available_after.notna()
    violations=int((meta.x_available_after[known]>meta.index[known]).sum())
    assert violations==0
    record={'day':day,'rows':len(cache),'fields':len(fields),'exact_native_cache_parity':True,'availability_violations':violations,'guards_match_missingness':True}
    checks.append(record);print(record,flush=True)
(ROOT/'NATIVE_FEATURE_PROBES.json').write_text(json.dumps({'checks':checks,'checked_cells':sum(x['rows']*x['fields'] for x in checks),'scope':'All 86 fields on six fixed days; not an exhaustive native-source check of every day'},indent=2),encoding='utf-8')
