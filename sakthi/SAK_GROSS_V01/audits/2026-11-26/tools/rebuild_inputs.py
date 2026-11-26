"""Independent reconstruction using the existing raw H4 arena and native release labels."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib,json,shutil
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

AUDIT=Path(__file__).resolve().parents[1]
SOURCE=AUDIT/'source/SAK_GROSS_V01_COMPLETE'
WORK=AUDIT/'independent_inputs'
H4=Path(r'D:\Team\November\HANDOFF_4_ML_RESEARCH_V1')
CACHE=Path(r'D:\APEX_ARENA_V2\BTCUSDT_1m_eligible')

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    WORK.mkdir(exist_ok=False)
    compact=WORK/'research_compact';compact.mkdir()
    paneldir=WORK/'research_week2_v2';paneldir.mkdir()
    fields=json.loads((SOURCE/'contracts/feature_config.json').read_text())
    names=[f['field_id'] for f in fields]
    meta=json.loads((SOURCE/'contracts/matrix_complete.json').read_text())
    registry={r['field_id']:r for r in json.loads((H4/'catalog/FEATURE_REGISTRY.json').read_text())}
    keys=['schema_key','source_field','validity_guards','available_after_rule','source_type']
    differences=[{'field':f['field_id'],'key':k} for f in fields for k in keys if f.get(k)!=registry[f['field_id']].get(k)]
    if differences:raise ValueError(f'Feature contract differs from native H4: {differences}')
    assert meta['columns'][:len(names)]==names
    grid=pd.date_range(meta['start'],periods=meta['shape'][0],freq='min')
    matrix=np.lib.format.open_memmap(compact/'X.npy',mode='w+',dtype='float32',shape=tuple(meta['shape']))
    matrix[:]=np.nan
    covered=np.zeros(len(grid),dtype=bool);files=[]
    for i,part in enumerate(sorted(CACHE.glob('date=*/part.parquet'))):
        day=pd.Timestamp(part.parent.name.split('=')[1],tz='UTC')
        if day>grid.max():continue
        frame=pq.read_table(part,columns=['decision_time']+names).to_pandas().set_index('decision_time')
        if not frame.index.is_unique:raise ValueError(f'Duplicate feature timestamps {part}')
        loc=grid.get_indexer(frame.index);good=loc>=0
        if covered[loc[good]].any():raise ValueError('Duplicate timestamps across cache partitions')
        covered[loc[good]]=True
        matrix[loc[good],:len(names)]=frame.iloc[np.flatnonzero(good)].to_numpy('float32')
        files.append({'path':str(part),'bytes':part.stat().st_size,'mtime_ns':part.stat().st_mtime_ns})
        if i%100==0:print('CACHE_PARTITIONS',i+1,flush=True)
    if not covered.all():raise ValueError(f'Missing cache rows: {int((~covered).sum())}')
    for k,driver in enumerate(meta['drivers']):
        series=pd.Series(np.array(matrix[:,names.index(driver)]))
        offset=len(names)+3*k
        matrix[:,offset]=series.shift(1)
        matrix[:,offset+1]=series.shift(5)
        matrix[:,offset+2]=series.rolling(15,min_periods=15).mean()
    matrix.flush();del matrix
    for name in ['matrix_complete.json','feature_config.json']:
        shutil.copy2(SOURCE/'contracts'/name,compact/name)
    calendar=pd.read_parquet(SOURCE/'contracts/decision_calendar.parquet').index
    old=pd.read_parquet(AUDIT.parent/'Apex R1/inputs/legacy_modelling/FAST_15M_V1/JOINED.parquet',columns=[]).index
    assert calendar.equals(old)
    sys.path.insert(0,str(H4))
    from ml.dataset import Dataset
    ds=Dataset(H4)
    y=ds.y('BTCUSDT',15,str(calendar.min()),str(calendar.max()+pd.Timedelta(minutes=17)))
    y=y.loc[y.label_status.eq('VALID')&~y.binding_crossing.astype(bool)].set_index('decision_time').sort_index()
    assert y.index.is_unique
    aligned=y.reindex(calendar+pd.Timedelta(minutes=1))
    panel=pd.DataFrame({'r':aligned.signed_return_simple_bps.to_numpy(), 'available_at':aligned.target_available_at.to_numpy()},index=calendar)
    panel.index.name='decision_time'
    assert (panel.available_at.dropna()>=panel.index[panel.available_at.notna()]+pd.Timedelta(minutes=16)).all()
    assert panel.r.notna().sum()==1418367
    panel.to_parquet(paneldir/'aligned_panel.parquet')
    identities=json.loads((SOURCE/'model_files/training_data_identity.json').read_text())
    hashes={name:{'actual':sha(WORK/name),'expected':identities[name],'exact_bytes':sha(WORK/name)==identities[name]} for name in identities if (WORK/name).exists()}
    result={'status':'COMPLETE','feature_contract_fields_matched':len(fields),'cache_partitions':len(files),'cache_rows_covered':int(covered.sum()),'shape':meta['shape'],'calendar_equals_apex_historical_index':True,'native_label_valid_rows':int(panel.r.notna().sum()),'training_input_hash_checks':hashes,'cache_partition_inventory':files,'scope':'Reconstructed full feature matrix from existing H4 arena plus native H4 target. Native-source feature probes are a separate check.'}
    (AUDIT/'INPUT_RECONSTRUCTION.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='cache_partition_inventory'},indent=2),flush=True)

if __name__=='__main__':main()
