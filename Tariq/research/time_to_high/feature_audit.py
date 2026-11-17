"""Colab sampled feature preflight. No model fitting or feature-performance ranking."""
import io
import json
import re
import hashlib
import zipfile
import calendar
from collections import Counter
import numpy as np
import pandas as pd

AUDIT_CONTRACT = {
    'version':'time_high_btc_feature_preflight_v2_f4z',
    'symbol':'BTCUSDT', 'target':'time_to_high_min',
    'horizons_minutes':[15,60,240,480,960,2880,4320,7200,12000,18000],
    'research_origin_end_exclusive':'2024-07-01T00:00:00Z',
    'reserved_start_inclusive':'2024-07-24T00:01:00Z',
    'source_months':['2022-04','2024-06'],
    'sampling':'Three predetermined source days per month: first available day, midpoint of available day range, last day. April 2022 starts day 22. Exact half-hour UTC origins only.',
    'purpose':'Sampled structural/coverage audit of 522 F4Z features only; not full-period eligibility, predictive ranking or scientific certification.',
    'feature_gate':'Finite value; true valid flag; known family available_after <= decision_time. F4Z numerical diagnostics additionally require baseline_n >= 1000; scientific statuses are reported, never relabelled approved.',
    'scope':'BTCUSDT only. No July 2024 or later source or target partitions opened. Metadata may list their names.',
    'next':'Review sampled diagnostics, propose first fixed individual-feature batch, require full-training coverage and provenance decisions before fits.',
    'storage':'Cloud memory only; one month at a time. One compact FEATURE_AUDIT_RESULTS.json in existing research Drive folder. Resume completed months with identical source identities/code/contract.',
    'reserve':'Final six months excluded; previous project exposure remains disclosed.',
    'model_fits':0,
}
FAMILIES=['basis','flow','funding','liquidation','oi','price','volatility']
SOURCE_IDS={'X':'1RXZDR_PaLkKMYtuswOBnl4_0eG9_LkNx','Y':'1y351d1ibxpAH-MxijvGDHPWEfqOJ70aM'}


def sha(data): return hashlib.sha256(data).hexdigest()


def sample_days(month):
    year,mo=map(int,month.split('-'));first=22 if month=='2022-04' else 1
    last=calendar.monthrange(year,mo)[1]
    return [f'{month}-{day:02}' for day in sorted({first,(first+last)//2,last})]


def sampled_index(days):
    values=[]
    for day in days:
        start=pd.Timestamp(day,tz='UTC')
        values.extend(pd.date_range(start+pd.Timedelta(minutes=30),start+pd.Timedelta(days=1),freq='30min'))
    result=pd.DatetimeIndex(values)
    return result[result<pd.Timestamp(AUDIT_CONTRACT['research_origin_end_exclusive'])]


def assess(values, flags, available, origins, baseline=None):
    v=pd.to_numeric(pd.Series(values),errors='coerce').to_numpy(dtype=float)
    valid=pd.Series(flags).fillna(False).astype(bool).to_numpy()
    a=pd.DatetimeIndex(pd.to_datetime(available,utc=True))
    known=~a.isna();late=known&(a>origins);finite=np.isfinite(v)
    good=finite&valid&known&~late
    low=np.zeros(len(v),dtype=bool)
    if baseline is not None:
        n=pd.to_numeric(pd.Series(baseline),errors='coerce').to_numpy(dtype=float)
        low=~np.isfinite(n)|(n<1000);good &= ~low
    admitted=v[good]
    return good,{'sample_rows':len(v),'finite_rows':int(finite.sum()),'valid_flag_rows':int(valid.sum()),
                 'availability_missing_rows':int((~known).sum()),'availability_late_rows':int(late.sum()),
                 'baseline_insufficient_rows':int(low.sum()),'usable_by_recorded_metadata_rows':int(good.sum()),
                 'minimum':float(admitted.min()) if len(admitted) else None,
                 'maximum':float(admitted.max()) if len(admitted) else None}


def merge_record(total, record):
    if not total: return json.loads(json.dumps(record))
    for key,value in record.items():
        if key=='minimum':
            if value is not None: total[key]=value if total[key] is None else min(total[key],value)
        elif key=='maximum':
            if value is not None: total[key]=value if total[key] is None else max(total[key],value)
        elif key=='scientific_status_counts':
            counts=Counter(total[key]);counts.update(value);total[key]=dict(counts)
        elif key=='target_matched_usable_rows':
            for h,n in value.items():total[key][h]+=n
        else: total[key]+=value
    return total


def migrate_f4z(previous, identity):
    """Reuse complete v1 months only; reject unrelated settings or source changes."""
    assert previous['signature']==sha(json.dumps(previous['identity'],sort_keys=True).encode()), 'Invalid saved identity'
    assert previous['identity']['runner_sha256']=='27e5d9a82f0c22f0ac3c594cc6d346f7936621254512f00df3a96c16a6cfc641', 'Unknown prior runner'
    expected=dict(AUDIT_CONTRACT)
    expected['version']='time_high_btc_feature_preflight_v1'
    expected['purpose']='Sampled structural/coverage audit of all 825 features; not full-period eligibility, predictive ranking or scientific certification.'
    assert previous['identity']['contract']==expected, 'Other contract settings changed'
    assert previous['identity']['sources']==identity['sources'], 'Source archive identity changed'
    state=json.loads(json.dumps(previous))
    retained={k:v for k,v in state['features'].items() if k.startswith('F4Z__')}
    assert not state['completed_months'] or len(retained)==522, 'Incomplete saved F4Z feature set'
    assert len(set(state['completed_months']))==len(state['completed_months'])
    for month in state['completed_months']:
        assert month in state['month_checks'] and state['month_checks'][month]['source_days']==sample_days(month)
    for stats in retained.values():
        for key in ['sample_screen','upstream_timing_certified','eligibility']:stats.pop(key,None)
    state['migration']={'previous_signature':state['signature'],'reused_completed_months':list(state['completed_months']),
                        'scope_change':'Retain F4Z statistics only; prior month receipts retain hashes of historical F4 reads. No new F4 data reads.'}
    state['features']=retained;state.pop('summary',None)
    state['identity']=identity;state['signature']=sha(json.dumps(identity,sort_keys=True).encode())
    state['status']='RUNNING_SAMPLED_AUDIT'
    return state


def self_test():
    assert sample_days('2022-04')==['2022-04-22','2022-04-26','2022-04-30']
    assert len(sampled_index(sample_days('2024-06')))==143
    t=pd.date_range('2023-01-01',periods=5,freq='30min',tz='UTC')
    a=list(t);a[1]=t[1]+pd.Timedelta(minutes=1);a[2]=pd.NaT
    mask,counts=assess([1,2,3,4,np.nan],[True]*5,a,t,[1000,1000,1000,999,1000])
    assert mask.tolist()==[True,False,False,False,False]
    assert counts['availability_late_rows']==1 and counts['availability_missing_rows']==1
    combined=merge_record(merge_record({},dict(counts)),dict(counts));assert combined['sample_rows']==10 and combined['minimum']==1
    print('PASS: fixed-date sampling, availability/missingness gates, baseline threshold and aggregation.',flush=True)


def run():
    from google.colab import auth
    import google.auth
    from google.auth.transport.requests import AuthorizedSession
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload
    import pyarrow.parquet as pq
    self_test();auth.authenticate_user();creds,_=google.auth.default()
    session=AuthorizedSession(creds);drive=build('drive','v3',credentials=creds,cache_discovery=False)
    metas={k:drive.files().get(fileId=v,fields='id,name,size,md5Checksum,modifiedTime').execute() for k,v in SOURCE_IDS.items()}
    identity={'contract':AUDIT_CONTRACT,'sources':metas,'runner_sha256':RUNNER_SHA256}
    signature=sha(json.dumps(identity,sort_keys=True).encode())
    folder='1DXhoOLybfhEIt3phvPfG3cL3M31FPFEC'
    files=drive.files().list(q=f"'{folder}' in parents and name='FEATURE_AUDIT_RESULTS.json' and trashed=false",fields='files(id)').execute()['files']
    assert len(files)<=1,'Ambiguous destination'
    file_id=files[0]['id'] if files else None
    state={'identity':identity,'signature':signature,'status':'RUNNING_SAMPLED_AUDIT','completed_months':[],
           'features':{},'month_checks':{},'model_fits':0,'reserved_partitions_read':0}
    if file_id:
        state=json.loads(drive.files().get_media(fileId=file_id).execute())
        if state['signature']!=signature:
            state=migrate_f4z(state,identity)
            print(f"Scope narrowed to F4Z; reusing {len(state['completed_months'])} completed months.",flush=True)
    def save():
        nonlocal file_id
        media=MediaIoBaseUpload(io.BytesIO(json.dumps(state,indent=2,allow_nan=False).encode()),mimetype='application/json')
        if file_id:drive.files().update(fileId=file_id,media_body=media,fields='id').execute()
        else:file_id=drive.files().create(body={'name':'FEATURE_AUDIT_RESULTS.json','parents':[folder]},media_body=media,fields='id').execute()['id']
    readers={k:DriveRangeReader(session,v,int(metas[k]['size'])) for k,v in SOURCE_IDS.items()}
    with zipfile.ZipFile(readers['X']) as xzip,zipfile.ZipFile(readers['Y']) as yzip:
        manifest_name=next(n for n in yzip.namelist() if n.endswith('/HANDOFF_2_MANIFEST.json'))
        root=manifest_name[:-len('HANDOFF_2_MANIFEST.json')]
        expected={root+e['path']:e['sha256'] for e in json.loads(yzip.read(manifest_name))['files']}
        raw= yzip.read(root+'FEATURE_DICTIONARY.json');assert sha(raw)==expected[root+'FEATURE_DICTIONARY.json']
        dictionary=json.loads(raw);assert len(dictionary)==825
        dictionary=[entry for entry in dictionary if entry['layer']=='F4Z']
        assert len(dictionary)==522
        groups={}
        for entry in dictionary:groups.setdefault((entry['layer'],entry['family']),[]).append(entry)
        checksum_name=next(n for n in xzip.namelist() if n=='FILES.sha256')
        checksums={}
        for line in xzip.read(checksum_name).decode().splitlines():
            if line.strip():
                h,name=line.split(maxsplit=1);checksums[name.lstrip('*')]=h
        # Resolve only research BTC feature members; no market values read during listing.
        source_map={}
        for name in xzip.namelist():
            if not name.endswith('.parquet') or 'symbol=BTCUSDT/' not in name:continue
            date=re.search(r'/dates/[^/=]+=(\d{4}-\d{2}-\d{2})/',name)
            family=re.search(r'/family=([^/]+)/',name)
            layer=re.search(r'/(F4Z)/',name)
            if date and family and layer and '2022-04-22'<=date[1]<='2024-06-30':
                key=(date[1],layer[1],family[1]);assert key not in source_map;source_map[key]=name
        required={(day,layer,family) for month in pd.period_range('2022-04','2024-06',freq='M').astype(str)
                  for day in sample_days(month) for layer,family in groups}
        assert required <= set(source_map), 'Missing sampled feature partitions; stop before reading data.'
        for month in pd.period_range('2022-04','2024-06',freq='M').astype(str):
            if month in state['completed_months']:
                print(month+': reused completed month.',flush=True);continue
            days=sample_days(month);origins=sampled_index(days);target_masks={};target_counts={};unit_features={};member_hashes={}
            for h in AUDIT_CONTRACT['horizons_minutes']:
                name=root+f'y/horizon_minutes={h}/symbol=BTCUSDT/source_month={month}/part.parquet'
                raw=yzip.read(name);assert sha(raw)==expected[name];member_hashes[name]=expected[name]
                frame=pq.read_table(io.BytesIO(raw),columns=['canonical_symbol','decision_time','horizon_minutes','time_to_high_min','label_status']).to_pandas()
                frame.index=pd.to_datetime(frame.decision_time,utc=True)
                assert frame.index.is_unique and origins.isin(frame.index).all()
                frame=frame.loc[origins];assert frame.canonical_symbol.eq('BTCUSDT').all() and frame.horizon_minutes.eq(h).all()
                good=frame.label_status.eq('VALID').to_numpy()
                vals=frame.time_to_high_min.to_numpy(dtype=float)
                assert np.isfinite(vals[good]).all() and ((vals[good]>=0)&(vals[good]<=h)).all()
                assert ((origins+pd.Timedelta(minutes=h))<pd.Timestamp(AUDIT_CONTRACT['reserved_start_inclusive'])).all()
                target_masks[str(h)]=good;target_counts[str(h)]=int(good.sum())
            print(month+': target alignment checked for all horizons.',flush=True)
            for (layer,family),entries in sorted(groups.items()):
                frames=[]
                for day in days:
                    name=source_map[(day,layer,family)]
                    raw=xzip.read(name);assert sha(raw)==checksums[name];member_hashes[name]=checksums[name]
                    frame=pq.read_table(io.BytesIO(raw)).to_pandas()
                    times=pd.DatetimeIndex(pd.to_datetime(frame.decision_time,utc=True))
                    expected_times=pd.date_range(pd.Timestamp(day,tz='UTC')+pd.Timedelta(minutes=1),periods=1440,freq='min')
                    assert times.equals(expected_times) and frame.canonical_symbol.eq('BTCUSDT').all()
                    frame.index=times;frames.append(frame.loc[times.isin(origins)])
                frame=pd.concat(frames).sort_index();assert frame.index.equals(origins)
                for entry in entries:
                    field=entry['source_field'];base=field[:-7]
                    good,stats=assess(frame[field],frame[base+'__valid'],frame.available_after,origins,
                                      frame[base+'__baseline_n'] if layer=='F4Z' else None)
                    stats['scientific_status_counts']=dict(Counter(frame[base+'__scientific_status'].fillna('MISSING').astype(str))) if layer=='F4Z' else {}
                    stats['target_matched_usable_rows']={h:int((good&mask).sum()) for h,mask in target_masks.items()}
                    unit_features[entry['column']]=stats
                del frame,frames,raw
                print(f'{month}: {layer}/{family} checked.',flush=True)
            assert len(unit_features)==522
            for col,stats in unit_features.items():state['features'][col]=merge_record(state['features'].get(col,{}),stats)
            state['month_checks'][month]={'source_days':days,'sample_origins':len(origins),'valid_target_rows':target_counts,'member_sha256':member_hashes}
            state['completed_months'].append(month)
            state['remote_bytes_read_this_session']={k:v.transferred for k,v in readers.items()};save()
    for k,file in SOURCE_IDS.items():
        assert drive.files().get(fileId=file,fields='id,name,size,md5Checksum,modifiedTime').execute()==metas[k]
    summary=Counter()
    for col,s in state['features'].items():
        if not s['usable_by_recorded_metadata_rows']:reason='NO_USABLE_SAMPLED_VALUES'
        elif s['minimum']==s['maximum']:reason='CONSTANT_IN_USABLE_SAMPLE'
        else:reason='VARIED_IN_SAMPLE_FULL_PERIOD_CHECK_REQUIRED'
        s['sample_screen']=reason;s['upstream_timing_certified']=False
        s['eligibility']='PENDING_UPSTREAM_NORMALIZATION_EVIDENCE' if col.startswith('F4Z__') else 'RESEARCH_CANDIDATE_PENDING_FULL_TRAINING_AND_PROVENANCE_CHECKS'
        summary[reason]+=1
    state['summary']=dict(summary);state['status']='COMPLETE_SAMPLED_PREFLIGHT_NOT_FULL_ELIGIBILITY';save()
    print('DONE: 522 F4Z features audited on fixed research dates. No model fits or predictive ranking.')
    print(json.dumps(state['summary']))
    print('Send this folder link: https://drive.google.com/drive/folders/'+folder)
