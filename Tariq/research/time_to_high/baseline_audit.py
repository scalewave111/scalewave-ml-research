"""Colab stage 1: research-only timing-label audit and no-feature baselines.
Requires DriveRangeReader (embedded in the companion notebook).
No model fitting, feature search, reserved-row reads, or large local writes.
"""
import io
import json
import hashlib
import zipfile
import re
import numpy as np
import pandas as pd

CONTRACT = {
    'version': 'time_high_btc_stage1_v2_mean',
    'symbol': 'BTCUSDT', 'target': 'time_to_high_min',
    'horizons_minutes': [15,60,240,480,960,2880,4320,7200,12000,18000],
    'reserved_source_partition_start': '2024-07-24',
    'reserved_utc_start_inclusive': '2024-07-24T00:01:00Z',
    'reserved_utc_end_inclusive': '2025-01-24T00:00:00Z',
    'research_label_end_rule': 'strictly before reserved_utc_start_inclusive',
    'stage1_origin_end_exclusive': '2024-07-01T00:00:00Z',
    'stage1_allowed_source_months': ['2022-04','2024-06'],
    'partial_month_policy': 'Skip all July 2024 partitions: no mixed reserved/research member read. July research days remain unused in stage 1.',
    'origin_cadence_minutes': 30,
    'validation_periods': [['2023-04-01','2023-07-01'],['2023-07-01','2023-10-01'],
                           ['2023-10-01','2024-01-01'],['2024-01-01','2024-04-01'],
                           ['2024-04-01','2024-07-01']],
    'training': 'Expanding earlier origins; label_end strictly before each validation start.',
    'baselines': ['training_median_time_to_high','horizon_midpoint','training_mean_time_to_high'],
    'primary_metric': 'MAE_minutes', 'secondary_metric': '100*MAE_minutes/horizon_minutes',
    'ties': 'Earliest occurrence of exact maximum, including minute zero.',
    'row_selection': 'VALID label, exact UTC half-hour origin, finite target in [0,horizon]. Same rows for all three baselines.',
    'test_policy': 'Six-month reserve excluded from all stage-1 reads/scoring/tuning. Final test requires separately frozen policy. Prior project exposure disclosed.',
    'feature_policy': 'No X values read in stage 1. F4/F4Z eligibility and upstream normalization unresolved; no automatic admission or F4Z leakage-free claim.',
    'storage': 'Archive ranges in cloud memory; one horizon at a time. Compact progress/results JSON in one dedicated Drive folder. No models or row-level prediction files.',
    'resume': 'Completed horizons reused only if contract, runner and source ZIP identity match; interrupted horizon restarts.',
    'next_stage': 'Review baseline matrix, then propose an explicit individual-feature batch before training.',
    'magnitude_incumbent': 'Unchanged; no previous magnitude experiments repeated.',
}


def digest(data): return hashlib.sha256(data).hexdigest()


def eligible_members(names, horizon):
    pattern = re.compile(r'(?:^|/)y/horizon_minutes=(\d+)/symbol=BTCUSDT/source_month=(\d{4}-\d{2})/part\.parquet$')
    found = []
    for name in names:
        match = pattern.search(name)
        if match and int(match[1]) == horizon and '2022-04' <= match[2] <= '2024-06':
            found.append((match[2], name))
    months = [m for m,n in found]
    expected = pd.period_range('2022-04', '2024-06', freq='M').astype(str).tolist()
    assert sorted(months) == expected, 'Missing or duplicated research partitions'
    return [n for m,n in sorted(found)]


def evaluate(frame, horizon):
    t = pd.to_datetime(frame.decision_time, utc=True)
    assert not t.duplicated().any()
    assert frame.canonical_symbol.eq('BTCUSDT').all() and frame.horizon_minutes.eq(horizon).all()
    boundary = pd.Timestamp(CONTRACT['reserved_utc_start_inclusive'])
    ends = t + pd.Timedelta(minutes=horizon)
    mask = (t < pd.Timestamp(CONTRACT['stage1_origin_end_exclusive'])) & (ends < boundary)
    mask &= t.dt.minute.mod(30).eq(0) & t.dt.second.eq(0) & t.dt.microsecond.eq(0)
    status = frame.label_status.value_counts().to_dict()
    mask &= frame.label_status.eq('VALID')
    y = frame.loc[mask,'time_to_high_min'].to_numpy(dtype=float)
    origins = pd.DatetimeIndex(t[mask]); label_end = origins+pd.Timedelta(minutes=horizon)
    assert len(y) and np.isfinite(y).all() and ((y>=0)&(y<=horizon)&(y==np.floor(y))).all()
    folds=[]; totals={n:[0.,0] for n in CONTRACT['baselines']}
    for first,last in CONTRACT['validation_periods']:
        start,stop=pd.Timestamp(first,tz='UTC'),pd.Timestamp(last,tz='UTC')
        train=(origins<start)&(label_end<start)
        valid=(origins>=start)&(origins<stop)
        assert train.sum()>=1000 and valid.sum()>0
        predictions=[float(np.median(y[train])),horizon/2,float(np.mean(y[train]))]
        scores={}
        for name,pred in zip(CONTRACT['baselines'],predictions):
            ae=np.abs(y[valid]-pred);total=float(ae.sum());n=int(valid.sum())
            scores[name]={'constant_prediction_minutes':pred,'MAE_minutes':total/n,'MAE_percent_of_horizon':100*total/n/horizon}
            totals[name][0]+=total;totals[name][1]+=n
        folds.append({'start':str(start),'end_exclusive':str(stop),'train_rows':int(train.sum()),
                      'validation_rows':int(valid.sum()),'latest_training_label_end':str(label_end[train].max()),'scores':scores})
    pooled={n:{'rows':v[1],'MAE_minutes':v[0]/v[1],'MAE_percent_of_horizon':100*v[0]/v[1]/horizon} for n,v in totals.items()}
    return {'horizon_minutes':horizon,'raw_partition_status_counts':status,'eligible_research_rows':len(y),
            'first_origin':str(origins.min()),'last_origin':str(origins.max()),'latest_label_end':str(label_end.max()),
            'zero_time_fraction':float(np.mean(y==0)),'endpoint_time_fraction':float(np.mean(y==horizon)),
            'folds':folds,'pooled':pooled}


def self_test():
    names=[f'root/y/horizon_minutes=15/symbol=BTCUSDT/source_month={m}/part.parquet' for m in pd.period_range('2022-04','2024-08',freq='M')]
    chosen=eligible_members(names,15)
    assert len(chosen)==27 and all('2024-07' not in n and '2024-08' not in n for n in chosen)
    start=pd.Timestamp('2023-04-01',tz='UTC')
    origins=pd.DatetimeIndex([start-pd.Timedelta(minutes=16),start-pd.Timedelta(minutes=15)])
    assert ((origins+pd.Timedelta(minutes=15))<start).tolist()==[True,False]
    t=pd.date_range('2022-04-22','2024-06-30',freq='30min',tz='UTC')
    frame=pd.DataFrame({'canonical_symbol':'BTCUSDT','decision_time':t,'horizon_minutes':15,'label_status':'VALID','time_to_high_min':3.})
    result=evaluate(frame,15)
    assert result['pooled']['training_median_time_to_high']['MAE_minutes']==0
    assert result['pooled']['horizon_midpoint']['MAE_minutes']==4.5
    assert result['pooled']['training_mean_time_to_high']['MAE_minutes']==0
    earlier=(t+pd.Timedelta(minutes=15))<start
    frame.loc[earlier,'time_to_high_min']=np.resize([0.,0.,15.],int(earlier.sum()))
    skewed=evaluate(frame,15)['folds'][0]['scores']
    expected=float(frame.loc[earlier,'time_to_high_min'].mean())
    assert skewed['training_mean_time_to_high']['constant_prediction_minutes']==expected
    assert abs(skewed['training_mean_time_to_high']['MAE_minutes']-abs(3-expected))<1e-10
    assert skewed['training_median_time_to_high']['constant_prediction_minutes']==0
    print('PASS: month exclusion, strict label maturity, exact baseline scoring.',flush=True)


def run():
    from google.colab import auth
    import google.auth
    from google.auth.transport.requests import AuthorizedSession
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload
    import pyarrow.parquet as pq
    self_test();auth.authenticate_user();credentials,_=google.auth.default()
    session=AuthorizedSession(credentials);drive=build('drive','v3',credentials=credentials,cache_discovery=False)
    source_id='1y351d1ibxpAH-MxijvGDHPWEfqOJ70aM'
    meta=drive.files().get(fileId=source_id,fields='id,name,size,md5Checksum,modifiedTime').execute()
    identity={'contract':CONTRACT,'source':meta,'runner_sha256':RUNNER_SHA256}
    signature=digest(json.dumps(identity,sort_keys=True).encode())
    folder_name='ScaleWave_Time_To_High_Research'
    folders=drive.files().list(q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",fields='files(id)').execute()['files']
    assert len(folders)<=1,'Ambiguous destination'
    folder=folders[0]['id'] if folders else drive.files().create(body={'name':folder_name,'mimeType':'application/vnd.google-apps.folder'},fields='id').execute()['id']
    matches=drive.files().list(q=f"'{folder}' in parents and name='BASELINE_RESULTS.json' and trashed=false",fields='files(id)').execute()['files']
    assert len(matches)<=1
    result_id=matches[0]['id'] if matches else None
    state={'signature':signature,'identity':identity,'completed':{},'status':'RUNNING','new_model_fits':0,'reserved_partitions_read':0}
    if result_id:
        previous=json.loads(drive.files().get_media(fileId=result_id).execute())
        if previous['signature']==signature:
            state=previous
        else:
            # One explicitly authorized migration: add mean to the verified v1 run.
            assert previous['signature']=='46e1e57083eafdba5f92b59658900d809825a511d152ecfbecf55d333559b6f2', 'Unknown previous run; stop for inspection.'
            assert digest(json.dumps(previous['identity'],sort_keys=True).encode())==previous['signature']
            assert previous['identity']['source']==meta, 'Source changed'
            assert previous['status']=='COMPLETE_RESEARCH_BASELINES_ONLY' and len(previous['completed'])==10
            state['prior_baselines_pending_verification']=previous['completed']
            state['migration']={'from_signature':previous['signature'],'change':'Add historical-mean MAE only; verify original scores and populations unchanged.'}
    def save():
        nonlocal result_id
        media=MediaIoBaseUpload(io.BytesIO(json.dumps(state,indent=2,allow_nan=False).encode()),mimetype='application/json',resumable=False)
        if result_id:drive.files().update(fileId=result_id,media_body=media,fields='id').execute()
        else:result_id=drive.files().create(body={'name':'BASELINE_RESULTS.json','parents':[folder]},media_body=media,fields='id').execute()['id']
    remote=DriveRangeReader(session,source_id,int(meta['size']))
    with zipfile.ZipFile(remote) as archive:
        manifest_name=next(n for n in archive.namelist() if n.endswith('/HANDOFF_2_MANIFEST.json'))
        prefix=manifest_name[:-len('HANDOFF_2_MANIFEST.json')]
        manifest=json.loads(archive.read(manifest_name));expected={prefix+x['path']:x['sha256'] for x in manifest['files']}
        for horizon in CONTRACT['horizons_minutes']:
            key=str(horizon)
            if key in state['completed']:
                print(f'{horizon}m: reusing completed audit.',flush=True);continue
            frames=[];hashes={}
            for number,name in enumerate(eligible_members(archive.namelist(),horizon),1):
                raw=archive.read(name)
                assert digest(raw)==expected[name], 'Source partition hash mismatch'
                hashes[name]=expected[name]
                frames.append(pq.read_table(io.BytesIO(raw),columns=['canonical_symbol','decision_time','horizon_minutes','time_to_high_min','label_status']).to_pandas())
                print(f'{horizon}m: research partition {number}/27 checked',flush=True)
            frame=pd.concat(frames,ignore_index=True);result=evaluate(frame,horizon)
            result['partition_sha256']=hashes
            pending=state.get('prior_baselines_pending_verification',{})
            if key in pending:
                original_columns=json.loads(json.dumps(result))
                original_columns['pooled'].pop('training_mean_time_to_high')
                for fold in original_columns['folds']:
                    fold['scores'].pop('training_mean_time_to_high')
                assert original_columns==pending[key], 'Original baseline evidence changed; not saving this horizon.'
                result['original_baselines_unchanged']=True
                del pending[key]
            state['completed'][key]=result
            state['remote_bytes_read_this_session']=remote.transferred
            save();del frames,frame,raw
            print(f"{horizon}m DONE: median MAE {result['pooled']['training_median_time_to_high']['MAE_minutes']:.3f} min",flush=True)
    latest=drive.files().get(fileId=source_id,fields='id,name,size,md5Checksum,modifiedTime').execute()
    assert latest==meta,'Source changed during run'
    assert not state.get('prior_baselines_pending_verification')
    state.pop('prior_baselines_pending_verification',None)
    state['status']='COMPLETE_RESEARCH_BASELINES_ONLY';save()
    print('\nHorizon(min)  Median MAE  Midpoint MAE  Mean MAE')
    for h in CONTRACT['horizons_minutes']:
        s=state['completed'][str(h)]['pooled'];print(f"{h:12} {s['training_median_time_to_high']['MAE_minutes']:11.3f} {s['horizon_midpoint']['MAE_minutes']:13.3f} {s['training_mean_time_to_high']['MAE_minutes']:9.3f}")
    print('DONE. Send this folder link to Codex: https://drive.google.com/drive/folders/'+folder)
    print('No feature/model fits. Reserved monthly partitions were never opened.')
