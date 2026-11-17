"""BTCUSDT timing feature screen: pilot by default, full scope only after review.
Colab supplies CONFIG, RUNNER_SHA256 and DriveRangeReader in the notebook.
"""
import io
import json
import re
import hashlib
import zipfile
import struct
import zlib
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd


def sha(data): return hashlib.sha256(data).hexdigest()
def signature(obj): return sha(json.dumps(obj,sort_keys=True,allow_nan=False).encode())
def utc(value): return pd.Timestamp(value,tz='UTC')


def partition_masks(times, horizon, first, last):
    start,stop=utc(first),utc(last)
    ends=times+pd.Timedelta(minutes=horizon)
    inner_start=start-pd.DateOffset(months=3)
    return {'inner_train':(times<inner_start)&(ends<inner_start),
            'inner_valid':(times>=inner_start)&(times<start)&(ends<start),
            'outer_train':(times<start)&(ends<start),
            'outer_valid':(times>=start)&(times<stop)}


def preprocess(x_train,x_predict,minimum):
    observed=np.isfinite(x_train)
    if observed.sum()<minimum or np.unique(x_train[observed]).size<2:
        return None
    median=float(np.median(x_train[observed]))
    def transform(x):
        missing=~np.isfinite(x)
        return np.column_stack([np.where(missing,median,x),missing.astype(float)]).astype(np.float32)
    return transform(x_train),transform(x_predict),median


def manual_predict(model,x):
    # Independent traversal checks saved sklearn leaf predictions without another fit.
    tree=model.tree_;nodes=np.zeros(len(x),dtype=np.int64)
    for _ in range(model.get_depth()+1):
        active=tree.children_left[nodes]!=-1
        if not active.any():break
        rows=np.flatnonzero(active);current=nodes[rows]
        left=x[rows,tree.feature[current]]<=tree.threshold[current]
        nodes[rows]=np.where(left,tree.children_left[current],tree.children_right[current])
    assert (tree.children_left[nodes]==-1).all()
    return tree.value[nodes,0,0]


def tree_fit(x_train,y_train,x_valid,y_valid,depth,leaf,minimum):
    from sklearn.tree import DecisionTreeRegressor
    start=time.perf_counter();prepared=preprocess(x_train,x_valid,minimum)
    coverage={'train_observed':int(np.isfinite(x_train).sum()),'train_rows':len(x_train),
              'validation_observed':int(np.isfinite(x_valid).sum()),'validation_rows':len(x_valid)}
    if prepared is None:return {'status':'INSUFFICIENT_TRAINING_COVERAGE_OR_VARIATION','fit_count':0,'coverage':coverage}
    train,valid,median=prepared
    model=DecisionTreeRegressor(criterion='absolute_error',max_depth=depth,min_samples_leaf=leaf,random_state=48)
    model.fit(train,y_train);prediction=model.predict(valid)
    assert np.allclose(prediction,manual_predict(model,valid),rtol=0,atol=1e-10)
    error=np.abs(prediction-y_valid);observed=np.isfinite(x_valid)
    return {'status':'PASS','fit_count':1,'depth':depth,'min_samples_leaf':leaf,
            'MAE_minutes':float(error.mean()),'absolute_error_sum':float(error.sum()),
            'observed_rows_MAE':float(error[observed].mean()) if observed.any() else None,
            'missing_rows_MAE':float(error[~observed].mean()) if (~observed).any() else None,
            'coverage':coverage,'imputation_median':median,'tree_nodes':int(model.tree_.node_count),
            'seconds':time.perf_counter()-start,'independent_tree_traversal':'PASS'}


def pick_setting(results):
    valid=[v for v in results if v['status']=='PASS']
    return min(valid,key=lambda v:(v['MAE_minutes'],v['depth'],-v['min_samples_leaf'])) if valid else None


def self_test():
    times=pd.date_range('2022-10-01','2023-04-02',freq='30min',tz='UTC')
    masks=partition_masks(times,18000,'2023-04-01','2023-07-01')
    ends=times+pd.Timedelta(minutes=18000)
    assert (ends[masks['inner_train']]<utc('2023-01-01')).all()
    assert (ends[masks['inner_valid']]<utc('2023-04-01')).all()
    assert (ends[masks['outer_train']]<utc('2023-04-01')).all()
    prep=preprocess(np.array([0.,2.,np.nan]),np.array([10000.,np.nan]),2)
    assert prep[2]==1 and prep[1][1,0]==1 and prep[1][1,1]==1
    assert preprocess(np.zeros(10),np.ones(2),2) is None
    x=np.repeat([0.,1.,2.,3.],100);y=np.where(x<2,1.,9.)
    for depth,leaf in [(2,64),(3,64),(5,256)]:
        score=tree_fit(x,y,np.array([0.,3.,np.nan]),np.array([1.,9.,1.]),depth,leaf,100)
        assert score['status']=='PASS' and score['coverage']['validation_rows']==3
    tied=[{'status':'PASS','MAE_minutes':1,'depth':3,'min_samples_leaf':64},
          {'status':'PASS','MAE_minutes':1,'depth':2,'min_samples_leaf':64},
          {'status':'PASS','MAE_minutes':1,'depth':2,'min_samples_leaf':256}]
    assert pick_setting(tied)['min_samples_leaf']==256
    print('PASS: temporal purges, train-only imputation, missing-row preservation, deterministic ties and tree replay.',flush=True)


class DriveStore:
    def __init__(self,service,folder):self.service,self.folder=service,folder;self.ids={}
    def find(self,name):
        if name not in self.ids:
            files=self.service.files().list(q=f"'{self.folder}' in parents and name='{name}' and trashed=false",fields='files(id)').execute()['files']
            assert len(files)<=1,'Ambiguous output: '+name
            self.ids[name]=files[0]['id'] if files else None
        return self.ids[name]
    def read(self,name):
        file=self.find(name)
        return self.service.files().get_media(fileId=file).execute() if file else None
    def write(self,name,data,mime='application/json'):
        from googleapiclient.http import MediaIoBaseUpload
        assert len(data)<=128*1024**2,'Compact cache limit exceeded'
        media=MediaIoBaseUpload(io.BytesIO(data),mimetype=mime,resumable=len(data)>5*1024**2)
        file=self.find(name)
        request=self.service.files().update(fileId=file,media_body=media,fields='id') if file else self.service.files().create(body={'name':name,'parents':[self.folder]},media_body=media,fields='id')
        self.ids[name]=request.execute()['id']
    def json(self,name,obj):self.write(name,json.dumps(obj,indent=2,allow_nan=False).encode())


class RemoteArchive:
    """Index once, read independent compressed members with bounded parallel I/O."""
    def __init__(self,credentials,meta):
        from google.auth.transport.requests import AuthorizedSession
        self.credentials,self.meta=credentials,meta;self.local=threading.local();self.lock=threading.Lock();self.bytes=0
        reader=DriveRangeReader(AuthorizedSession(credentials),meta['id'],int(meta['size']))
        with zipfile.ZipFile(reader) as archive:self.members={x.filename:x for x in archive.infolist()}
        self.bytes=reader.transferred
    def read_range(self,start,length):
        from google.auth.transport.requests import AuthorizedSession
        if not hasattr(self.local,'session'):self.local.session=AuthorizedSession(self.credentials)
        with self.lock:
            assert self.bytes+length<=12*1024**3,'Remote session byte budget exceeded'
            self.bytes+=length
        reader=DriveRangeReader(self.local.session,self.meta['id'],int(self.meta['size']));reader.seek(start)
        return reader.read(length)
    def read(self,name):
        info=self.members[name];header=self.read_range(info.header_offset,30)
        fields=struct.unpack('<4s5H3I2H',header);assert fields[0]==b'PK\x03\x04' and not fields[2]&1
        raw=self.read_range(info.header_offset+30+fields[-2]+fields[-1],info.compress_size)
        if info.compress_type==zipfile.ZIP_DEFLATED:data=zlib.decompress(raw,-15)
        elif info.compress_type==zipfile.ZIP_STORED:data=raw
        else:raise RuntimeError('Unsupported compression method; no full download fallback')
        assert len(data)==info.file_size and zlib.crc32(data)&0xffffffff==info.CRC
        return data


def read_cache(store,name,expected):
    raw=store.read(name)
    if raw is None:return {'identity':expected,'completed':[],'member_hashes':{}},None,None
    with np.load(io.BytesIO(raw),allow_pickle=False) as cache:
        meta=json.loads(str(cache['metadata'].item()));assert meta['identity']==expected,'Cache identity changed: '+name
        times=cache['times'];values=cache['values']
        assert meta['array_sha256']==sha(times.tobytes()+values.tobytes()),'Cache content hash failed'
    return meta,times,values


def write_cache(store,name,meta,times,values):
    meta['array_sha256']=sha(times.tobytes()+values.tobytes())
    buf=io.BytesIO();np.savez_compressed(buf,metadata=np.array(json.dumps(meta)),times=times,values=values)
    store.write(name,buf.getvalue(),'application/octet-stream')


def load_target(archive,store,root,manifest,horizon,source,config):
    import pyarrow.parquet as pq
    name=f'SCREEN_Y_{horizon}m.npz';identity={'type':'valid_time_high_research_v1','source':source,'horizon':horizon,'origin_end':config['origin_end_exclusive'],'cadence':30}
    meta,times,values=read_cache(store,name,identity)
    ts=[] if times is None else [times];ys=[] if values is None else [values]
    for month in config['months']:
        if month in meta['completed']:continue
        member=root+f'y/horizon_minutes={horizon}/symbol=BTCUSDT/source_month={month}/part.parquet'
        raw=archive.read(member);assert sha(raw)==manifest[member]
        frame=pq.read_table(io.BytesIO(raw),columns=['canonical_symbol','decision_time','horizon_minutes','time_to_high_min','label_status']).to_pandas()
        t=pd.DatetimeIndex(pd.to_datetime(frame.decision_time,utc=True));assert t.is_unique and frame.canonical_symbol.eq('BTCUSDT').all() and frame.horizon_minutes.eq(horizon).all()
        keep=(t.asi8%(30*60*10**9)==0)&(t< pd.Timestamp(config['origin_end_exclusive']))&frame.label_status.eq('VALID').to_numpy()
        t=t[keep];y=frame.loc[keep,'time_to_high_min'].to_numpy(dtype=float)
        assert np.isfinite(y).all() and ((y>=0)&(y<=horizon)).all()
        assert ((t+pd.Timedelta(minutes=horizon))<pd.Timestamp(config['reserve_start'])).all()
        ts.append(t.asi8);ys.append(y);meta['completed'].append(month);meta['member_hashes'][member]=manifest[member]
        write_cache(store,name,meta,np.concatenate(ts),np.concatenate(ys))
        print(f'Y {horizon}m: cached research month {month}',flush=True)
    times=np.concatenate(ts);values=np.concatenate(ys)
    assert len(times)==len(np.unique(times)) and (np.diff(times)>0).all()
    return pd.DatetimeIndex(times,tz='UTC'),values,meta['array_sha256']


def load_family(archive,store,family,columns,source,checksums,config,workers):
    import pyarrow as pa
    import pyarrow.parquet as pq
    name=f'SCREEN_X_F4Z_{family}.npz'
    identity={'type':'f4z_research_cache_v1','source':source,'columns':columns,'origin_end':config['origin_end_exclusive'],
              'cadence':30,'minimum_baseline_n':1000,'allowed_source_status':['PROVISIONAL']}
    meta,times,values=read_cache(store,name,identity)
    ts=[] if times is None else [times];xs=[] if values is None else [values]
    members={}
    for member in archive.members:
        match=re.search(r'/dates/[^/=]+=(\d{4}-\d{2}-\d{2})/',member)
        if match and '/F4Z/' in member and f'/family={family}/' in member and '/symbol=BTCUSDT/' in member and member.endswith('.parquet'):
            if '2022-04-22'<=match[1]<='2024-06-30':assert match[1] not in members;members[match[1]]=member
    fields=[c.split('__',2)[2] for c in columns]
    wanted=['decision_time','canonical_symbol','available_after']
    for field in fields:wanted += [field,field[:-7]+'__valid',field[:-7]+'__baseline_n',field[:-7]+'__scientific_status']
    def read_day(day):
        member=members[day];raw=archive.read(member);assert sha(raw)==checksums[member]
        table=pq.read_table(io.BytesIO(raw),columns=wanted)
        t=pd.DatetimeIndex(pd.to_datetime(table['decision_time'].to_pandas(),utc=True))
        expected=pd.date_range(utc(day)+pd.Timedelta(minutes=1),periods=1440,freq='min')
        assert t.equals(expected) and set(table['canonical_symbol'].to_pylist())=={'BTCUSDT'}
        keep=(t.asi8%(30*60*10**9)==0)&(t<pd.Timestamp(config['origin_end_exclusive']))
        table=table.take(pa.array(np.flatnonzero(keep)));t=t[keep]
        available=pd.DatetimeIndex(pd.to_datetime(table['available_after'].to_pandas(),utc=True));causal=~available.isna()&(available<=t)
        data=np.empty((len(t),len(fields)),dtype=np.float64)
        for i,field in enumerate(fields):
            base=field[:-7];v=table[field].cast(pa.float64()).to_numpy(zero_copy_only=False)
            valid=table[base+'__valid'].to_pandas().fillna(False).to_numpy(dtype=bool)
            n=table[base+'__baseline_n'].cast(pa.float64()).to_numpy(zero_copy_only=False)
            status=np.array(table[base+'__scientific_status'].to_pylist())
            good=causal&valid&np.isfinite(v)&np.isfinite(n)&(n>=1000)&(status=='PROVISIONAL')
            data[:,i]=np.where(good,v,np.nan)
        return t.asi8,data,member,checksums[member]
    for month in config['months']:
        if month in meta['completed']:continue
        days=pd.date_range(max(utc(month+'-01'),utc('2022-04-22')),utc(month+'-01')+pd.offsets.MonthEnd(0),freq='D').strftime('%Y-%m-%d').tolist()
        assert set(days)<=set(members)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for count,(t,x,member,h) in enumerate(executor.map(read_day,days),1):
                ts.append(t);xs.append(x);meta['member_hashes'][member]=h
                if count%5==0:print(f'X {family} {month}: {count}/{len(days)} days read',flush=True)
        meta['completed'].append(month);write_cache(store,name,meta,np.concatenate(ts),np.concatenate(xs))
        print(f'X {family}: cached research month {month}',flush=True)
    times=np.concatenate(ts);values=np.concatenate(xs)
    assert len(times)==len(np.unique(times)) and (np.diff(times)>0).all()
    return pd.DatetimeIndex(times,tz='UTC'),values,meta['array_sha256']


def check_baselines(times,y,horizon,baseline,config):
    for i,(first,last) in enumerate(config['folds']):
        masks=partition_masks(times,horizon,first,last);train=masks['outer_train'];valid=masks['outer_valid']
        expected=baseline['completed'][str(horizon)]['folds'][i]
        assert train.sum()==expected['train_rows'] and valid.sum()==expected['validation_rows']
        constants=[np.median(y[train]),horizon/2,np.mean(y[train])]
        for name,pred in zip(['training_median_time_to_high','horizon_midpoint','training_mean_time_to_high'],constants):
            assert abs(float(np.abs(y[valid]-pred).mean())-expected['scores'][name]['MAE_minutes'])<1e-8


def update_matrix(state,baseline,config):
    for feature in config['candidate_features']:
        for horizon in config['horizons']:
            keys=[f'{feature}|{horizon}|{i}' for i in range(len(config['folds']))]
            if not all(key in state['folds'] for key in keys):continue
            folds=[state['folds'][key] for key in keys]
            count=sum(v['validation_rows'] for v in folds);mae=sum(v['absolute_error_sum'] for v in folds)/count
            base=baseline['completed'][str(horizon)]['pooled']
            state['matrix'][f'{feature}|{horizon}']={'feature':feature,'horizon_minutes':horizon,'rows':count,'MAE_minutes':mae,
                'MAE_percent_of_horizon':100*mae/horizon,'delta_MAE_vs_baselines':{n:mae-v['MAE_minutes'] for n,v in base.items()},
                'folds_using_tree':sum(v['status']=='PASS' for v in folds),'folds_using_fallback':sum(v['status']!='PASS' for v in folds),
                'status':'PROVISIONAL_DEVELOPMENT_SCREEN_NOT_TEST_PROMOTION'}


def run(mode='pilot'):
    import re
    import platform
    import sklearn
    import pyarrow
    import os
    from google.colab import auth
    import google.auth
    from googleapiclient.discovery import build
    from threadpoolctl import threadpool_limits
    assert mode in ('pilot','full');self_test();auth.authenticate_user();credentials,_=google.auth.default()
    service=build('drive','v3',credentials=credentials,cache_discovery=False);store=DriveStore(service,CONFIG['folder'])
    sources={k:service.files().get(fileId=v,fields='id,name,size,md5Checksum,modifiedTime').execute() for k,v in CONFIG['source_ids'].items()}
    assert sources==CONFIG['audit_sources'],'Source archives changed since the feature audit'
    versions={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'pyarrow':pyarrow.__version__}
    identity={'config':CONFIG,'sources':sources,'runner_sha256':RUNNER_SHA256,'versions':versions};sig=signature(identity)
    prior=store.read('SCREEN_RESULTS.json')
    state=json.loads(prior) if prior else {'signature':sig,'identity':identity,'jobs':{},'matrix':{},'status':'RUNNING_PILOT','new_model_fits_total':0,'reserved_partitions_read':0}
    assert state['signature']==sig,'Screen identity changed; inspect results before replacement.'
    baseline=json.loads(store.read('BASELINE_RESULTS.json'))
    assert signature(baseline['identity'])==baseline['signature']==CONFIG['baseline_signature']
    assert baseline['identity']['source']==sources['Y']
    features=CONFIG['pilot_features'] if mode=='pilot' else CONFIG['candidate_features']
    horizons=CONFIG['pilot_horizons'] if mode=='pilot' else CONFIG['horizons']
    todo=[(f,h) for f in features for h in horizons if f'{f}|{h}' not in state['jobs']]
    if not todo:
        state['status']='PILOT_COMPLETE_REVIEW_BEFORE_FULL' if mode=='pilot' else 'FULL_INDIVIDUAL_SCREEN_COMPLETE'
        store.json('SCREEN_RESULTS.json',state)
        print('Requested scope already complete. Reusing saved results; no archive reads.');return
    fit_workers=min(2,os.cpu_count() or 1);io_workers=min(4,os.cpu_count() or 1)
    state['workers']={'fits':fit_workers,'remote_reads':io_workers}
    print(f'Scope={mode}; {len(features)} features x {len(horizons)} horizons. Fits={fit_workers} workers; reads={io_workers} workers.',flush=True)
    folders=service.files().list(q=f"'{CONFIG['folder']}' in parents and name='Timing_Screen_Fit_Scores' and mimeType='application/vnd.google-apps.folder' and trashed=false",fields='files(id)').execute()['files']
    assert len(folders)<=1
    fit_folder=folders[0]['id'] if folders else service.files().create(body={'name':'Timing_Screen_Fit_Scores','parents':[CONFIG['folder']],'mimeType':'application/vnd.google-apps.folder'},fields='id').execute()['id']
    fit_store=DriveStore(service,fit_folder)
    start=time.perf_counter();archives={k:RemoteArchive(credentials,v) for k,v in sources.items()}
    yarchive=archives['Y'];manifest_name=next(n for n in yarchive.members if n.endswith('/HANDOFF_2_MANIFEST.json'))
    root=manifest_name[:-len('HANDOFF_2_MANIFEST.json')];manifest={root+e['path']:e['sha256'] for e in json.loads(yarchive.read(manifest_name))['files']}
    raw=archives['X'].read('FILES.sha256');checksums={}
    for line in raw.decode().splitlines():
        if line.strip():h,name=line.split(maxsplit=1);checksums[name.lstrip('*')]=h
    targets={}
    target_start=time.perf_counter()
    for horizon in sorted({h for f,h in todo}):
        targets[horizon]=load_target(yarchive,store,root,manifest,horizon,sources['Y'],CONFIG)
        check_baselines(targets[horizon][0],targets[horizon][1],horizon,baseline,CONFIG)
    state['target_loading_seconds_this_session']=time.perf_counter()-target_start
    def save():
        state['session_elapsed_seconds']=time.perf_counter()-start
        state['remote_bytes_this_session']={k:v.bytes for k,v in archives.items()}
        state['new_model_fits_total']=sum(v['fit_count'] for v in state['jobs'].values())
        state['summed_fit_seconds']=sum(v.get('fit_seconds',0) for v in state['jobs'].values())
        try:
            import resource
            state['peak_process_RSS_MiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        except ImportError:state['peak_process_RSS_MiB']=None
        store.json('SCREEN_RESULTS.json',state)
    for family in sorted({f.split('__')[1] for f,h in todo}):
        columns=[c for c in CONFIG['candidate_features'] if c.split('__')[1]==family]
        family_start=time.perf_counter()
        xt,xdata,xhash=load_family(archives['X'],store,family,columns,sources['X'],checksums,CONFIG,io_workers)
        state.setdefault('family_loading_seconds_this_session',{})[family]=time.perf_counter()-family_start
        for feature in [f for f in features if f.split('__')[1]==family]:
            xcol=xdata[:,columns.index(feature)]
            for horizon in horizons:
                if (feature,horizon) not in todo:continue
                times,y,yhash=targets[horizon];indices=xt.get_indexer(times);assert (indices>=0).all();x=xcol[indices]
                job_id=feature+'|'+str(horizon);job_name='job_'+sha(job_id.encode())[:24]+'.json'
                saved_job=fit_store.read(job_name)
                job=json.loads(saved_job) if saved_job else {'screen_signature':sig,'feature':feature,'horizon_minutes':horizon,'X_cache_sha256':xhash,'Y_cache_sha256':yhash,'fits':{},'folds':{},'matrix':{}}
                assert job['screen_signature']==sig and job['feature']==feature and job['horizon_minutes']==horizon
                assert job['X_cache_sha256']==xhash and job['Y_cache_sha256']==yhash
                def save_job():fit_store.json(job_name,job)
                for fold,(first,last) in enumerate(CONFIG['folds']):
                    key=f'{feature}|{horizon}|{fold}'
                    if key in job['folds']:continue
                    masks=partition_masks(times,horizon,first,last);it,iv=masks['inner_train'],masks['inner_valid'];ot,ov=masks['outer_train'],masks['outer_valid']
                    assert it.sum() and iv.sum() and ot.sum() and ov.sum()
                    pending=[];inner=[]
                    with threadpool_limits(limits=1),ThreadPoolExecutor(max_workers=fit_workers) as executor:
                        for depth,leaf in CONFIG['grid']:
                            fit_key=f'{key}|inner|{depth}|{leaf}'
                            if fit_key in job['fits']:inner.append(job['fits'][fit_key]);continue
                            future=executor.submit(tree_fit,x[it],y[it],x[iv],y[iv],depth,leaf,CONFIG['minimum_observed_training'])
                            pending.append((future,fit_key))
                        lookup={future:name for future,name in pending}
                        for future in as_completed(lookup):
                            score=future.result();job['fits'][lookup[future]]=score;inner.append(score);save_job()
                            print(f'{horizon}m {feature} fold {fold+1}: inner setting saved',flush=True)
                    selected=pick_setting(inner);fit_key=key+'|outer'
                    if fit_key not in job['fits']:
                        outer=tree_fit(x[ot],y[ot],x[ov],y[ov],selected['depth'],selected['min_samples_leaf'],CONFIG['minimum_observed_training']) if selected else {'status':'INNER_FEATURE_INELIGIBLE','fit_count':0}
                        if outer['status']!='PASS':
                            pred=float(np.median(y[ot]));errors=np.abs(y[ov]-pred)
                            outer.update({'fallback':'training_median_time_to_high','MAE_minutes':float(errors.mean()),'absolute_error_sum':float(errors.sum())})
                        job['fits'][fit_key]=outer;save_job()
                    outer=job['fits'][fit_key]
                    job['folds'][key]={'feature':feature,'horizon_minutes':horizon,'fold':fold+1,
                        'train_rows':int(ot.sum()),'validation_rows':int(ov.sum()),'inner_train_rows':int(it.sum()),'inner_validation_rows':int(iv.sum()),
                        'inner_latest_training_label_end':str((times[it]+pd.Timedelta(minutes=horizon)).max()),
                        'inner_latest_validation_label_end':str((times[iv]+pd.Timedelta(minutes=horizon)).max()),
                        'outer_latest_training_label_end':str((times[ot]+pd.Timedelta(minutes=horizon)).max()),
                        'selected':None if selected is None else {'depth':selected['depth'],'min_samples_leaf':selected['min_samples_leaf']},
                        'MAE_minutes':outer['MAE_minutes'],'absolute_error_sum':outer['absolute_error_sum'],
                        'status':outer['status'],'X_cache_sha256':xhash,'Y_cache_sha256':yhash,'baseline_reproduction':'PASS'}
                    save_job();print(f'{horizon}m {feature} fold {fold+1}: outer MAE={outer["MAE_minutes"]:.3f}',flush=True)
                update_matrix(job,baseline,dict(CONFIG,candidate_features=[feature],horizons=[horizon]))
                save_job();state['matrix'].update(job['matrix'])
                state['jobs'][job_id]={'fit_count':sum(v.get('fit_count',0) for v in job['fits'].values()),'fit_seconds':sum(v.get('seconds',0) for v in job['fits'].values()),'score_file':job_name,'score_file_id':fit_store.ids[job_name],
                                      'score_sha256':signature(job),'status':'COMPLETE'}
                save()
        del xdata
    for k,v in CONFIG['source_ids'].items():assert service.files().get(fileId=v,fields='id,name,size,md5Checksum,modifiedTime').execute()==sources[k]
    state['status']='PILOT_COMPLETE_REVIEW_BEFORE_FULL' if mode=='pilot' else 'FULL_INDIVIDUAL_SCREEN_COMPLETE';save()
    print('\nDONE: '+state['status'])
    print('Horizon | feature | MAE minutes | delta vs lowest baseline (negative is better)')
    for feature in features:
        for horizon in horizons:
            row=state['matrix'][f'{feature}|{horizon}'];print(horizon,feature,round(row['MAE_minutes'],3),round(max(row['delta_MAE_vs_baselines'].values()),3))
    print('Send only this folder link: https://drive.google.com/drive/folders/'+CONFIG['folder'])
