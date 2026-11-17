"""1h-only resumable screen. REF/OPT/REPORT namespaces supplied by notebook."""
import json
import time
import threading
import platform
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import numpy as np
import pandas as pd


class BudgetStop(Exception):
    pass


def pack(job):
    return {'job':job,'sha256':REF['signature'](job)}


def unpack(raw, signature, feature, xhash, yhash):
    record=json.loads(raw); job=record['job']
    assert REF['signature'](job)==record['sha256'], 'Job checksum mismatch'
    assert job['screen_signature']==signature and job['feature']==feature and job['horizon_minutes']==60
    assert job['X_cache_sha256']==xhash and job['Y_cache_sha256']==yhash
    return job


def recover_indexed(record, info, signature, key):
    job=record['job']
    assert REF['signature'](job)==record['sha256']
    if info['complete_folds']==5: assert record['sha256']==info['sha256']
    assert job['screen_signature']==signature and job['feature']+'|60'==key
    assert len(job['folds'])>=info['complete_folds'] and job['horizon_minutes']==60
    info.update(sha256=record['sha256'],complete_folds=len(job['folds']))
    return job


def fit_job(job, x, times, y, masks, config, checkpoint, deadline, clock=time.monotonic):
    """One worker owns one job. Saved scores skip fits; incomplete aggregates recover."""
    feature=job['feature']; minimum=config['minimum_observed_training']
    for i,m in enumerate(masks):
        prefix=f'{feature}|60|{i}'
        if prefix in job['folds']: continue
        it,iv,ot,ov=[m[k] for k in ('inner_train','inner_valid','outer_train','outer_valid')]
        assert it.any() and iv.any() and ot.any() and ov.any()
        inner=[]; prepared=None
        for depth,leaf in config['grid']:
            key=f'{prefix}|inner|{depth}|{leaf}'
            if key not in job['fits']:
                if clock()>=deadline: return job
                if prepared is None: prepared=OPT['prepare'](x[it],y[it],x[iv],y[iv],minimum)
                score,prediction=OPT['prepared_fit'](prepared,depth,leaf); del prediction
                job['fits'][key]=score; checkpoint(job)
                print(f'{feature} fold {i+1}: inner {depth}/{leaf} complete',flush=True)
            inner.append(job['fits'][key])
        selected=REF['pick_setting'](inner); key=prefix+'|outer'
        if key not in job['fits']:
            if clock()>=deadline: return job
            coverage={'train_observed':int(np.isfinite(x[ot]).sum()),'train_rows':int(ot.sum()),
                      'validation_observed':int(np.isfinite(x[ov]).sum()),'validation_rows':int(ov.sum())}
            if selected:
                score,prediction=OPT['prepared_fit'](OPT['prepare'](x[ot],y[ot],x[ov],y[ov],minimum),selected['depth'],selected['min_samples_leaf']); del prediction
            else: score={'status':'INNER_FEATURE_INELIGIBLE','fit_count':0,'coverage':coverage}
            if score['status']!='PASS':
                errors=np.abs(y[ov]-float(np.median(y[ot])))
                score.update(fallback='training_median_time_to_high',MAE_minutes=float(errors.mean()),absolute_error_sum=float(errors.sum()))
            job['fits'][key]=score; checkpoint(job)
        outer=job['fits'][key]
        job['folds'][prefix]={'feature':feature,'horizon_minutes':60,'fold':i+1,
            'train_rows':int(ot.sum()),'validation_rows':int(ov.sum()),'inner_train_rows':int(it.sum()),'inner_validation_rows':int(iv.sum()),
            'inner_latest_training_label_end':str((times[it]+pd.Timedelta(minutes=60)).max()),
            'inner_latest_validation_label_end':str((times[iv]+pd.Timedelta(minutes=60)).max()),
            'outer_latest_training_label_end':str((times[ot]+pd.Timedelta(minutes=60)).max()),
            'selected':None if selected is None else {'depth':selected['depth'],'min_samples_leaf':selected['min_samples_leaf']},
            'MAE_minutes':outer['MAE_minutes'],'absolute_error_sum':outer['absolute_error_sum'],
            'status':outer['status'],'X_cache_sha256':job['X_cache_sha256'],'Y_cache_sha256':job['Y_cache_sha256'],
            'baseline_reproduction':'PASS'}
        checkpoint(job)
        print(f'{feature} fold {i+1}/5 complete',flush=True)
    return job


def folder(service,parent,name):
    found=service.files().list(q=f"'{parent}' in parents and name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",fields='files(id)').execute()['files']
    assert len(found)<=1
    return found[0]['id'] if found else service.files().create(body={'name':name,'parents':[parent],'mimeType':'application/vnd.google-apps.folder'},fields='id').execute()['id']


class IsolatedStore:
    """Write only to this horizon; borrow only complete shared caches read-only."""
    def __init__(self, private, shared, months):
        self.private,self.shared,self.months=private,shared,months
    def read(self,name):
        raw=self.private.read(name)
        if raw is not None: return raw
        if not (name.startswith('SCREEN_X_F4Z_') or name=='SCREEN_Y_60m.npz'):
            return None
        raw=self.shared.read(name)
        if raw is None: return None
        import io
        with np.load(io.BytesIO(raw),allow_pickle=False) as cache:
            meta=json.loads(str(cache['metadata'].item()))
        if meta['completed']!=self.months: return None
        # REF.read_cache verifies source identity and full array checksum before use.
        return raw
    def write(self,name,data,mime='application/json'):
        return self.private.write(name,data,mime)
    def json(self,name,value):
        return self.private.json(name,value)


def run_1h(config, expected_pilot, hashes, session_hours=4):
    import sklearn
    import pyarrow
    import joblib
    from sklearn.tree import DecisionTreeRegressor
    from google.colab import auth
    import google.auth
    from googleapiclient.discovery import build
    from google_auth_httplib2 import AuthorizedHttp
    import httplib2
    from threadpoolctl import threadpool_limits
    assert config['target']=='time_to_high_min' and 60 in config['horizons']
    assert 0<session_hours<=4
    start=time.monotonic(); deadline=start+session_hours*3600
    auth.authenticate_user(); credentials,_=google.auth.default()
    def service_new():
        return build('drive','v3',http=AuthorizedHttp(credentials,http=httplib2.Http(timeout=90)),cache_discovery=False)
    service=service_new(); store=REF['DriveStore'](service,config['folder'])
    print('1h ONLY: checking identities, pilot receipts and verified implementation.',flush=True)
    pilot=json.loads(store.read('SCREEN_RESULTS.json'))
    versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,pyarrow=pyarrow.__version__)
    assert pilot['identity']==expected_pilot and REF['signature'](expected_pilot)==pilot['signature']
    assert expected_pilot['config']==config and expected_pilot['versions']==versions
    for k,file_id in config['source_ids'].items():
        assert service.files().get(fileId=file_id,fields='id,name,size,md5Checksum,modifiedTime').execute()==config['audit_sources'][k]
    receipt=json.loads(store.read('THROUGHPUT_CHECK.json'))
    assert receipt['status']=='PASS_EXACT_PREDICTIONS_SCORES_SELECTION_AND_SAVED_PILOT'
    assert receipt['screen_signature']==pilot['signature'] and receipt['reference_sha256']==hashes['reference'] and receipt['optimized_sha256']==hashes['optimized']
    workers_receipt=json.loads(store.read('WORKER_CHECK.json'))
    assert workers_receipt['status']=='PASS_WORKER_EQUIVALENCE'
    assert REF['signature'](workers_receipt['identity'])==workers_receipt['signature']
    assert workers_receipt['identity']['pilot_signature']==pilot['signature']
    assert workers_receipt['identity']['hashes']['reference']==hashes['reference'] and workers_receipt['identity']['hashes']['optimized']==hashes['optimized']
    baseline=json.loads(store.read('BASELINE_RESULTS.json'))
    assert REF['signature'](baseline['identity'])==baseline['signature']==config['baseline_signature']
    assert baseline['identity']['source']==config['audit_sources']['Y']
    shared_store=store
    work_root=folder(service,config['folder'],'Timing_1h_Run')
    store=IsolatedStore(REF['DriveStore'](service,work_root),shared_store,config['months'])
    identity={'target':config['target'],'horizon_minutes':60,'pilot_signature':pilot['signature'],
              'implementation_hashes':{k:hashes[k] for k in ('reference','optimized','runner','range_reader')},'versions':versions,
              'compatibility':'Reuse only individually hash-verified legacy jobs under their original identity; new jobs use this identity. Historical runner hash discrepancy retained, never rewritten.'}
    sig=REF['signature'](identity)
    raw=store.read('SCREEN_1h_RESULTS.json')
    state=json.loads(raw) if raw else {'identity':identity,'signature':sig,'jobs':{},'status':'RUNNING_1h_ONLY'}
    assert state['identity']==identity and state['signature']==sig, '1h implementation changed; review compatibility before reuse'
    scores_folder=folder(service,work_root,'Timing_1h_Fit_Scores')
    reports_folder=folder(service,work_root,'Timing_Research_Reports')
    read_scores=REF['DriveStore'](service,scores_folder)
    report_store=REF['DriveStore'](service,reports_folder)
    # One HTTP client owned by one asynchronous writer. Main reads use another.
    write_scores=REF['DriveStore'](service_new(),scores_folder)
    all_jobs={}
    for key,info in pilot['jobs'].items():
        job=json.loads(service.files().get_media(fileId=info['score_file_id']).execute())
        assert REF['signature'](job)==info['score_sha256'] and job['screen_signature']==pilot['signature']
        all_jobs[key]=job
    def save_reports():
        for name,data in REPORT['artifacts'](config,baseline,all_jobs,60).items():
            report_store.write(name,data,'application/pdf' if name.endswith('.pdf') else 'text/csv')
        print('Reports updated: '+f'https://drive.google.com/drive/folders/{reports_folder}',flush=True)
    # Validate completed new jobs before skipping. Load partials later using cache hashes.
    for key,info in state['jobs'].items():
        raw=service.files().get_media(fileId=info['file_id']).execute(); record=json.loads(raw)
        # A resumed partial job can advance before its aggregate is saved.
        # Completed jobs are immutable; partial aggregates may lag durable fits.
        job=recover_indexed(record,info,sig,key)
        all_jobs[key]=job
    pending=[f for f in config['candidate_features'] if len(all_jobs.get(f+'|60',{}).get('folds',{}))!=5]
    if not pending:
        state['status']='COMPLETE_1h_ONLY'; store.json('SCREEN_1h_RESULTS.json',state); save_reports(); return
    yidentity=dict(type='valid_time_high_research_v1',source=config['audit_sources']['Y'],horizon=60,origin_end=config['origin_end_exclusive'],cadence=30)
    ym,ts,y=REF['read_cache'](store,'SCREEN_Y_60m.npz',yidentity)
    if ym['completed']!=config['months']:
        class TargetBudgetStore:
            def read(self,name): return store.read(name)
            def write(self,name,data,mime='application/json'):
                store.write(name,data,mime)
                if time.monotonic()>=deadline: raise BudgetStop('Target month saved; resume next session')
        try:
            if time.monotonic()>=deadline: raise BudgetStop('Session budget reached')
            print('Loading 1h target via remote ZIP ranges; research months only.',flush=True)
            yarchive=REF['RemoteArchive'](credentials,config['audit_sources']['Y'])
            manifest_name=next(n for n in yarchive.members if n.endswith('/HANDOFF_2_MANIFEST.json'))
            yroot=manifest_name[:-len('HANDOFF_2_MANIFEST.json')]
            manifest={yroot+entry['path']:entry['sha256'] for entry in json.loads(yarchive.read(manifest_name))['files']}
            loaded_times,y,yhash=REF['load_target'](yarchive,TargetBudgetStore(),yroot,manifest,60,config['audit_sources']['Y'],config)
            ts=loaded_times.asi8
            ym,_,_=REF['read_cache'](store,'SCREEN_Y_60m.npz',yidentity)
        except BudgetStop:
            state['status']='PAUSED_TARGET_LOADING_1h_ONLY'
            store.json('SCREEN_1h_RESULTS.json',state); save_reports()
            print('Target-loading progress saved. Rerun the same 1h notebook to resume.',flush=True)
            return
    assert ym['completed']==config['months'] and ts is not None

    times=pd.DatetimeIndex(ts,tz='UTC'); yhash=ym['array_sha256']
    assert times.is_unique and times.is_monotonic_increasing and (times<pd.Timestamp(config['origin_end_exclusive'])).all()
    assert ((times+pd.Timedelta(minutes=60))<pd.Timestamp(config['reserve_start'])).all()
    assert np.isfinite(y).all() and ((y>=0)&(y<=60)).all()
    REF['check_baselines'](times,y,60,baseline,config)
    masks=[REF['partition_masks'](times,60,*fold) for fold in config['folds']]
    archive=None; checksums=None
    class BudgetStore:
        def read(self,name): return store.read(name)
        def write(self,name,data,mime='application/json'):
            store.write(name,data,mime)
            if time.monotonic()>=deadline: raise BudgetStop('Saved completed month; session budget reached')
    fit_workers=min(2,joblib.cpu_count())
    families=sorted({f.split('__')[1] for f in pending},key=lambda f:(f!='flow',f))
    save_reports()
    for family in families:
        if time.monotonic()>=deadline: break
        columns=[f for f in config['candidate_features'] if f.split('__')[1]==family]
        xid=dict(type='f4z_research_cache_v1',source=config['audit_sources']['X'],columns=columns,
                 origin_end=config['origin_end_exclusive'],cadence=30,minimum_baseline_n=1000,allowed_source_status=['PROVISIONAL'])
        print(f'Loading family {family}; completed caches are reused.',flush=True)
        xm,xt,xx=REF['read_cache'](store,f'SCREEN_X_F4Z_{family}.npz',xid)
        if xm['completed']!=config['months']:
            if archive is None:
                archive=REF['RemoteArchive'](credentials,config['audit_sources']['X'])
                checksums={}
                for line in archive.read('FILES.sha256').decode().splitlines():
                    if line.strip(): h,name=line.split(maxsplit=1); checksums[name.lstrip('*')]=h
            try:
                xt,xx,xhash=REF['load_family'](archive,BudgetStore(),family,columns,config['audit_sources']['X'],checksums,config,min(2,joblib.cpu_count()))
            except BudgetStop:
                break
        else:
            xt=pd.DatetimeIndex(xt,tz='UTC'); xhash=xm['array_sha256']
        assert xt.is_unique and xt.is_monotonic_increasing and (xt<pd.Timestamp(config['origin_end_exclusive'])).all()
        index=xt.get_indexer(times); assert (index>=0).all()
        names=iter(f for f in pending if f.split('__')[1]==family)
        writer=OPT['OrderedWriter'](write_scores.write,capacity=2)
        active={}; finished=[]
        try:
            with threadpool_limits(limits=1),ThreadPoolExecutor(max_workers=fit_workers) as pool:
                def submit_next():
                    if time.monotonic()>=deadline: return False
                    feature=next(names,None)
                    if feature is None: return False
                    filename='job_'+REF['sha']((feature+'|60').encode())[:24]+'.json'
                    raw=read_scores.read(filename)
                    job=unpack(raw,sig,feature,xhash,yhash) if raw else {'screen_signature':sig,'feature':feature,'horizon_minutes':60,
                        'X_cache_sha256':xhash,'Y_cache_sha256':yhash,'fits':{},'folds':{},'matrix':{}}
                    def checkpoint(value): writer.submit(filename,pack(value))
                    future=pool.submit(fit_job,job,xx[index,columns.index(feature)],times,y,masks,config,checkpoint,deadline)
                    active[future]=(feature,filename); return True
                for _ in range(fit_workers): submit_next()
                while active:
                    done,_=wait(active,timeout=20,return_when=FIRST_COMPLETED)
                    if not done: print(f'{family}: {len(active)} active features; elapsed {(time.monotonic()-start)/60:.1f} minutes',flush=True)
                    for future in done:
                        feature,filename=active.pop(future); job=future.result()
                        all_jobs[feature+'|60']=job; finished.append((feature,filename,job))
                        submit_next()
        finally:
            writer.close()  # Failure must propagate; no completed-job promotion before flush.
        # Read back durable snapshots before recording completion in the aggregate.
        for feature,filename,job in finished:
            read_scores.ids.pop(filename,None)
            raw=read_scores.read(filename)
            if raw is None: continue  # Budget ended before first fit.
            stored=unpack(raw,sig,feature,xhash,yhash)
            assert REF['signature'](stored)==REF['signature'](job)
            state['jobs'][feature+'|60']={'file_id':read_scores.ids[filename],'sha256':REF['signature'](stored),
                                         'complete_folds':len(stored['folds'])}
        state['status']='PARTIAL_1h_ONLY'; store.json('SCREEN_1h_RESULTS.json',state); save_reports()
    complete=sum(len(all_jobs.get(f+'|60',{}).get('folds',{}))==5 for f in config['candidate_features'])
    state['status']='COMPLETE_1h_ONLY' if complete==len(config['candidate_features']) else 'PAUSED_SESSION_BUDGET_1h_ONLY'
    state['completed_features']=complete; state['session_seconds']=time.monotonic()-start
    for k,file_id in config['source_ids'].items():
        assert service.files().get(fileId=file_id,fields='id,name,size,md5Checksum,modifiedTime').execute()==config['audit_sources'][k]
    store.json('SCREEN_1h_RESULTS.json',state); save_reports()
    print(f'{complete}/{len(config["candidate_features"])} complete. Rerun this same notebook to resume 1h only.',flush=True)
