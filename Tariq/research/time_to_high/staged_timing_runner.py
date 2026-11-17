"""All-horizon staged run, 15m first. Sole active notebook; no time cutoff."""
import json
import platform
import time
import numpy as np
import pandas as pd


def folder(service,parent,name,create=True):
    found=service.files().list(q=f"'{parent}' in parents and name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",fields='files(id)').execute()['files']
    assert len(found)<=1, 'Ambiguous folder: '+name
    if found: return found[0]['id']
    if not create: return None
    return service.files().create(body={'name':name,'parents':[parent],'mimeType':'application/vnd.google-apps.folder'},fields='id').execute()['id']


def unpack(raw,signature,horizon):
    envelope=json.loads(raw); state=envelope['state']
    assert REF['signature'](state)==envelope['sha256'], 'Staged checkpoint checksum mismatch'
    assert state['signature']==signature and state['horizon_minutes']==horizon
    return state


def run(config,expected_pilot,hashes):
    import sklearn
    import pyarrow
    import joblib
    from sklearn.tree import DecisionTreeRegressor
    from google.colab import auth
    import google.auth
    from googleapiclient.discovery import build
    from google_auth_httplib2 import AuthorizedHttp
    import httplib2
    auth.authenticate_user(); credentials,_=google.auth.default()
    service=build('drive','v3',http=AuthorizedHttp(credentials,http=httplib2.Http(timeout=90)),cache_discovery=False)
    shared=REF['DriveStore'](service,config['folder'])
    print('Read-only preflight: versions, sources, contracts and previous results.',flush=True)
    pilot=json.loads(shared.read('SCREEN_RESULTS.json'))
    versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,pyarrow=pyarrow.__version__)
    assert pilot['identity']==expected_pilot and REF['signature'](expected_pilot)==pilot['signature'], 'Pilot identity changed'
    assert expected_pilot['config']==config, 'Frozen configuration changed; no fitting started'
    assert expected_pilot['versions']==versions, f'Runtime versions changed; no fitting started. Expected {expected_pilot["versions"]}; observed {versions}'
    observed={}
    for key,file_id in config['source_ids'].items():
        actual=service.files().get(fileId=file_id,fields='id,name,size,md5Checksum,modifiedTime').execute()
        SOURCE['compare'](actual,config['audit_sources'][key]); observed[key]=actual
    baseline=json.loads(shared.read('BASELINE_RESULTS.json'))
    assert REF['signature'](baseline['identity'])==baseline['signature']==config['baseline_signature']
    assert baseline['identity']['source']==config['audit_sources']['Y']
    legacy15raw=shared.read('SCREEN_15m_RESULTS.json')
    legacy15=json.loads(legacy15raw) if legacy15raw else None
    if legacy15:
        assert REF['signature'](legacy15['identity'])==legacy15['signature']
        assert legacy15['identity']['pilot_signature']==pilot['signature']
        assert legacy15['identity']['implementation_hashes']['runner']==SOURCE['APPROVED']['15m']
        assert legacy15['identity']['implementation_hashes']['reference']==hashes['reference']
        assert legacy15['identity']['implementation_hashes']['optimized']==hashes['optimized']
        assert legacy15['identity']['versions']==versions
    legacy_folder=folder(service,config['folder'],'Timing_15m_Fit_Scores',False)
    legacy_store=REF['DriveStore'](service,legacy_folder) if legacy_folder else None
    print('Preflight passed. Original model results will not be overwritten.',flush=True)
    root=folder(service,config['folder'],'Timing_Staged_All_Horizons')
    output=REF['DriveStore'](service,root)
    identity={'method':CORE['METHOD'],'config':config,'pilot_signature':pilot['signature'],'versions':versions,
              'hashes':{k:hashes[k] for k in ('reference','optimized','core','runner','range_reader')}}
    signature=REF['signature'](identity)
    contract=output.read('CONTRACT.json')
    if contract:
        assert json.loads(contract)['identity']==identity, 'Staged code/contract changed; stop for review'
    else: output.json('CONTRACT.json',{'identity':identity,'signature':signature,'observed_source_metadata':observed})
    states={}; arrays={}; xhashes={}; archives={}; target_root=None; manifest=None; checksums=None
    families=sorted({f.split('__')[1] for f in config['candidate_features']},key=lambda f:(f!='flow',f))
    def archive(key):
        if key not in archives: archives[key]=REF['RemoteArchive'](credentials,config['audit_sources'][key])
        return archives[key]
    def family_data(family):
        nonlocal checksums
        if family in arrays: return arrays[family]
        columns=[f for f in config['candidate_features'] if f.split('__')[1]==family]
        expected=dict(type='f4z_research_cache_v1',source=config['audit_sources']['X'],columns=columns,origin_end=config['origin_end_exclusive'],cadence=30,minimum_baseline_n=1000,allowed_source_status=['PROVISIONAL'])
        meta,ts,x=REF['read_cache'](shared,f'SCREEN_X_F4Z_{family}.npz',expected)
        if meta['completed']!=config['months']:
            print(f'Loading missing research months: {family}. Original completed months are reused.',flush=True)
            xa=archive('X')
            if checksums is None:
                checksums={}
                for line in xa.read('FILES.sha256').decode().splitlines():
                    if line.strip(): checksum,name=line.split(maxsplit=1); checksums[name.lstrip('*')]=checksum
            times,x,digest=REF['load_family'](xa,shared,family,columns,config['audit_sources']['X'],checksums,config,2)
        else: times=pd.DatetimeIndex(ts,tz='UTC'); digest=meta['array_sha256']
        assert times.is_unique and times.is_monotonic_increasing and (times<pd.Timestamp(config['origin_end_exclusive'])).all()
        assert sum(v[1].nbytes for v in arrays.values())+x.nbytes<512*1024**2, 'Feature RAM budget exceeded; stop rather than drop rows'
        arrays[family]=(times,x,columns); xhashes[family]=digest
        return arrays[family]
    def target_data(h):
        nonlocal target_root,manifest
        expected=dict(type='valid_time_high_research_v1',source=config['audit_sources']['Y'],horizon=h,origin_end=config['origin_end_exclusive'],cadence=30)
        meta,ts,y=REF['read_cache'](shared,f'SCREEN_Y_{h}m.npz',expected)
        if meta['completed']!=config['months']:
            ya=archive('Y')
            if manifest is None:
                name=next(n for n in ya.members if n.endswith('/HANDOFF_2_MANIFEST.json'))
                target_root=name[:-len('HANDOFF_2_MANIFEST.json')]
                manifest={target_root+e['path']:e['sha256'] for e in json.loads(ya.read(name))['files']}
            times,y,digest=REF['load_target'](ya,shared,target_root,manifest,h,config['audit_sources']['Y'],config)
        else: times=pd.DatetimeIndex(ts,tz='UTC'); digest=meta['array_sha256']
        assert times.is_unique and times.is_monotonic_increasing
        assert (times<pd.Timestamp(config['origin_end_exclusive'])).all() and ((times+pd.Timedelta(minutes=h))<pd.Timestamp(config['reserve_start'])).all()
        assert np.isfinite(y).all() and ((y>=0)&(y<=h)).all()
        REF['check_baselines'](times,y,h,baseline,config)
        return times,y,digest
    def report():
        for name,data in REPORT['make'](config,baseline,states).items():
            output.write(name,data,'application/pdf' if name.endswith('.pdf') else 'text/csv')
        print('Reports: https://drive.google.com/drive/folders/'+root,flush=True)
    for h in config['horizons']:
        raw=output.read(f'horizon_{h}m.json')
        if raw: states[str(h)]=unpack(raw,signature,h)
    report()
    for h in config['horizons']:
        state=states.setdefault(str(h),{'signature':signature,'horizon_minutes':h,'screen':{},'tuning':{},'policy_folds':{},'X_hashes':{},'status':'RUNNING'})
        if state.get('status')=='COMPLETE':
            assert set(state['screen'])==set(config['candidate_features']) and len(state['policy_folds'])==5, 'Incomplete horizon marked complete'
            print(f'{h}m already complete; no fits repeated.',flush=True); continue
        print(f'=== {h}m horizon: cheap screen then nested shortlist ===',flush=True)
        times,y,yhash=target_data(h)
        assert state.get('Y_hash',yhash)==yhash; state['Y_hash']=yhash
        def save(): output.json(f'horizon_{h}m.json',{'state':state,'sha256':REF['signature'](state)})
        indices={}
        for family in families:
            xt,xx,columns=family_data(family)
            assert state['X_hashes'].get(family,xhashes[family])==xhashes[family]
            state['X_hashes'][family]=xhashes[family]
            index=xt.get_indexer(times); assert (index>=0).all(); indices[family]=index
            started=time.perf_counter(); added=0
            for feature in columns:
                if feature in state['screen']: continue
                x=xx[index,columns.index(feature)]
                result=CORE['screen_feature'](x,times,y,h,config)
                # Identical outer populations are mandatory, even for fallbacks.
                for i,fold in enumerate(result['folds']):
                    assert fold['outer']['validation_rows']==baseline['completed'][str(h)]['folds'][i]['validation_rows']
                state['screen'][feature]=result; added+=1
                if added%10==0:
                    save(); print(f'{h}m: {len(state["screen"])}/{len(config["candidate_features"])} features cheaply screened',flush=True)
            save(); report()
            print(f'{h}m {family}: {added} new cheap screens in {time.perf_counter()-started:.1f}s (loading excluded)',flush=True)
        assert set(state['screen'])==set(config['candidate_features'])
        def get_x(feature):
            family=feature.split('__')[1]; xt,xx,columns=arrays[family]
            return xx[indices[family],columns.index(feature)]
        legacy_cache={}
        def legacy(feature):
            if feature in legacy_cache: return legacy_cache[feature]
            key=f'{feature}|{h}'; job=None
            if key in pilot['jobs']:
                info=pilot['jobs'][key]
                job=json.loads(service.files().get_media(fileId=info['score_file_id']).execute())
                assert REF['signature'](job)==info['score_sha256'] and job['screen_signature']==pilot['signature']
            elif h==15 and legacy_store and legacy15:
                name='job_'+REF['sha'](key.encode())[:24]+'.json'; raw=legacy_store.read(name)
                if raw:
                    record=json.loads(raw); job=record['job']
                    assert REF['signature'](job)==record['sha256'] and job['screen_signature']==legacy15['signature']
                    prior_info=legacy15['jobs'].get(key)
                    if prior_info and prior_info['complete_folds']==5: assert record['sha256']==prior_info['sha256']
            if job:
                assert job['feature']==feature and job['horizon_minutes']==h
                assert job['X_cache_sha256']==xhashes[feature.split('__')[1]] and job['Y_cache_sha256']==yhash
            legacy_cache[feature]=job
            return job
        CORE['refine'](state,times,y,h,config,get_x,save,legacy,min(2,joblib.cpu_count()))
        assert len(state['policy_folds'])==5
        state['status']='COMPLETE'; save(); report()
    for key,file_id in config['source_ids'].items():
        SOURCE['compare'](service.files().get(fileId=file_id,fields='id,name,size,md5Checksum,modifiedTime').execute(),config['audit_sources'][key])
    output.json('COMPLETION.json',{'status':'COMPLETE_ALL_TEN_HORIZONS_STAGED','signature':signature,'horizons':config['horizons'],
                                  'method':CORE['METHOD'],'not_exhaustive':True,'reserved_rows_used':False})
    print('All ten horizons complete for the staged method. Original exhaustive experiment remains separate.',flush=True)
