"""Exercise orchestration/resume with tiny fake artifacts; never train a forest."""
import importlib.util,json,tempfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('offline',HERE/'run_remaining_batches.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with tempfile.TemporaryDirectory(prefix='offline_test_',dir=HERE) as folder:
 root=Path(folder);sub=root/'submissions';active=sub/'magnitude_active';current=sub/'magnitude_current/run_001'
 active.mkdir(parents=True);current.mkdir(parents=True)
 m.ROOT=root;m.SUB=sub;m.HERE=active;m.OUT=active/'run_001';m.OUT.mkdir()
 m.LEADER=active/'context_leader';m.STATE=active/'offline_state.json'
 names=sum(m.BATCHES,[])
 inventory={'features':[{'pdf_name':n,'column':n,'status':'PENDING'} for n in names]}
 m.write(active/'pdf_feature_inventory.json',inventory)
 m.write(current/'config.json',{'anchor_features':['a'],'additions':{},'forest':{'n_jobs':8,'n_estimators':128},'seeds':[48,17,91]})
 m.write(sub/'magnitude_benchmark_registry.json',{'current':{'MAE':100.,'Spearman_IC':.1},'cumulative_counted_steps':0})
 replay=sub/'magnitude_current/replay.py';replay.write_text("from pathlib import Path\nHERE=Path(__file__).resolve().parent\nOUT=HERE/'run_001'\nassert (OUT/'RESULT.json').exists()\n")
 m.preflight=lambda:{f['pdf_name']:f for f in m.read(active/'pdf_feature_inventory.json')['features']}
 m.archive_previous=lambda:None
 attempts=[];finished=[];interrupted=[False]
 def fake_run(script):
  if script.name=='context_feature.py':
   c=m.read(active/'context_config.json');name=c['version'].removeprefix('ensemble_context_');attempts.append(name)
   if name==names[2] and not interrupted[0]:
    interrupted[0]=True;m.write(m.OUT/'checkpoint.json',{'preserved':True});raise KeyboardInterrupt
   if name==names[2]:assert m.read(m.OUT/'checkpoint.json')['preserved']
   expected=100. if name in m.BATCHES[0] else 99.96
   assert abs(c['benchmark_mae']-expected)<1e-10
   mae=c['benchmark_mae']-.01*(m.BATCHES[0 if name in m.BATCHES[0] else 1].index(name)+1)
   m.write(m.OUT/'config.json',c)
   m.write(m.OUT/'RESULT.json',{'experiment':c['version'],'metrics':{'prediction':{'MAE':mae,'Spearman_IC':.1}},'cumulative_counted_steps':c['prior_fit_count']+28})
   m.write(m.OUT/'verification.json',{'status':'PASS'});m.write(m.OUT/'inner_scores.json',[])
   (m.OUT/'fold_MAE.csv').write_text('fold,incumbent,prediction\n1,100,99\n')
   finished.append(name)
  elif script.name=='finalize_context_batch.py':
   batch=m.read(active/'context_results.json');winner=min(batch['results'],key=lambda e:e['result']['metrics']['prediction']['MAE'])
   assert m.read(m.OUT/'RESULT.json')['experiment']==winner['experiment']
   m.write(current/'config.json',winner['config'])
   m.write(sub/'magnitude_benchmark_registry.json',{'current':winner['result']['metrics']['prediction'],'cumulative_counted_steps':batch['cumulative_counted_steps']})
   inv=m.read(active/'pdf_feature_inventory.json')
   for f in inv['features']:
    if f['pdf_name'] in batch['plan']['candidates']:f['status']='TESTED'
   m.write(active/'pdf_feature_inventory.json',inv);m.clean(m.OUT)
  else:raise AssertionError(script)
 m.run=fake_run
 try:m.main()
 except KeyboardInterrupt:pass
 else:raise AssertionError('Interruption was not exercised')
 assert m.read(m.STATE)['candidate']==2
 m.main()
 assert finished==names and len(attempts)==9
 assert m.read(m.STATE)['phase']=='complete'
 assert abs(m.read(sub/'magnitude_benchmark_registry.json')['current']['MAE']-99.92)<1e-10
 assert not list(m.OUT.iterdir()) and not m.LEADER.exists()
 before=len(attempts);m.main();assert len(attempts)==before
 print('PASS: two batches, fixed baseline per batch, winner carried forward, checkpoint resume, cleanup, and completed-run restart without refitting.')
