import copy
from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch
import sys

import pipeline as p
from monitoring import Monitor, error_text
from dashboard.state import monitor_status


class MonitoringTests(unittest.TestCase):
    def setUp(self):
        self.config = p.read(p.ROOT/'config.json')
        self.folder = p.ROOT/'.verification'/self._testMethodName
        self.folder.mkdir(parents=True,exist_ok=True)

    def test_missing_artifact_skips_models(self):
        self.config['models']['direction']['artifact_dir']='.verification/nonexistent_artifacts'
        with patch.object(p.subprocess,'run',side_effect=AssertionError('Must not execute')):
            result=p.run(self.config,'replay',sys.executable,self.folder)
        monitor=p.read(self.folder/'monitor.json')
        steps={s['id']:s for s in monitor['steps']}
        self.assertEqual(monitor['status'],'FAILED')
        self.assertEqual(steps['files']['status'],'FAILED')
        self.assertEqual(steps['direction']['status'],'SKIPPED')
        self.assertEqual(steps['save']['status'],'PASSED')
        self.assertNotIn('outputs',result)

    def test_input_failure_skips_models(self):
        checks=[{'check':'example_input','status':'FAIL','detail':'scale_bps missing'}]
        with patch.object(p,'inspect_package',return_value=checks),patch.object(p.subprocess,'run',side_effect=AssertionError('Must not execute')):
            p.run(self.config,'replay',sys.executable,self.folder)
        steps={s['id']:s for s in p.read(self.folder/'monitor.json')['steps']}
        self.assertEqual(steps['inputs']['status'],'FAILED')
        self.assertIn('scale_bps',steps['inputs']['detail'])
        self.assertEqual(steps['magnitude']['status'],'SKIPPED')

    def test_model_failure(self):
        with patch.object(p,'inspect_package',return_value=[]), patch.object(p,'environment_check',side_effect=ValueError('Dependency mismatch')):
            result=p.run(self.config,'replay',sys.executable,self.folder)
        steps={s['id']:s for s in p.read(self.folder/'monitor.json')['steps']}
        self.assertEqual(steps['direction']['status'],'FAILED')
        self.assertEqual(steps['magnitude']['status'],'SKIPPED')
        self.assertNotIn('predicted_signed_score_bps',result)

    def test_keyboard_interrupt(self):
        with patch.object(p,'inspect_package',return_value=[]), patch.object(p,'environment_check',side_effect=KeyboardInterrupt()):
            result=p.run(self.config,'replay',sys.executable,self.folder)
        self.assertEqual(result['status'],'INTERRUPTED')
        self.assertEqual(p.read(self.folder/'monitor.json')['status'],'INTERRUPTED')

    def test_stale_heartbeat(self):
        now=datetime.now(timezone.utc)
        record={'status':'RUNNING','heartbeat_at_utc':(now-timedelta(seconds=30)).isoformat()}
        self.assertEqual(monitor_status(record,now),'POSSIBLY_INTERRUPTED')
        record['heartbeat_at_utc']=now.isoformat()
        self.assertEqual(monitor_status(record,now),'RUNNING')
        record['status']='COMPLETED'
        self.assertEqual(monitor_status(record,now+timedelta(days=1)),'COMPLETED')

    def test_check_mode_skips_inference(self):
        with patch.object(p,'inspect_package',return_value=[]):
            result=p.run(self.config,'check',sys.executable,self.folder)
        mon=p.read(self.folder/'monitor.json')
        self.assertEqual(mon['status'],'COMPLETED')
        self.assertEqual(result['attempt_id'],mon['attempt_id'])
        self.assertTrue(all(s['status']=='SKIPPED' for s in mon['steps'] if s['id'] in ('direction','magnitude','combine')))

    def test_redaction(self):
        self.assertNotIn('example-secret',error_text(ValueError('password=example-secret')))

    def test_save_failure_is_recorded(self):
        original=p.atomic_json
        def broken(path,value):
            if path.name=='config.json': raise OSError('Simulated disk write failure')
            return original(path,value)
        with patch.object(p,'atomic_json',side_effect=broken):
            with self.assertRaises(OSError): p.run(self.config,'check',sys.executable,self.folder)
        mon=p.read(self.folder/'monitor.json')
        self.assertEqual(mon['status'],'FAILED')
        self.assertEqual(next(s['status'] for s in mon['steps'] if s['id']=='save'),'FAILED')

if __name__=='__main__': unittest.main()
