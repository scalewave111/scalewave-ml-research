"""Offline read-only tests: no UI server, MLflow access or generated fixtures."""
import ast
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from fixtures import state


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Synthetic records exercise the UI contract without local run history.
        import hashlib
        config = json.loads((state.ROOT / 'config/models.json').read_text())
        outputs = {role: {'symbol': config['symbol'], 'horizon': config['horizon'],
                          'forecast_time_utc': '2024-10-01T00:00:00Z', 'model_version': model['version'],
                          model['output_field']: 0.5071603369109142 if role == 'direction' else 7.983212754702233}
                   for role, model in config['models'].items()}
        result = {
            'status': 'REPLAY_PASSED_DEVELOPMENT_ONLY',
            'prediction_source': 'EXECUTED_FROZEN_MODELS',
            'config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            'outputs': outputs,
            'checks': {role: [{'check': 'prediction_replay', 'status': 'PASS'}] for role in outputs},
            'predicted_signed_score_bps': (2 * outputs['direction']['p_up'] - 1) * outputs['magnitude']['net_magnitude_bps'],
        }
        original = state.read_json
        def read(path, kind, warnings):
            if path.name == 'result.json': return result
            if path.name == 'monitor.json': return {}
            if path.name == 'history.json': return []
            return original(path, kind, warnings)
        with patch.object(state, 'read_json', side_effect=read):
            cls.baseline = state.snapshot()

    def simulate_record(self, result=None, config=None, history=None, monitor=None):
        config = copy.deepcopy(self.baseline['config'] if config is None else config)
        result = copy.deepcopy(self.baseline['result'] if result is None else result)
        def read(path, kind, warnings):
            if path == state.ROOT / 'config/models.json': return config
            if path.name == 'result.json': return result
            if path.name == 'history.json': return [] if history is None else history
            if path.name == 'monitor.json': return {} if monitor is None else monitor
            return {}
        original_exists = Path.exists
        def exists(path):
            if path.name == 'monitor.json': return monitor is not None
            return original_exists(path)
        with patch.object(state, 'read_json', side_effect=read), patch.object(Path, 'exists', exists):
            return state.snapshot()

    def test_current_verified_record(self):
        self.assertTrue(self.baseline['verified_outputs'])
        self.assertAlmostEqual(self.baseline['score'], 0.11432498591035122)
        self.assertEqual(len(self.baseline['checks']), 2)

    def test_short_long_and_boundaries(self):
        for probability, magnitude, score, interpretation in [(.8,20,12,'LONG'),(.2,20,-12,'SHORT'),(.75,20,10,'NO TRADE'),(.25,20,-10,'NO TRADE'),(.5,20,0,'NO TRADE')]:
            s = state.simulate(probability,magnitude,10)
            self.assertAlmostEqual(s['score'],score)
            self.assertIn(interpretation,s['interpretation'])

    def test_invalid_simulator_inputs(self):
        for p,m,t in [(float('nan'),20,10),(1.1,20,10),(.5,-1,10),(.5,20,0),(True,20,10)]:
            with self.assertRaises(ValueError): state.simulate(p,m,t)

    def test_failed_result_withholds_old_outputs(self):
        result=copy.deepcopy(self.baseline['result']); result['status']='FAILED'
        self.assertIsNone(self.simulate_record(result=result)['score'])

    def test_new_configuration_withholds_old_outputs(self):
        config=copy.deepcopy(self.baseline['config']); config['formula_version']='changed'
        data=self.simulate_record(config=config)
        self.assertFalse(data['matches']); self.assertIsNone(data['score'])

    def test_formula_demo_is_not_replay(self):
        result=copy.deepcopy(self.baseline['result']); result['prediction_source']='SUBMITTED_EXPECTED_OUTPUTS_ONLY'
        self.assertFalse(self.simulate_record(result=result)['verified_outputs'])

    def test_invalid_score_withheld(self):
        result=copy.deepcopy(self.baseline['result']); result['predicted_signed_score_bps']=100
        self.assertIsNone(self.simulate_record(result=result)['score'])

    def test_wrong_output_version_withheld(self):
        result=copy.deepcopy(self.baseline['result']); result['outputs']['direction']['model_version']='old'
        self.assertFalse(self.simulate_record(result=result)['verified_outputs'])

    def test_missing_checks_withheld(self):
        result=copy.deepcopy(self.baseline['result']); result['checks']={}
        self.assertFalse(self.simulate_record(result=result)['verified_outputs'])

    def test_failed_check_withheld(self):
        result=copy.deepcopy(self.baseline['result']); result['checks']['direction'][0]['status']='FAIL'
        self.assertFalse(self.simulate_record(result=result)['verified_outputs'])

    def test_missing_and_malformed_json(self):
        warnings=[]
        with patch.object(Path,'read_bytes',side_effect=FileNotFoundError('Missing')):
            self.assertEqual(state.read_json(Path('missing'),dict,warnings),{})
        with patch.object(Path,'read_bytes',return_value=b'{invalid'):
            self.assertEqual(state.read_json(Path('partial'),dict,warnings),{})
        self.assertEqual(len(warnings),2)

    def test_history_bounded(self):
        self.assertEqual(len(self.simulate_record(history=[{}]*200)['history']),100)

    def test_source_syntax(self):
        for name in ('app.py','launch.py','state.py'):
            ast.parse((state.ROOT/'dashboard'/name).read_text(encoding='utf-8'))

    def test_running_monitor_withholds_previous_result(self):
        from datetime import datetime, timezone
        record={'attempt_id':'new-attempt','status':'RUNNING','heartbeat_at_utc':datetime.now(timezone.utc).isoformat()}
        self.assertFalse(self.simulate_record(monitor=record)['verified_outputs'])

    def test_failed_save_withholds_result(self):
        record={'attempt_id':self.baseline['result'].get('attempt_id'),'status':'FAILED','steps':[]}
        self.assertFalse(self.simulate_record(monitor=record)['verified_outputs'])


if __name__ == '__main__': unittest.main()
