"""Offline tests: no trained-model execution and no MLflow connection."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import uuid

sys.dont_write_bytecode = True
import pipeline as p


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = p.read(p.ROOT / 'config.json')
        self.d = p.read(p.ROOT / self.config['models']['direction']['package_dir'] / 'expected_output.json')
        self.m = p.read(p.ROOT / self.config['models']['magnitude']['package_dir'] / 'expected_output.json')

    def test_formula(self):
        self.d['p_up'] = .3
        self.m['net_magnitude_bps'] = 50
        self.assertAlmostEqual(p.combine(self.d, self.m, self.config), -20)

    def test_neutral_and_extremes(self):
        for prob, expected in [(0.5, 0), (1, self.m['net_magnitude_bps']), (0, -self.m['net_magnitude_bps'])]:
            self.d['p_up'] = prob
            self.assertEqual(p.combine(self.d, self.m, self.config), expected)

    def test_bad_probability(self):
        for value in [-0.01, 1.01, float('nan'), float('inf'), True]:
            self.d['p_up'] = value
            with self.assertRaises(ValueError): p.combine(self.d, self.m, self.config)

    def test_bad_magnitude(self):
        for value in [-1, float('nan'), float('inf'), True]:
            self.m['net_magnitude_bps'] = value
            with self.assertRaises(ValueError): p.combine(self.d, self.m, self.config)

    def test_different_timestamp(self):
        self.m['forecast_time_utc'] = '2025-01-01T00:30:00Z'
        with self.assertRaises(ValueError): p.combine(self.d, self.m, self.config)

    def test_wrong_grid(self):
        self.d['forecast_time_utc'] = '2025-01-01T00:15:01Z'
        with self.assertRaises(ValueError): p.combine(self.d, self.m, self.config)

    def test_missing_timezone(self):
        self.d['forecast_time_utc'] = '2025-01-01T00:00:00'
        with self.assertRaises(ValueError): p.combine(self.d, self.m, self.config)

    def test_wrong_identity(self):
        for key in ['symbol', 'horizon', 'model_version']:
            bad = dict(self.d, **{key: 'wrong'})
            with self.assertRaises(ValueError): p.combine(bad, self.m, self.config)

    def test_features(self):
        model = self.config['models']['direction']
        package = p.ROOT / model['package_dir']
        example = p.read(package / 'example_input.json')
        columns = p.input_columns(package / 'predict.py')
        p.validate_input(example, columns, self.config, model)
        example['features']['unexpected'] = 0
        with self.assertRaises(ValueError): p.validate_input(example, columns, self.config, model)

    def test_traversal(self):
        with self.assertRaises(ValueError): p.inside(p.ROOT, '../elsewhere')

    def test_package_tamper(self):
        self.config['models']['direction']['package_sha256']['predict.py'] = '0'*64
        checks = p.inspect_package(self.config, 'direction')
        self.assertIn({'check': 'package:predict.py', 'status': 'FAIL', 'detail': 'predict.py differs from pinned value'}, checks)

    def result_path(self):
        return p.ROOT / '.verification' / self._testMethodName

    def test_reuse_clears_stale_outputs(self):
        output = self.result_path()
        output.mkdir(parents=True, exist_ok=True)
        (output / 'direction_output.json').write_text('{"stale": true}')
        result = p.run(self.config, 'check', sys.executable, output)
        self.assertFalse((output / 'direction_output.json').exists())
        self.assertNotIn('outputs', result)
        p.run(self.config, 'check', sys.executable, output)
        self.assertTrue((output / 'result.json').exists())

    def test_reported_example_never_executes(self):
        with patch.object(p.subprocess, 'run', side_effect=AssertionError('Must not execute')):
            result = p.run(self.config, 'reported-example', sys.executable, self.result_path())
        self.assertEqual(result['status'], 'FORMULA_DEMO_NOT_MODEL_REPLAY')
        windows = [m['target_window'] for m in self.config['models'].values()]
        self.assertEqual(result['target_alignment'], 'MATCH' if windows[0] == windows[1] else 'MISMATCH')
        self.assertFalse(result['trading_enabled'])

    def test_v2_contract(self):
        c = p.read(p.ROOT / 'config.v002.json')
        self.assertEqual(p.evaluation_alignment(c)['status'], 'MATCH')
        for role, model in c['models'].items():
            checks = p.inspect_package(c, role)
            self.assertFalse([x for x in checks if x['status'] == 'FAIL' and not x['check'].startswith('artifact:')])
            package = p.ROOT / model['package_dir']
            example = p.read(package / 'example_input.json')
            example['forecast_time_utc'] = '2024-10-01T00:01:00Z'
            p.validate_input(example, p.input_columns(package / 'predict.py'), c, model)

    def test_v2_scale_required(self):
        c = p.read(p.ROOT / 'config.v002.json')
        model = c['models']['magnitude']
        package = p.ROOT / model['package_dir']
        example = p.read(package / 'example_input.json')
        for value in [None, 0, -1, True, float('nan')]:
            example['scale_bps'] = value
            with self.assertRaises(ValueError):
                p.validate_input(example, p.input_columns(package / 'predict.py'), c, model)

    def test_upgrade_refuses_demo(self):
        from upgrade_v002 import verify_ready
        with self.assertRaises(ValueError):
            verify_ready(p.read(p.ROOT / 'config.v002.json'), {'status': 'FORMULA_DEMO_NOT_MODEL_REPLAY'})

    def test_upgrade_cleanup_targets_only_old_files(self):
        from upgrade_v002 import old_file_plan
        c = p.read(p.ROOT / 'config.v002.json')
        paths = old_file_plan(self.config, c)
        self.assertTrue(all(path.is_relative_to(p.ROOT) and 'v002' not in path.parts for path in paths))

    def test_replay_refuses_missing_artifacts(self):
        config = copy.deepcopy(self.config)
        for model in config['models'].values(): model['artifact_dir'] = '.verification/missing-' + uuid.uuid4().hex
        with patch.object(p.subprocess, 'run', side_effect=AssertionError('Must not execute')):
            result = p.run(config, 'replay', sys.executable, self.result_path())
        self.assertEqual(result['status'], 'FAILED')
        self.assertNotIn('predicted_signed_score_bps', result)

    def test_simulated_replay(self):
        def fake_run(args, **kwargs):
            role = Path(args[1]).parent.name
            expected = self.d if role == 'direction' else self.m
            Path(args[args.index('--output') + 1]).write_text(json.dumps(expected))
            return type('Completed', (), {'returncode': 0})()
        with patch.object(p, 'inspect_package', return_value=[]), patch.object(p, 'environment_check'), patch.object(p.subprocess, 'run', side_effect=fake_run):
            result = p.run(self.config, 'replay', sys.executable, self.result_path())
        self.assertEqual(result['status'], 'REPLAY_PASSED_DEVELOPMENT_ONLY')
        self.assertFalse(result['trading_enabled'])


if __name__ == '__main__':
    unittest.main()
