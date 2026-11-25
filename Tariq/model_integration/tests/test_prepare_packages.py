import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fixtures import p, prepare_fixture
import prepare_packages as setup


class PackageSetupTests(unittest.TestCase):
    def setUp(self):
        self.config = prepare_fixture(self)
        self.sources = {}
        for role, model in self.config['models'].items():
            for name in model['package_sha256']:
                self.sources[model['source_dir'] + '/' + name] = (p.ROOT / model['package_dir'] / name).read_bytes()
            model['package_dir'] = 'prepared/' + role
        self.patcher = patch.object(setup, 'ROOT', p.ROOT)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def fetch(self, args, **kwargs):
        source = args[-1].split(':', 1)[1]
        return SimpleNamespace(returncode=0, stdout=self.sources[source])

    def test_exact_bytes_and_reuse(self):
        with patch.object(setup.subprocess, 'run', side_effect=self.fetch) as git:
            count = setup.prepare(self.config, p.ROOT)
            self.assertGreater(count, 0)
            git.reset_mock()
            self.assertEqual(setup.prepare(self.config, p.ROOT), 0)
            git.assert_not_called()

    def test_bad_source_writes_nothing(self):
        with patch.object(setup.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=b'wrong')):
            with self.assertRaises(ValueError): setup.prepare(self.config, p.ROOT)
        self.assertFalse((p.ROOT / 'prepared').exists())

    def test_escape_refused(self):
        self.config['models']['direction']['package_dir'] = '../outside'
        with self.assertRaises(ValueError): setup.prepare(self.config, p.ROOT)


if __name__ == '__main__':
    unittest.main()
