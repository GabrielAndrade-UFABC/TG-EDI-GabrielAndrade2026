import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main


class SlottedMetaTest(unittest.TestCase):
    def test_class_without_slots_is_rejected(self):
        with self.assertRaisesRegex(TypeError, 'Loose.*__slots__'):
            class Loose(main.Slotted):
                pass

    def test_instances_do_not_accept_undeclared_attributes(self):
        class Point(main.Slotted):
            __slots__ = ('x',)
        point = Point()
        point.x = 1
        with self.assertRaises(AttributeError):
            point.y = 2


class DependencyCheckTest(unittest.TestCase):
    def check(self, absent=(), answer='', install_code=0, interactive=True, installs_fix=True):
        self.installed = []
        self.asked = []
        missing = set(absent)

        def install(command):
            self.installed.append(command)
            if install_code == 0 and installs_fix:
                missing.clear()
            return install_code

        def ask(prompt):
            self.asked.append(prompt)
            return answer

        return main.DependencyCheck({'torch': 'torch', 'yaml': 'PyYAML'},
                                    find_spec=lambda module: None if module in missing else object(),
                                    install=install, ask=ask, interactive=lambda: interactive,
                                    executable='/py')

    def test_missing_reports_pip_package_names(self):
        self.assertEqual(self.check(absent={'yaml'}).missing(), ['PyYAML'])

    def test_nothing_missing_does_nothing(self):
        self.check().ensure()
        self.assertEqual((self.installed, self.asked), ([], []))

    def test_non_interactive_exits_with_install_command(self):
        with redirect_stderr(io.StringIO()), self.assertRaisesRegex(SystemExit, '/py -m pip install torch'):
            self.check(absent={'torch'}, interactive=False).ensure()
        self.assertEqual((self.installed, self.asked), ([], []))

    def test_accepted_install_that_works_returns(self):
        with redirect_stderr(io.StringIO()):
            self.check(absent={'torch'}, answer='s').ensure()
        self.assertEqual(self.installed, [['/py', '-m', 'pip', 'install', 'torch']])

    def test_failed_install_exits_and_says_so(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit):
            self.check(absent={'torch'}, install_code=1).ensure()
        self.assertIn('A instalação falhou.', stderr.getvalue())

    def test_install_that_leaves_packages_missing_exits(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.check(absent={'torch'}, installs_fix=False).ensure()

    def test_declined_install_exits_without_installing(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.check(absent={'torch'}, answer='n').ensure()
        self.assertEqual(self.installed, [])


class ModelCheckTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        present = self.root / 'det'
        present.mkdir()
        (present / 'model.safetensors').write_bytes(b'')
        self.models = main.ModelCheck(self.root, {'detection': present, 'cwe': self.root / 'cwe',
                                                  'fix': self.root / 'fix'})

    def tearDown(self):
        self.directory.cleanup()

    def test_lists_only_absent_checkpoints(self):
        self.assertEqual(self.models.absent([]), ['cwe', 'fix'])

    def test_overridden_checkpoints_are_not_required(self):
        self.assertEqual(self.models.absent(['--cwe-model', 'x', '--fix-model=y']), [])

    def test_ensure_exits_listing_absent_checkpoints(self):
        with self.assertRaisesRegex(SystemExit, r'(?s)Checkpoints não encontrados.*cwe.*fix'):
            self.models.ensure([])

    def test_ensure_passes_when_all_present(self):
        self.models.ensure(['--cwe-model', 'x', '--fix-model', 'y'])


class LauncherTest(unittest.TestCase):
    class Recorder:
        def __init__(self, calls, name):
            self.calls, self.name = calls, name

        def ensure(self, *args):
            self.calls.append((self.name, *args))

    def launcher(self, argv):
        self.calls = []

        def entry(args):
            self.calls.append(('entry', args))
            return 7

        return main.Launcher(argv, self.Recorder(self.calls, 'deps'), self.Recorder(self.calls, 'models'), entry)

    def test_runs_checks_in_order_then_returns_pipeline_code(self):
        argv = ['--input', 'x.csv']
        self.assertEqual(self.launcher(argv).run(), 7)
        self.assertEqual(self.calls, [('deps',), ('models', argv), ('entry', argv)])

    def test_help_skips_checks(self):
        for flag in ('-h', '--help'):
            self.assertEqual(self.launcher([flag]).run(), 7)
            self.assertEqual(self.calls, [('entry', [flag])])

    def test_help_runs_without_model_dependencies(self):
        # Integração real: --help funciona mesmo sem torch instalado.
        result = subprocess.run([sys.executable, str(ROOT / 'main.py'), '--help'],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--resume-from', result.stdout)


if __name__ == '__main__':
    unittest.main()
