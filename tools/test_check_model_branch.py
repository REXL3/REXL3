"""check_model_branch.py: one failing case per check (contract section 6) plus a passing control.

Uses disposable Git repos with the synthetic seal; no network, credentials or native code.
"""
import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools import check_model_branch as c
from tools import release_evidence as e
from tools.test_approve_profile import SyntheticSeal
from tools.test_release_evidence import encoded, sha, write

FAMILY, MODEL = 'qwen3.8', 'qwen3.8-27b'


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), '-c', 'user.name=test', '-c',
                           'user.email=test@example.invalid', *map(str, args)],
                          capture_output=True, check=True)


class ModelBranchFixture:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.f = SyntheticSeal(str(self.root))
        self.repo = self.f.repo
        self.approval, self.doc, self.registry = self.f.published_fixture_model(FAMILY, MODEL)
        self.registry_path = self.repo / 'runtimes/templates/APPROVALS.json'
        write(self.repo, 'runtimes/templates/APPROVALS.json', encoded(self.registry))
        write(self.repo, 'runtimes/templates/APPROVED.json',
             encoded({'schema': 'rexl3.approved-templates.v1', 'templates': self.f.approved}))
        pin = sha(self.registry_path.read_bytes())
        verify = ('name: verify\non:\n  pull_request_target:\n    branches: [main]\nenv:\n'
                  f'  TRUSTED_APPROVALS_SHA256: {pin}\n').encode()
        write(self.repo, '.github/workflows/verify.yml', verify)
        write(self.repo, '.github/CODEOWNERS', b'* @am423 @vcruz305\n')
        self.assets = self.root / 'assets'
        from tools.test_release_evidence import stage_assets
        stage_assets(self.assets / FAMILY / MODEL / 'v1', self.f, self.doc)

    def add_model(self):
        self.tree = self.repo / 'runtimes' / FAMILY / MODEL / 'v1'
        for name, data in self.f.source.items():
            write(self.tree, name, data).chmod(self.doc['source_asset']['members'][name]['mode'])
        write(self.tree, 'approval.json', self.approval.read_bytes())
        write(self.tree.parent, 'MODEL.json', e.json_bytes(self.f.model_doc(FAMILY, MODEL)))
        write(self.tree.parent, 'LATEST', b'v1\n')
        write(self.repo, f'docs/packs/{MODEL}.md', b'# Pack sheet\n')
        write(self.repo, f'changes/unreleased/{MODEL}.md', b'## Highlights\n- Model added.\n')

    def commit(self, message):
        git(self.repo, 'add', '.')
        git(self.repo, 'commit', '-qm', message)

    def init(self):
        git(self.repo, 'init', '-q')
        git(self.repo, 'add', '.')
        git(self.repo, 'commit', '-qm', 'base')


class CheckModelBranchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def fixture(self, sub='ok'):
        return ModelBranchFixture(self.tmp.name + '/' + sub)

    def run_check(self, fx, *args, branch=f'model/{MODEL}'):
        return c.main(['--repo-root', str(fx.repo), '--base', 'HEAD~1', '--head', 'HEAD',
                       '--branch', branch, '--assets-dir', str(fx.assets), *args])

    def output(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            pass
        return ''

    def captured(self, fn):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = fn()
        return code, out.getvalue().splitlines()

    def test_passing_model_branch(self):
        fx = self.fixture()
        fx.init()
        fx.add_model()
        fx.commit('model')
        code, lines = self.captured(lambda: self.run_check(fx))
        self.assertEqual(code, 0, '\n'.join(lines))
        self.assertEqual([line for line in lines if line.startswith('FAIL')], [])

    def test_failing_case_for_every_check(self):
        cases = {
            'branch/model': (['--branch', 'model/other-model'], 'branch/model'),
            'scope': ([], 'scope'),
            'MODEL.json': ([], 'MODEL.json'),
            'LATEST': ([], 'LATEST'),
            'tree': ([], 'tree'),
            'source commit': (['--rc-commit', '0' * 40], 'source commit'),
            'capability contract': ([], 'capability contract'),
            'docs and fragment': ([], 'docs and fragment'),
        }
        for name, (extra, check_name) in cases.items():
            with self.subTest(check=name):
                fx = self.fixture(name.replace(' ', '-'))
                fx.init()
                fx.add_model()
                if name == 'scope':
                    write(fx.repo, 'tools/unexpected.txt', b'x')
                elif name == 'MODEL.json':
                    write(fx.tree.parent, 'MODEL.json', e.json_bytes(fx.f.model_doc(FAMILY, 'other-model')))
                elif name == 'LATEST':
                    write(fx.tree.parent, 'LATEST', b'v2\n')
                elif name == 'tree':
                    write(fx.tree, 'launch.sh', b'forged').chmod(0o755)
                elif name == 'capability contract':
                    profile = e.load_json(fx.tree / 'profile.json')
                    profile['profiles']['single']['capabilities']['prefix_cache'] = False
                    write(fx.tree, 'profile.json', encoded(profile))
                elif name == 'docs and fragment':
                    (fx.repo / f'changes/unreleased/{MODEL}.md').unlink()
                fx.commit('model')
                code, lines = self.captured(lambda: self.run_check(fx, *extra))
                self.assertEqual(code, 1, '\n'.join(lines))
                fail_lines = [line for line in lines if line.startswith('FAIL')]
                self.assertTrue(any(line.startswith('FAIL ' + check_name) for line in fail_lines),
                                f'expected FAIL {check_name} in:\n' + '\n'.join(lines))

    def test_no_op_success_for_a_non_model_branch(self):
        fx = self.fixture('engine')
        fx.init()
        write(fx.repo, 'tools/check.txt', b'x')
        fx.commit('engine')
        code, lines = self.captured(lambda: self.run_check(fx, branch='engine/topic'))
        self.assertEqual(code, 0, '\n'.join(lines))
        self.assertTrue(all(line.startswith('PASS') for line in lines), lines)


if __name__ == '__main__':
    unittest.main()
