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
    def __init__(self, root, **seal):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.f = SyntheticSeal(str(self.root), **seal)
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

    def fixture(self, sub='ok', **seal):
        return ModelBranchFixture(self.tmp.name + '/' + sub, **seal)

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


def vision_off_profile(*names):
    caps = {key: key not in ('audio', 'video', 'json_schema', 'images') for key in (
        'text', 'sse', 'reasoning', 'tools', 'images', 'video', 'audio', 'json_schema', 'prefix_cache')}
    return {'schema': 'rexl3.runtime-profile.v1', 'state': 'final', 'profiles': {
        name: {'kind': 'single', 'argv': ['--draft', '@DRAFT_DIR@', '--spec', 'auto', '--no-vision'],
               'capabilities': dict(caps)}
        for name in names}}


OWNER_EXCEPTION = dict(granted_by='am423', granted='2026-11-01', expires='2099-12-31',
                       reason='synthetic owner exception')
LOCK = {'schema': 'rexl3.model-lock.v1', 'packs': {'target': {'repo': 'synthetic/model'},
                                                   'draft': {'repo': 'synthetic/draft'}}}


class VisionExceptionContractTest(unittest.TestCase):
    """Vision off: a does-not-fit justification, or an unexpired owner exception sealed in every receipt."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def vision_problems(self, exceptions, names=('single',), model_doc=None):
        profile = vision_off_profile(*names)
        problems = c.resolve_vision_off(c.capability_problems(profile, model_doc, models=LOCK), profile, exceptions)
        return [problem for problem in problems if problem.startswith('vision is off')]

    def test_the_resolved_finding_is_the_one_capability_problems_reports(self):
        self.assertEqual(c.capability_problems(vision_off_profile('single'), None, models=LOCK), [c.VISION_OFF_PROBLEM])
        self.assertEqual(c.resolve_vision_off(['other'], vision_off_profile('single'), None), ['other'])

    def test_a_valid_owner_exception_on_every_profile_is_accepted(self):
        self.assertEqual(self.vision_problems({'single': OWNER_EXCEPTION}), [])
        self.assertEqual(self.vision_problems({'single': OWNER_EXCEPTION, 'tp2': dict(OWNER_EXCEPTION,
                                               granted_by='vcruz305')}, names=('single', 'tp2')), [])

    def test_a_missing_expired_or_malformed_exception_is_refused(self):
        cases = {
            'no receipts read': None,
            'no record': {'single': None},
            'expired': {'single': dict(OWNER_EXCEPTION, expires='2020-01-01')},
            'not a maintainer': {'single': dict(OWNER_EXCEPTION, granted_by='mallory')},
            'no reason': {'single': dict(OWNER_EXCEPTION, reason=' ')},
            'bad date': {'single': dict(OWNER_EXCEPTION, expires='2026-13-01')},
            'extra field': {'single': dict(OWNER_EXCEPTION, tracking='x')},
        }
        for name, exceptions in cases.items():
            with self.subTest(case=name):
                problems = self.vision_problems(exceptions)
                self.assertEqual(len(problems), 1, problems)
                self.assertIn('no valid owner vision exception is sealed (single: ', problems[0])

    def test_every_vision_off_profile_needs_its_own_exception(self):
        problems = self.vision_problems({'single': OWNER_EXCEPTION}, names=('single', 'tp2'))
        self.assertEqual(len(problems), 1, problems)
        self.assertIn('(tp2: ', problems[0])
        self.assertNotIn('single: ', problems[0])

    def test_a_does_not_fit_justification_still_suffices(self):
        model_doc = {'pack': {'format': 'exl3', 'vision_does_not_fit': 'measured: the tower does not fit'}}
        self.assertEqual(self.vision_problems(None, model_doc=model_doc), [])

    def test_sealed_receipts_are_read_only_when_their_bytes_match_the_approval(self):
        receipt = write(Path(self.tmp.name), 'r/qualification.json', encoded({'vision_exception': OWNER_EXCEPTION}))
        approval = {'profiles': {'single': {'receipt': {'path': 'r/qualification.json',
                                                        'sha256': sha(receipt.read_bytes())}}}}
        self.assertEqual(c.sealed_vision_exceptions(approval, Path(self.tmp.name)), {'single': OWNER_EXCEPTION})
        approval['profiles']['single']['receipt']['sha256'] = '0' * 64
        self.assertEqual(c.sealed_vision_exceptions(approval, Path(self.tmp.name)), {'single': None})
        self.assertEqual(c.sealed_vision_exceptions(None, Path(self.tmp.name)), {})
        self.assertEqual(c.sealed_vision_exceptions(approval, None), {})

    def run_check(self, fx):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = c.main(['--repo-root', str(fx.repo), '--base', 'HEAD~1', '--head', 'HEAD',
                           '--branch', f'model/{MODEL}', '--assets-dir', str(fx.assets)])
        return code, out.getvalue().splitlines()

    def test_a_vision_off_model_branch_passes_with_the_sealed_owner_exception(self):
        fx = ModelBranchFixture(self.tmp.name + '/vision-off', images=False)
        self.assertNotIn('vision_does_not_fit', fx.f.model_doc(FAMILY, MODEL)['pack'])
        fx.init()
        fx.add_model()
        fx.commit('model')
        code, lines = self.run_check(fx)
        self.assertEqual(code, 0, '\n'.join(lines))
        self.assertIn('PASS capability contract', lines)

    def test_a_tampered_exception_in_the_assets_fails_the_tree_and_the_contract(self):
        fx = ModelBranchFixture(self.tmp.name + '/tampered', images=False)
        fx.init()
        fx.add_model()
        fx.commit('model')
        receipt_path = fx.assets / FAMILY / MODEL / 'v1' / 'qualification.json'
        receipt = e.load_json(receipt_path)
        receipt['vision_exception'] = dict(OWNER_EXCEPTION, expires='2020-01-01')
        receipt_path.write_bytes(encoded(receipt))
        code, lines = self.run_check(fx)
        self.assertEqual(code, 1, '\n'.join(lines))
        self.assertTrue(any(line.startswith('FAIL tree') for line in lines), lines)
        self.assertTrue(any(line.startswith('FAIL capability contract') and 'no valid owner vision exception' in line
                            for line in lines), lines)


if __name__ == '__main__':
    unittest.main()
