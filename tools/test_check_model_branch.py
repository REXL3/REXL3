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


# Every serving profile on REXL3-dev main fe729abf: the argv copied verbatim from
# runtimes/<family>/<model>/profile.json and the roles of that entry's model.lock.json packs.
DEV_MAIN_PROFILES = {
    # runtimes/deepseek-v4.1/deepseek-v4.1-flash-1017/profile.json, profile dsv41-single-262k
    'dsv41-single-262k': (
        ['--model', '@MODEL_DIR@', '--host', '@HTTP_HOST@', '--port', '@HTTP_PORT@', '--slots', '1',
         '--max-model-len', '262144', '--kv-dtype', 'q4', '--no-video', '--spec', 'auto', '--prefix-cache',
         'on', '--served-model-name', 'DSV4.1-Flash-1017-SAGE-REXL3-1.55bpw'],
        ('target', 'dsv41-1.62')),
    # runtimes/deepseek-v4.1/deepseek-v4.1-flash-1017/profile.json, profile dsv41-tp2-1m
    'dsv41-tp2-1m': (
        ['--model', '@MODEL_DIR@', '--slots', '1', '--max-model-len', '1048576', '--kv-dtype', 'q4',
         '--no-video', '--spec', 'auto', '--prefix-cache', 'on', '--served-model-name',
         'DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw', '--host', '@HTTP_HOST@', '--port', '@HTTP_PORT@',
         '--tp-ranks', '2', '--tp-rank', '0', '--tp-peer', '@NODES@:@TP_PORT@', '--tp-connect-timeout',
         '300', '--tp-ready-timeout', '900', '--tp-peer-timeout', '60', '--shutdown-grace', '5'],
        ('target', 'dsv41-1.62')),
    # runtimes/glm-5.3/glm-5.3-flash/profile.json, profile glm53-flash-single
    'glm53-flash-single': (
        ['--model', '@MODEL_DIR@', '--host', '@HTTP_HOST@', '--port', '@HTTP_PORT@', '--slots', '1',
         '--max-model-len', '262144', '--spec', 'auto', '--prefix-cache', 'on', '--prefix-cache-mib', '3584',
         '--served-model-name', 'GLM-5.3-Flash-EXL3-2.25bpw-sm121'],
        ('target',)),
    # runtimes/glm-5.3/glm-5.3-flash/profile.json, profile glm53-flash-tp2-1m
    'glm53-flash-tp2-1m': (
        ['--model', '@MODEL_DIR@', '--slots', '1', '--max-model-len', '1048576', '--kv-tokens', '1048576',
         '--spec', 'auto', '--prefix-cache', 'on', '--prefix-cache-mib', '13440', '--served-model-name',
         'GLM-5.3-Flash-EXL3-2.25bpw-sm121', '--host', '@HTTP_HOST@', '--port', '@HTTP_PORT@', '--tp-ranks',
         '2', '--tp-rank', '0', '--tp-peer', '@NODES@:@TP_PORT@', '--tp-connect-timeout', '300',
         '--tp-ready-timeout', '900', '--tp-peer-timeout', '60', '--shutdown-grace', '5'],
        ('target',)),
    # runtimes/glm-5.3/red-snow-5.3-flash/profile.json, profile redsnow-single
    'redsnow-single': (
        ['--model', '@MODEL_DIR@', '--host', '@HTTP_HOST@', '--port', '@HTTP_PORT@', '--slots', '1',
         '--max-model-len', '262144', '--no-vision', '--spec', 'auto', '--prefix-cache', 'on',
         '--prefix-cache-mib', '1024', '--served-model-name', 'RED-SNOW-5.3-FLASH-EXL3-SAGE-2.49bpw'],
        ('target',)),
    # runtimes/ling-3.1/ling-3.1-flash-1019/profile.json, profile ling31-single-32k
    # (no model.lock.json on dev main yet; the target role is assumed)
    'ling31-single-32k': (
        ['--model', '@MODEL_DIR@', '--host', '@HTTP_HOST@', '--port', '@HTTP_PORT@', '--max-model-len',
         '32768', '--slots', '1', '--spec', 'mtp', '--prefix-cache', 'on', '--heartbeat-stale-secs', '1800',
         '--request-timeout', '7200', '--served-model-name', 'Ling-3.1-flash-1019-EXL3-SAGE-1.51bpw'],
        ('target',)),
    # runtimes/nemotron-3.5/nemotron-3.5-super-vl/profile.json, profile nemotron35-super-vl-single-1m
    'nemotron35-super-vl-single-1m': (
        ['--model', '@MODEL_DIR@', '--slots', '16', '--max-model-len', '1048576', '--spec', 'auto',
         '--prefix-cache', 'on', '--request-timeout', '7200', '--queue-timeout', '6900',
         '--served-model-name', 'nemotron-3.5-super-vl', '--host', '@HTTP_HOST@', '--port', '@HTTP_PORT@'],
        ('target',)),
    # runtimes/qwen3.8/qwen3.8-27b/profile.json, profile qwen38-single-262k
    'qwen38-single-262k': (
        ['--model', '@MODEL_DIR@', '--draft', '@DRAFT_DIR@', '--host', '@HTTP_HOST@', '--port',
         '@HTTP_PORT@', '--slots', '6', '--max-model-len', '262144', '--spec', 'auto', '--prefix-cache',
         'on', '--prefix-cache-mib', '40960', '--default-reasoning-effort', 'low', '--served-model-name',
         'qwen38-27b-exl3'],
        ('target', 'draft')),
    # runtimes/qwen3.8/qwen3.8-flash-next/profile.json, profile qwen38-flash-next-single-262k
    'qwen38-flash-next-single-262k': (
        ['--model', '@MODEL_DIR@', '--max-model-len', '262144', '--spec', 'mtp', '--prefix-cache', 'on',
         '--prefix-cache-mib', '22528', '--no-vision', '--served-model-name', 'default', '--slots', '2',
         '--queue-timeout', '1500', '--host', '@HTTP_HOST@', '--port', '@HTTP_PORT@'],
        ('target',)),
}


def sealed_profile(argv):
    """A sealed profile doc with one choice whose only open capability question is speculation."""
    caps = {key: key not in ('audio', 'video', 'json_schema') for key in (
        'text', 'sse', 'reasoning', 'tools', 'images', 'video', 'audio', 'json_schema', 'prefix_cache')}
    return {'schema': 'rexl3.runtime-profile.v1', 'state': 'final',
            'profiles': {'choice': {'kind': 'single', 'argv': list(argv), 'capabilities': caps}}}


def lock(*roles):
    return {'schema': 'rexl3.model-lock.v1', 'packs': {role: {'repo': 'synthetic/' + role} for role in roles}}


class SpeculationRuleTest(unittest.TestCase):
    """The capability contract reads speculative decoding the way the engine parses argv."""

    def spec_problems(self, argv, roles=('target',)):
        problems = c.capability_problems(sealed_profile(argv), None, models=lock(*roles))
        return [problem for problem in problems if 'speculative decoding' in problem]

    def test_every_dev_main_profile_counts_as_speculative(self):
        for name, (argv, roles) in DEV_MAIN_PROFILES.items():
            with self.subTest(profile=name):
                self.assertEqual(c.capability_problems(sealed_profile(argv), None, models=lock(*roles)), [])

    def test_spec_auto_and_mtp_count_without_a_draft_pack(self):
        for mode in ('auto', 'mtp'):
            with self.subTest(mode=mode):
                self.assertEqual(self.spec_problems(['--max-model-len', '262144', '--spec', mode]), [])

    def test_argv_that_does_not_draft_with_a_model_fails(self):
        cases = {
            'no flag and no draft pack': [],
            'spec none': ['--spec', 'none'],
            'spec copy': ['--spec', 'copy'],
            'no-drafts': ['--spec', 'auto', '--no-drafts'],
            'no-drafts first': ['--no-drafts', '--spec', 'mtp'],
            'last spec wins': ['--spec', 'mtp', '--spec', 'none'],
            'draft with copy': ['--draft', '@DRAFT_DIR@', '--spec', 'copy'],
            'draft with none': ['--draft', '@DRAFT_DIR@', '--spec', 'none'],
            'draft with no-drafts': ['--draft', '@DRAFT_DIR@', '--no-drafts'],
            'spec without a value': ['--max-model-len', '262144', '--spec'],
            'spec followed by a flag': ['--spec', '--prefix-cache', 'on'],
            'draft without a value': ['--draft'],
            'unknown spec value': ['--spec', 'off'],
            'spec value is case-sensitive': ['--spec', 'AUTO'],
            'no equals form in the engine': ['--spec=auto'],
            'not a list': '--spec auto',
        }
        for name, argv in cases.items():
            with self.subTest(case=name):
                problems = c.capability_problems(
                    {'schema': 'rexl3.runtime-profile.v1', 'state': 'final', 'profiles': {'choice': dict(
                        sealed_profile([])['profiles']['choice'], argv=argv)}}, None, models=lock('target'))
                self.assertEqual(problems, ['profile choice: speculative decoding is not on and proven'])

    def test_the_last_spec_value_wins(self):
        self.assertEqual(self.spec_problems(['--spec', 'none', '--spec', 'auto']), [])

    def test_an_alternate_target_pack_is_not_a_draft(self):
        argv, roles = DEV_MAIN_PROFILES['dsv41-tp2-1m']
        self.assertEqual(roles, ('target', 'dsv41-1.62'))
        at = argv.index('--spec')
        without_spec = argv[:at] + argv[at + 2:]
        serial = argv[:at + 1] + ['none'] + argv[at + 2:]
        for name, mutated in (('no --spec', without_spec), ('--spec none', serial)):
            with self.subTest(argv=name):
                self.assertEqual(self.spec_problems(mutated, roles),
                                 ['profile choice: speculative decoding is not on and proven'])

    def test_a_draft_role_in_the_lock_still_counts_unless_drafting_is_off(self):
        self.assertEqual(self.spec_problems(['--max-model-len', '262144'], ('target', 'draft')), [])
        self.assertEqual(self.spec_problems(['--max-model-len', '262144', '--no-drafts'], ('target', 'draft')),
                         ['profile choice: speculative decoding is not on and proven'])
        self.assertEqual(self.spec_problems(['--spec', 'copy'], ('target', 'draft')),
                         ['profile choice: speculative decoding is not on and proven'])

    def test_every_choice_is_checked_on_its_own(self):
        profile = sealed_profile(DEV_MAIN_PROFILES['glm53-flash-single'][0])
        profile['profiles']['second'] = dict(profile['profiles']['choice'], argv=['--spec', 'none'])
        self.assertEqual(c.capability_problems(profile, None, models=lock('target')),
                         ['profile second: speculative decoding is not on and proven'])


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

    def test_a_model_whose_profiles_pin_their_own_pack_and_runtime_id_passes(self):
        # The DeepSeek shape: a second profile with its own lock role and runtime id (dev #159).
        fx = self.fixture('per-profile', alt_profile=dict(pack='alt', runtime_id='synthetic-alt'))
        fx.init()
        fx.add_model()
        fx.commit('model')
        code, lines = self.captured(lambda: self.run_check(fx))
        self.assertEqual(code, 0, '\n'.join(lines))
        self.assertIn('PASS tree', lines)

    def test_failing_case_for_every_check(self):
        cases = {
            'branch/model': (['--branch', 'model/other-model'], 'branch/model'),
            'scope': ([], 'scope'),
            'MODEL.json': ([], 'MODEL.json'),
            'LATEST': ([], 'LATEST'),
            'tree': ([], 'tree'),
            'source commit': (['--rc-commit', '0' * 40], 'source commit'),
            'capability contract': ([], 'capability contract'),
            'capability contract (speculation)': ([], 'capability contract'),
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
                elif name == 'capability contract (speculation)':
                    profile = e.load_json(fx.tree / 'profile.json')
                    profile['profiles']['single']['argv'].append('--no-drafts')
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

    def test_a_packaged_profile_the_approval_does_not_name_is_not_judged(self):
        # tp1 approved alone (D38/D53): the unapproved tp2 choice needs no owner exception and its flags are not judged.
        profile = vision_off_profile('single', 'tp2')
        approval = {'profiles': {'single': {}}}
        view = c.approved_view(profile, approval)
        self.assertEqual(list(view['profiles']), ['single'])
        problems = c.resolve_vision_off(c.capability_problems(view, None, models=LOCK), view, {'single': OWNER_EXCEPTION})
        self.assertEqual([p for p in problems if p.startswith('vision is off')], [])
        # Without an approval (or one naming nothing) every packaged choice is still judged.
        for none in (None, {}, {'profiles': {}}):
            self.assertIs(c.approved_view(profile, none), profile)

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
