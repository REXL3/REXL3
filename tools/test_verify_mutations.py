"""Mutation tests for the production gate.

Every test below breaks exactly one property of an otherwise valid input and asserts that the
verifier rejects it for that property's reason (the named problem text), not for an unrelated hash
drift. v2 fixtures are synthetic (disposable Git objects, never qualification evidence); the legacy
tests run against the real committed v1 tree, its real registry and its real pinned hash.

This file is production-only. It has no counterpart in the dev repository.
"""
import contextlib
import copy
import hashlib
import io
import os
from pathlib import Path
import re
import shutil
import stat
import tarfile
import tempfile
import unittest
from unittest import mock

from tools import promote_runtime as p
from tools import release_evidence as e
from tools import verify_runtime_tree as v
from tools.test_approve_profile import SyntheticSeal
from tools.test_release_evidence import archive, encoded, sha, write

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ID = 'dsv41-flash-1017-sage-1.62'
REGISTRY = ROOT / 'runtimes/templates/APPROVALS.json'
ALLOW_LIST = ROOT / 'runtimes/templates/APPROVED.json'
WORKFLOW = ROOT / '.github/workflows/verify.yml'


def run_cli(*argv):
    """Run the real CLI entry point; return (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = v.main([str(a) for a in argv])
    return code, out.getvalue(), err.getvalue()


def normalized_copy(source, dest):
    """Copy a tree and give every file the mode Git records (644, or 755 when executable)."""
    shutil.copytree(source, dest, symlinks=True)
    for path in Path(dest).rglob('*'):
        if path.is_file() and not path.is_symlink():
            executable = path.stat().st_mode & stat.S_IXUSR
            path.chmod(0o755 if executable else 0o644)


class RealLegacyTree(unittest.TestCase):
    """The committed v1 tree, verified exactly as the workflow does (CLI, committed registry, pinned hash)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / 'candidate'
        self.tree = self.root / 'runtimes' / RUNTIME_ID / 'v1'
        normalized_copy(ROOT / 'runtimes' / RUNTIME_ID / 'v1', self.tree)
        self.assets = self.base / 'dev-assets'
        self.assets.mkdir()
        self.pin = sha(REGISTRY.read_bytes())

    def cli(self, *extra, registry=None, pin=None, legacy=True):
        args = ['--root', self.root, '--approved', ALLOW_LIST, '--trusted-approvals', registry or REGISTRY,
                '--trusted-approvals-sha256', pin or self.pin, '--assets-dir', self.assets]
        return run_cli(*args, *(['--allow-legacy'] if legacy else []), *extra)

    def test_real_v1_tree_passes_in_legacy_mode(self):
        code, out, err = self.cli()
        self.assertEqual((code, err), (0, ''), out + err)
        self.assertIn('1 tree(s), 0 problem(s); legacy allowed=True', out)

    def test_real_v1_tree_is_rejected_without_allow_legacy(self):
        code, out, err = self.cli(legacy=False)
        self.assertEqual(code, 1, out + err)
        self.assertIn('LEGACY history requires explicit --allow-legacy', out)

    def test_runtime_file_edited_together_with_its_manifest_entry(self):
        launch = self.tree / 'launch.sh'
        launch.write_bytes(launch.read_bytes() + b'\necho tampered\n')
        manifest_path = self.tree / 'source.manifest.json'
        manifest = e.loads(manifest_path.read_bytes())
        entry = manifest['files']['launch.sh']
        entry['sha256'], entry['bytes'] = sha(launch.read_bytes()), launch.stat().st_size
        manifest_path.write_bytes(encoded(manifest))
        # The tree is now self-consistent: the old unanchored check would have passed it.
        code, out, _ = self.cli()
        self.assertEqual(code, 1, out)
        self.assertIn('LEGACY tree hash/bytes/mode/inventory drift', out)

    def test_self_edited_approval_is_not_self_authorizing(self):
        approval = self.tree / 'approval.json'
        document = e.loads(approval.read_bytes())
        document['approved_by'] = 'someone-else'
        approval.write_bytes(encoded(document))
        code, out, _ = self.cli()
        self.assertEqual(code, 1, out)
        self.assertIn('LEGACY approval hash drift', out)

    def test_attacker_registry_that_blesses_the_edit_fails_the_pinned_hash(self):
        approval = self.tree / 'approval.json'
        document = e.loads(approval.read_bytes())
        document['approved_by'] = 'someone-else'
        approval.write_bytes(encoded(document))
        forged = e.loads(REGISTRY.read_bytes())
        entry = forged['legacy'][RUNTIME_ID + '/v1']
        entry['approval_sha256'] = sha(approval.read_bytes())
        entry['files']['approval.json'] = e.record(approval)
        forged_path = self.base / 'forged-registry.json'
        forged_path.write_bytes(encoded(forged))
        # Same forged registry with the hash computed from it would verify, which is why the pin
        # has to come from protected policy: with the real pin the forged registry is refused.
        code, out, err = self.cli(registry=forged_path)
        self.assertEqual(code, 2, out + err)
        self.assertIn('trusted registry hash mismatch', err)
        with self.assertRaisesRegex(ValueError, 'trusted registry hash mismatch'):
            e.load_registry(forged_path, self.pin)

    def test_extra_missing_mode_and_symlink_files_are_rejected(self):
        cases = {
            'extra file': lambda t: write(t, 'extra.txt', b'x'),
            'missing file': lambda t: (t / 'KNOWN_ISSUES.md').unlink(),
            'mode change': lambda t: (t / 'launch.sh').chmod(0o644),
            'symlink': lambda t: ((t / 'README.md').unlink(), (t / 'README.md').symlink_to('LICENSE')),
        }
        for name, mutate in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / 'candidate'
                tree = root / 'runtimes' / RUNTIME_ID / 'v1'
                normalized_copy(self.tree, tree)
                mutate(tree)
                code, out, _ = run_cli('--root', root, '--approved', ALLOW_LIST, '--trusted-approvals', REGISTRY,
                                       '--trusted-approvals-sha256', self.pin, '--assets-dir', self.assets, '--allow-legacy')
                self.assertEqual(code, 1, out)

    def test_unknown_version_tree_has_no_legacy_anchor(self):
        normalized_copy(self.tree, self.tree.with_name('v2'))
        code, out, _ = self.cli()
        self.assertEqual(code, 1, out)
        self.assertIn('LEGACY history has no trusted anchor', out)

    def test_empty_runtimes_directory_is_refused(self):
        shutil.rmtree(self.tree)
        code, out, err = self.cli()
        self.assertEqual(code, 2, out + err)
        self.assertIn('no runtime trees found', err)


class V2Mutations(unittest.TestCase):
    """A sealed synthetic v2 tree, then one rebound mutation at a time."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f = SyntheticSeal(self.tmp.name)
        self.approval, self.doc, self.registry = self.f.published_fixture()
        self.tree = Path(self.tmp.name) / 'prod/runtimes/synthetic-demo/v1'
        for name, data in self.f.source.items():
            write(self.tree, name, data).chmod(self.doc['source_asset']['members'][name]['mode'])
        write(self.tree, 'approval.json', self.approval.read_bytes())

    def verify(self, **kwargs):
        kwargs.setdefault('assets_dir', self.f.assets)
        return v.verify_tree(self.tree, kwargs.pop('approved', self.f.approved),
                             trusted_approvals=kwargs.pop('registry', self.registry), **kwargs)

    def assertRejected(self, pattern, **kwargs):
        problems = self.verify(**kwargs)
        self.assertEqual(len(problems), 1, problems)
        self.assertRegex(problems[0], pattern)

    def rebind(self, mutate):
        """Mutate the receipt, then rebind every downstream hash so only the mutation is invalid."""
        receipt = e.load_json(self.f.receipt_path)
        mutate(receipt)
        self.f.receipt_path.write_bytes(encoded(receipt))
        self.doc['profiles']['single']['receipt']['sha256'] = sha(self.f.receipt_path.read_bytes())
        self.doc['evidence_asset'] = e.evidence_asset(self.doc, self.f.assets)
        (self.f.assets / 'evidence.tar.gz').unlink()
        e.write_evidence_archive(self.f.assets, self.doc['evidence_asset']['members'], self.f.assets / 'evidence.tar.gz')
        self.approval.write_bytes(encoded(self.doc))
        self.registry['approvals']['synthetic-demo/v1']['approval_sha256'] = sha(self.approval.read_bytes())
        write(self.tree, 'approval.json', self.approval.read_bytes())

    def test_control_unmutated_tree_passes(self):
        self.assertEqual(self.verify(), [])

    def test_fail_gate(self):
        def mutate(receipt):
            raw_path = self.f.assets / 'doctor.json'
            raw = e.load_json(raw_path)
            raw['cases'][0]['status'] = 'FAIL'
            raw_path.write_bytes(encoded(raw))
            receipt['gates']['doctor'] = dict(status='FAIL', passed=0, total=1, failed=1, skipped=0,
                                              evidence=[{'path': 'doctor.json', 'sha256': sha(raw_path.read_bytes())}])
        self.rebind(mutate)
        self.assertRejected('gate not PASS: doctor')

    def test_skip_gate(self):
        def mutate(receipt):
            receipt['gates']['exactness'] = dict(status='NOT_RUN', passed=0, total=1, failed=0, skipped=1, evidence=[])
        self.rebind(mutate)
        self.assertRejected('gate not PASS: exactness')

    def test_pass_label_over_a_skipped_case_is_a_count_mismatch(self):
        def mutate(receipt):
            gate = receipt['gates']['quality']
            gate.update(total=2, skipped=1)
        self.rebind(mutate)
        self.assertRejected('gate status/count mismatch: quality')

    def test_zero_tests_gate(self):
        def mutate(receipt):
            receipt['gates']['lifecycle'] = dict(status='NOT_RUN', passed=0, total=0, failed=0, skipped=0, evidence=[])
        self.rebind(mutate)
        self.assertRejected('gate not PASS: lifecycle')

    def test_pass_label_over_zero_tests_is_a_count_mismatch(self):
        def mutate(receipt):
            receipt['gates']['performance'] = dict(status='PASS', passed=0, total=0, failed=0, skipped=0, evidence=[])
        self.rebind(mutate)
        self.assertRejected('gate status/count mismatch: performance')

    def test_missing_gate(self):
        self.rebind(lambda receipt: receipt['gates'].pop('vision'))
        self.assertRejected('missing required gates')

    def test_wrong_template_digest(self):
        self.assertRejected('template not approved', approved={'v1': 'sha256:' + '0' * 64})
        self.assertRejected('template not approved', approved={})

    def test_runtime_file_edited_together_with_its_manifest_entry(self):
        write(self.tree, 'launch.sh', b'forged').chmod(0o755)
        manifest = e.load_json(self.tree / 'source.manifest.json')
        manifest['files']['launch.sh']['sha256'] = sha(b'forged')
        manifest['files']['launch.sh']['bytes'] = len(b'forged')
        write(self.tree, 'source.manifest.json', encoded(manifest))
        self.assertRejected(r'launch\.sh: hash/bytes/mode != approved')

    def test_self_edited_approval_in_tree_and_assets(self):
        forged = dict(self.doc, approved_by='forged')
        for target in (self.f.assets / 'approval.json', self.tree / 'approval.json'):
            target.write_bytes(encoded(forged))
        self.assertRejected('untrusted approval bytes')

    def test_self_edited_approval_in_tree_only(self):
        write(self.tree, 'approval.json', encoded(dict(self.doc, approved_by='forged')))
        self.assertRejected('tree/asset approval inequality')

    def test_attacker_registry_fails_the_pinned_hash(self):
        forged = copy.deepcopy(self.registry)
        forged['approvals']['synthetic-demo/v1']['approval_sha256'] = '0' * 64
        path = write(self.tmp.name, 'forged-registry.json', encoded(forged))
        with self.assertRaisesRegex(ValueError, 'trusted registry hash mismatch'):
            e.load_registry(path, sha(encoded(self.registry)))

    def test_approval_absent_from_the_registry(self):
        self.assertRejected('approval not in independently trusted registry',
                            registry={'schema': 'rexl3.approval-registry.v1', 'approvals': {}})

    def test_no_registry_and_no_assets_fail_closed(self):
        self.assertRejected('independently trusted approval registry required', registry=None)
        self.assertRejected('authenticated dev assets directory required', assets_dir=None)

    def test_v1_schema_without_allow_legacy(self):
        legacy = dict(self.doc, schema='rexl3.profile-approval.v1')
        write(self.tree, 'approval.json', encoded(legacy))
        self.assertRejected('LEGACY history requires explicit --allow-legacy')
        self.assertRejected('LEGACY history has no trusted anchor', allow_legacy=True)

    def test_traversal_and_link_tars_are_refused_as_assets(self):
        for member, kind in [('../escape', tarfile.REGTYPE), ('/absolute', tarfile.REGTYPE),
                             ('a/../b', tarfile.REGTYPE), ('link', tarfile.SYMTYPE), ('link', tarfile.LNKTYPE)]:
            for asset in (self.f.archive, self.f.source_asset):
                with self.subTest(member=member, kind=kind, asset=asset.name):
                    original = asset.read_bytes()
                    self.addCleanup(asset.write_bytes, original)
                    archive(asset, [('profile.json', b'{}', tarfile.REGTYPE), (member, b'x', kind)])
                    problems = self.verify()
                    self.assertEqual(len(problems), 1, problems)
                    self.assertRegex(problems[0], r'unsafe (relative path|archive member)')
                    asset.write_bytes(original)

    def test_traversal_and_link_tars_write_nothing_when_extracted(self):
        for member, kind in [('../escape', tarfile.REGTYPE), ('link', tarfile.SYMTYPE)]:
            with self.subTest(member=member, kind=kind):
                bad = Path(self.tmp.name) / 'bad.tgz'
                archive(bad, [('ok', b'ok', tarfile.REGTYPE), (member, b'x', kind)])
                dest = Path(self.tmp.name) / 'extracted'
                with self.assertRaises(ValueError):
                    p.extract_tree(bad, dest, self.doc, approval_path=self.approval)
                self.assertFalse(dest.exists())
                self.assertFalse((Path(self.tmp.name) / 'escape').exists())

    def test_cli_fails_closed_without_authenticated_assets_and_passes_with_them(self):
        # CI has no read access to the private dev release, so it passes an empty assets directory.
        root = Path(self.tmp.name) / 'prod'
        registry = write(self.tmp.name, 'registry.json', encoded(self.registry))
        allow = write(self.tmp.name, 'approved.json', encoded({'templates': self.f.approved}))
        args = ['--root', root, '--approved', allow, '--trusted-approvals', registry,
                '--trusted-approvals-sha256', sha(registry.read_bytes())]
        empty = Path(self.tmp.name) / 'empty-assets'
        empty.mkdir()
        code, out, _ = run_cli(*args, '--assets-dir', empty, '--allow-legacy')
        self.assertEqual(code, 1, out)
        self.assertIn('missing/nonregular/hardlinked file: approval.json', out)
        staged = Path(self.tmp.name) / 'staged-assets'
        shutil.copytree(self.f.assets, staged / 'synthetic-demo/v1')
        code, out, err = run_cli(*args, '--assets-dir', staged)
        self.assertEqual((code, err), (0, ''), out)
        self.assertIn('1 tree(s), 0 problem(s)', out)


class WorkflowPolicy(unittest.TestCase):
    """The workflow is the other half of the gate: pin it, because nothing else tests it."""

    @classmethod
    def setUpClass(cls):
        cls.text = WORKFLOW.read_text()

    def test_pinned_registry_hash_matches_the_registry(self):
        match = re.search(r'^\s*TRUSTED_APPROVALS_SHA256:\s*"?([0-9a-f]{64})"?\s*$', self.text, re.M)
        self.assertIsNotNone(match, 'workflow must pin the registry hash')
        self.assertEqual(match[1], hashlib.sha256(REGISTRY.read_bytes()).hexdigest())

    def test_no_merge_and_no_write_permission_anywhere(self):
        for forbidden in ('gh pr merge', '--auto', 'auto-merge', 'merge:', 'contents: write', 'pull-requests: write',
                          'workflow_dispatch', 'github.head_ref'):
            self.assertNotIn(forbidden, self.text)
        self.assertFalse((ROOT / '.github/workflows/promote.yml').exists())
        for path in (ROOT / '.github/workflows').glob('*.y*ml'):
            self.assertNotIn('gh pr merge', path.read_text(), path.name)

    def test_the_only_secret_is_the_dev_read_token_on_the_download_step(self):
        self.assertEqual(re.findall(r'secrets\.[A-Za-z0-9_]+', self.text), ['secrets.DEV_ASSETS_TOKEN'])
        steps = re.split(r'(?m)^      - name: ', self.text)[1:]
        holders = [step.splitlines()[0] for step in steps if 'secrets.' in step]
        self.assertEqual(holders, ['Download the authenticated dev release assets for v2 trees'])
        download = next(step for step in steps if step.startswith('Download the authenticated'))
        self.assertIn('base/tools/fetch_dev_assets.py', download)
        self.assertNotIn('candidate/', download.replace('--root candidate', ''))

    def test_runs_in_the_base_context_and_never_executes_candidate_code(self):
        self.assertRegex(self.text, r'(?m)^\s+pull_request_target:')
        self.assertNotRegex(self.text, r'(?m)^\s+pull_request:')
        self.assertRegex(self.text, r'(?m)^permissions:\s*\n\s+contents: read\s*$')
        for line in self.text.splitlines():
            stripped = line.strip()
            if re.match(r'(python3?|bash|sh|\./|source|\.\s)', stripped) or ' python' in stripped:
                self.assertNotRegex(stripped, r'candidate/(tools|\.github|scripts)', stripped)
        self.assertNotRegex(self.text, r'(?m)^\s*(cd|working-directory:)\s+\.?/?candidate')
        self.assertEqual(self.text.count('persist-credentials: false'), self.text.count('actions/checkout@'))

    def test_every_action_is_pinned_by_full_commit_sha(self):
        uses = re.findall(r'(?m)^\s*-?\s*uses:\s*(\S+)', self.text)
        self.assertTrue(uses)
        for ref in uses:
            self.assertRegex(ref, r'^[\w.-]+/[\w.-]+@[0-9a-f]{40}$')

    def test_codeowners_covers_workflow_tools_and_policy(self):
        owners = (ROOT / '.github/CODEOWNERS').read_text().splitlines()
        rules = [line.split() for line in owners if line.strip() and not line.startswith('#')]
        self.assertIn(['*', '@am423', '@vcruz305'], rules)
        # Later rules override earlier ones; a narrower rule must not drop an owner.
        for rule in rules:
            self.assertGreaterEqual(set(rule[1:]), {'@am423', '@vcruz305'})


if __name__ == '__main__':
    unittest.main()
