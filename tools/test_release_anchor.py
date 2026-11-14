"""New approval anchors arrive only in a release PR of this repository whose manifest lists them.

The head repository and head ref come from the pull-request event and are attacker-controlled, so the
rule is tested as plain Python here: unit tests of the comparison, then the two tools that enforce it
(fetch_dev_assets before any download, verify_runtime_tree before any tree is judged), then the
workflow wiring. Fixtures are synthetic; there is no network access.
"""
import contextlib
import copy
import io
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import fetch_dev_assets as f
from tools import release_anchor as ra
from tools import test_verify_runtime_tree as tv
from tools import verify_runtime_tree as v
from tools.test_release_evidence import encoded, write

REPO = 'REXL3/REXL3'
FAMILY, MODEL = tv.FAMILY, tv.MODEL
KEY = f'{FAMILY}/{MODEL}/v1'
WORKFLOWS = Path(__file__).resolve().parents[1] / '.github/workflows'
RELEASE = dict(base_repo=REPO, head_repo=REPO, head_ref='release/v0.1.2')
BAD_REFS = (None, '', 'main', 'engine/verify-release-head', 'model/qwen3.8-27b', 'model/qwen3.8-27b-approval',
            'release/', 'release/v0.1', 'release/0.1.2', 'release/v0.1.2-rc1', 'release/v0.1.2/extra', 'release/v01.1.2',
            'release/v0.1.2\n', ' release/v0.1.2', 'refs/heads/release/v0.1.2', 'Release/v0.1.2', 'release/v0.1.2;id',
            'x/release/v0.1.2', '--full', 0, ['release/v0.1.2'])
BAD_HEAD_REPOS = (None, '', 'attacker/REXL3', 'REXL3/REXL3-dev', 'rexl3/rexl3', 'REXL3/REXL3 ', 'REXL3/REXL3\n', 'REXL3/REXL3/x')


def manifest(entry, release='v0.1.2', **changes):
    model = {'family': FAMILY, 'model': MODEL, 'version': 'v1', 'runtime_id': entry['runtime_id'],
             'approval_sha256': entry['approval_sha256']}
    model.update(changes)
    return {'schema': 'rexl3.release-manifest.v1', 'version': release, 'notes': 'synthetic',
            'engine': {'dev_commit': '4208d848' * 5, 'dev_tree': 'b' * 40, 'engine_release_tag': 'engine-v0.1.2-4208d848',
                       'artifacts': {'rexl3-serve': 'd' * 64}},
            'models': [model]}


class ReleaseHeadTest(unittest.TestCase):
    def test_a_release_branch_of_the_base_repository_gives_its_version(self):
        self.assertEqual(ra.release_head_version(REPO, REPO, 'release/v0.1.2'), 'v0.1.2')
        self.assertEqual(ra.release_head_version(REPO, REPO, 'release/v10.20.30'), 'v10.20.30')

    def test_any_other_head_ref_is_refused(self):
        for ref in BAD_REFS:
            with self.subTest(ref=ref), self.assertRaisesRegex(ValueError, 'only from a release/v<X.Y.Z> head branch'):
                ra.release_head_version(REPO, REPO, ref)

    def test_a_fork_or_any_other_head_repository_is_refused(self):
        for repo in BAD_HEAD_REPOS:
            with self.subTest(repo=repo), self.assertRaisesRegex(ValueError, 'not from a fork'):
                ra.release_head_version(REPO, repo, 'release/v0.1.2')

    def test_a_missing_or_malformed_base_repository_is_refused(self):
        # an empty head repository must not equal an empty base repository
        for base, head in ((None, None), ('', ''), ('REXL3', 'REXL3'), ('REXL3/REXL3\n', 'REXL3/REXL3\n'), ('a b/c', 'a b/c')):
            with self.subTest(base=base), self.assertRaisesRegex(ValueError, 'need the pull request context'):
                ra.release_head_version(base, head, 'release/v0.1.2')

    def test_refusals_never_echo_the_untrusted_strings(self):
        marker = '::error::injected'
        for args in ((REPO, marker, 'release/v0.1.2'), (REPO, REPO, marker), (marker, marker, marker)):
            with self.assertRaises(ValueError) as caught:
                ra.release_head_version(*args)
            self.assertNotIn('injected', str(caught.exception))


class RequireReleasePrTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'candidate'
        self.base = Path(self.tmp.name) / 'base'
        self.root.mkdir()
        self.base.mkdir()
        self.entry = {'runtime_id': 'synthetic-demo', 'approval_sha256': 'a' * 64, 'source_commit': 'b' * 40, 'source_tree': 'c' * 40}
        self.registry = {'schema': 'rexl3.approval-registry.v1', 'approvals': {KEY: self.entry}}
        self.put(manifest(self.entry))

    def put(self, doc, version='v0.1.2', root=None):
        return write(root or self.root, f'releases/{version}/manifest.json', encoded(doc))

    def check(self, keys=(KEY,), **changes):
        return ra.require_release_pr(set(keys), self.registry, root=self.root, base=self.base, **dict(RELEASE, **changes))

    def test_a_listed_anchor_in_a_release_pr_is_accepted(self):
        self.assertEqual(self.check(), 'v0.1.2')

    def test_without_new_anchors_nothing_is_required(self):
        # every engine/* and model/* pull request, every fork pull request and every push to main
        for context in ({}, dict(base_repo='', head_repo='', head_ref=''),
                        dict(base_repo=REPO, head_repo='attacker/REXL3', head_ref='engine/anything')):
            self.assertIsNone(ra.require_release_pr(set(), self.registry, root=self.root, base=self.base, **context))

    def test_new_anchors_without_the_context_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'need the pull request context'):
            ra.require_release_pr({KEY}, self.registry, root=self.root, base=self.base)

    def test_a_fork_and_a_non_release_branch_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'not from a fork'):
            self.check(head_repo='attacker/REXL3')
        for ref in BAD_REFS:
            with self.subTest(ref=ref), self.assertRaisesRegex(ValueError, 'only from a release/v<X.Y.Z> head branch'):
                self.check(head_ref=ref)

    def test_the_manifest_must_exist_be_valid_and_be_this_release(self):
        with self.assertRaisesRegex(ValueError, 'missing/nonregular/hardlinked file'):
            self.check(head_ref='release/v0.1.3')
        self.put(manifest(self.entry), version='v0.1.3')  # says v0.1.2 inside
        with self.assertRaisesRegex(ValueError, 'manifest version differs'):
            self.check(head_ref='release/v0.1.3')
        self.put({'schema': 'other'})
        with self.assertRaisesRegex(ValueError, 'release manifest schema mismatch'):
            self.check()

    def test_a_linked_manifest_is_refused(self):
        real = self.put(manifest(self.entry), root=Path(self.tmp.name) / 'elsewhere')
        shutil.rmtree(self.root / 'releases')
        (self.root / 'releases').mkdir()
        (self.root / 'releases/v0.1.2').symlink_to(real.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.check()

    def test_an_unlisted_anchor_is_refused(self):
        other = 'qwen3.8/qwen3.8-flash-next/v1'
        self.registry['approvals'][other] = dict(self.entry)
        with self.assertRaisesRegex(ValueError, 'not listed in the release manifest: ' + re.escape(other)):
            self.check(keys=(KEY, other))
        for change in ({'version': 'v2'}, {'model': 'qwen3.8-flash-next'}, {'family': 'glm-5.3'}):
            self.put(manifest(self.entry, **change))
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'not listed in the release manifest'):
                self.check()

    def test_a_listing_with_another_approval_hash_or_runtime_id_is_refused(self):
        for change in ({'approval_sha256': 'f' * 64}, {'runtime_id': 'other-runtime'}):
            self.put(manifest(self.entry, **change))
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'does not match the new approval anchor'):
                self.check()

    def test_a_release_that_is_already_on_the_base_is_refused(self):
        self.put(manifest(self.entry), root=self.base)
        with self.assertRaisesRegex(ValueError, 'release already has files on the protected base'):
            self.check()


class VerifyNewAnchorTest(unittest.TestCase):
    """verify_runtime_tree.main with a real synthetic seal: the candidate adds the anchor of its tree."""

    setUp = tv.VerifyTest.setUp
    diff_fixture = tv.VerifyTest.diff_fixture

    def fixture(self, name, listed=True):
        candidate, base = self.diff_fixture(name)
        write(base, 'runtimes/templates/APPROVALS.json',
              encoded({'schema': 'rexl3.approval-registry.v1', 'approvals': {}, 'legacy': {}}))
        if listed:
            write(candidate, 'releases/v0.1.2/manifest.json', encoded(manifest(self.registry['approvals'][KEY])))
        return candidate, base

    def run_verify(self, candidate, base, *context):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = v.main(['--root', str(candidate), '--base', str(base),
                           '--approved', str(base / 'runtimes/templates/APPROVED.json'),
                           '--trusted-approvals', str(base / 'runtimes/templates/APPROVALS.json'),
                           '--assets-dir', str(self.assets), *context])
        return code, out.getvalue() + err.getvalue()

    def test_a_release_pr_that_lists_the_anchor_verifies(self):
        candidate, base = self.fixture('release')
        code, text = self.run_verify(candidate, base, '--base-repo=' + REPO, '--head-repo=' + REPO, '--head-ref=release/v0.1.2')
        self.assertEqual(code, 0, text)
        self.assertIn('1 tree(s), 0 problem(s); 0 unchanged/skipped', text)

    def test_a_real_seal_from_a_fork_or_another_branch_is_refused(self):
        candidate, base = self.fixture('refused')
        cases = (
            ((), 'need the pull request context'),
            (('--base-repo=', '--head-repo=', '--head-ref='), 'need the pull request context'),
            (('--base-repo=' + REPO, '--head-repo=attacker/REXL3', '--head-ref=release/v0.1.2'), 'not from a fork'),
            (('--base-repo=' + REPO, '--head-repo=' + REPO, '--head-ref=engine/release-flow'), 'only from a release/v<X.Y.Z> head branch'),
            (('--base-repo=' + REPO, '--head-repo=' + REPO, '--head-ref=model/qwen3.8-27b-approval'), 'only from a release/v<X.Y.Z> head branch'),
            (('--base-repo=' + REPO, '--head-repo=' + REPO, '--head-ref=release/v0.1.3'), 'missing/nonregular/hardlinked file'),
            (('--base-repo=' + REPO, '--head-repo=' + REPO, '--head-ref=--full'), 'only from a release/v<X.Y.Z> head branch'))
        for context, reason in cases:
            with self.subTest(context=context):
                code, text = self.run_verify(candidate, base, *context)
                self.assertEqual(code, 2, text)
                self.assertIn('TREE_REFUSED: ', text)
                self.assertIn(reason, text)

    def test_a_release_pr_whose_manifest_omits_the_anchor_is_refused(self):
        candidate, base = self.fixture('unlisted', listed=False)
        doc = manifest(self.registry['approvals'][KEY], model='qwen3.8-flash-next')
        write(candidate, 'releases/v0.1.2/manifest.json', encoded(doc))
        code, text = self.run_verify(candidate, base, '--base-repo=' + REPO, '--head-repo=' + REPO, '--head-ref=release/v0.1.2')
        self.assertEqual(code, 2, text)
        self.assertIn('new approval anchor is not listed in the release manifest: ' + KEY, text)

    def test_an_unchanged_registry_needs_no_context(self):
        candidate, base = self.diff_fixture('unchanged')
        code, text = self.run_verify(candidate, base, '--base-repo=' + REPO, '--head-repo=attacker/REXL3', '--head-ref=anything')
        self.assertEqual(code, 0, text)
        self.assertIn('1 unchanged/skipped', text)


class FetchNewAnchorTest(unittest.TestCase):
    """fetch_dev_assets holds the dev-read secret: the rule must refuse before any download."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'candidate'
        self.base = Path(self.tmp.name) / 'base'
        for root in (self.root, self.base):
            tree = root / 'runtimes' / FAMILY / MODEL / 'v1'
            write(tree, 'approval.json', encoded({'schema': 'rexl3.profile-approval.v2', 'runtime_id': 'synthetic-demo'}))
        self.entry = {'approval_sha256': 'a' * 64, 'source_commit': 'b' * 40, 'source_tree': 'c' * 40, 'runtime_id': 'synthetic-demo'}
        self.empty = {'schema': 'rexl3.approval-registry.v1', 'approvals': {}, 'legacy': {}}
        self.candidate = {'schema': 'rexl3.approval-registry.v1', 'approvals': {KEY: self.entry}}
        write(self.root, 'releases/v0.1.2/manifest.json', encoded(manifest(self.entry)))

    def fetch(self, **context):
        with (patch.object(f.e, 'load_registry', side_effect=[copy.deepcopy(self.empty), copy.deepcopy(self.candidate)]),
              patch.object(f.e, 'trusted_entry', return_value={}),
              patch.object(f, 'local_metadata_problems', return_value=[]),
              patch.object(f.p, 'download_release') as download):
            try:
                return f.fetch(self.root, Path(self.tmp.name) / 'assets', self.empty, {}, base=self.base, **context), download
            except ValueError as error:
                return error, download

    def test_a_release_pr_downloads_the_new_anchor(self):
        result, download = self.fetch(**RELEASE)
        self.assertEqual(result, ['approve-synthetic-demo-1'])
        download.assert_called_once()

    def test_nothing_is_downloaded_for_a_fork_another_branch_or_an_unlisted_anchor(self):
        cases = ({}, dict(RELEASE, head_repo='attacker/REXL3'), dict(RELEASE, head_ref='engine/x'),
                 dict(RELEASE, head_ref='model/qwen3.8-27b-approval'), dict(RELEASE, head_ref='release/v0.1.3'))
        for context in cases:
            with self.subTest(context=context):
                result, download = self.fetch(**context)
                self.assertIsInstance(result, ValueError)
                download.assert_not_called()
        write(self.root, 'releases/v0.1.2/manifest.json', encoded(manifest(self.entry, approval_sha256='f' * 64)))
        result, download = self.fetch(**RELEASE)
        self.assertIn('does not match the new approval anchor', str(result))
        download.assert_not_called()

    def test_the_command_line_carries_the_context(self):
        argv = ['--root', str(self.root), '--base', str(self.base), '--approved', 'unused', '--trusted-approvals', 'unused',
                '--assets-dir', 'unused', '--base-repo=' + REPO, '--head-repo=' + REPO, '--head-ref=release/v0.1.2']
        with (patch.object(f.e, 'load_registry', return_value=self.empty), patch.object(f.e, 'load_json', return_value={'templates': {}}),
              patch.object(f, 'fetch', return_value=[]) as fetch, contextlib.redirect_stdout(io.StringIO())):
            self.assertEqual(f.main(argv), 0)
        self.assertEqual({k: fetch.call_args.kwargs[k] for k in RELEASE}, RELEASE)


class WorkflowWiringTest(unittest.TestCase):
    CONTEXT = '--base-repo="$REXL3_BASE_REPO" --head-repo="$REXL3_HEAD_REPO" --head-ref="$REXL3_HEAD_REF"'

    def test_every_diff_scoped_call_passes_the_pull_request_context(self):
        for name, calls in (('verify.yml', 2), ('model-completeness.yml', 1)):
            body = (WORKFLOWS / name).read_text()
            tools = re.findall(r'python3 -B -E -s base/tools/(?:fetch_dev_assets|verify_runtime_tree)\.py(?:[^\n]*\\\n)*[^\n]*\n', body)
            self.assertEqual(len(tools), calls, name)
            for call in tools:
                self.assertIn('--base base', call, name)
                self.assertIn(self.CONTEXT, call, name)

    def test_the_context_comes_from_env_in_the_option_value_form(self):
        for name, steps in (('verify.yml', 2), ('model-completeness.yml', 1)):
            body = (WORKFLOWS / name).read_text()
            self.assertEqual(body.count('REXL3_BASE_REPO: ${{ github.repository }}'), steps, name)
            self.assertEqual(body.count('REXL3_HEAD_REPO: ${{ github.event.pull_request.head.repo.full_name }}'), steps, name)
            self.assertGreaterEqual(body.count('REXL3_HEAD_REF: ${{ github.event.pull_request.head.ref }}'), steps, name)
            # `--head-ref "$VALUE"` would let a branch named like an option be read as one
            self.assertNotRegex(body, r'--(?:base-repo|head-repo|head-ref)[ "]')


if __name__ == '__main__':
    unittest.main()
