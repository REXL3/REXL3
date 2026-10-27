"""governance-audit: read-only failure conditions (contract section 5). All API inputs are injected."""
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from tools import governance_audit as g
from tools.test_release_evidence import write

REPO = 'REXL3/REXL3'
OWNERS = {'am423', 'vcruz305'}
NOW = datetime.now(timezone.utc)


def merged_pr(number, author='vcruz305', days=1):
    return {'number': number, 'merged_at': (NOW - timedelta(days=days)).isoformat(),
            'user': {'login': author}}


def review(login, state='APPROVED'):
    return {'state': state, 'user': {'login': login}}


class ReleaseCheckTest(unittest.TestCase):
    def test_non_product_tag_fails(self):
        problems = g.check_releases(Path('.'), REPO, 'tok', releases=[{'tag_name': 'prod/x/v1', 'draft': False}])
        self.assertTrue(any('not a v<X.Y.Z>' in p for p in problems))

    def test_missing_manifest_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            problems = g.check_releases(Path(tmp), REPO, 'tok', releases=[{'tag_name': 'v0.1.0', 'draft': False}])
            self.assertTrue(any('manifest.json' in p for p in problems))

    def test_manifest_present_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(tmp, 'releases/v0.1.0/manifest.json', b'{}')
            problems = g.check_releases(Path(tmp), REPO, 'tok', releases=[{'tag_name': 'v0.1.0', 'draft': False}])
            self.assertEqual(problems, [])


class MergedPrCheckTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_no_approving_non_author_review_fails(self):
        problems = g.check_merged_prs(self.root, REPO, 'tok', OWNERS, pulls=[merged_pr(1)],
                                      reviews_for=lambda n: [review('am423', 'CHANGES_REQUESTED')])
        self.assertTrue(any('PR #1' in p for p in problems))

    def test_maintainer_review_passes(self):
        problems = g.check_merged_prs(self.root, REPO, 'tok', OWNERS, pulls=[merged_pr(1)],
                                      reviews_for=lambda n: [review('am423')])
        self.assertEqual(problems, [])

    def test_author_review_does_not_qualify(self):
        problems = g.check_merged_prs(self.root, REPO, 'tok', OWNERS, pulls=[merged_pr(1)],
                                      reviews_for=lambda n: [review('vcruz305')])
        self.assertTrue(any('PR #1' in p for p in problems))

    def test_incident_record_exempts(self):
        write(self.tmp.name, 'docs/incidents/2026-10-26-unreviewed-release.md', b'PRs #12-#28 merged unreviewed.\n')
        problems = g.check_merged_prs(self.root, REPO, 'tok', OWNERS, pulls=[merged_pr(20)],
                                      reviews_for=lambda n: [])
        self.assertEqual(problems, [])

    def test_incident_record_through_range_exempts(self):
        write(self.tmp.name, 'docs/incidents/2026-10-26-unreviewed-release.md', b'PRs #1 through #28 merged unreviewed.\n')
        problems = g.check_merged_prs(self.root, REPO, 'tok', OWNERS, pulls=[merged_pr(15)],
                                      reviews_for=lambda n: [])
        self.assertEqual(problems, [])

    def test_incident_record_single_number_exempts(self):
        write(self.tmp.name, 'docs/incidents/2026-10-26-unreviewed-release.md', b'PR #33 merged unreviewed.\n')
        problems = g.check_merged_prs(self.root, REPO, 'tok', OWNERS, pulls=[merged_pr(33)],
                                      reviews_for=lambda n: [])
        self.assertEqual(problems, [])

    def test_old_merge_is_ignored(self):
        problems = g.check_merged_prs(self.root, REPO, 'tok', OWNERS, pulls=[merged_pr(1, days=30)],
                                      reviews_for=lambda n: [])
        self.assertEqual(problems, [])


class RulesetCheckTest(unittest.TestCase):
    def test_live_matches_committed(self):
        with tempfile.TemporaryDirectory() as tmp:
            committed = {'name': 'main-guard', 'target': 'branch', 'enforcement': 'active',
                         'conditions': {'ref_name': {'include': ['refs/heads/main'], 'exclude': []}},
                         'bypass_actors': [], 'rules': [{'type': 'deletion'}]}
            write(tmp, '.github/rulesets/main-guard.json', json.dumps(committed).encode())
            live = [dict(committed)]
            self.assertEqual(g.check_rulesets(Path(tmp), REPO, 'tok', live=live), [])

    def test_integration_id_is_not_a_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            committed = {'name': 'n', 'rules': [{'type': 'required_status_checks', 'parameters': {
                'required_status_checks': [{'context': 'verify'}]}}]}
            write(tmp, '.github/rulesets/n.json', json.dumps(committed).encode())
            live = [{'name': 'n', 'rules': [{'type': 'required_status_checks', 'parameters': {
                'required_status_checks': [{'context': 'verify', 'integration_id': 15368}]}}]}]
            self.assertEqual(g.check_rulesets(Path(tmp), REPO, 'tok', live=live), [])

    def test_incomplete_live_ruleset_view_is_skipped(self):
        # The workflow token sees ruleset details without conditions/rules; that must be skipped
        # with an INFO line, not reported as drift (and not flagged as "not live").
        with tempfile.TemporaryDirectory() as tmp:
            write(tmp, '.github/rulesets/n.json', b'{"name": "n", "target": "branch", "enforcement": "active", "rules": []}')
            live = [{'name': 'n', 'target': 'branch', 'enforcement': 'active'}]
            problems = g.check_rulesets(Path(tmp), REPO, 'tok', live=live)
            self.assertEqual(problems, [])

    def test_unreadable_live_ruleset_detail_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(tmp, '.github/rulesets/n.json', b'{"name": "n", "rules": []}')
            listing = json.dumps([{'id': 1, 'name': 'n'}]).encode()
            def fake_api(path, token, **kwargs):
                if path == '/repos/REXL3/REXL3/rulesets':
                    return listing
                raise ValueError('GitHub API 403 for ' + path)
            with patch('tools.governance_audit.api', side_effect=fake_api):
                problems = g.check_rulesets(Path(tmp), REPO, 'tok')
            self.assertEqual(problems, [])

    def test_null_echoed_lists_are_not_differences(self):
        # The API may echo absent lists as null under tokens with limited view; those must compare
        # equal to the committed empty lists.
        with tempfile.TemporaryDirectory() as tmp:
            committed = {'name': 'n', 'target': 'branch', 'enforcement': 'active',
                         'conditions': {'ref_name': {'include': ['refs/heads/model/*'], 'exclude': []}},
                         'bypass_actors': [],
                         'rules': [{'type': 'pull_request', 'parameters': {
                             'allowed_merge_methods': ['squash', 'rebase']}}]}
            write(tmp, '.github/rulesets/n.json', json.dumps(committed).encode())
            live = [{'name': 'n', 'target': 'branch', 'enforcement': 'active',
                     'conditions': {'ref_name': {'include': ['refs/heads/model/*'], 'exclude': None}},
                     'bypass_actors': None,
                     'rules': [{'type': 'pull_request', 'parameters': {
                         'allowed_merge_methods': ['squash', 'rebase'], 'required_reviewers': None}}]}]
            self.assertEqual(g.check_rulesets(Path(tmp), REPO, 'tok', live=live), [])

    def test_live_difference_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(tmp, '.github/rulesets/n.json', b'{"name": "n", "rules": [{"type": "deletion"}]}')
            live = [{'name': 'n', 'rules': []}]
            problems = g.check_rulesets(Path(tmp), REPO, 'tok', live=live)
            self.assertTrue(any('differs' in p for p in problems))

    def test_unreadable_live_rulesets_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            problems = g.check_rulesets(Path(tmp), REPO, 'tok')
            self.assertTrue(any('not readable' in p for p in problems))

    def test_committed_without_live_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(tmp, '.github/rulesets/n.json', b'{"name": "n", "rules": []}')
            problems = g.check_rulesets(Path(tmp), REPO, 'tok', live=[])
            self.assertTrue(any('not live' in p for p in problems))

    def test_volatile_metadata_and_api_defaults_are_not_differences(self):
        with tempfile.TemporaryDirectory() as tmp:
            committed = {'name': 'main-guard', 'target': 'branch', 'enforcement': 'active',
                         'conditions': {'ref_name': {'include': ['refs/heads/main'], 'exclude': []}},
                         'bypass_actors': [],
                         'rules': [{'type': 'deletion'},
                                   {'type': 'pull_request', 'parameters': {
                                       'required_approving_review_count': 1,
                                       'require_code_owner_review': True,
                                       'require_last_push_approval': True,
                                       'dismiss_stale_reviews_on_push': True,
                                       'required_review_thread_resolution': True,
                                       'allowed_merge_methods': ['squash', 'rebase']}}]}
            write(tmp, '.github/rulesets/main-guard.json', json.dumps(committed).encode())
            live = [{'id': 25786347, 'name': 'main-guard', 'target': 'branch',
                     'source_type': 'Repository', 'source': 'REXL3/REXL3', 'enforcement': 'active',
                     'node_id': 'N', 'created_at': '2026-10-24T00:00:00Z',
                     'updated_at': '2026-10-26T00:00:00Z', 'current_user_can_bypass': 'never',
                     '_links': {}, 'bypass_actors': [],
                     'conditions': {'ref_name': {'include': ['refs/heads/main'], 'exclude': []}},
                     'rules': [{'type': 'deletion'},
                               {'type': 'pull_request', 'parameters': {
                                   'required_approving_review_count': 1,
                                   'require_code_owner_review': True,
                                   'require_last_push_approval': True,
                                   'dismiss_stale_reviews_on_push': True,
                                   'required_review_thread_resolution': True,
                                   'allowed_merge_methods': ['squash', 'rebase'],
                                   'required_reviewers': [],
                                   'dismissal_restriction': {'enabled': False, 'allowed_actors': []},
                                   'require_extra_approval_for_unattributed_changes': True}}]}]
            self.assertEqual(g.check_rulesets(Path(tmp), REPO, 'tok', live=live), [])

    def test_committed_policy_param_that_live_weakens_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            committed = {'name': 'main-guard', 'target': 'branch', 'enforcement': 'active',
                         'conditions': {'ref_name': {'include': ['refs/heads/main'], 'exclude': []}},
                         'bypass_actors': [],
                         'rules': [{'type': 'pull_request', 'parameters': {
                             'require_code_owner_review': True}}]}
            write(tmp, '.github/rulesets/main-guard.json', json.dumps(committed).encode())
            live = [{'name': 'main-guard', 'target': 'branch', 'enforcement': 'active',
                     'conditions': {'ref_name': {'include': ['refs/heads/main'], 'exclude': []}},
                     'bypass_actors': [],
                     'rules': [{'type': 'pull_request', 'parameters': {
                         'require_code_owner_review': False}}]}]
            problems = g.check_rulesets(Path(tmp), REPO, 'tok', live=live)
            self.assertTrue(any('differs' in p for p in problems))

    def test_live_list_endpoint_details_are_fetched_per_id(self):
        # The /rulesets list endpoint omits conditions/rules; the check must fetch each detail.
        listing = json.dumps([{'id': 25935530, 'name': 'model-branches'}]).encode()
        detail = json.dumps({'id': 25935530, 'name': 'model-branches', 'target': 'branch',
                             'enforcement': 'active',
                             'conditions': {'ref_name': {'include': ['refs/heads/model/*'], 'exclude': []}},
                             'bypass_actors': [],
                             'rules': [{'type': 'deletion'}, {'type': 'non_fast_forward'}]}).encode()
        calls = []
        def fake_api(path, token, **kwargs):
            calls.append(path)
            if path == '/repos/REXL3/REXL3/rulesets':
                return listing
            if path == '/repos/REXL3/REXL3/rulesets/25935530':
                return detail
            raise AssertionError('unexpected API call: ' + path)
        with tempfile.TemporaryDirectory() as tmp:
            write(tmp, '.github/rulesets/model-branches.json', detail)
            with patch('tools.governance_audit.api', side_effect=fake_api):
                problems = g.check_rulesets(Path(tmp), REPO, 'tok')
            self.assertEqual(problems, [])
            self.assertEqual(calls, ['/repos/REXL3/REXL3/rulesets',
                                     '/repos/REXL3/REXL3/rulesets/25935530'])


class ImmutableCheckTest(unittest.TestCase):
    def test_enabled_passes(self):
        with patch('tools.governance_audit.api', return_value=b'{"enabled": true}'):
            self.assertEqual(g.check_immutable_releases(REPO, 'tok')[0], [])

    def test_explicitly_disabled_fails(self):
        with patch('tools.governance_audit.api', return_value=b'{"enabled": false}'):
            problems = g.check_immutable_releases(REPO, 'tok')[0]
            self.assertTrue(any('disabled' in p for p in problems))

    def test_unreadable_fails_closed(self):
        with patch('tools.governance_audit.api', side_effect=ValueError('HTTP 500')):
            problems = g.check_immutable_releases(REPO, 'tok')[0]
            self.assertTrue(any('not readable' in p for p in problems))

    def test_http403_is_skipped(self):
        # The workflow token cannot read the repo-settings endpoint (403); the audit skips it with
        # an informational line and does not fail.
        with patch('tools.governance_audit.api', side_effect=ValueError('GitHub API 403 for /repos/REXL3/REXL3/immutable-releases')):
            problems, note = g.check_immutable_releases(REPO, 'tok')
            self.assertEqual(problems, [])
            self.assertIn('INFO', note)


if __name__ == '__main__':
    unittest.main()
