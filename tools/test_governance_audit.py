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


class ImmutableCheckTest(unittest.TestCase):
    def test_enabled_passes(self):
        with patch('tools.governance_audit.api', return_value=b'{"enabled": true}'):
            self.assertEqual(g.check_immutable_releases(REPO, 'tok')[0], [])

    def test_explicitly_disabled_fails(self):
        with patch('tools.governance_audit.api', return_value=b'{"enabled": false}'):
            problems = g.check_immutable_releases(REPO, 'tok')[0]
            self.assertTrue(any('disabled' in p for p in problems))

    def test_unreadable_fails_closed(self):
        with patch('tools.governance_audit.api', side_effect=ValueError('HTTP 403')):
            problems = g.check_immutable_releases(REPO, 'tok')[0]
            self.assertTrue(any('not readable' in p for p in problems))


if __name__ == '__main__':
    unittest.main()
