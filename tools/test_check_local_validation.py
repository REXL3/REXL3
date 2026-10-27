"""local-validation: LOCAL-VALIDATION v1 parsing, reviewer qualification, head/model binding, status attach.

No network: check_local_validation.api_json is patched; the status POST is captured via a fake opener.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import check_local_validation as l

REPO = 'REXL3/REXL3'
PR = 7
HEAD = 'a' * 40
OWNERS = {'am423', 'vcruz305'}


def review(login, state='APPROVED', body=None):
    return {'state': state, 'user': {'login': login}, 'body': body}


def block(head=HEAD, model='qwen3.8/qwen3.8-27b', pr=PR, **overrides):
    fields = {'pr': str(pr), 'head': head, 'model': model, 'host': 'gb10-1',
              'check_model_branch': 'PASS', 'assets': 'PASS', 'doctor': 'PASS', 'smoke': 'PASS',
              'gates': 'doctor lifecycle capability vision context exactness prefix_cache performance quality'}
    fields.update(overrides)
    return 'LOCAL-VALIDATION v1\n' + '\n'.join(f'{k}: {v}' for k, v in fields.items())


def pr_doc(state='open', author='vcruz305', ref='model/qwen3.8-27b'):
    return {'state': state, 'user': {'login': author}, 'head': {'ref': ref}}


class ParseBlockTest(unittest.TestCase):
    def test_valid_block(self):
        fields = l.parse_block(block())
        self.assertIsNotNone(fields)
        self.assertEqual(fields['head'], HEAD)
        self.assertEqual(fields['model'], 'qwen3.8/qwen3.8-27b')
        self.assertEqual(fields['check_model_branch'], 'PASS')

    def test_missing_or_extra_keys(self):
        # parse_block returns a dict whenever the header and required keys are present; semantic
        # validation happens at evaluation time (PASS fields, gates text).
        self.assertIsNone(l.parse_block('no block here'))
        self.assertIsNone(l.parse_block('LOCAL-VALIDATION v1\nhead: x'))
        fields = l.parse_block(block(pr='wrong'))
        self.assertIsNotNone(fields)
        self.assertEqual(fields['pr'], 'wrong')
        fields = l.parse_block(block(gates=''))
        self.assertIsNotNone(fields)
        self.assertEqual(fields['gates'], '')


class CheckTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.owners_file = Path(self.tmp.name) / 'CODEOWNERS'
        self.owners_file.write_text('* @am423 @vcruz305\n')
        self.reviews = []

    def api_fake(self, path, token, **kwargs):
        if '/reviews' in path:
            return self.reviews
        raise AssertionError('unexpected API call: ' + path)

    def run_check(self, files, pr_doc, reviews=None):
        if reviews is not None:
            self.reviews = reviews
        with patch.object(l, 'api_json', side_effect=self.api_fake):
            return l.check_local_validation(REPO, PR, HEAD, 'token', OWNERS,
                                            pr_files=files, pr_doc=pr_doc)

    def test_not_a_model_pr_is_a_no_op_success(self):
        state, _ = self.run_check([{'filename': 'tools/check.py'}], pr_doc())
        self.assertEqual(state, 'success')

    def test_approval_registration_pr_is_a_no_op_success(self):
        files = [{'filename': 'runtimes/templates/APPROVALS.json'}]
        state, _ = self.run_check(files, pr_doc(ref='model/qwen3.8-27b-approval'))
        self.assertEqual(state, 'success')

    def test_model_pr_without_a_qualifying_review_fails(self):
        files = [{'filename': 'runtimes/qwen3.8/qwen3.8-27b/v1/approval.json'}]
        state, _ = self.run_check(files, pr_doc(), reviews=[review('am423', body='lgtm')])
        self.assertEqual(state, 'failure')

    def test_qualifying_review_passes(self):
        files = [{'filename': 'runtimes/qwen3.8/qwen3.8-27b/v1/approval.json'}]
        state, _ = self.run_check(files, pr_doc(), reviews=[review('am423', body=block())])
        self.assertEqual(state, 'success')

    def test_author_or_non_maintainer_reviews_do_not_qualify(self):
        files = [{'filename': 'runtimes/qwen3.8/qwen3.8-27b/v1/approval.json'}]
        for login in ('vcruz305', 'somebody'):
            state, _ = self.run_check(files, pr_doc(), reviews=[review(login, body=block())])
            self.assertEqual(state, 'failure', login)

    def test_stale_head_or_wrong_model_block_fails(self):
        files = [{'filename': 'runtimes/qwen3.8/qwen3.8-27b/v1/approval.json'}]
        for body in (block(head='b' * 40), block(model='qwen3.8/other'), block(pr=9)):
            with self.subTest(body=body):
                state, _ = self.run_check(files, pr_doc(), reviews=[review('am423', body=body)])
                self.assertEqual(state, 'failure')

    def test_malformed_block_is_ignored(self):
        files = [{'filename': 'runtimes/qwen3.8/qwen3.8-27b/v1/approval.json'}]
        state, _ = self.run_check(files, pr_doc(),
                                  reviews=[review('am423', body='LOCAL-VALIDATION v1\nhead: ' + HEAD + '\npr: 7\nmodel: x')])
        self.assertEqual(state, 'failure')

    def test_status_attaches_to_the_head_sha(self):
        posted = []

        class FakeOpener:
            def urlopen(self, request, timeout=60):
                posted.append((request.full_url, request.data.decode()))
                class R:
                    status = 201
                    def __enter__(self):
                        return self
                    def __exit__(self, *exc):
                        return False
                return R()

        with patch('urllib.request.urlopen', FakeOpener().urlopen):
            l.set_status(REPO, HEAD, 'token', 'success', 'ok')
        self.assertEqual(len(posted), 1)
        self.assertIn(f'/repos/{REPO}/statuses/{HEAD}', posted[0][0])
        payload = json.loads(posted[0][1])
        self.assertEqual(payload['context'], 'local-validation')
        self.assertEqual(payload['state'], 'success')


if __name__ == '__main__':
    unittest.main()
