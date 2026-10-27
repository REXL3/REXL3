"""Workflow contract tests (section 5): thin YAML, base-checkout execution, every check reports on every PR.

Structural regex tests only; YAML is kept thin so behavior stays in the tested stdlib Python.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / '.github/workflows'


def text(name):
    return (WORKFLOWS / name).read_text()


class WorkflowStructureTest(unittest.TestCase):
    def test_workflows_exist(self):
        for name in ('verify.yml', 'pr-rules.yml', 'model-completeness.yml', 'local-validation.yml', 'governance-audit.yml'):
            self.assertTrue((WORKFLOWS / name).is_file(), name)

    def test_no_workflow_level_filters(self):
        # Required checks must report on every PR to main: no paths/branch filters on triggers or jobs.
        for name in ('verify.yml', 'pr-rules.yml', 'model-completeness.yml', 'local-validation.yml'):
            body = text(name)
            self.assertNotRegex(body, r'(?m)^\s+paths:', name)
            self.assertNotRegex(body, r'(?m)^\s+paths-ignore:', name)
            self.assertRegex(body, r'(?m)^\s+branches:\s*\[main\]$', name)
            self.assertNotIn('if:', body, name)

    def test_every_action_pinned_by_full_commit_sha(self):
        for path in WORKFLOWS.glob('*.y*ml'):
            uses = re.findall(r'(?m)^\s*-?\s*uses:\s*(\S+)', path.read_text())
            self.assertTrue(uses, path.name)
            for ref in uses:
                self.assertRegex(ref, r'^[\w.-]+/[\w.-]+@[0-9a-f]{40}$', path.name + ': ' + ref)

    def test_checkouts_persist_credentials_false(self):
        for path in WORKFLOWS.glob('*.y*ml'):
            body = path.read_text()
            self.assertEqual(body.count('persist-credentials: false'), body.count('actions/checkout@'), path.name)

    def test_candidate_code_is_never_executed(self):
        for name in ('verify.yml', 'pr-rules.yml', 'model-completeness.yml', 'local-validation.yml'):
            body = text(name)
            for line in body.splitlines():
                stripped = line.strip()
                if re.match(r'(python3?|bash|sh|\./|source|\.\s)', stripped) or ' python' in stripped:
                    self.assertNotRegex(stripped, r'candidate/(tools|\.github|scripts)', name + ': ' + stripped)
            self.assertNotRegex(body, r'(?m)^\s*(cd|working-directory:)\s+\.?/?candidate', name)

    def test_base_tools_only(self):
        for name in ('verify.yml', 'pr-rules.yml', 'model-completeness.yml', 'local-validation.yml', 'governance-audit.yml'):
            body = text(name)
            for call in re.findall(r'python3 -B -E -s ([\w./-]+)', body):
                self.assertTrue(call.startswith(('base/', 'repo/')), name + ': ' + call)

    def test_local_validation_triggers_and_status(self):
        body = text('local-validation.yml')
        self.assertRegex(body, r'(?m)^\s+pull_request_target:')
        self.assertRegex(body, r'(?m)^\s+pull_request_review:')
        self.assertRegex(body, r'(?m)^\s+statuses: write$')
        self.assertRegex(body, r'(?m)^\s+types: \[submitted, dismissed\]$')
        self.assertRegex(body, r'(?m)^\s+types: \[opened, synchronize, reopened\]$')
        self.assertIn('base/tools/check_local_validation.py', body)
        self.assertNotIn('pull_request:', body)

    def test_governance_audit_is_daily_read_only(self):
        body = text('governance-audit.yml')
        self.assertRegex(body, r'(?m)^\s+schedule:')
        self.assertRegex(body, r'(?m)^\s+- cron:')
        self.assertRegex(body, r'(?m)^\s+workflow_dispatch:')
        self.assertRegex(body, r'(?m)^permissions:\s*\n\s+contents: read')
        self.assertNotIn('statuses: write', body)
        self.assertNotIn('pull_request_target', body)

    def test_model_completeness_downloads_then_runs_base_checker(self):
        body = text('model-completeness.yml')
        self.assertIn('base/tools/fetch_dev_assets.py', body)
        self.assertIn('base/tools/check_model_branch.py', body)
        self.assertIn('--base-tree-dir base', body)
        self.assertIn('--assets-dir "$ASSETS_DIR"', body)
        self.assertNotIn('--local', body)

    def test_pr_rules_reads_title_and_branch_from_env(self):
        body = text('pr-rules.yml')
        self.assertIn('REXL3_TITLE:', body)
        self.assertIn('REXL3_HEAD_REF:', body)
        self.assertIn('base/tools/check_pr_rules.py', body)
        self.assertNotIn('pull-requests: write', body)

    def test_model_completeness_passes_untrusted_values_via_env(self):
        body = text('model-completeness.yml')
        for var in ('REXL3_BASE_SHA', 'REXL3_HEAD_SHA', 'REXL3_HEAD_REF'):
            self.assertIn(var + ':', body)
        self.assertIn('--branch "$REXL3_HEAD_REF"', body)
        self.assertIn('--base "$REXL3_BASE_SHA"', body)
        self.assertIn('--head "$REXL3_HEAD_SHA"', body)

    def test_no_run_step_interpolates_pr_event_fields_inline(self):
        # Untrusted PR values (title, head ref, head sha) must arrive via env, never inline in a
        # shell script: ${{ ... }} inside run: is expanded before the shell sees it and a crafted
        # branch name (spaces, ;, $, backticks are all legal in git refs) becomes command injection.
        # env: assignments in the same step are the sanctioned channel, so only the run body is
        # examined here.
        for name in ('verify.yml', 'pr-rules.yml', 'model-completeness.yml', 'local-validation.yml'):
            lines = text(name).splitlines()
            i = 0
            while i < len(lines):
                match = re.match(r'^(\s+)run: \|', lines[i])
                if match is None:
                    i += 1
                    continue
                indent = len(match.group(1))
                body = []
                j = i + 1
                while j < len(lines):
                    line = lines[j]
                    if line.strip() and len(line) - len(line.lstrip()) <= indent:
                        break
                    body.append(line)
                    j += 1
                self.assertNotIn('${{ github.event.pull_request', '\n'.join(body),
                                 name + ' run body interpolates PR event fields inline')
                i = j

    def test_ruleset_copies_match_section_8(self):
        import json
        main = json.loads((ROOT / '.github/rulesets/main-guard.json').read_text())
        pr = next(rule for rule in main['rules'] if rule['type'] == 'pull_request')['parameters']
        self.assertEqual(pr['required_approving_review_count'], 1)
        self.assertTrue(pr['require_code_owner_review'])
        self.assertTrue(pr['require_last_push_approval'])
        self.assertTrue(pr['dismiss_stale_reviews_on_push'])
        self.assertTrue(pr['required_review_thread_resolution'])
        self.assertEqual(pr['allowed_merge_methods'], ['squash', 'rebase'])
        checks = next(rule for rule in main['rules'] if rule['type'] == 'required_status_checks')['parameters']['required_status_checks']
        self.assertEqual({c['context'] for c in checks}, {'verify', 'pr-rules', 'model-completeness', 'local-validation'})
        self.assertEqual(main['bypass_actors'], [])
        tags = json.loads((ROOT / '.github/rulesets/protected-tags.json').read_text())
        self.assertEqual(tags['conditions']['ref_name']['include'], ['~ALL'])
        self.assertEqual({r['type'] for r in tags['rules']}, {'creation', 'update', 'deletion'})
        self.assertEqual(tags['bypass_actors'], [])
        model = json.loads((ROOT / '.github/rulesets/model-branches.json').read_text())
        self.assertEqual(model['conditions']['ref_name']['include'], ['refs/heads/model/*'])
        self.assertEqual({r['type'] for r in model['rules']}, {'deletion', 'non_fast_forward'})
        self.assertEqual(model['bypass_actors'], [])


if __name__ == '__main__':
    unittest.main()
