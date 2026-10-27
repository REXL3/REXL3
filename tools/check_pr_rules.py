#!/usr/bin/env python3
"""PR rules check (contract sections 3 and 5): title, branch, path scope per branch type, fragment rule.

Runs from the protected base revision on pull_request_target. All inputs come via environment
variables (never inline workflow expressions). It reports PASS/FAIL lines; a job-level no-op is not
possible for this check because section 3 applies to every PR to main. Exit 0 all pass, 1 any fail,
2 usage error.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

try:
    from . import model_layout as m
except ImportError:
    import model_layout as m

TYPES = ('FIX', 'FEAT', 'BUILD', 'CI', 'TEST', 'BUMP', 'DOCS')
TITLE_RE = re.compile(r'^(FIX|FEAT|BUILD|CI|TEST|BUMP|DOCS): ([A-Z][a-z]*)( .*)?$')
PAST_IRREGULAR = {'built', 'made', 'wrote', 'rewrote', 'split', 'set', 'ran', 'kept', 'cut',
                  'put', 'found', 'held', 'led', 'left', 'read', 'sent', 'took', 'saw', 'gave',
                  'brought', 'wrote', 'got', 'sent', 'did', 'went', 'came', 'became', 'began',
                  'drew', 'fell', 'felt', 'grew', 'knew', 'met', 'paid', 'said', 'spoke', 'stood',
                  'told', 'thought', 'won', 'worn'}
PRODUCT_VERSION_RE = re.compile(r'v[0-9]+\.[0-9]+\.[0-9]+\Z')
SEALED_TREE_RE = re.compile(r'^runtimes/[^/]+/[^/]+/v[1-9][0-9]*/')


def api(path, token, *, method='GET', data=None, accept=None):
    url = 'https://api.github.com' + path
    headers = {'Authorization': 'Bearer ' + token, 'User-Agent': 'rexl3-pr-rules',
               'Accept': accept or 'application/vnd.github+json'}
    body = json.dumps(data).encode() if data is not None else None
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        raise ValueError(f'GitHub API {error.code} for {path}') from None


def api_json(path, token, **kwargs):
    raw = api(path, token, **kwargs)
    return json.loads(raw.decode('utf-8'))


def changed_files(repo, pr, token):
    files = []
    page = 1
    while True:
        batch = api_json(f'/repos/{repo}/pulls/{pr}/files?per_page=100&page={page}', token)
        files.extend(batch)
        if len(batch) < 100:
            return files
        page += 1


def file_at(repo, path, ref, token):
    try:
        raw = api(f'/repos/{repo}/contents/{path}?ref={urllib.parse.quote(ref, safe="")}',
                  token, accept='application/vnd.github.raw')
    except ValueError:
        return None
    return raw.decode('utf-8')


def verify_yml_diff_ok(files, repo, pr, token, head_ref=None, base_sha=None):
    """For model/*-approval: verify.yml may only change its TRUSTED_APPROVALS_SHA256 line.

    The GitHub files API normally includes a unified ``patch``; it is omitted for very large
    files, in which case the base and head file contents are compared line-by-line. The base
    content is read at the PR's base commit and the head content at the branch head — never at
    the same ref.
    """
    for item in files:
        if item['filename'] != '.github/workflows/verify.yml':
            continue
        patch = item.get('patch')
        if patch:
            lines = [line for line in patch.splitlines()
                     if line.startswith(('+', '-')) and not line.startswith(('+++', '---'))]
        else:
            if base_sha is None or head_ref is None:
                doc = api_json(f'/repos/{repo}/pulls/{pr}', token)
                base_sha = base_sha or doc.get('base', {}).get('sha')
                head_ref = head_ref or doc.get('head', {}).get('ref')
            base = file_at(repo, item['filename'], base_sha, token)
            head = file_at(repo, item['filename'], head_ref, token)
            if head is None:
                return False, 'verify.yml unreadable'
            base_lines = (base or '').splitlines()
            head_lines = head.splitlines()
            added = [line for line in head_lines if line not in base_lines]
            removed = [line for line in base_lines if line not in head_lines]
            lines = ['+' + line for line in added] + ['-' + line for line in removed]
        for line in lines:
            if 'TRUSTED_APPROVALS_SHA256' not in line:
                return False, 'verify.yml changed beyond the TRUSTED_APPROVALS_SHA256 pin line'
    return True, ''


def classify_branch(head_ref):
    if not isinstance(head_ref, str):
        return None, 'branch name missing'
    if head_ref.startswith('model/'):
        parts = head_ref.split('/')
        if len(parts) == 2:
            model = parts[1]
            if model.endswith('-approval'):
                base = model[:-len('-approval')]
                if base and m.MODEL_NAME_RE.fullmatch(base):
                    return 'model-approval', base
            elif m.MODEL_NAME_RE.fullmatch(model):
                return 'model', model
        return None, 'invalid model branch name'
    if head_ref.startswith('engine/'):
        topic = head_ref[len('engine/'):]
        if m.TOPIC_RE.fullmatch(topic):
            return 'engine', topic
        return None, 'invalid engine topic'
    if head_ref.startswith('release/'):
        version = head_ref[len('release/'):]
        if PRODUCT_VERSION_RE.fullmatch(version):
            return 'release', version
        return None, 'invalid release branch version'
    return None, 'branch is not a section 3 type (model/*, engine/*, release/v<X.Y.Z>)'


def title_problems(title):
    problems = []
    if not isinstance(title, str) or not title.strip():
        return ['title missing']
    if len(title) > 90:
        problems.append('title exceeds 90 characters')
    if title.endswith('.'):
        problems.append('title has a trailing period')
    match = TITLE_RE.fullmatch(title)
    if match is None:
        problems.append('title is not TYPE: Past-tense summary (TYPE one of FIX FEAT BUILD CI TEST BUMP DOCS)')
        return problems
    word = match.group(2)
    if not (word.endswith('ed') or word.lower() in PAST_IRREGULAR):
        problems.append(f'title verb {word!r} is not past tense')
    return problems


def fragment_status(files):
    """Return (kind, name) for changes/unreleased changes: 'added-mod', 'deleted', or None."""
    result = None
    for item in files:
        if not item['filename'].startswith('changes/unreleased/'):
            continue
        if item['status'] == 'removed':
            if result is None:
                result = ('deleted', item['filename'])
        else:
            result = ('added-mod', item['filename'])
    return result


def check_pr_rules(repo, pr, title, head_ref, token, files=None):
    results = []

    def report(name, ok, reason):
        results.append((name, ok, reason))

    kind, model = classify_branch(head_ref)
    if kind is None:
        report('branch', False, model)
    else:
        report('branch', True, head_ref)

    problems = title_problems(title)
    report('title', not problems, '' if not problems else '; '.join(problems))

    if files is None:
        files = changed_files(repo, pr, token)
    names = {item['filename'] for item in files}

    if kind is None:
        return results

    if kind == 'model-approval':
        allowed = {'runtimes/templates/APPROVALS.json', '.github/workflows/verify.yml'}
        bad = sorted(names - allowed)
        report('scope', not bad, '' if not bad else 'out of scope: ' + ', '.join(bad))
        ok, reason = verify_yml_diff_ok(files, repo, pr, token, head_ref=head_ref)
        report('verify pin only', ok, reason)
        return results

    if kind == 'model':
        scope_bad = []
        allow_prefix = None
        for item in files:
            path = item['filename']
            if path.startswith('runtimes/'):
                match = re.fullmatch(r'runtimes/([^/]+)/' + re.escape(model) + r'(?:/.*)?', path)
                if match is None:
                    scope_bad.append(path)
                else:
                    prefix = f'runtimes/{match[1]}/{model}/'
                    if allow_prefix is not None and allow_prefix != prefix:
                        scope_bad.append(path)
                    allow_prefix = prefix
            elif path not in (f'docs/packs/{model}.md', f'changes/unreleased/{model}.md'):
                scope_bad.append(path)
        report('scope', not scope_bad, '' if not scope_bad else 'out of scope: ' + ', '.join(scope_bad))
        fragment = fragment_status(files)
        if fragment is None or fragment[0] != 'added-mod' or fragment[1] != f'changes/unreleased/{model}.md':
            report('fragment', False, f'changes/unreleased/{model}.md must be present')
        else:
            report('fragment', True, '')
        return results

    if kind == 'engine':
        scope_bad = []
        for item in files:
            path = item['filename']
            if SEALED_TREE_RE.match(path) or path == 'runtimes/templates/APPROVALS.json':
                scope_bad.append(path)
        report('scope', not scope_bad, '' if not scope_bad else 'forbidden paths: ' + ', '.join(scope_bad))
        behavior = any(path.startswith(('tools/', '.github/')) for path in names)
        pure_docs = title.startswith('DOCS:') if isinstance(title, str) else False
        if behavior and not pure_docs:
            fragment = fragment_status(files)
            if fragment is None or fragment[0] != 'added-mod':
                report('fragment', False, 'engine behavior change needs a changes/unreleased/ fragment')
            else:
                report('fragment', True, '')
        else:
            report('fragment', True, 'not applicable (DOCS-only or no behavior change)')
        return results

    # release/v<X.Y.Z>
    version = model
    scope_bad = []
    for item in files:
        path = item['filename']
        if path.startswith(f'releases/{version}/') or path == 'CHANGELOG.md':
            continue
        if path.startswith('changes/unreleased/'):
            if item['status'] != 'removed':
                scope_bad.append(path + ' (must be a deletion)')
            continue
        scope_bad.append(path)
    report('scope', not scope_bad, '' if not scope_bad else 'out of scope: ' + ', '.join(scope_bad))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo')
    parser.add_argument('--pr', type=int)
    parser.add_argument('--title')
    parser.add_argument('--head-ref')
    parser.add_argument('--token')
    args = parser.parse_args(argv)
    repo = args.repo or os.environ.get('REXL3_REPO')
    pr = args.pr if args.pr is not None else (int(os.environ['REXL3_PR_NUMBER']) if os.environ.get('REXL3_PR_NUMBER') else None)
    title = args.title if args.title is not None else os.environ.get('REXL3_TITLE')
    head_ref = args.head_ref if args.head_ref is not None else os.environ.get('REXL3_HEAD_REF')
    token = args.token or os.environ.get('GH_TOKEN')
    if not (repo and pr and token):
        print('usage: check_pr_rules.py --repo OWNER/REPO --pr N --title T --head-ref REF --token TOKEN', file=sys.stderr)
        return 2
    try:
        results = check_pr_rules(repo, pr, title, head_ref, token)
    except (ValueError, KeyError, TypeError, OSError) as error:
        print('FAIL internal: ' + str(error))
        return 1
    failed = 0
    for name, ok, reason in results:
        line = ('PASS' if ok else 'FAIL') + ' ' + name
        if reason:
            line += ': ' + reason
        print(line)
        if not ok:
            failed += 1
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
