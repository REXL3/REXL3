#!/usr/bin/env python3
"""local-validation check (contract sections 4 and 5).

Attaches a commit status named `local-validation` to the PR head SHA. A model PR passes only when an
APPROVED review by a maintainer other than the PR author carries a LOCAL-VALIDATION v1 block for the
current head. The result re-evaluates when the head changes (pull_request_target synchronize) and when
a review is submitted or dismissed (pull_request_review). Non-model PRs and model/*-approval PRs get
a no-op success status. Code and policy come from the protected base revision only; the PR's data is
read through the GitHub REST API and never executed.
"""
import argparse
import json
import os
import sys
import urllib.request

try:
    from . import model_layout as m
    from .check_pr_rules import api_json, changed_files
except ImportError:
    import model_layout as m
    from check_pr_rules import api_json, changed_files

LOCAL_VALIDATION = 'LOCAL-VALIDATION v1'
PASS_FIELDS = ('check_model_branch', 'assets', 'doctor', 'smoke')
REQUIRED_FIELDS = ('pr', 'head', 'model', 'host', 'gates')


def parse_block(body):
    """Return the LOCAL-VALIDATION v1 dict, or None when the body has none."""
    if not isinstance(body, str):
        return None
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if line.strip() != LOCAL_VALIDATION:
            continue
        fields = {}
        for following in lines[i + 1:]:
            stripped = following.strip()
            if not stripped:
                break
            if ':' not in stripped:
                return None
            key, value = stripped.split(':', 1)
            key = key.strip()
            if key in fields:
                return None
            fields[key] = value.strip()
        if all(key in fields for key in REQUIRED_FIELDS):
            return fields
        return None
    return None


def model_folder(pr_files):
    folders = set()
    for item in pr_files:
        path = item['filename']
        parts = path.split('/')
        if len(parts) >= 4 and parts[0] == 'runtimes' and parts[1] != 'templates':
            folders.add((parts[1], parts[2]))
    if len(folders) == 1:
        return next(iter(folders))
    return None


def maintainers(path):
    owners = set()
    for line in path.read_text(errors='replace').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        fields = line.split()
        if fields[0] == '*':
            owners.update(f.lstrip('@') for f in fields[1:] if f.startswith('@'))
    return owners


def check_local_validation(repo, pr, head_sha, token, owners, pr_files=None, pr_doc=None):
    if pr_files is None:
        pr_files = changed_files(repo, pr, token)
    if pr_doc is None:
        pr_doc = api_json(f'/repos/{repo}/pulls/{pr}', token)
    if pr_doc.get('state') != 'open':
        return ('success', 'PR is not open')
    author = pr_doc.get('user', {}).get('login')
    folder = model_folder(pr_files)
    if folder is None:
        return ('success', 'not a model PR (no single model folder)')
    family, model = folder
    branch = pr_doc.get('head', {}).get('ref', '')
    if branch.startswith('model/') and branch.endswith('-approval'):
        return ('success', 'model approval registration PR')
    # Reviews: only APPROVED reviews by a maintainer other than the author.
    reviews = []
    page = 1
    while True:
        batch = api_json(f'/repos/{repo}/pulls/{pr}/reviews?per_page=100&page={page}', token)
        reviews.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    for review in reviews:
        if review.get('state') != 'APPROVED':
            continue
        login = review.get('user', {}).get('login')
        if login in (None, author) or login not in owners:
            continue
        fields = parse_block(review.get('body'))
        if fields is None:
            continue
        if fields['head'] != head_sha:
            continue
        if fields['model'] != f'{family}/{model}':
            continue
        if str(fields['pr']) != str(pr):
            continue
        if all(fields.get(f) == 'PASS' for f in PASS_FIELDS) and fields.get('gates'):
            return ('success', 'LOCAL-VALIDATION v1 for the current head')
    return ('failure', 'no APPROVED maintainer review with a LOCAL-VALIDATION v1 block for the current head')


def set_status(repo, sha, token, state, description):
    url = f'https://api.github.com/repos/{repo}/statuses/{sha}'
    payload = json.dumps({'state': state, 'context': 'local-validation', 'description': description[:140]}).encode()
    request = urllib.request.Request(url, data=payload, method='POST', headers={
        'Authorization': 'Bearer ' + token, 'User-Agent': 'rexl3-local-validation',
        'Accept': 'application/vnd.github+json'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo')
    parser.add_argument('--pr', type=int)
    parser.add_argument('--head-sha')
    parser.add_argument('--token')
    parser.add_argument('--codeowners')
    args = parser.parse_args(argv)
    repo = args.repo or os.environ.get('REXL3_REPO')
    pr = args.pr if args.pr is not None else (int(os.environ['REXL3_PR_NUMBER']) if os.environ.get('REXL3_PR_NUMBER') else None)
    head_sha = args.head_sha or os.environ.get('REXL3_HEAD_SHA')
    token = args.token or os.environ.get('GH_TOKEN')
    codeowners = args.codeowners or os.environ.get('REXL3_CODEOWNERS')
    if not (repo and pr and head_sha and token and codeowners):
        print('usage: check_local_validation.py --repo OWNER/REPO --pr N --head-sha SHA --codeowners PATH --token TOKEN',
              file=sys.stderr)
        return 2
    try:
        owners = maintainers(__import__('pathlib').Path(codeowners))
        state, description = check_local_validation(repo, pr, head_sha, token, owners)
        set_status(repo, head_sha, token, state, description)
        print(f'{state.upper()} {description}')
    except (ValueError, KeyError, TypeError, OSError) as error:
        # A failure to evaluate must be visible, not silently green.
        try:
            set_status(repo, head_sha, token, 'failure', 'local-validation could not evaluate: ' + str(error)[:120])
        except Exception:
            pass
        print('FAIL internal: ' + str(error))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
