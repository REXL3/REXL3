#!/usr/bin/env python3
"""Governance audit (contract section 5, not required): daily schedule + workflow_dispatch, read-only.

Fails when:
- a release exists whose tag is not v<X.Y.Z> or has no matching releases/v<X.Y.Z>/manifest.json on main;
- a PR merged to main in the last 14 days has no approving review by a non-author maintainer and is not
  listed in an incident record (docs/incidents/*.md);
- the live rulesets differ from .github/rulesets/*.json;
- immutable releases are disabled (or cannot be verified: fail closed).
Everything here is read-only. Exit 0 all pass, 1 any fail, 2 usage error.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import sys

try:
    from .check_pr_rules import api_json, api
    from . import model_layout as m
except ImportError:
    from check_pr_rules import api_json, api
    import model_layout as m

PRODUCT_VERSION_RE = re.compile(r'v[0-9]+\.[0-9]+\.[0-9]+\Z')
INCIDENT_RANGE = re.compile(r'#(\d+)(?:-#(\d+))?')


def days_ago(days):
    return datetime.now(timezone.utc) - timedelta(days=days)


def incident_covers(incident_text, number):
    for match in INCIDENT_RANGE.finditer(incident_text):
        start, end = int(match[1]), int(match[2] or match[1])
        if start <= number <= end:
            return True
    return False


def check_releases(repo_root, repo, token, releases=None):
    if releases is None:
        releases = []
        page = 1
        while True:
            batch = api_json(f'/repos/{repo}/releases?per_page=100&page={page}', token)
            releases.extend(batch)
            if len(batch) < 100:
                break
            page += 1
    problems = []
    for release in releases:
        if release.get('draft'):
            continue
        tag = release.get('tag_name')
        if not isinstance(tag, str) or not PRODUCT_VERSION_RE.fullmatch(tag):
            problems.append(f'release `{tag}` is not a v<X.Y.Z> tag')
            continue
        manifest = repo_root / 'releases' / tag / 'manifest.json'
        if not manifest.is_file():
            problems.append(f'release `{tag}` has no releases/{tag}/manifest.json on main')
    return problems


def check_merged_prs(repo_root, repo, token, owners, *, since=None, pulls=None, reviews_for=None):
    if since is None:
        since = days_ago(14)
    if pulls is None:
        pulls = []
        page = 1
        while True:
            batch = api_json(f'/repos/{repo}/pulls?state=closed&base=main&sort=updated&direction=desc&per_page=100&page={page}', token)
            pulls.extend(batch)
            if len(batch) < 100 or not batch or batch[-1].get('updated_at') is None:
                break
            if batch[-1].get('merged_at') and parse_iso(batch[-1]['merged_at']) < since:
                break
            page += 1
    incidents = ''
    for path in sorted((repo_root / 'docs/incidents').glob('*.md')) if (repo_root / 'docs/incidents').is_dir() else []:
        incidents += path.read_text(errors='replace')
    problems = []
    for pull in pulls:
        merged_at = pull.get('merged_at')
        if not merged_at or parse_iso(merged_at) < since:
            continue
        number = pull.get('number')
        author = pull.get('user', {}).get('login')
        reviews = reviews_for(number) if reviews_for is not None else fetch_reviews(repo, number, token)
        approved = any(r.get('state') == 'APPROVED' and r.get('user', {}).get('login') not in (None, author)
                       and r.get('user', {}).get('login') in owners for r in reviews)
        if not approved and not incident_covers(incidents, number):
            problems.append(f'PR #{number} merged without an approving review by a non-author maintainer and is not listed in an incident record')
    return problems


def parse_iso(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def fetch_reviews(repo, number, token):
    reviews = []
    page = 1
    while True:
        batch = api_json(f'/repos/{repo}/pulls/{number}/reviews?per_page=100&page={page}', token)
        reviews.extend(batch)
        if len(batch) < 100:
            return reviews
        page += 1


def strip_integration(value):
    if isinstance(value, dict):
        return {k: strip_integration(v) for k, v in value.items() if k != 'integration_id'}
    if isinstance(value, list):
        return [strip_integration(v) for v in value]
    return value


def check_rulesets(repo_root, repo, token, live=None):
    if live is None:
        try:
            raw = api(f'/repos/{repo}/rulesets', token)
        except ValueError as error:
            return [f'live rulesets are not readable: {error}']
        live = json.loads(raw.decode('utf-8'))
    committed = {}
    for path in sorted((repo_root / '.github/rulesets').glob('*.json')):
        try:
            committed[path.stem] = json.loads(path.read_text())
        except (ValueError, OSError) as error:
            return [f'committed ruleset {path.name} is unreadable: {error}']
    problems = []
    seen = set()
    for ruleset in live:
        name = ruleset.get('name')
        if name is None:
            continue
        seen.add(name)
        expected = committed.get(name)
        if expected is None:
            problems.append(f'live ruleset `{name}` has no committed copy in .github/rulesets/')
        elif strip_integration(ruleset) != strip_integration(expected):
            problems.append(f'live ruleset `{name}` differs from .github/rulesets/{name}.json')
    for name in committed:
        if name not in seen:
            problems.append(f'committed ruleset {name}.json is not live')
    return problems


def check_immutable_releases(repo, token):
    """Read the repo-level immutable-releases setting; fail closed when unreadable or explicitly disabled."""
    try:
        raw = api(f'/repos/{repo}/immutable-releases', token)
    except ValueError as error:
        return [f'immutable releases setting is not readable: {error}'], None
    try:
        doc = json.loads(raw.decode('utf-8'))
    except (ValueError, UnicodeError):
        return ['immutable releases setting returned an unreadable response'], None
    if not isinstance(doc, dict) or doc.get('enabled') is False:
        return ['immutable releases are disabled'], None
    return [], raw.decode('utf-8', 'replace')[:80]


def audit(repo_root, repo, token, owners, releases=None, pulls=None, reviews_for=None, live_rulesets=None):
    problems = []
    problems.extend(check_releases(repo_root, repo, token, releases=releases))
    problems.extend(check_merged_prs(repo_root, repo, token, owners, pulls=pulls, reviews_for=reviews_for))
    problems.extend(check_rulesets(repo_root, repo, token, live=live_rulesets))
    problems.extend(check_immutable_releases(repo, token)[0])
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', default='.')
    parser.add_argument('--repo')
    parser.add_argument('--token')
    parser.add_argument('--codeowners', help='defaults to <repo-root>/.github/CODEOWNERS')
    args = parser.parse_args(argv)
    repo = args.repo or os.environ.get('REXL3_REPO')
    token = args.token or os.environ.get('GH_TOKEN')
    if not (repo and token):
        print('usage: governance_audit.py --repo-root PATH --repo OWNER/REPO --token TOKEN', file=sys.stderr)
        return 2
    root = Path(args.repo_root)
    codeowners = Path(args.codeowners) if args.codeowners else root / '.github/CODEOWNERS'
    owners = set()
    for line in codeowners.read_text(errors='replace').splitlines():
        line = line.strip()
        if line.startswith('*') and not line.startswith('*.'):
            owners.update(f.lstrip('@') for f in line.split()[1:] if f.startswith('@'))
    try:
        problems = audit(root, repo, token, owners)
    except (ValueError, KeyError, TypeError, OSError) as error:
        print('FAIL internal: ' + str(error))
        return 1
    for problem in problems:
        print('FAIL ' + problem)
    if not problems:
        print('PASS governance-audit')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
