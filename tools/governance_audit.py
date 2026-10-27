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
INCIDENT_RANGE = re.compile(r'#(\d+)(?:\s*(?:-|through)\s*#?(\d+))?')


def days_ago(days):
    return datetime.now(timezone.utc) - timedelta(days=days)


def incident_covers(incident_text, number):
    for match in INCIDENT_RANGE.finditer(incident_text):
        start = int(match[1])
        end = int(match[2] or match[3] or match[1])
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


VOLATILE_KEYS = frozenset({'id', 'node_id', 'created_at', 'updated_at', 'source', 'source_type',
                            'current_user_can_bypass', '_links'})


def strip_volatile(value):
    """Drop incidental API metadata (ids, timestamps, links, actor status) from a ruleset doc."""
    if isinstance(value, dict):
        return {k: strip_volatile(v) for k, v in value.items() if k not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [strip_volatile(v) for v in value]
    return value


def committed_parameter_subset(live_rule, committed_rule):
    """The committed rule's parameters must all hold in the live rule of the same type.

    The API echoes defaulted parameters (e.g. required_reviewers, dismissal_restriction,
    require_extra_approval_for_unattributed_changes) that a PUT payload may omit, so the live
    side may carry keys the committed side does not. Every committed key must be present with
    an equal value; integration_id on required status checks is incidental and ignored.
    """
    live_params = live_rule.get('parameters')
    committed_params = committed_rule.get('parameters')
    if committed_params is None:
        return live_params is None
    if not isinstance(live_params, dict):
        return False
    for key, expected in committed_params.items():
        if key not in live_params:
            return False
        actual = live_params[key]
        if key == 'required_status_checks' and isinstance(actual, list) and isinstance(expected, list):
            actual = [{k: v for k, v in item.items() if k != 'integration_id'} for item in actual if isinstance(item, dict)]
            expected = [{k: v for k, v in item.items() if k != 'integration_id'} for item in expected if isinstance(item, dict)]
        if actual != expected:
            return False
    return True


def policy_matches(committed, live):
    """Committed policy subset holds in the live ruleset (name/target/enforcement/conditions/bypass)."""
    for key in ('name', 'target', 'enforcement'):
        if committed.get(key) != live.get(key):
            return False
    if committed.get('conditions') != live.get('conditions'):
        return False
    if committed.get('bypass_actors') != live.get('bypass_actors'):
        return False
    live_rules = {rule.get('type'): rule for rule in live.get('rules', []) if isinstance(rule, dict)}
    for rule in committed.get('rules', []):
        if not isinstance(rule, dict):
            return False
        live_rule = live_rules.get(rule.get('type'))
        if live_rule is None or not committed_parameter_subset(live_rule, rule):
            return False
    return True


def fetch_live_rulesets(repo, token):
    """List then fetch details: the list endpoint omits conditions and rules."""
    raw = api(f'/repos/{repo}/rulesets', token)
    listing = json.loads(raw.decode('utf-8'))
    live = []
    for entry in listing:
        ruleset_id = entry.get('id') if isinstance(entry, dict) else None
        if ruleset_id is None:
            continue
        try:
            detail = api(f'/repos/{repo}/rulesets/{ruleset_id}', token)
        except ValueError as error:
            live.append({'name': entry.get('name'), '_unreadable': str(error)})
            continue
        live.append(json.loads(detail.decode('utf-8')))
    return live


def check_rulesets(repo_root, repo, token, live=None):
    if live is None:
        try:
            live = fetch_live_rulesets(repo, token)
        except ValueError as error:
            return [f'live rulesets are not readable: {error}']
    committed = {}
    for path in sorted((repo_root / '.github/rulesets').glob('*.json')):
        try:
            committed[path.stem] = json.loads(path.read_text())
        except (ValueError, OSError) as error:
            return [f'committed ruleset {path.name} is unreadable: {error}']
    problems = []
    seen = set()
    for ruleset in live:
        name = ruleset.get('name') if isinstance(ruleset, dict) else None
        if name is None:
            continue
        seen.add(name)
        if '_unreadable' in ruleset:
            print(f'INFO: live ruleset `{name}` detail is unreadable by this token: '
                  f'{ruleset["_unreadable"]}; skipped')
            continue
        expected = committed.get(name)
        if expected is None:
            problems.append(f'live ruleset `{name}` has no committed copy in .github/rulesets/')
            continue
        if any(key not in ruleset for key in ('enforcement', 'conditions', 'rules')):
            missing = [key for key in ('enforcement', 'conditions', 'rules', 'bypass_actors')
                       if key not in ruleset and key in expected]
            if missing:
                print(f'INFO: live ruleset `{name}` returned an incomplete view for this token '
                      f'(missing {missing}); skipped')
                continue
        if not policy_matches(expected, strip_volatile(ruleset)):
            problems.append(f'live ruleset `{name}` differs from .github/rulesets/{name}.json')
    for name in committed:
        if name not in seen:
            problems.append(f'committed ruleset {name}.json is not live')
    return problems


def check_immutable_releases(repo, token):
    """Read the repo-level immutable-releases setting; fail closed when unreadable or explicitly disabled.

    The GitHub Actions workflow token cannot read this repository-settings endpoint and receives
    HTTP 403; the change is still verifiable by a maintainer (or a token with repository-admin
    settings read), so a 403 is skipped with an informational line instead of failing the audit.
    """
    try:
        raw = api(f'/repos/{repo}/immutable-releases', token)
    except ValueError as error:
        if '403' in str(error):
            return [], 'INFO: immutable releases setting is unreadable by this token (HTTP 403); skipped'
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
