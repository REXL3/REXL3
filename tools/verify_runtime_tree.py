#!/usr/bin/env python3
"""Verify production trees using protected code and independently anchored dev assets.

The candidate's manifest cannot authenticate itself. --trusted-approvals-sha256
must come from a maintainer-reviewed immutable mapping/protected workflow input.
Legacy mode verifies exact pinned history only; it never authorizes promotion.
"""
import argparse
import os
from pathlib import Path
import re
import sys
try:
    from . import release_evidence as e
except ImportError:
    import release_evidence as e

SCHEMA = e.SCHEMA
sha256_file = e.sha256_file


def tree_inventory(tree):
    tree = e.checked_path(tree)
    e.require(tree.is_dir(), 'runtime tree missing')
    files = {}
    for base, dirs, names in os.walk(tree):
        for name in dirs:
            e.checked_path(Path(base) / name)
        for name in names:
            path = Path(base) / name
            rel = e.relative_path(path.relative_to(tree).as_posix())
            files[rel] = e.record(e.safe_file(tree, rel))
    return files


def verify_tree(tree, approved_templates, *, trusted_approvals=None, assets_dir=None, allow_legacy=False):
    """Return named problems (empty means PASS). All inputs are read-only."""
    try:
        tree = e.checked_path(tree)
        version = re.fullmatch(r'v([1-9][0-9]*)', tree.name)
        e.require(version is not None, 'invalid runtime version directory')
        runtime_id, number = tree.parent.name, int(version[1])
        key = f'{runtime_id}/v{number}'
        approval_path = e.safe_file(tree, 'approval.json')
        doc = e.load_json(approval_path)
        e.require(doc.get('runtime_id') == runtime_id, 'runtime path identity mismatch')
        e.require(isinstance(trusted_approvals, dict) and trusted_approvals.get('schema') == 'rexl3.approval-registry.v1',
                  'independently trusted approval registry required')
        actual = tree_inventory(tree)
        if doc.get('schema') == 'rexl3.profile-approval.v1':
            e.require(allow_legacy, 'LEGACY history requires explicit --allow-legacy; never promotion')
            entry = trusted_approvals.get('legacy', {}).get(key)
            e.require(isinstance(entry, dict), 'LEGACY history has no trusted anchor')
            e.require(e.sha256_file(approval_path) == e.full_hash(entry.get('approval_sha256')), 'LEGACY approval hash drift')
            e.require(all(doc.get(k) == entry.get(k) and e.HEX40.fullmatch(str(doc.get(k))) for k in ('source_commit', 'source_tree')),
                      'LEGACY source identity drift')
            expected = entry.get('files')
            e.require(isinstance(expected, dict) and expected, 'LEGACY exact inventory required')
            for rel, item in expected.items():
                e.relative_path(rel)
                e.validate_record(item)
            e.require(actual == expected, 'LEGACY tree hash/bytes/mode/inventory drift')
            return []
        e.require(assets_dir is not None, 'authenticated dev assets directory required')
        asset_approval = e.safe_file(assets_dir, 'approval.json')
        e.require(approval_path.read_bytes() == asset_approval.read_bytes(), 'tree/asset approval inequality')
        approved, _ = e.verify_assets(asset_approval, registry=trusted_approvals, approved_templates=approved_templates,
                                      engine_dir=Path(assets_dir) / 'engine')
        e.require(approved['version'] == number, 'runtime path version mismatch')
        expected = dict(approved['source_asset']['members'])
        expected['approval.json'] = dict(e.record(approval_path), mode=0o644)
        for rel in sorted(set(expected) | set(actual)):
            e.require(rel in actual, rel + ' is listed but missing')
            e.require(rel in expected, rel + ' is not in the approved inventory')
            e.require(actual[rel] == expected[rel], rel + ': hash/bytes/mode != approved')
        return []
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        return [str(tree) + ': ' + str(error)]


def discover(root):
    runtimes = e.checked_path(Path(root) / 'runtimes')
    if not runtimes.exists():
        return []
    trees = []
    for base in sorted(runtimes.iterdir()):
        e.checked_path(base)
        if base.name in ('templates', 'template-v1', 'models') or not base.is_dir():
            continue
        for version in sorted(base.iterdir()):
            e.checked_path(version)
            if re.fullmatch(r'v[1-9][0-9]*', version.name) and version.is_dir():
                trees.append(version)
    return trees


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--approved', required=True, help='protected template allow-list JSON')
    parser.add_argument('--trusted-approvals', required=True)
    parser.add_argument('--trusted-approvals-sha256', required=True)
    parser.add_argument('--assets-dir', required=True, help='asset root containing <runtime-id>/v<N>/approval.json')
    parser.add_argument('--allow-legacy', action='store_true')
    args = parser.parse_args(argv)
    try:
        registry = e.load_registry(args.trusted_approvals, args.trusted_approvals_sha256)
        approved = e.load_json(args.approved)['templates']
        trees = discover(args.root)
        # Zero trees is a pass only while the pinned registry anchors nothing; a registry that anchors a tree
        # which is missing, or a wrong --root, still stops here.
        # A legacy anchor is history: its tree may be removed by a reviewed change. A v2 approval anchor is not, so a
        # registry that anchors one still refuses a root with no trees (a wrong --root or a deleted tree).
        anchored = bool(registry.get('approvals'))
        e.require(trees or not anchored, 'no runtime trees found but the registry anchors some')
        problems = []
        for tree in trees:
            problems.extend(verify_tree(tree, approved, trusted_approvals=registry,
                assets_dir=Path(args.assets_dir) / tree.parent.name / tree.name, allow_legacy=args.allow_legacy))
        for problem in problems:
            print(problem)
        print(f'{len(trees)} tree(s), {len(problems)} problem(s); legacy allowed={args.allow_legacy}')
        return 1 if problems else 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print('TREE_REFUSED: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
