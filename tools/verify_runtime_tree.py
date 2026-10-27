#!/usr/bin/env python3
"""Verify production trees using protected code and independently anchored dev assets.

The candidate's manifest cannot authenticate itself. --trusted-approvals-sha256
must come from a maintainer-reviewed immutable mapping/protected workflow input.
Layout is family/model: runtimes/<family>/<model>/v<N> with MODEL.json
(schema rexl3.prod-model.v1). The old flat runtimes/<runtime-id>/v<N> layout is
rejected; the registry key is <family>/<model>/v<N> and carries runtime_id.
"""
import argparse
import os
from pathlib import Path
import re
import sys
try:
    from . import release_evidence as e
    from . import model_layout as m
except ImportError:
    import release_evidence as e
    import model_layout as m

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


def verify_tree(tree, approved_templates, *, trusted_approvals=None, assets_dir=None, family=None, model=None,
                precomputed=None):
    """Return named problems (empty means PASS). All inputs are read-only.

    ``precomputed`` (per-archive approval-listed member inventories) is passed
    through to the asset verifier; the fresh streaming digest of every archive
    is still required at this boundary.
    """
    try:
        tree = e.checked_path(tree)
        family = str(m.model_name(family, 'family'))
        model = str(m.model_name(model))
        version = re.fullmatch(r'v([1-9][0-9]*)', tree.name)
        e.require(version is not None, 'invalid runtime version directory')
        number = int(version[1])
        key = m.registry_key(family, model, number)
        e.require(tree.parent.name == model and tree.parent.parent.name == family,
                  'tree path does not match family/model')
        approval_path = e.safe_file(tree, 'approval.json')
        doc = e.load_json(approval_path)
        e.require(doc.get('schema') == SCHEMA, 'v2 approval required')
        e.require(isinstance(trusted_approvals, dict) and trusted_approvals.get('schema') == 'rexl3.approval-registry.v1',
                  'independently trusted approval registry required')
        entry = trusted_approvals.get('approvals', {}).get(key)
        e.require(isinstance(entry, dict), 'approval not in independently trusted registry: ' + key)
        e.require(entry.get('runtime_id') == doc.get('runtime_id'),
                  'registry runtime_id differs from the sealed approval')
        model_path = e.safe_file(tree.parent, 'MODEL.json')
        model_doc = m.validate_model_doc(m.parse_model_json(model_path), family, model)
        e.require(model_doc.get('runtime_id') == doc.get('runtime_id'),
                  'MODEL.json runtime_id differs from the sealed approval')
        actual = tree_inventory(tree)
        e.require(assets_dir is not None, 'authenticated dev assets directory required')
        asset_root = Path(assets_dir) / family / model / f'v{number}'
        asset_approval = e.safe_file(asset_root, 'approval.json')
        e.require(approval_path.read_bytes() == asset_approval.read_bytes(), 'tree/asset approval inequality')
        approved, _ = e.verify_assets(asset_approval, registry=trusted_approvals, approved_templates=approved_templates,
                                      engine_dir=Path(asset_root) / 'engine', key=key, precomputed=precomputed)
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
    """Return [(family, model, version_dir)] for the family/model layout; refuse the flat layout."""
    runtimes = e.checked_path(Path(root) / 'runtimes')
    if not runtimes.exists():
        return []
    trees = []
    for base in sorted(runtimes.iterdir()):
        e.checked_path(base)
        if base.name in ('templates', 'template-v1', 'models') or not base.is_dir():
            continue
        direct = [entry for entry in base.iterdir()
                  if re.fullmatch(r'v[1-9][0-9]*', entry.name) and entry.is_dir()]
        e.require(not direct, 'old flat runtime layout rejected: runtimes/<runtime-id>/v<N>')
        m.model_name(base.name, 'family')
        for model in sorted(base.iterdir()):
            e.checked_path(model)
            if not model.is_dir():
                e.require(False, 'non-directory entry under family folder: ' + model.name)
            m.model_name(model.name)
            for version in sorted(model.iterdir()):
                e.checked_path(version)
                if re.fullmatch(r'v[1-9][0-9]*', version.name) and version.is_dir():
                    trees.append((base.name, model.name, version))
    return trees


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--approved', required=True, help='protected template allow-list JSON')
    parser.add_argument('--trusted-approvals', required=True)
    parser.add_argument('--trusted-approvals-sha256', required=True)
    parser.add_argument('--assets-dir', required=True, help='asset root containing <family>/<model>/v<N>/approval.json')
    args = parser.parse_args(argv)
    try:
        registry = e.load_registry(args.trusted_approvals, args.trusted_approvals_sha256)
        approved = e.load_json(args.approved)['templates']
        trees = discover(args.root)
        # Zero trees is a pass only while the pinned registry anchors nothing; a registry that anchors a tree
        # which is missing, or a wrong --root, still stops here.
        anchored = bool(registry.get('approvals'))
        e.require(trees or not anchored, 'no runtime trees found but the registry anchors some')
        problems = []
        for family, model, tree in trees:
            problems.extend(verify_tree(tree, approved, trusted_approvals=registry,
                                       assets_dir=args.assets_dir, family=family, model=model))
        for problem in problems:
            print(problem)
        print(f'{len(trees)} tree(s), {len(problems)} problem(s)')
        return 1 if problems else 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print('TREE_REFUSED: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
