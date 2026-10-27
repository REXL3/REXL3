#!/usr/bin/env python3
"""Download the authenticated dev release assets that production `verify` needs for each runtime tree.

Runs from the protected base revision only. The candidate tree is read as data: the tool takes the
family/model/version from its directory names, looks up the registry entry the tree claims (the same
entry the verifier anchors it with), derives the dev release tag from that entry's runtime_id
(approve-<runtime_id>-<version>, never from folder names) and never reads a tag, URL or command from
the candidate. Every download goes through promote_runtime.download_release, which checks the archive
hashes against the pinned approval registry and refuses anything else. The GitHub token is read from
the environment by `gh`.
"""
import argparse
import sys
from pathlib import Path
try:
    from . import release_evidence as e
    from . import promote_runtime as p
    from . import model_layout as m
    from .verify_runtime_tree import discover
except ImportError:
    import release_evidence as e
    import promote_runtime as p
    import model_layout as m
    from verify_runtime_tree import discover


def v2_trees(root):
    """(family, model, version) for every tree (approvals are v2 by requirement)."""
    found = []
    for family, model, tree in discover(root):
        number = int(tree.name[1:])
        key = m.registry_key(family, model, number)
        doc = e.load_json(e.safe_file(tree, 'approval.json'))
        e.require(doc.get('schema') == e.SCHEMA, 'legacy v1 approval is not accepted in the family/model layout')
        found.append((family, model, number, key, doc))
    return found


def fetch(root, assets_dir, registry, approved):
    fetched = []
    for family, model, version, key, doc in v2_trees(root):
        entry = registry.get('approvals', {}).get(key)
        e.require(isinstance(entry, dict), 'approval not in independently trusted registry: ' + key)
        runtime_id = entry.get('runtime_id')
        e.require(runtime_id == doc.get('runtime_id'), 'candidate approval runtime_id differs from the registry')
        tag = f'approve-{runtime_id}-{version}'
        e.validate_request(runtime_id, tag, version)
        p.download_release(tag, Path(assets_dir) / family / model / f'v{version}',
                           registry=registry, approved_templates=approved, key=key)
        fetched.append(tag)
    return fetched


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--approved', required=True)
    parser.add_argument('--trusted-approvals', required=True)
    parser.add_argument('--trusted-approvals-sha256', required=True)
    parser.add_argument('--assets-dir', required=True)
    args = parser.parse_args(argv)
    try:
        registry = e.load_registry(args.trusted_approvals, args.trusted_approvals_sha256)
        approved = e.load_json(args.approved)['templates']
        fetched = fetch(args.root, args.assets_dir, registry, approved)
        print(f'{len(fetched)} dev release(s) downloaded and verified: {", ".join(fetched) or "none needed"}')
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        # Content-free on purpose: the message of a failed download can carry command lines.
        print('ASSETS_REFUSED: ' + (str(error) if isinstance(error, ValueError) else type(error).__name__), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
