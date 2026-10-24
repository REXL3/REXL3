#!/usr/bin/env python3
"""Download the authenticated dev release assets that production `verify` needs for each v2 runtime tree.

Runs from the protected base revision only. The candidate tree is read as data: this tool takes the runtime id and
version from its directory names, derives the release tag from them (approve-<id>-<version>, the same rule as
tools/promote_runtime.py validate_request) and never reads a tag, URL or command from the candidate. Every download
goes through promote_runtime.download_release, which checks the archive hashes against the pinned approval registry
and refuses anything else. Legacy v1 trees need no assets. The GitHub token is read from the environment by `gh`.
"""
import argparse
import sys
from pathlib import Path
try:
    from . import release_evidence as e
    from . import promote_runtime as p
    from .verify_runtime_tree import discover
except ImportError:
    import release_evidence as e
    import promote_runtime as p
    from verify_runtime_tree import discover


def v2_trees(root):
    """(runtime_id, version) for every tree whose approval is not the legacy v1 schema."""
    found = []
    for tree in discover(root):
        doc = e.load_json(e.safe_file(tree, 'approval.json'))
        if doc.get('schema') == 'rexl3.profile-approval.v1':
            continue
        found.append((tree.parent.name, int(tree.name[1:])))
    return found


def fetch(root, assets_dir, registry, approved):
    fetched = []
    for runtime_id, version in v2_trees(root):
        tag = f'approve-{runtime_id}-{version}'
        e.validate_request(runtime_id, tag, version)
        p.download_release(tag, Path(assets_dir) / runtime_id / f'v{version}', registry=registry, approved_templates=approved)
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
