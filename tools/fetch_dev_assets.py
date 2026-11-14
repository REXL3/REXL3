#!/usr/bin/env python3
"""Download authenticated dev assets for changed production runtime trees.

Protected base code treats the candidate as inert data. Legacy tags are derived
from validated registry runtime ids; unified engine-v<X.Y.Z>-<8hex> tags come only
from a schema-validated registry approve_tag. With --base, unchanged tree bytes
are skipped, and a new registry anchor is refused before any download unless the
pull request is a release/v<X.Y.Z> branch of the base repository whose manifest lists
it (--base-repo, --head-repo, --head-ref; tools/release_anchor.py). Changed downloads run concurrently and share transport files through
a thread-safe cache, while every model is independently authenticated and verified.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path
try:
    from . import release_evidence as e
    from . import promote_runtime as p
    from . import model_layout as m
    from . import release_anchor
    from .verify_runtime_tree import discover, local_metadata_problems, tree_inventory, tree_key
except ImportError:
    import release_evidence as e
    import promote_runtime as p
    import model_layout as m
    import release_anchor
    from verify_runtime_tree import discover, local_metadata_problems, tree_inventory, tree_key


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


def unchanged_tree(tree, base):
    if base is None:
        return False
    previous = Path(base) / 'runtimes' / tree.parent.parent.name / tree.parent.name / tree.name
    return previous.is_dir() and tree_inventory(previous) == tree_inventory(tree)


def fetch(root, assets_dir, registry, approved, *, base=None, full=False, candidate_registry=None,
          base_repo=None, head_repo=None, head_ref=None):
    jobs = []
    cache = p.ReleaseAssetCache(Path(assets_dir) / '.shared')
    rows = v2_trees(root)
    active_registry = candidate_registry if candidate_registry is not None else registry
    new_keys = set()
    trusted_source_registry = None
    if base is not None:
        base_registry = e.load_registry(Path(base) / 'runtimes/templates/APPROVALS.json')
        e.require(registry.get('schema') == 'rexl3.approval-registry.v1'
                  and all(base_registry.get(section, {}) == registry.get(section, {})
                          for section in ('approvals', 'legacy')),
                  'trusted approval registry differs from the protected base checkout')
        disk_candidate = e.load_registry(Path(root) / 'runtimes/templates/APPROVALS.json')
        if candidate_registry is not None:
            e.require(candidate_registry.get('schema') == 'rexl3.approval-registry.v1'
                      and all(candidate_registry.get(section, {}) == disk_candidate.get(section, {})
                              for section in ('approvals', 'legacy')),
                      'candidate registry argument differs from the candidate checkout')
        active_registry = disk_candidate
        discovered = {tree_key(family, model, Path(root) / 'runtimes' / family / model / f'v{version}')
                      for family, model, version, _, _ in rows}
        new_keys = e.validate_registry_transition(base_registry, disk_candidate, discovered)
        # Before any download: a new anchor must arrive in a release PR of this repository that lists it.
        release_anchor.require_release_pr(new_keys, disk_candidate, root=root, base=base,
                                          base_repo=base_repo, head_repo=head_repo, head_ref=head_ref)
        metadata = local_metadata_problems(
            [(family, model, Path(root) / 'runtimes' / family / model / f'v{version}')
             for family, model, version, _, _ in rows], active_registry)
        e.require(not metadata, 'candidate local metadata refused: ' + '; '.join(metadata))
        trusted_source_registry = base_registry
    for family, model, version, key, doc in rows:
        tree = Path(root) / 'runtimes' / family / model / f'v{version}'
        if base is not None and not full and key not in new_keys and unchanged_tree(tree, base):
            continue
        entry = active_registry.get('approvals', {}).get(key)
        e.require(isinstance(entry, dict), 'approval not in independently trusted registry: ' + key)
        e.validate_registry_entry(key, entry)
        runtime_id = entry.get('runtime_id')
        e.require(runtime_id == doc.get('runtime_id'), 'candidate approval runtime_id differs from the registry')
        tag = entry.get('approve_tag') or f'approve-{runtime_id}-{version}'
        e.validate_request(runtime_id, tag, version)
        if candidate_registry is not None or e.is_engine_tag(tag):
            # Fail before network access on malformed candidate anchors and bind the
            # candidate approval bytes/source identity to the selected registry entry.
            e.trusted_entry(e.safe_file(tree, 'approval.json'), active_registry, key=key)
        jobs.append((family, model, version, key, tag, active_registry))

    def download(job):
        family, model, version, key, tag, active_registry = job
        kwargs = dict(registry=active_registry, approved_templates=approved, key=key)
        if trusted_source_registry is not None:
            kwargs['trusted_source_registry'] = trusted_source_registry
        if e.is_engine_tag(tag):
            kwargs['cache'] = cache
        p.download_release(tag, Path(assets_dir) / family / model / f'v{version}', **kwargs)
        return tag

    if not jobs:
        return []
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(download, jobs))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--approved', required=True)
    parser.add_argument('--trusted-approvals', required=True)
    parser.add_argument('--trusted-approvals-sha256')
    parser.add_argument('--assets-dir', required=True)
    parser.add_argument('--base', help='protected base checkout for diff-scoped verification')
    parser.add_argument('--full', action='store_true', help='download every tree even when unchanged')
    release_anchor.add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        registry = e.load_registry(args.trusted_approvals, args.trusted_approvals_sha256)
        candidate_registry = None
        if args.base is not None:
            candidate_registry = e.load_registry(Path(args.root) / 'runtimes/templates/APPROVALS.json')
        approved = e.load_json(args.approved)['templates']
        fetched = fetch(args.root, args.assets_dir, registry, approved,
                        base=args.base, full=args.full, candidate_registry=candidate_registry,
                        **release_anchor.context(args))
        print(f'{len(fetched)} dev release asset set(s) downloaded and verified: {", ".join(fetched) or "none needed"}')
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        # Content-free on purpose: the message of a failed download can carry command lines.
        print('ASSETS_REFUSED: ' + (str(error) if isinstance(error, ValueError) else type(error).__name__), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
