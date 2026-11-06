#!/usr/bin/env python3
"""Seal actual qualification receipts into v2 approval; never publish or run native code.

Output is a candidate for independent maintainer review, NOT its own trust anchor.
All assets/receipts must be sibling-contained beneath --evidence-root. CLI copies
only the bound bytes to durable --out-dir and prints explicit publication commands.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shlex
import sys

try:
    from . import release_evidence as e
    from .runtime_template_digest import verified_identity
    from .corresponding_source_identity import verify_git_projection
except ImportError:
    import release_evidence as e
    from runtime_template_digest import verified_identity
    from corresponding_source_identity import verify_git_projection

SCHEMA = e.SCHEMA
sha256_file = e.sha256_file


def engine_records(files, tag):
    """``engine_binaries`` for an approval whose bundle leaves the native executables to an engine release.

    ``files`` maps an install path (bin/rexl3-serve) to the engine release asset on disk, named exactly as the release
    names it; ``tag`` is that release's tag. Returns the records and the directory holding the files."""
    e.require(isinstance(files, dict) and files and isinstance(tag, str), 'engine binary files and release tag are required together')
    paths = {install: e.checked_path(path) for install, path in files.items()}
    directories = {path.parent for path in paths.values()}
    e.require(len(directories) == 1, 'engine binary files must share one directory')
    records = {install: dict(release_tag=tag, asset=path.name, sha256=e.sha256_file(path), bytes=path.stat().st_size, mode=0o755)
               for install, path in sorted(paths.items())}
    return e.validate_engine_binaries(records), directories.pop()


def build_approval(*, runtime_id, commit, tree, template_dir, archive, source_asset, engine_source_asset,
                   receipts, evidence_root, repo, version, approved_by, engine_binary_files=None, engine_release_tag=None):
    e.validate_request(runtime_id, f'approve-{runtime_id}-{version}', version)
    e.require(isinstance(approved_by, str) and approved_by.strip(), 'named approver required')
    repo = e.checked_path(repo)
    e.git_identity(repo, commit, tree)
    engine_inventory = verify_git_projection(engine_source_asset, repo=repo, commit=commit, tree=tree)
    template = verified_identity(template_dir, repo=repo, commit=commit)
    engine_binaries = engine_dir = None
    if engine_binary_files is not None or engine_release_tag is not None:
        engine_binaries, engine_dir = engine_records(engine_binary_files, engine_release_tag)
    profile, identities, inventory, source_inventory = e.bundle_identity(
        archive, source_asset, runtime_id=runtime_id, commit=commit, tree=tree,
        template=template, version=version, engine_binaries=engine_binaries, engine_dir=engine_dir)
    # Source declarations must name actual local Git objects, not 40-hex inventions.
    for filename in ('profile.json', 'model.lock.json'):
        raw = e.git_file(repo, commit, f'runtimes/models/{runtime_id}/{filename}')
        e.require(hashlib.sha256(raw).hexdigest() == inventory[filename]['sha256'], 'profile/model not from source commit')
    source = profile.get('source')
    e.require(isinstance(source, dict) and source.get('repository') == 'REXL3/REXL3-dev', 'dev profile source required')
    e.git_identity(repo, source.get('commit'))
    raw = e.git_file(repo, source['commit'], source.get('path'))
    e.require(hashlib.sha256(raw).hexdigest() == e.full_hash(source.get('sha256')), 'profile source asset mismatch')
    root, selected = e.checked_path(evidence_root), {}
    e.require(isinstance(receipts, (list, tuple)) and receipts, 'actual qualification receipt files required')
    for path in receipts:
        path = e.checked_path(path)
        e.require(path.is_relative_to(root), 'receipt must be beneath evidence root')
        receipt = e.load_json(path)
        e.validate_identity(receipt)
        name = receipt['profile']
        e.require(name in profile['profiles'] and name not in selected, 'unknown/duplicate receipt profile')
        choice = profile['profiles'][name]
        context, ranks = e.profile_context(choice, archive)
        expected = dict(runtime_id=choice.get('runtime_id', runtime_id), profile=name,
            pack=choice.get('pack', 'target'), engine_commit=commit, engine_tree=tree,
            effective_profile_digest=receipt['effective_profile_digest'], effective_profile_source='owned-launcher-observation',
            effective_max_context=context, capabilities=choice['capabilities'], topology={'ranks': ranks},
            **{k: v for k, v in identities.items() if k != 'source_manifest_sha256'})
        e.validate_receipt(path, evidence_root=root, expected=expected)
        observation = e.load_json(e.safe_file(root, receipt['identity_receipt']['path']))
        e.validate_observed_profile(observation, choice)
        selected[name] = dict(identity=expected, receipt={'path': path.relative_to(root).as_posix(), 'sha256': e.sha256_file(path)})
        carried = e.carried_gates(receipt)
        if carried:
            selected[name]['carried_gates'] = carried
            selected[name]['rerun_gates'] = e.rerun_gates(receipt, carried)
    e.require(set(selected) == set(profile['profiles']), 'every packaged profile requires qualification')
    approval = dict(schema=SCHEMA, runtime_id=runtime_id, version=version, approve_tag=f'approve-{runtime_id}-{version}',
        source_commit=commit, source_tree=tree, template=template, identities=identities, profiles=selected,
        bundle=e.asset_record(archive, inventory), source_asset=e.asset_record(source_asset, source_inventory),
        engine_source_asset=e.asset_record(engine_source_asset, engine_inventory),
        approved_by=approved_by, approved_at=datetime.now(timezone.utc).isoformat())
    if engine_binaries is not None:
        approval['engine_binaries'] = engine_binaries
    approval['evidence_asset'] = e.evidence_asset(approval, root)
    e.approval_assets(approval)  # all names distinct, including reserved transport/state names
    return approval


def evidence_paths(approval, root):
    """Exact sibling evidence set; approval/archives cannot refer to themselves."""
    return set(e.receipt_files(approval, root))


def seal_to_directory(approval, *, archive, source_asset, engine_source_asset, evidence_root, out_dir):
    """Exclusive output files; exact-byte retries reuse existing files, never overwrite."""
    out = e.checked_path(out_dir)
    e.approval_assets(approval)
    sources = {}
    for key, path in (('bundle', archive), ('source_asset', source_asset), ('engine_source_asset', engine_source_asset)):
        # Do not copy changed bytes under a previously validated seal.
        e.require(e.asset_record(path) == approval[key], 'seal input asset drift: ' + key)
        sources[approval[key]['archive']] = Path(path)
    for rel in evidence_paths(approval, evidence_root):
        sources[rel] = e.safe_file(evidence_root, rel)
    e.require(e.receipt_files(approval, evidence_root) == approval['evidence_asset']['members'], 'seal evidence drift')
    expected = {asset['archive']: asset for asset in e.approval_assets(approval) if asset != approval['evidence_asset']}
    expected.update(approval['evidence_asset']['members'])
    e.require(set(sources) == set(expected), 'unhandled seal asset')
    # Preflight every existing target before any copy. Inputs are already validated.
    for rel, source in sources.items():
        target = e.checked_path(out / rel)
        if target.exists():
            e.require(e.sha256_file(source) == e.sha256_file(target), 'output collision: ' + rel)
    approval_path = e.checked_path(out / 'approval.json')
    data = e.json_bytes(approval)
    if approval_path.exists():
        previous = e.load_json(approval_path)
        e.require({k: v for k, v in previous.items() if k != 'approved_at'} ==
                  {k: v for k, v in approval.items() if k != 'approved_at'}, 'existing seal differs')
        data = approval_path.read_bytes()
    out.mkdir(parents=True, exist_ok=True)
    transport = e.checked_path(out / 'evidence.tar.gz')
    if transport.exists():
        e.require(e.asset_record(transport) == approval['evidence_asset'], 'existing evidence transport differs')
    else:
        e.write_evidence_archive(evidence_root, approval['evidence_asset']['members'], transport)
        e.require(e.asset_record(transport) == approval['evidence_asset'], 'evidence transport copy drift')
    for rel, source in sources.items():
        target = out / rel
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open('rb') as incoming, target.open('xb') as stream:
                import shutil
                shutil.copyfileobj(incoming, stream)
            target.chmod(0o644)
        e.require(e.sha256_file(target) == expected[rel]['sha256'] and target.stat().st_size == expected[rel]['bytes'],
                  'seal copy drift: ' + rel)
    if not approval_path.exists():
        with approval_path.open('xb') as stream:
            stream.write(data)
        approval_path.chmod(0o644)
    return approval_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-id', required=True)
    parser.add_argument('--target-commit', '--commit', dest='commit', required=True)
    parser.add_argument('--tree', required=True)
    parser.add_argument('--repo', default='.')
    parser.add_argument('--version', required=True, type=int)
    parser.add_argument('--template-dir', required=True)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--source-asset', required=True, help='runtime wrapper projection (not engine source)')
    parser.add_argument('--engine-source-asset', required=True, help='exact Git-bound native engine corresponding-source archive')
    parser.add_argument('--engine-binary', action='append', default=[], metavar='INSTALL_PATH=FILE',
                        help='native executable supplied by an engine release instead of the bundle, e.g. '
                             'bin/rexl3-serve=/dir/rexl3-serve-0.1.2-4208d848-linux-aarch64-sm121 (file named as the release asset)')
    parser.add_argument('--engine-release-tag', help='tag of the engine release that holds the --engine-binary assets')
    parser.add_argument('--receipt', action='append', required=True)
    parser.add_argument('--evidence-root', required=True)
    parser.add_argument('--approved-by', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--no-publish', action='store_true', help='explicit offline mode (publication is always print-only)')
    args = parser.parse_args(argv)
    try:
        engine_files = None
        if args.engine_binary or args.engine_release_tag:
            pairs = [item.partition('=') for item in args.engine_binary]
            e.require(args.engine_binary and args.engine_release_tag and all(sep and key and value for key, sep, value in pairs)
                      and len({key for key, _, _ in pairs}) == len(pairs),
                      '--engine-binary INSTALL_PATH=FILE and --engine-release-tag are required together')
            engine_files = {key: value for key, _, value in pairs}
        doc = build_approval(runtime_id=args.runtime_id, commit=args.commit, tree=args.tree, template_dir=args.template_dir,
            archive=args.archive, source_asset=args.source_asset, engine_source_asset=args.engine_source_asset,
            receipts=args.receipt, evidence_root=args.evidence_root,
            repo=args.repo, version=args.version, approved_by=args.approved_by,
            engine_binary_files=engine_files, engine_release_tag=args.engine_release_tag)
        path = seal_to_directory(doc, archive=args.archive, source_asset=args.source_asset,
                                 engine_source_asset=args.engine_source_asset,
                                 evidence_root=args.evidence_root, out_dir=args.out_dir)
        # Re-read the seal and every bound file before reporting a durable result.
        # Carry-forward receipts reference a registry-pinned prior approval, so seed the
        # self-check registry with the repo's pinned approvals (when present) as well as
        # this approval's own entry.
        registry = {'schema': 'rexl3.approval-registry.v1', 'approvals': {}}
        registry_path = Path(args.repo) / 'runtimes/templates/APPROVALS.json'
        if registry_path.is_file():
            try:
                prior = e.load_json(registry_path)
            except (ValueError, OSError):
                prior = None
            if isinstance(prior, dict) and isinstance(prior.get('approvals'), dict):
                registry['approvals'].update(prior['approvals'])
        registry['approvals'][f'{args.runtime_id}/v{args.version}'] = {
            'approval_sha256': e.sha256_file(path), 'source_commit': args.commit, 'source_tree': args.tree}
        engine_dir = engine_records(engine_files, args.engine_release_tag)[1] if engine_files else None
        e.verify_assets(path, registry=registry, approved_templates={doc['template']['version']: doc['template']['digest']},
                        engine_dir=engine_dir)
        assets = [path] + [path.parent / asset['archive'] for asset in e.approval_assets(doc)]
        # Nested evidence is transported as a separate tar; GitHub assets cannot preserve directories.
        # promote_runtime also accepts a local authenticated asset directory after unpacking evidence.
        print('SEALED_NOT_PUBLISHED ' + str(path))
        print('approval_sha256=' + e.sha256_file(path))
        print('Review the immutable registry entry before promotion. Publication command (not executed):')
        print(shlex.join(['gh', 'release', 'create', doc['approve_tag'], *map(str, assets), '-R', 'REXL3/REXL3-dev',
                          '--target', args.commit, '--title', 'approve ' + args.runtime_id]))
        print('evidence.tar.gz carries the exact bound sibling receipt/observation/raw files.')
        for install, item in doc.get('engine_binaries', {}).items():
            print(f"engine binary {install} is not a release asset here: it comes from engine release {item['release_tag']} "
                  f"(asset {item['asset']}, sha256 {item['sha256']}); that release must exist before this one is promoted.")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print('APPROVAL_REFUSED: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
