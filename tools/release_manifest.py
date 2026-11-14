"""Release manifest schema (rexl3.release-manifest.v1) shared by notes assembly and cut_release.

    python3 tools/release_manifest.py --repo-root PATH --version vX.Y.Z --source-archives

prints the optional `engine.source_archives` block for the models the manifest lists: one shared
engine corresponding-source archive per distinct sealed digest (docs/RELEASING.md). Read-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

try:
    from . import release_evidence as e
    from . import model_layout as m
except ImportError:
    import release_evidence as e
    import model_layout as m

MANIFEST_SCHEMA = 'rexl3.release-manifest.v1'
MODEL_KEYS = ('family', 'model', 'version', 'runtime_id', 'approval_sha256')
ENGINE_KEYS = ('dev_commit', 'dev_tree', 'engine_release_tag', 'artifacts')
# Change-scoped carry-forward: a model whose family/kernel scope is untouched by the
# newer engine commits in this release carries forward its prior sealed approval instead of re-running
# a full qualification chain. T0 is the only prod carry-forward tier (the prior sealed approval travels
# as-is in the manifest); T1/T2 re-run the required gates on the new engine binary and seal a NEW
# approval on the new engine build, so they are manifest carry-forwards. T3 (model
# math/weights/sampling changes or a new model) never carries forward and must re-seal. The qualified
# engine release tag must be the exact dev engine release the carried approval was sealed against.
QUALIFICATION_TIERS = ('T0',)
CARRY_FORWARD_KEYS = ('qualified_engine_release_tag', 'qualification_tier', 'scope_diff_base')
# Optional `engine.source_archives`: {shared asset name: {sha256, bytes}}. A manifest that carries it
# publishes each named engine corresponding-source archive once, in place of one copy per model.
# Every model's sealed approval must then name an engine source with exactly one entry's digest and
# size. A manifest without the key keeps the per-model engine-source assets (v0.1.0, v0.1.1).
SOURCE_ARCHIVES = 'source_archives'
# The name release_evidence.shared_engine_source_name gives an engine-v<X.Y.Z>-<8hex> release.
SOURCE_ARCHIVE_NAME = re.compile(r'rexl3-engine-source-[0-9]+\.[0-9]+\.[0-9]+-[0-9a-f]{8}\.tar\.gz')


def validate_manifest(doc, version):
    e.require(isinstance(doc, dict), 'release manifest object required')
    e.require(doc.get('schema') == MANIFEST_SCHEMA, 'release manifest schema mismatch')
    m.product_version(version)
    e.require(doc.get('version') == version, 'manifest version differs from the release version')
    engine = doc.get('engine')
    e.require(isinstance(engine, dict), 'manifest engine required')
    e.require(all(key in engine for key in ENGINE_KEYS), 'incomplete manifest engine')
    e.full_hash(engine['dev_commit'], e.HEX40)
    e.full_hash(engine['dev_tree'], e.HEX40)
    e.require(isinstance(engine['engine_release_tag'], str) and engine['engine_release_tag'], 'engine release tag required')
    artifacts = engine['artifacts']
    e.require(isinstance(artifacts, dict) and artifacts, 'engine artifact digests required')
    for name, digest in artifacts.items():
        e.relative_path(name)
        e.full_hash(digest)
    if SOURCE_ARCHIVES in engine:
        sources = engine[SOURCE_ARCHIVES]
        e.require(isinstance(sources, dict) and sources, 'engine source archives must be a non-empty object')
        digests = set()
        for name, record in sources.items():
            e.require(isinstance(name, str) and SOURCE_ARCHIVE_NAME.fullmatch(name) and name not in artifacts,
                      'invalid shared engine-source asset name')
            e.require(isinstance(record, dict) and set(record) == {'sha256', 'bytes'},
                      'invalid engine source archive record: ' + name)
            e.full_hash(record['sha256'])
            e.require(type(record['bytes']) is int and record['bytes'] > 0, 'invalid engine source archive size: ' + name)
            e.require(record['sha256'] not in digests, 'engine source archives repeat a digest: ' + name)
            digests.add(record['sha256'])
    # The primary engine release tag names the engine dev commit: its eight-hex commit
    # suffix must be exactly the dev_commit prefix, so the tag cannot name a different
    # commit than the manifest's engine source.
    e.require(engine['dev_commit'].startswith(engine['engine_release_tag'][-8:]),
              'engine dev_commit prefix must match the engine release tag commit')
    models = doc.get('models')
    e.require(isinstance(models, list) and models, 'manifest models required')
    for item in models:
        e.require(isinstance(item, dict) and all(key in item for key in MODEL_KEYS), 'incomplete model entry')
        m.model_name(item['family'], 'family')
        m.model_name(item['model'])
        e.require(isinstance(item['version'], str) and m.VERSION_RE.fullmatch(item['version']), 'invalid model version')
        e.require(isinstance(item['runtime_id'], str) and e.ID.fullmatch(item['runtime_id']), 'invalid runtime_id')
        e.full_hash(item['approval_sha256'])
        carried = item.get('carried_forward')
        if 'carried_forward' in item:
            e.require(isinstance(carried, bool), 'carried_forward must be a boolean')
        if carried is True:
            e.require(all(key in item for key in CARRY_FORWARD_KEYS), 'incomplete carried-forward model entry')
            e.require(isinstance(item['qualified_engine_release_tag'], str)
                      and e.ENGINE_TAG.fullmatch(item['qualified_engine_release_tag']), 'invalid qualified engine release tag')
            e.require(item['qualification_tier'] in QUALIFICATION_TIERS, 'invalid qualification tier')
            e.full_hash(item['scope_diff_base'], e.HEX40)
        else:
            e.require(not any(key in item for key in CARRY_FORWARD_KEYS), 'carry-forward fields require carried_forward: true')
    seen = set()
    for item in models:
        pair = (item['family'], item['model'])
        e.require(pair not in seen, 'duplicate (family, model) entry: ' + '/'.join(pair))
        seen.add(pair)
    e.require(isinstance(doc.get('notes'), str) and doc['notes'].strip(), 'manifest notes required')
    return doc


def manifest_models(manifest):
    """The manifest as a {version: {model: entry}} map."""
    by_version = {}
    for item in manifest['models']:
        by_version.setdefault(item['version'], {})[item['model']] = item
    return by_version


def approval_engine_tag(approval):
    """The dev engine release a sealed approval was built on: its unified tag, or the one its engine binaries name."""
    tag = approval.get('approve_tag')
    if not e.is_engine_tag(tag):
        engine = approval.get('engine_binaries')
        e.require(engine is not None, 'approval names no engine release, so its engine source cannot be shared')
        tag = next(iter(e.validate_engine_binaries(engine).values()))['release_tag']
    e.full_hash(approval.get('source_commit'), e.HEX40)
    e.require(tag.endswith('-' + approval['source_commit'][:8]), 'engine release tag does not name the approval commit')
    return tag


def shared_source_record(approval):
    """(shared asset name, {sha256, bytes}) of one sealed approval's engine corresponding source."""
    asset = approval['engine_source_asset']
    e.full_hash(asset.get('sha256'))
    e.require(type(asset.get('bytes')) is int and asset['bytes'] > 0, 'invalid engine source size')
    return e.shared_engine_source_name(approval_engine_tag(approval)), {'sha256': asset['sha256'], 'bytes': asset['bytes']}


def shared_source_name(approval, sources):
    """The manifest's shared archive for one approval: same engine release name, same digest, same size."""
    name, record = shared_source_record(approval)
    e.require(sources.get(name) == record, 'engine source differs from the shared archive the manifest names: ' + name)
    return name


def _manifest_approvals(manifest, repo_root):
    root = Path(repo_root)
    for item in manifest['models']:
        path = root / 'runtimes' / item['family'] / item['model'] / item['version'] / 'approval.json'
        # The manifest pins the approval bytes; a different local file must not choose the asset set.
        e.require(e.sha256_file(path) == item['approval_sha256'],
                  f"local approval bytes differ from the manifest: {item['family']}/{item['model']} {item['version']}")
        approval = e.load_json(path)
        e.approval_assets(approval)
        yield (item['family'], item['model'], item['version']), approval


def shared_engine_sources(manifest, repo_root):
    """{(family, model, version): shared asset name}; empty for a manifest without `engine.source_archives`.

    With the key, every model must map to one entry by digest and size, and every entry must be used.
    """
    sources = manifest['engine'].get(SOURCE_ARCHIVES)
    if sources is None:
        return {}
    e.require(repo_root is not None, 'shared engine source archives need the production tree')
    mapped = {key: shared_source_name(approval, sources) for key, approval in _manifest_approvals(manifest, repo_root)}
    unused = set(sources) - set(mapped.values())
    e.require(not unused, 'manifest names an engine source archive no model was sealed from: ' + ', '.join(sorted(unused)))
    return mapped


def derive_source_archives(manifest, repo_root):
    """The `engine.source_archives` block for the manifest's models: one entry per distinct sealed digest."""
    sources = {}
    for _, approval in _manifest_approvals(manifest, repo_root):
        name, record = shared_source_record(approval)
        e.require(sources.setdefault(name, record) == record,
                  'approvals name the same engine source archive with different bytes: ' + name)
    return sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', default='.')
    parser.add_argument('--version', required=True)
    parser.add_argument('--source-archives', action='store_true', required=True,
                        help='print the engine.source_archives block derived from the sealed approvals')
    args = parser.parse_args(argv)
    try:
        path = Path(args.repo_root) / 'releases' / args.version / 'manifest.json'
        manifest = validate_manifest(e.load_json(path), args.version)
        print(json.dumps({SOURCE_ARCHIVES: derive_source_archives(manifest, args.repo_root)}, indent=2, sort_keys=True))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print('MANIFEST_REFUSED: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
