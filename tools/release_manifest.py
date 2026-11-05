"""Release manifest schema (rexl3.release-manifest.v1) shared by notes assembly and cut_release."""
from __future__ import annotations

try:
    from . import release_evidence as e
    from . import model_layout as m
except ImportError:
    import release_evidence as e
    import model_layout as m

MANIFEST_SCHEMA = 'rexl3.release-manifest.v1'
MODEL_KEYS = ('family', 'model', 'version', 'runtime_id', 'approval_sha256')
ENGINE_KEYS = ('dev_commit', 'dev_tree', 'engine_release_tag', 'artifacts')
# Change-scoped carry-forward (decision D37): a model whose family/kernel scope is untouched by the
# newer engine commits in this release carries forward its prior sealed approval instead of re-running
# a full qualification chain (T0/T1/T2). T3 (model math/weights/sampling changes or a new model) never
# carries forward and must re-seal. The qualified engine release tag must be the exact dev engine
# release the carried approval was sealed against.
QUALIFICATION_TIERS = ('T0', 'T1', 'T2')
CARRY_FORWARD_KEYS = ('qualified_engine_release_tag', 'qualification_tier', 'scope_diff_base')


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
        if carried is True:
            e.require(all(key in item for key in CARRY_FORWARD_KEYS), 'incomplete carried-forward model entry')
            e.require(isinstance(item['qualified_engine_release_tag'], str)
                      and e.ENGINE_TAG.fullmatch(item['qualified_engine_release_tag']), 'invalid qualified engine release tag')
            e.require(item['qualification_tier'] in QUALIFICATION_TIERS, 'invalid qualification tier')
            e.full_hash(item['scope_diff_base'], e.HEX40)
        else:
            e.require(carried is None, 'carried_forward must be true when present')
            e.require(not any(key in item for key in CARRY_FORWARD_KEYS), 'carry-forward fields require carried_forward: true')
    e.require(isinstance(doc.get('notes'), str) and doc['notes'].strip(), 'manifest notes required')
    return doc


def manifest_models(manifest):
    """The manifest as a {version: {model: entry}} map."""
    by_version = {}
    for item in manifest['models']:
        by_version.setdefault(item['version'], {})[item['model']] = item
    return by_version
