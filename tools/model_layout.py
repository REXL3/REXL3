"""Prod model layout: runtimes/<family>/<model>/v<N> and MODEL.json (schema rexl3.prod-model.v1).

One prod folder per political model; pack details (quantization format, bit width, draft/speculation
method) live in MODEL.json and in the sealed approval, never in folder or branch names. Dev release
tags derive from the registry's runtime_id, never from folder names.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

try:
    from . import release_evidence as e
except ImportError:
    import release_evidence as e

MODEL_SCHEMA = 'rexl3.prod-model.v1'
# Section 2: lowercase, `^[a-z0-9]+([.-][a-z0-9]+)*$` for family and model.
MODEL_NAME_RE = re.compile(r'[a-z0-9]+([.-][a-z0-9]+)*\Z')
# Section 3: topic `^[a-z0-9]+(-[a-z0-9]+)*$`.
TOPIC_RE = re.compile(r'[a-z0-9]+(-[a-z0-9]+)*\Z')
VERSION_RE = re.compile(r'v([1-9][0-9]*)\Z')
PRODUCT_VERSION_RE = re.compile(r'v[0-9]+\.[0-9]+\.[0-9]+\Z')

MODEL_REQUIRED = ('schema', 'family', 'model', 'display_name', 'architectures',
                  'engine_family', 'runtime_id', 'pack')


def model_name(value, what='model'):
    e.require(isinstance(value, str) and MODEL_NAME_RE.fullmatch(value),
              f'invalid {what} name: {value!r}')
    return value


def product_version(value):
    e.require(isinstance(value, str) and PRODUCT_VERSION_RE.fullmatch(value), 'invalid product version')
    return value


def registry_key(family, model, version):
    model_name(family, 'family')
    model_name(model)
    e.require(type(version) is int and version > 0, 'positive version required')
    return f'{family}/{model}/v{version}'


def parse_model_json(path):
    doc = e.load_json(path)
    e.require(doc.get('schema') == MODEL_SCHEMA, 'MODEL.json schema must be ' + MODEL_SCHEMA)
    return doc


def validate_model_doc(doc, family, model):
    """Full MODEL.json validation: schema, folder agreement and typed required keys."""
    e.require(isinstance(doc, dict), 'MODEL.json object required')
    e.require(doc.get('schema') == MODEL_SCHEMA, 'MODEL.json schema mismatch')
    model_name(family, 'family')
    model_name(model)
    e.require(doc.get('family') == family, 'MODEL.json family differs from the folder')
    e.require(doc.get('model') == model, 'MODEL.json model differs from the folder')
    e.require(isinstance(doc.get('display_name'), str) and doc['display_name'].strip(),
              'MODEL.json display_name required')
    architectures = doc.get('architectures')
    e.require(isinstance(architectures, list) and architectures
              and all(isinstance(a, str) and a for a in architectures),
              'MODEL.json architectures must be a non-empty list of names')
    e.require(isinstance(doc.get('engine_family'), str) and e.ID.fullmatch(doc['engine_family']),
              'MODEL.json engine_family invalid')
    e.require(isinstance(doc.get('runtime_id'), str) and e.ID.fullmatch(doc['runtime_id']),
              'MODEL.json runtime_id invalid')
    pack = doc.get('pack')
    e.require(isinstance(pack, dict) and isinstance(pack.get('format'), str) and pack['format'].strip(),
              'MODEL.json pack.format required')
    for key in doc:
        if key not in MODEL_REQUIRED:
            e.require(isinstance(doc[key], (str, int, float, bool, list, dict)) and doc[key] is not None,
                      f'MODEL.json extra key {key} has a non-descriptive value')
    return doc


def model_json_doc(*, family, model, display_name, architectures, engine_family, runtime_id, pack):
    for value, what in ((family, 'family'), (model, 'model')):
        model_name(value, what)
    e.require(isinstance(display_name, str) and display_name.strip(), 'display_name required')
    e.require(isinstance(architectures, list) and architectures
              and all(isinstance(a, str) and a for a in architectures), 'architectures required')
    e.require(isinstance(engine_family, str) and e.ID.fullmatch(engine_family), 'engine_family required')
    e.require(isinstance(runtime_id, str) and e.ID.fullmatch(runtime_id), 'runtime_id required')
    e.require(isinstance(pack, dict) and isinstance(pack.get('format'), str) and pack['format'].strip(),
              'pack.format required')
    return dict(schema=MODEL_SCHEMA, family=family, model=model, display_name=display_name,
                architectures=list(architectures), engine_family=engine_family,
                runtime_id=runtime_id, pack=dict(pack))


def write_or_validate_model_json(path, *, family, model, doc):
    """Write MODEL.json when absent; validate bytes for byte when present. Never overwrite."""
    path = e.checked_path(path)
    validate_model_doc(doc, family, model)
    data = e.json_bytes(doc)
    if path.exists():
        e.require(path.read_bytes() == data, 'existing MODEL.json differs from the derived metadata')
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)
    path.chmod(0o644)
    e.require(path.read_bytes() == data, 'MODEL.json write drift')
    return path


def capability_contract(approval, model_doc):
    """Section 6 check 7: speculative decoding on, prefix cache on, vision on unless the tree records a
    does-not-fit justification. The sealed profile is authoritative; the justification lives in MODEL.json."""
    e.require(isinstance(approval, dict) and approval.get('schema') == e.SCHEMA, 'sealed approval required')
    problems = []
    profiles = approval.get('profiles')
    if not isinstance(profiles, dict) or not profiles:
        problems.append('sealed approval has no profiles')
        return problems
    vision_on = any(isinstance(selected.get('identity'), dict)
                    and selected['identity'].get('capabilities', {}).get('images') is True
                    for selected in profiles.values())
    for name, selected in profiles.items():
        if not isinstance(selected, dict):
            problems.append(f'profile {name} is malformed')
            continue
        identity = selected.get('identity')
        if not isinstance(identity, dict):
            problems.append(f'profile {name} has no identity')
            continue
        caps = identity.get('capabilities')
        if not isinstance(caps, dict):
            problems.append(f'profile {name} capabilities missing')
            continue
        if caps.get('prefix_cache') is not True:
            problems.append(f'profile {name}: prefix cache is not on and proven')
    # Speculative decoding is a serving configuration recorded in the sealed profile argv (--draft).
    # The profile json is inside the tree; here the approval's receipts only carry capabilities, so the
    # argv check happens on the off profile (check_model_branch passes it explicitly).
    if not vision_on:
        justification = (model_doc or {}).get('pack', {}).get('vision_does_not_fit')
        if not isinstance(justification, str) or not justification.strip():
            problems.append('vision is off and MODEL.json records no does-not-fit justification')
    return problems
