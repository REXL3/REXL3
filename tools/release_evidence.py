"""Stdlib release evidence boundary. Hash integrity is NOT evidence provenance.

Only authenticated dev assets plus an independently reviewed immutable approval
registry authorize promotion. Never execute code from incoming bundles.
"""
from __future__ import annotations

import hashlib
import gzip
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import subprocess
import tarfile
import tempfile

SCHEMA = 'rexl3.profile-approval.v2'
RECEIPT_SCHEMA = 'rexl3.qualification.v1'
GATES = ('doctor', 'lifecycle', 'capability', 'vision', 'context', 'exactness',
         'prefix_cache', 'performance', 'quality')
IDENTITY_KEYS = ('runtime_id', 'profile', 'engine_commit', 'engine_tree', 'binary_sha256',
                 'profile_sha256', 'model_lock_sha256', 'release_lock_sha256',
                 'effective_profile_digest', 'effective_profile_source', 'effective_max_context', 'capabilities', 'topology')
HEX40 = re.compile(r'[0-9a-f]{40}\Z')
HEX64 = re.compile(r'[0-9a-f]{64}\Z')
ID = re.compile(r'[a-z0-9][a-z0-9._-]{0,127}\Z')
MAX_FILES, MAX_BYTES, MAX_FILE_BYTES = 10000, 8 * 1024**3, 2 * 1024**3
MAX_JSON_BYTES = 16 * 1024**2
# Native executables a runtime may take from an engine release instead of carrying them in its bundle.
ENGINE_PATHS = ('bin/rexl3-serve', 'bin/rexl3-cluster')
ENGINE_TAG = re.compile(r'engine-v[0-9]+\.[0-9]+\.[0-9]+-[0-9a-f]{8}\Z')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def full_hash(value, pattern=HEX64):
    require(isinstance(value, str) and pattern.fullmatch(value), 'invalid full hash')
    return value


def relative_path(value):
    require(isinstance(value, str) and bool(value) and len(value) <= 4096, 'invalid relative path')
    require(not value.startswith('/') and '\\' not in value and ':' not in value,
            'unsafe relative path: ' + value)
    require(all(p not in ('', '.', '..', '.git') for p in value.split('/'))
            and not any(ord(c) < 32 or ord(c) == 127 for c in value), 'unsafe relative path: ' + value)
    return value


def checked_path(path):
    path = Path(os.path.abspath(path))
    for entry in reversed((path, *path.parents)):
        if entry.exists() or entry.is_symlink():
            mode = entry.lstat().st_mode
            require(not stat.S_ISLNK(mode), 'symlink refused: ' + str(entry))
            require(stat.S_ISDIR(mode) or entry == path and stat.S_ISREG(mode), 'special path refused')
    return path


def safe_file(root, rel):
    relative_path(rel)
    root = checked_path(root)
    path = checked_path(root / rel)
    require(path.is_file() and path.stat().st_nlink == 1, 'missing/nonregular/hardlinked file: ' + rel)
    return path


def sha256_file(path):
    path = checked_path(path)
    require(path.is_file() and path.stat().st_nlink == 1, 'regular single-link file required')
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def loads(data):
    require(len(data) <= MAX_JSON_BYTES, 'JSON size limit')
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON: ' + value)
    value = json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)
    require(isinstance(value, dict), 'JSON object required')
    json.dumps(value, allow_nan=False)  # Also refuses exponent overflow (1e999).
    return value


def load_json(path):
    p = checked_path(path)
    safe_file(p.parent, p.name)
    require(p.stat().st_size <= MAX_JSON_BYTES, 'JSON size limit')
    return loads(p.read_bytes())


def record(path):
    p = checked_path(path)
    return dict(sha256=sha256_file(p), bytes=p.stat().st_size, mode=stat.S_IMODE(p.stat().st_mode))


def validate_record(item):
    require(isinstance(item, dict) and set(item) == {'sha256', 'bytes', 'mode'}, 'invalid file record')
    full_hash(item['sha256'])
    require(type(item['bytes']) is int and item['bytes'] >= 0, 'invalid byte count')
    require(type(item['mode']) is int and item['mode'] in (0o644, 0o755), 'invalid mode')


def validate_identity(doc):
    require(isinstance(doc, dict), 'identity object required')
    require(all(key in doc for key in IDENTITY_KEYS), 'incomplete receipt identity')
    for key in ('runtime_id', 'profile'):
        require(isinstance(doc[key], str) and ID.fullmatch(doc[key]), 'invalid ' + key)
    for key in ('engine_commit', 'engine_tree'):
        full_hash(doc[key], HEX40)
    for key in ('binary_sha256', 'profile_sha256', 'model_lock_sha256', 'release_lock_sha256'):
        full_hash(doc[key])
    value = doc['effective_profile_digest']
    require(isinstance(value, str) and value.startswith('sha256:'), 'effective digest required')
    full_hash(value[7:])
    require(doc['effective_profile_source'] == 'owned-launcher-observation', 'owned launcher observation required')
    require(type(doc['effective_max_context']) is int and doc['effective_max_context'] > 0, 'invalid effective context')
    caps = doc['capabilities']
    require(isinstance(caps, dict) and caps and all(isinstance(k, str) and type(v) is bool for k, v in caps.items()),
            'typed capabilities required')
    topology = doc['topology']
    require(isinstance(topology, dict) and set(topology) == {'ranks'} and type(topology['ranks']) is int
            and topology['ranks'] in (1, 2), 'invalid topology')


def validate_observation(path, expected, *, choice=None):
    """Validate v1 observed bytes (or an already parsed object), never synthesize them.

    Pass the pinned profile choice to enforce its serving contract as well as the
    receipt identity. The same pure checks serve qualification, sealing and retry
    verification; this function neither launches nor probes a process.
    """
    observation = path if isinstance(path, dict) else load_json(path)
    _validate_observation(observation, expected)
    if choice is not None:
        _validate_observed_profile(observation, choice)
    return observation


def _validate_observation(observation, expected):
    require(isinstance(observation, dict), 'launcher observation object required')
    require(observation.get('schema') == 'rexl3.launcher-observation.v1', 'launcher observation schema required')
    validate_identity(observation)
    validate_identity(expected)
    require(all(observation[k] == expected[k] for k in IDENTITY_KEYS), 'observation identity drift')
    resolved = observation.get('resolved_config')
    require(isinstance(resolved, dict) and set(resolved) == {'max_context', 'capabilities', 'ranks'}
            and resolved.get('max_context') == expected['effective_max_context']
            and type(resolved['max_context']) is int
            and json_bytes(resolved.get('capabilities')) == json_bytes(expected['capabilities']),
            'resolved configuration identity mismatch')
    digest = hashlib.sha256(json.dumps(resolved, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    require(expected['effective_profile_digest'] == 'sha256:' + digest, 'observed resolved configuration digest drift')
    ranks = observation.get('ranks')
    require(isinstance(ranks, list) and len(ranks) == expected['topology']['ranks'], 'all observed ranks required')
    seen, hosts, projections = set(), set(), []
    for rank in ranks:
        require(isinstance(rank, dict) and type(rank.get('rank')) is int and rank['rank'] not in seen, 'unique observed ranks required')
        seen.add(rank['rank'])
        for key in ('binary_sha256', 'engine_commit', 'engine_tree'):
            require(rank.get(key) == expected[key], 'installed rank binary/source drift')
        require(all(type(rank.get(k)) is int and rank[k] > 0 for k in ('pid', 'start_time_ticks'))
                and isinstance(rank.get('hostname'), str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,253}', rank['hostname'])
                and rank.get('ready') is True,
                'owned ready process identity required')
        require(rank['hostname'].lower().rstrip('.') not in hosts, 'distinct rank hosts required')
        hosts.add(rank['hostname'].lower().rstrip('.'))
        require(isinstance(rank.get('argv'), list) and rank['argv'] and all(isinstance(v, str) for v in rank['argv']), 'observed argv required')
        argv = rank['argv']
        windows = []
        for i, arg in enumerate(argv):
            if arg == '--max-model-len':
                require(i + 1 < len(argv), 'missing observed context argument')
                windows.append(argv[i + 1])
            elif arg.startswith('--max-model-len='):
                windows.append(arg.split('=', 1)[1])
        require(windows == [str(expected['effective_max_context'])], 'actual process context differs from resolved claim')
        for capability, flag in (('images', '--no-vision'), ('video', '--no-video')):
            require(not expected['capabilities'].get(capability) or not any(
                v == flag or v.startswith(flag + '=') for v in argv), 'actual process disables claimed modality')
        if expected['capabilities'].get('images') is False:
            # The reverse direction: a vision-off release must really run vision-off,
            # as one bare switch; an attached or repeated form is never an alias.
            require(argv.count('--no-vision') == 1 and not any(v.startswith('--no-vision=') for v in argv),
                    'images=false requires exactly one bare --no-vision in every rank argv')
        require(isinstance(rank.get('env'), dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in rank['env'].items()), 'observed environment required')
        projections.append({k: rank[k] for k in ('rank', 'argv', 'env')})
    require(seen == set(range(len(ranks))) and json_bytes(resolved.get('ranks')) == json_bytes(projections),
            'resolved config not tied to observed rank argv/env')
    return observation


TOKEN = re.compile(r'(@[A-Z][A-Z0-9_]*@)')
PATH_TOKENS = {'@MODEL_DIR@', '@DRAFT_DIR@', '@BUNDLE_DIR@', '@STATE_DIR@'}
BOOTSTRAP_FIXED = {'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'LANG': 'C.UTF-8',
                   'LC_ALL': 'C.UTF-8', 'PYTHONDONTWRITEBYTECODE': '1'}
BOOTSTRAP_PATHS = {'HOME', 'PWD', 'TMPDIR', '_'}


def _absolute_site_path(value):
    require(isinstance(value, str) and 1 < len(value) <= 4096 and value.startswith('/')
            and all(p not in ('', '.', '..') for p in value[1:].split('/'))
            and not any(ord(c) < 32 or ord(c) == 127 or c in '\\@$' for c in value), 'invalid observed site path')


def _site_value(token, value, kind):
    if token in PATH_TOKENS:
        _absolute_site_path(value)
    elif token in ('@HTTP_HOST@', '@NODES@'):
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            raise ValueError('invalid observed site address') from None
        require(not address.is_unspecified and not address.is_multicast, 'invalid observed site address')
        require(kind != 'single' or address.is_loopback, 'single profile requires loopback')
        require(kind != 'cluster' or address.version == 4, 'native site requires IPv4')
    elif token in ('@HTTP_PORT@', '@TP_PORT@'):
        require(re.fullmatch(r'[1-9][0-9]{0,4}', value) and int(value) <= 65535, 'invalid observed site port')
    elif token == '@FABRIC_INTERFACE@':
        require(re.fullmatch(r'[A-Za-z0-9_.-]{1,32}', value), 'invalid observed fabric interface')
    elif token == '@HCA@':
        devices = value.split(',')
        require(len(devices) == len(set(devices)) and all(re.fullmatch(r'[A-Za-z0-9_]+:[1-9][0-9]*', d)
                                                       for d in devices), 'invalid observed HCA')
    else:
        raise ValueError('undefined profile substitution')


def _site_tokens(choice):
    fields = choice.get('site_fields', [])
    require(isinstance(fields, list) and all(isinstance(f, str) for f in fields), 'invalid site fields')
    allowed = set(PATH_TOKENS) if choice['kind'] == 'single' else set()
    for field, tokens in {'nodes': {'@MODEL_DIR@', '@HTTP_HOST@', '@NODES@'},
                          'http_host': {'@HTTP_HOST@'}, 'http_port': {'@HTTP_PORT@'},
                          'tp_port': {'@TP_PORT@'}, 'fabric_interface': {'@FABRIC_INTERFACE@'},
                          'hca': {'@HCA@'}}.items():
        if field in fields:
            allowed.update(tokens)
    return allowed


def _match_profile_value(want, got, *, allowed, bindings, kind):
    require(isinstance(want, str) and isinstance(got, str), 'profile value must be text')
    pieces = TOKEN.split(want)
    tokens = pieces[1::2]
    require(set(tokens) <= allowed, 'undefined profile substitution')
    pattern = ''.join('(.+?)' if i % 2 else re.escape(piece) for i, piece in enumerate(pieces))
    match = re.fullmatch(pattern, got)
    require(match is not None, 'observed value differs from immutable profile')
    for token, value in zip(tokens, match.groups()):
        _site_value(token, value, kind)
        require(token not in bindings or bindings[token] == value, 'inconsistent observed site substitution')
        bindings[token] = value


def _flag_index(argv, flag):
    require(argv.count(flag) == 1 and not any(v.startswith(flag + '=') for v in argv), 'one explicit native option required')
    index = argv.index(flag)
    require(index + 1 < len(argv) and not argv[index + 1].startswith('--'), 'native option value required')
    return index


def _profile_rank_argv(choice):
    """Normalized rank-0 profile -> native rank_args, with no knob exceptions.

    Native TP2 adds HTTP binding only on the leader; rank id alone differs on
    the follower. Per-node model locations are explicit @MODEL_DIR@ placement.
    profile_context independently checks these against the pinned native config.
    """
    require(isinstance(choice, dict) and choice.get('kind') in ('single', 'cluster'), 'unknown profile kind')
    argv = choice.get('argv')
    require(isinstance(argv, list) and all(isinstance(v, str) and v for v in argv), 'profile argv required')
    context_argument(argv)
    if choice['kind'] == 'single':
        return [list(argv)]
    require(argv[_flag_index(argv, '--tp-ranks') + 1] == '2'
            and argv[_flag_index(argv, '--tp-rank') + 1] == '0', 'normalized native TP2 leader required')
    for flag in ('--model', '--host', '--port', '--tp-peer', '--tp-connect-timeout',
                 '--tp-ready-timeout', '--tp-peer-timeout', '--shutdown-grace'):
        _flag_index(argv, flag)
    follower = list(argv)
    for flag in ('--host', '--port'):
        index = _flag_index(follower, flag)
        del follower[index:index + 2]
    follower[_flag_index(follower, '--tp-rank') + 1] = '1'
    return [list(argv), follower]


def validate_observed_profile(observation, choice):
    """Pure admission of actual rank argv/env against a pinned normalized profile.

    Safe to call directly: the observation's schema/digest/rank projections are
    revalidated too. Host/process provenance must still be collected by the owned
    launcher; passing fixture checks is not a hardware qualification.
    """
    _validate_observation(observation, observation)
    _validate_observed_profile(observation, choice)


def _validate_observed_profile(observation, choice):
    expected = _profile_rank_argv(choice)
    require(observation['topology']['ranks'] == len(expected), 'profile topology differs from observation')
    require(observation['effective_max_context'] == context_argument(choice['argv'])
            and observation['capabilities'] == choice.get('capabilities'), 'observed profile context/capabilities drift')
    environment = choice.get('env')
    require(isinstance(environment, dict) and all(isinstance(k, str) and isinstance(v, str)
                                                for k, v in environment.items()), 'profile environment required')
    allowed, shared = _site_tokens(choice), {}
    for rank in observation['ranks']:
        actual, wanted = rank['argv'][1:], expected[rank['rank']]
        require(len(actual) == len(wanted), 'observed argv length differs from immutable profile')
        bindings = dict(shared)  # topology is shared, paths may differ by node
        for want, got in zip(wanted, actual):
            _match_profile_value(want, got, allowed=allowed, bindings=bindings, kind=choice['kind'])
        require(set(environment) <= set(rank['env']), 'missing immutable environment')
        for key, value in rank['env'].items():
            if key in environment:
                _match_profile_value(environment[key], value, allowed=allowed, bindings=bindings, kind=choice['kind'])
            elif key in BOOTSTRAP_FIXED:
                require(value == BOOTSTRAP_FIXED[key], 'bootstrap environment drift')
            elif key in BOOTSTRAP_PATHS:
                _absolute_site_path(value)
            elif key == 'SHLVL':
                require(re.fullmatch(r'[0-9]{1,3}', value), 'bootstrap environment drift')
            else:
                raise ValueError('additional unpinned environment')
        shared.update({k: v for k, v in bindings.items() if k not in PATH_TOKENS})
    require('@HTTP_PORT@' not in shared or '@TP_PORT@' not in shared
            or shared['@HTTP_PORT@'] != shared['@TP_PORT@'], 'HTTP/TP port collision')


def validate_gate_evidence(gate, result, *, evidence_root, expected, allow_incomplete=False):
    """One measured-case contract for both imports and final sealing.

    Incomplete imports may retain failed executed cases and unexecuted skips;
    they never qualify a release. Final sealing requires every case to pass.
    """
    require(isinstance(gate, str) and bool(gate), 'gate name required')
    require(type(allow_incomplete) is bool, 'invalid incomplete policy')
    root = checked_path(evidence_root)
    validate_identity(expected)
    require(isinstance(result, dict) and set(result) ==
            {'status', 'passed', 'total', 'failed', 'skipped', 'evidence'}, 'typed gate required: ' + gate)
    require(result['status'] in ('PASS', 'FAIL', 'NOT_RUN'), 'invalid gate status: ' + gate)
    require(all(type(result[k]) is int and result[k] >= 0
                for k in ('passed', 'total', 'failed', 'skipped')), 'integer counts required')
    require(result['total'] == result['passed'] + result['failed'] + result['skipped'], 'gate count mismatch: ' + gate)
    derived_status = ('FAIL' if result['failed'] else
                      'PASS' if result['total'] > 0 and result['passed'] == result['total'] else 'NOT_RUN')
    require(result['status'] == derived_status, 'gate status/count mismatch: ' + gate)
    if not allow_incomplete:
        require(result['status'] == 'PASS', 'gate not PASS: ' + gate)
        require(result['total'] > 0 and result['passed'] == result['total']
                and result['failed'] == result['skipped'] == 0, 'nonpositive/failing/skipped gate: ' + gate)
    executed = result['passed'] + result['failed']
    refs = result['evidence']
    require(isinstance(refs, list) and (bool(refs) or executed == 0), 'gate evidence required')
    seen, cases = set(), []
    for ref in refs:
        require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'invalid evidence reference')
        rel = relative_path(ref['path'])
        full_hash(ref['sha256'])
        require(rel not in seen, 'duplicate/self evidence')
        seen.add(rel)
        evidence = safe_file(root, rel)
        require(sha256_file(evidence) == ref['sha256'], 'evidence hash drift: ' + rel)
        if rel.endswith('.json'):
            raw = load_json(evidence)
            if isinstance(raw, dict) and raw.get('schema') == 'rexl3.gate-evidence.v1':
                validate_identity(raw)
                require(raw.get('gate') == gate and all(raw.get(k) == expected[k] for k in IDENTITY_KEYS),
                        'raw evidence identity drift')
                require(isinstance(raw.get('cases'), list), 'raw cases required')
                cases.extend(raw['cases'])
    require(len(cases) == executed, 'raw case count mismatch: ' + gate)
    names, passed, failed = set(), 0, 0
    for case in cases:
        require(isinstance(case, dict) and isinstance(case.get('name'), str) and case['name']
                and case['name'] not in names, 'unique case name required')
        names.add(case['name'])
        require(case.get('status') in ('PASS', 'FAIL') and case.get('expectation') in ('supported', 'refused')
                and case.get('modality') in ('text', 'image', 'video', 'audio', 'system'), 'invalid raw case')
        passed += case['status'] == 'PASS'
        failed += case['status'] == 'FAIL'
    require(passed == result['passed'] and failed == result['failed'], 'raw case verdict/count mismatch: ' + gate)
    if gate == 'vision' and result['status'] == 'PASS':
        images = expected['capabilities'].get('images')
        if images is True:
            require(any(c['status'] == 'PASS' and c['modality'] == 'image' and c['expectation'] == 'supported'
                        for c in cases),
                    'positive supported image evidence required; audio refusal is not vision')
        else:
            require(images is False, 'positive supported image evidence required; audio refusal is not vision')
            # Vision-off release: the live server must refuse an image cleanly. An
            # image the server was claimed NOT to serve can never be a supported case.
            require(not any(c['modality'] == 'image' and c['expectation'] == 'supported' for c in cases),
                    'images=false forbids supported image cases')
            refused = [c for c in cases if c['status'] == 'PASS' and c['modality'] == 'image'
                       and c['expectation'] == 'refused']
            require(refused, 'images=false requires a PASS refused image case (clean refusal evidence)')
            for case in refused:
                _require_clean_image_refusal(case, refs, root)
    return result


IMAGE_REFUSAL_MAX_SECONDS = 60


def _require_clean_image_refusal(case, refs, root):
    """A refused image case must carry a measured 4xx refusal, never a 5xx/timeout.

    The typed case summary is only a pointer: the bound detail file must record the
    HTTP status, a refusal message, a bounded wall time and a healthy server after.
    """
    ref = case.get('result')
    require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'image refusal detail reference required')
    require(ref in refs, 'image refusal detail must be referenced gate evidence')
    detail = load_json(safe_file(root, relative_path(ref['path'])))
    require(detail.get('error') is None and isinstance(detail.get('details'), dict),
            'image refusal detail records a failure')
    facts = detail['details']
    status = facts.get('http_status')
    require(type(status) is int and 400 <= status <= 499, 'image refusal must be an HTTP 4xx, not a 5xx or timeout')
    error = facts.get('error')
    require(isinstance(error, dict) and isinstance(error.get('message'), str) and error['message'].strip(),
            'image refusal must carry an error message')
    wall = facts.get('wall_s')
    require(type(wall) in (int, float) and 0 <= wall < IMAGE_REFUSAL_MAX_SECONDS, 'image refusal hung or took too long')
    require(facts.get('server_healthy_after') is True and facts.get('positive_capability') is False,
            'server must stay healthy after the image refusal')


def validate_receipt(path, *, evidence_root, expected):
    """Validate typed PASS counts, every referenced byte, identity and raw case summaries.

    Each gate references >=1 rexl3.gate-evidence.v1 JSON summary: gate, all receipt
    identity fields, cases[{name,status,modality,expectation}]. Additional raw files
    may also be referenced. Case names are unique per gate and counts must agree.
    Positive image/supported cases, not audio/refused cases, satisfy vision when
    capabilities.images is true. When it is false, a PASS image/refused case with a
    measured HTTP 4xx detail satisfies vision and no supported image case may exist.
    """
    path = checked_path(path)
    root = checked_path(evidence_root)
    require(path.is_relative_to(root), 'receipt escapes evidence root')
    doc = load_json(path)
    require(doc.get('schema') == RECEIPT_SCHEMA, 'qualification schema required')
    validate_identity(doc)
    validate_identity(expected)
    require(all(doc[key] == expected[key] for key in IDENTITY_KEYS), 'receipt identity drift')
    ref = doc.get('identity_receipt')
    require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'identity receipt required; naked digest refused')
    observation = safe_file(root, ref['path'])
    require(observation != path and sha256_file(observation) == full_hash(ref['sha256']), 'identity receipt hash drift')
    validate_observation(observation, expected)
    gates = doc.get('gates')
    require(isinstance(gates, dict) and set(GATES) <= set(gates), 'missing required gates')
    for gate, result in gates.items():
        validate_gate_evidence(gate, result, evidence_root=root, expected=expected)
        require(all(root / relative_path(ref['path']) != path for ref in result['evidence']), 'duplicate/self evidence')
    return doc


def inspect_archive(path):
    """Read and validate the ENTIRE tar before any extraction; return exact member records."""
    path = checked_path(path)
    safe_file(path.parent, path.name)
    inventory, total = {}, 0
    try:
        with tarfile.open(path, 'r:*') as tar:
            for member in tar:
                name = relative_path(member.name)
                require(member.isfile() and not member.issparse() and not member.pax_headers, 'unsafe archive member: ' + name)
                require(name not in inventory, 'duplicate archive member: ' + name)
                require(member.mode in (0o644, 0o755), 'unsafe archive mode')
                total += member.size
                require(0 <= member.size <= MAX_FILE_BYTES and total <= MAX_BYTES and len(inventory) < MAX_FILES, 'archive size limit')
                digest, count = hashlib.sha256(), 0
                with tar.extractfile(member) as stream:
                    for chunk in iter(lambda: stream.read(1 << 20), b''):
                        digest.update(chunk)
                        count += len(chunk)
                require(count == member.size, 'short archive member')
                inventory[name] = dict(sha256=digest.hexdigest(), bytes=count, mode=member.mode)
        require(inventory, 'empty archive')
        for name in inventory:
            require(not any(str(p) in inventory for p in PurePosixPath(name).parents), 'archive path collision')
    except (tarfile.TarError, EOFError) as error:
        raise ValueError('invalid archive') from error
    return inventory


def archive_json(path, name):
    with tarfile.open(path, 'r:*') as tar:
        member = tar.getmember(name)
        require(member.isfile() and member.size <= MAX_JSON_BYTES, 'invalid archive JSON')
        with tar.extractfile(member) as stream:
            return loads(stream.read())


def asset_record(path, inventory=None):
    path = Path(path)
    relative_path(path.name)
    members = inspect_archive(path) if inventory is None else inventory
    return dict(archive=path.name, sha256=sha256_file(path), bytes=path.stat().st_size,
                files=len(members), members=members)


def validate_asset(asset):
    require(isinstance(asset, dict) and set(asset) == {'archive', 'sha256', 'bytes', 'files', 'members'}, 'invalid asset record')
    relative_path(asset['archive'])
    require('/' not in asset['archive'], 'asset must be a basename')
    full_hash(asset['sha256'])
    require(type(asset['bytes']) is int and asset['bytes'] > 0 and type(asset['files']) is int and asset['files'] > 0, 'invalid asset counts')
    require(isinstance(asset['members'], dict) and len(asset['members']) == asset['files'], 'asset file count mismatch')
    for name, item in asset['members'].items():
        relative_path(name)
        validate_record(item)


def approval_assets(approval):
    """Every transport record, with four mandatory distinct v2 asset roles."""
    require(all(key in approval for key in ('bundle', 'source_asset', 'engine_source_asset', 'evidence_asset')),
            'missing release asset (engine corresponding source is mandatory)')
    assets = [value for key, value in approval.items() if key == 'bundle' or key.endswith('_asset')]
    names = {'approval.json', '.dev-download-identity.json'}
    for asset in assets:
        validate_asset(asset)
        require(asset['archive'] not in names, 'release asset collision')
        names.add(asset['archive'])
    require(approval['evidence_asset']['archive'] == 'evidence.tar.gz', 'bound evidence transport required')
    return assets


def receipt_files(approval, root):
    paths = set()
    forbidden = {'approval.json', 'evidence.tar.gz', '.dev-download-identity.json'} | {
        asset['archive'] for key, asset in approval.items() if key == 'bundle' or key.endswith('_asset')}
    for selected in approval['profiles'].values():
        rel = relative_path(selected['receipt']['path'])
        receipt = load_json(safe_file(root, rel))
        paths.add(rel)
        paths.add(relative_path(receipt['identity_receipt']['path']))
        paths.update(relative_path(ref['path']) for gate in receipt['gates'].values() for ref in gate['evidence'])
    require(not paths & forbidden, 'evidence asset collision')
    return {name: dict(record(safe_file(root, name)), mode=0o644) for name in sorted(paths)}


def write_evidence_archive(root, members, output):
    """Deterministic transport only; evidence remains a sibling, never inside the runtime."""
    with Path(output).open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, filename='', mode='wb', mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w', format=tarfile.USTAR_FORMAT) as tar:
                for name, item in sorted(members.items()):
                    path = safe_file(root, name)
                    require(sha256_file(path) == item['sha256'] and path.stat().st_size == item['bytes'], 'evidence changed during seal')
                    member = tarfile.TarInfo(name)
                    member.mode, member.size = 0o644, item['bytes']
                    with path.open('rb') as stream:
                        tar.addfile(member, stream)


def evidence_asset(approval, root):
    members = receipt_files(approval, root)
    with tempfile.TemporaryDirectory(prefix='.seal-', dir=root) as work:
        path = Path(work) / 'evidence.tar.gz'
        write_evidence_archive(root, members, path)
        return asset_record(path)


def extract_files(archive, dest, expected, *, selected=None):
    """Full header/content preflight and destination preflight before ANY write.

    Intended for exclusively owned private staging directories, not shared writable
    trees. Existing exact files are reusable; conflicts/links are never overwritten.
    """
    require(inspect_archive(archive) == expected, 'archive inventory drift')
    names = set(expected) if selected is None else set(selected)
    require(names <= set(expected), 'unknown selected member')
    dest = checked_path(dest)
    for name in names:
        target = checked_path(dest / relative_path(name))
        if target.exists():
            require(record(target) == expected[name], 'destination conflict: ' + name)
    with tarfile.open(archive, 'r:*') as tar:
        for name in sorted(names):
            target = dest / name
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(name) as incoming, target.open('xb') as output:
                import shutil
                shutil.copyfileobj(incoming, output, 1 << 20)
            target.chmod(expected[name]['mode'])
            require(record(target) == expected[name], 'extracted bytes drift')


def git(repo, *args):
    result = subprocess.run(['git', '--no-replace-objects', '-C', str(repo), *args], capture_output=True, check=False)
    require(result.returncode == 0, 'git failed: ' + ' '.join(args) + ': ' + result.stderr.decode(errors='replace'))
    return result.stdout


def git_identity(repo, commit, tree=None):
    full_hash(commit, HEX40)
    require(git(repo, 'cat-file', '-t', commit).strip() == b'commit', 'real commit object required')
    actual = git(repo, 'rev-parse', commit + '^{tree}').decode().strip()
    full_hash(actual, HEX40)
    require(git(repo, 'cat-file', '-t', actual).strip() == b'tree', 'real tree required')
    if tree is not None:
        full_hash(tree, HEX40)
        require(actual == tree, 'commit/tree mismatch')
    return actual


def git_file(repo, commit, name):
    relative_path(name)
    rows = git(repo, 'ls-tree', '-z', commit, '--', name).split(b'\0')
    require(len(rows) == 2 and rows[0], 'tracked source file missing: ' + name)
    header, found = rows[0].split(b'\t', 1)
    mode, kind, oid = header.split()
    require(found.decode() == name and mode in (b'100644', b'100755') and kind == b'blob', 'regular tracked blob required')
    return git(repo, 'cat-file', 'blob', oid.decode())


def validate_request(runtime_id, approve_tag, version):
    require(isinstance(runtime_id, str) and ID.fullmatch(runtime_id), 'invalid runtime id')
    require(type(version) is int and version > 0, 'positive version required')
    require(approve_tag == f'approve-{runtime_id}-{version}', 'runtime/tag/version mismatch')


def context_argument(argv):
    require(isinstance(argv, list) and all(isinstance(v, str) for v in argv), 'context argv required')
    require(not any(v.startswith('--max-model-len=') for v in argv)
            and argv.count('--max-model-len') == 1, 'one explicit context argument required')
    index = argv.index('--max-model-len')
    require(index + 1 < len(argv) and re.fullmatch(r'[1-9][0-9]*', argv[index + 1]), 'invalid context argv')
    return int(argv[index + 1])


def _validate_native_profile(choice, config, group, by_name):
    """Compare normalized argv/env to schema.rs::rank_args/rank_env on BOTH ranks.

    The attested adapter changes placement only; node/leader serving overrides and
    external env files/libraries are not supported by runtime.py. No incoming
    configuration is imported or executed here.
    """
    expected = _profile_rank_argv(choice)
    serve, tp, nccl = config.get('serve'), config.get('tp'), config.get('nccl')
    require(all(isinstance(v, dict) for v in (serve, tp, nccl)), 'explicit native launch configuration required')
    require(not nccl.get('env_file'), 'unbound native environment file')
    for key in ('connect_timeout_s', 'ready_timeout_s', 'peer_timeout_s', 'shutdown_grace_s'):
        require(type(tp.get(key)) is int and tp[key] >= 0, 'explicit native timeout required')
    require(type(nccl.get('max_nchannels')) is int and nccl['max_nchannels'] > 0, 'explicit native NCCL pin required')
    require(isinstance(nccl.get('hca'), list) and all(isinstance(v, str) for v in nccl['hca']), 'native HCA list required')
    for env in (serve.get('env', {}), nccl.get('env', {})):
        require(isinstance(env, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()), 'native environment required')
    attested = choice.get('cluster_adapter') == 'attested-v1'
    if attested:
        require(group['ranks'] == ['leader', 'follower'] and list(by_name) == ['leader', 'follower']
                and not group.get('leader_flags') and not nccl.get('lib'), 'unsupported attested native configuration')
        require(all(n.get('local') is False and not n.get('flags') and not n.get('env')
                    for n in by_name.values()), 'unsupported attested rank override')
    fields = set(choice.get('site_fields', [])) if attested else set()
    def place(field, token, value):
        return token if field in fields else value
    leader = by_name[group['ranks'][0]]
    model = place('nodes', '@MODEL_DIR@', config.get('pack'))
    host = place('nodes', '@HTTP_HOST@', group.get('http_host'))
    port = place('http_port', '@HTTP_PORT@', str(group.get('http_port')))
    peer = place('nodes', '@NODES@', leader.get('fabric_address', leader.get('address')))
    tp_port = place('tp_port', '@TP_PORT@', str(group.get('tp_port')))
    require(all(isinstance(v, str) and v for v in (model, host, peer)), 'native placement required')
    environment = {'NCCL_MAX_NCHANNELS': str(nccl['max_nchannels'])}
    if nccl.get('socket_ifname'):
        environment['NCCL_SOCKET_IFNAME'] = place('fabric_interface', '@FABRIC_INTERFACE@', nccl['socket_ifname'])
    if nccl['hca']:
        environment['NCCL_IB_HCA'] = '=' + place('hca', '@HCA@', ','.join(nccl['hca']))
    environment.update(nccl.get('env', {}))
    if nccl.get('lib'):
        environment['REXL3_NCCL_LIB'] = nccl['lib']
    environment.update(serve.get('env', {}))
    for rank, name in enumerate(group['ranks']):
        node = by_name[name]
        args = ['--model', model, *serve['flags'], *node.get('flags', [])]
        if rank == 0:
            args += [*group.get('leader_flags', []), '--host', host, '--port', port]
        args += ['--tp-ranks', '2', '--tp-rank', str(rank), '--tp-peer', peer + ':' + tp_port,
                 '--tp-connect-timeout', str(tp['connect_timeout_s']), '--tp-ready-timeout', str(tp['ready_timeout_s']),
                 '--tp-peer-timeout', str(tp['peer_timeout_s']), '--shutdown-grace', str(tp['shutdown_grace_s'])]
        require(args == expected[rank], 'native rank argv differs from normalized profile')
        require(isinstance(node.get('env', {}), dict) and
                dict(environment, **node.get('env', {})) == choice.get('env'), 'native rank environment differs from normalized profile')


def profile_context(choice, archive):
    context = context_argument(choice.get('argv'))
    if choice.get('kind') == 'single':
        return context, 1
    require(choice.get('kind') == 'cluster', 'unknown profile kind')
    config = archive_json(archive, relative_path(choice.get('cluster_file')))
    require(isinstance(config, dict) and type(config.get('schema_version')) is int and config['schema_version'] == 1, 'native cluster schema required')
    groups = config.get('groups')
    if not isinstance(groups, list) or not all(isinstance(g, dict) for g in groups):
        raise ValueError('native grouped cluster required')
    selected = [g for g in groups if g.get('name') == choice.get('group')]
    require(len(selected) == 1, 'unique native cluster group required')
    group = selected[0]
    ranks = group.get('ranks')
    require(isinstance(ranks, list) and len(ranks) == 2 and all(isinstance(n, str) for n in ranks)
            and len(set(ranks)) == 2, 'two distinct ordered native ranks required')
    nodes = config.get('nodes')
    if not isinstance(nodes, list) or not all(isinstance(n, dict) and isinstance(n.get('name'), str) for n in nodes):
        raise ValueError('native nodes required')
    by_name = {n['name']: n for n in nodes}
    require(len(by_name) == len(nodes) and set(ranks) <= set(by_name), 'missing or duplicate native nodes')
    serve = config.get('serve', {})
    require(isinstance(serve, dict) and isinstance(serve.get('flags'), list), 'native serve flags required')
    for rank, name in enumerate(ranks):
        extra = by_name[name].get('flags', [])
        leader = group.get('leader_flags', []) if rank == 0 else []
        require(isinstance(extra, list) and isinstance(leader, list), 'native rank flags required')
        actual = context_argument(serve['flags'] + extra + leader)
        require(actual == context, 'cluster context differs from immutable profile')
    _validate_native_profile(choice, config, group, by_name)
    return context, 2


def validate_engine_binaries(engine):
    """Strict shape of an approval's ``engine_binaries``: install path -> the engine release asset that supplies it."""
    require(isinstance(engine, dict) and 'bin/rexl3-serve' in engine and set(engine) <= set(ENGINE_PATHS),
            'engine_binaries must name bin/rexl3-serve and only known native paths')
    for path, item in engine.items():
        require(isinstance(item, dict) and set(item) == {'release_tag', 'asset', 'sha256', 'bytes', 'mode'},
                'invalid engine binary record: ' + path)
        require(isinstance(item['release_tag'], str) and ENGINE_TAG.fullmatch(item['release_tag']), 'invalid engine release tag')
        require(isinstance(item['asset'], str), 'invalid engine asset name')
        relative_path(item['asset'])
        require('/' not in item['asset'], 'engine asset must be a basename')
        full_hash(item['sha256'])
        require(type(item['bytes']) is int and item['bytes'] > 0, 'invalid engine binary size')
        require(type(item['mode']) is int and item['mode'] == 0o755, 'engine binary mode must be 0755')
    require(len({item['release_tag'] for item in engine.values()}) == 1, 'one engine release supplies every native binary')
    require(len({item['asset'] for item in engine.values()}) == len(engine), 'engine assets must be distinct')
    return engine


def native_elf_header(header, machine):
    return (len(header) == 64 and header[:7] == b'\x7fELF\x02\x01\x01' and
            struct.unpack_from('<H', header, 16)[0] in (2, 3) and
            struct.unpack_from('<H', header, 18)[0] == {'aarch64': 183, 'x86_64': 62}[machine])


def engine_binary_files(engine, engine_dir, machine):
    """The real engine files behind ``engine_binaries``: exact bytes, size and a native ELF header for this platform."""
    require(engine_dir is not None, 'engine binary files are required to verify an approval that references an engine release')
    for path, item in engine.items():
        file = safe_file(engine_dir, item['asset'])
        require(file.stat().st_size == item['bytes'] and sha256_file(file) == item['sha256'], 'engine binary hash/size mismatch: ' + path)
        with file.open('rb') as stream:
            header = stream.read(64)
        require(native_elf_header(header, machine), 'native ELF/platform mismatch: ' + path)


def bundle_identity(archive, source_asset, *, runtime_id, commit, tree, template, version, engine_binaries=None, engine_dir=None):
    """Identity of a bundle. With ``engine_binaries`` the native executables are not in the archive: the approval
    names the engine release assets that supply them and ``engine_dir`` holds those files."""
    inv = inspect_archive(archive)
    if engine_binaries is None:
        full = inv
    else:
        validate_engine_binaries(engine_binaries)
        require(not any(name.startswith('bin/') for name in inv), 'bundle must not carry native executables the engine release supplies')
        full = dict(inv, **{path: {k: item[k] for k in ('sha256', 'bytes', 'mode')} for path, item in engine_binaries.items()})
    require({'source.manifest.json', 'profile.json', 'model.lock.json', 'release.lock.json', 'bin/rexl3-serve'} <= set(full), 'incomplete bundle')
    profile = archive_json(archive, 'profile.json')
    models = archive_json(archive, 'model.lock.json')
    release = archive_json(archive, 'release.lock.json')
    manifest = archive_json(archive, 'source.manifest.json')
    require(profile.get('schema') == 'rexl3.runtime-profile.v1' and profile.get('state') == 'final', 'final runtime profile required')
    require(models.get('schema') == 'rexl3.model-lock.v1' and models.get('packs'), 'model lock required')
    require(release.get('schema') == 'rexl3.release.v1' and manifest.get('schema') == 'rexl3.generated-source.v1', 'bundle schema mismatch')
    require(profile.get('runtime_id') == release.get('runtime_id') == runtime_id, 'runtime identity mismatch')
    require(release.get('production_commit') == commit and release.get('production_tree') == tree, 'release source identity mismatch')
    require(profile.get('target_repo') == release.get('target_repo') == models['packs'].get('target', {}).get('repo'), 'model identity mismatch')
    require(release.get('files') == {k: v for k, v in full.items() if k != 'release.lock.json'}, 'release inventory mismatch')
    for records in (release['files'], manifest.get('files', {})):
        for name, item in records.items():
            relative_path(name)
            validate_record(item)
    native = release.get('native_version_fields', {})
    require(native.get('commit') == commit, 'native source identity mismatch')
    # The tester-side installer requires release.version == native_version_fields.version (the engine's own
    # version). The promotion number lives only in the approval and the release tag, never in release.lock.json.
    require(release.get('version') == native.get('version'), 'release version must equal the native engine version')
    require(set(native) == {'version', 'commit', 'system_fingerprint', 'nvcc', 'cuda_arch', 'kernel_hash', 'cudart', 'cuda_driver', 'cublas'}, 'complete measured native identity required')
    require(all(isinstance(v, str) and v and v.lower() not in ('none', 'unknown', 'unavailable', 'pending')
                and 'dirty' not in v.lower() and 'no_cuda' not in v.lower() for v in native.values()), 'native CPU stub/dirty identity refused')
    require(re.fullmatch(r'[0-9a-f]{16}', native['kernel_hash']), 'native kernel hash is FNV64, not SHA256')
    platform = release.get('platform', {})
    require(platform.get('system') == 'Linux' and platform.get('machine') in ('aarch64', 'x86_64'), 'native platform required')
    arch = platform.get('gpu_compute_capability')
    require(isinstance(arch, str) and re.fullmatch(r'[1-9][0-9]*\.[0-9]+', arch), 'native GPU architecture required')
    require(native['cuda_arch'] in (arch.replace('.', ''), 'sm_' + arch.replace('.', ''), 'compute_' + arch.replace('.', '')), 'native architecture mismatch')
    require({name for name in full if name.startswith('bin/')} <= set(ENGINE_PATHS), 'unexpected native executable')
    if engine_binaries is None:
        with tarfile.open(archive, 'r:*') as tar:
            for name in inv:
                if name.startswith('bin/'):
                    with tar.extractfile(name) as stream:
                        header = stream.read(64)
                    require(native_elf_header(header, platform['machine']) and inv[name]['mode'] == 0o755,
                            'native ELF/platform/mode mismatch')
    else:
        # The tag names the engine, never a model: engine version and the first eight hex digits of its commit.
        require({item['release_tag'] for item in engine_binaries.values()} == {'engine-v%s-%s' % (native['version'], commit[:8])},
                'engine release tag does not name the native engine version and commit')
        engine_binary_files(engine_binaries, engine_dir, platform['machine'])
    source_inv = {k: v for k, v in inv.items() if not k.startswith('bin/') and k != 'release.lock.json'}
    require(inspect_archive(source_asset) == source_inv, 'runtime wrapper source asset mismatch')
    require(manifest.get('files') == {k: v for k, v in source_inv.items() if k != 'source.manifest.json'}, 'source inventory mismatch')
    require(manifest.get('template') == {k: template[k] for k in ('version', 'digest')}, 'manifest template mismatch')
    identities = {key: full[name]['sha256'] for key, name in (
        ('profile_sha256', 'profile.json'), ('model_lock_sha256', 'model.lock.json'),
        ('release_lock_sha256', 'release.lock.json'), ('source_manifest_sha256', 'source.manifest.json'),
        ('binary_sha256', 'bin/rexl3-serve'))}
    for key in ('profile_sha256', 'model_lock_sha256'):
        require(release.get(key) == identities[key], 'release hash alias mismatch')
    choices = profile.get('profiles')
    require(isinstance(choices, dict) and choices, 'profile choices required')
    for name, choice in choices.items():
        require(isinstance(name, str) and ID.fullmatch(name) and isinstance(choice, dict), 'invalid profile')
        require(isinstance(choice.get('files'), dict), 'config inventory required')
        for rel, digest in choice['files'].items():
            require(rel in inv and inv[rel]['sha256'] == digest, 'profile config drift')
        context, ranks = profile_context(choice, archive)
        require(context > 0, 'positive context required')
        if ranks == 2:
            require('bin/rexl3-cluster' in full, 'cluster binary missing')
    return profile, identities, inv, source_inv


def trusted_entry(approval_path, registry, *, allow_legacy=False, key=None):
    approval = load_json(approval_path)
    require(isinstance(registry, dict) and registry.get('schema') == 'rexl3.approval-registry.v1', 'trusted approval registry required')
    legacy = approval.get('schema') == 'rexl3.profile-approval.v1'
    require(approval.get('schema') == SCHEMA or legacy and allow_legacy, 'weak/unknown approval schema refused')
    if key is None:
        key = f"{approval.get('runtime_id')}/v{approval.get('version')}"
        explicit_key = False
    else:
        require(not legacy, 'legacy approvals have no family/model registry key')
        require(isinstance(key, str) and key.count('/') == 2, 'invalid registry key: ' + str(key))
        require(isinstance(approval.get('runtime_id'), str), 'approval runtime_id required')
        explicit_key = True
    entries = registry.get('legacy' if legacy else 'approvals', {})
    entry = entries.get(key)
    require(isinstance(entry, dict), 'approval not in independently trusted registry: ' + key)
    if explicit_key and not legacy:
        require(entry.get('runtime_id') == approval['runtime_id'],
                'registry runtime_id differs from the sealed approval')
    full_hash(entry.get('approval_sha256'))
    require(sha256_file(approval_path) == entry['approval_sha256'], 'untrusted approval bytes')
    for field in ('source_commit', 'source_tree'):
        full_hash(approval.get(field), HEX40)
        require(approval[field] == entry.get(field), 'trusted source identity mismatch')
    return approval


def verify_assets(approval_path, *, registry, approved_templates, engine_dir=None, key=None):
    """Authenticate approval, then every archive and receipt, including engine source.

    ``key`` is the family/model registry key (``<family>/<model>/v<N>``) when verifying a prod tree;
    by default the runtime-id key inside the approval is used (dev-side fixture and sealer self-check).
    Consumer source validation is approval-bound integrity, not independent proof
    of private Git-object existence. Only the Git-backed sealer makes that proof.
    """
    approval_path = checked_path(approval_path)
    root = approval_path.parent
    doc = trusted_entry(approval_path, registry, key=key)
    validate_request(doc.get('runtime_id'), doc.get('approve_tag'), doc.get('version'))
    template = doc.get('template', {})
    require(isinstance(template, dict) and approved_templates.get(template.get('version')) == template.get('digest')
            and template.get('digest') is not None, 'template not approved')
    paths = {}
    for asset in approval_assets(doc):
        path = safe_file(root, asset['archive'])
        require(asset_record(path) == asset, 'asset hash/bytes/inventory mismatch: ' + asset['archive'])
        paths[asset['archive']] = path
    archive, wrapper = (paths[doc[key]['archive']] for key in ('bundle', 'source_asset'))
    try:
        from .corresponding_source_identity import validate_bound_archive
    except ImportError:
        from corresponding_source_identity import validate_bound_archive
    validate_bound_archive(paths[doc['engine_source_asset']['archive']], commit=doc['source_commit'],
                           tree=doc['source_tree'], asset=doc['engine_source_asset'])
    evidence = doc.get('evidence_asset')
    require(receipt_files(doc, root) == evidence['members'], 'evidence inventory drift')
    engine = doc.get('engine_binaries')
    require('engine_binaries' not in doc or isinstance(engine, dict), 'engine_binaries must be an object')
    profile, identities, _, _ = bundle_identity(archive, wrapper, runtime_id=doc['runtime_id'], commit=doc['source_commit'],
        tree=doc['source_tree'], template=template, version=doc['version'], engine_binaries=engine, engine_dir=engine_dir)
    require(identities == doc.get('identities'), 'approval bundle identities mismatch')
    profiles = doc.get('profiles')
    require(isinstance(profiles, dict) and set(profiles) == set(profile['profiles']), 'receipt profile set mismatch')
    for name, selected in profiles.items():
        require(isinstance(selected, dict) and set(selected) == {'receipt', 'identity'}, 'invalid profile approval')
        expected = selected['identity']
        validate_identity(expected)
        context, ranks = profile_context(profile['profiles'][name], archive)
        derived = dict(runtime_id=doc['runtime_id'], profile=name, engine_commit=doc['source_commit'],
                       engine_tree=doc['source_tree'], effective_max_context=context,
                       topology={'ranks': ranks}, capabilities=profile['profiles'][name]['capabilities'],
                       **{k: v for k, v in identities.items() if k != 'source_manifest_sha256'})
        require(all(expected.get(k) == v for k, v in derived.items()), 'approved profile identity drift')
        ref = selected['receipt']
        require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'invalid receipt reference')
        receipt = safe_file(root, ref['path'])
        require(sha256_file(receipt) == full_hash(ref['sha256']), 'receipt bytes drift')
        qualified = validate_receipt(receipt, evidence_root=root, expected=expected)
        observation = load_json(safe_file(root, qualified['identity_receipt']['path']))
        validate_observed_profile(observation, profile['profiles'][name])
    return doc, archive


def load_registry(path, expected_sha256):
    require(sha256_file(path) == full_hash(expected_sha256), 'trusted registry hash mismatch')
    return load_json(path)
