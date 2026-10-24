"""Engine corresponding-source identity, separate from the runtime wrapper.

The sealing boundary proves membership in real Git objects through the protected
exporter. The consumer boundary proves internal integrity ONLY AFTER a protected
approval registry has authenticated the exact archive inventory/hash. It does
not independently prove that a private commit/tree exists. No incoming code is
imported or executed, and no source/privacy policy can be supplied by the caller.
"""
from __future__ import annotations

import gzip
import io
from pathlib import Path
import tarfile
import zlib

try:
    from . import build_corresponding_source as exporter
    from . import release_evidence as e
except ImportError:
    import build_corresponding_source as exporter
    import release_evidence as e

MANIFEST = 'SOURCE_IDENTITY.json'
GENERATED = {'SOURCE_BUILD.md', '.rexl3_commit'}


def _read_members(path):
    """Safe, bounded, regular files only; never extract an engine archive."""
    inventory = e.inspect_archive(path)
    e.require(sum(r['bytes'] for r in inventory.values()) <= exporter.MAX_SOURCE_BYTES + e.MAX_JSON_BYTES,
              'engine source size limit')
    e.require(all(r['bytes'] <= (e.MAX_JSON_BYTES if n == MANIFEST else exporter.MAX_BLOB_BYTES)
                  for n, r in inventory.items()), 'engine source member size limit')
    members = {}
    with tarfile.open(path, 'r:*') as tar:
        for member in tar:
            exporter.validate_path(member.name)
            e.require(member.uid == member.gid == member.mtime == 0 and not member.uname and not member.gname,
                      'engine source header identity mismatch')
            with tar.extractfile(member) as stream:
                data = stream.read(inventory[member.name]['bytes'] + 1)
            e.require(inventory[member.name] == dict(sha256=exporter.digest(data), bytes=len(data), mode=member.mode),
                      'engine source changed during validation')
            members[member.name] = (member.mode, data)
    return inventory, members


def _expected_members(data):
    # data comes ONLY from the protected exporter, never from caller serialization.
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
        return {member.name: (member.mode, tar.extractfile(member).read()) for member in tar}


def _compare_projection(path, members, expected):
    e.require(members == _expected_members(expected), 'engine source differs from exact Git projection')
    # Tar readers stop at the first end marker. An otherwise exact archive must
    # not smuggle unlisted payloads after that marker or in a second gzip stream.
    # Compare canonical *uncompressed* USTAR, retaining compression portability.
    raw = gzip.decompress(expected)  # trusted exporter, already bounded
    decoder, actual = zlib.decompressobj(16 + zlib.MAX_WBITS), bytearray()
    try:
        with Path(path).open('rb') as stream:
            e.require(stream.read(4) == b'\x1f\x8b\x08\x00', 'engine gzip must have no optional metadata')
            stream.seek(0)
            for chunk in iter(lambda: stream.read(1 << 20), b''):
                actual.extend(decoder.decompress(chunk, len(raw) + 1 - len(actual)))
                e.require(len(actual) <= len(raw) and not decoder.unconsumed_tail and not decoder.unused_data,
                          'unlisted engine archive data')
        e.require(decoder.eof and actual == raw, 'noncanonical engine tar or hidden trailing data')
    except zlib.error:
        raise ValueError('invalid engine source compression') from None


def _trusted_template(data):
    path = Path(exporter.__file__).resolve().parents[1] / exporter.TEMPLATE
    e.require(data == e.safe_file(path.parent, path.name).read_bytes(),
              'engine build instructions differ from protected template')


def _required_inputs(selected):
    e.require(set(exporter.REQUIRED) | {exporter.TEMPLATE} <= set(selected),
              'engine source missing required build/license inputs')
    e.require(any(p.startswith('crates/') and p.endswith('.rs') for p in selected), 'engine crates missing')
    e.require(any(p.startswith('crates/') and Path(p).suffix.lower() in ('.cu', '.c', '.cc', '.cpp', '.h', '.hpp', '.s')
                  for p in selected), 'engine native inputs missing')
    e.require(any(p.startswith('third_party/') for p in selected), 'engine vendor inputs missing')


def verify_git_projection(path, *, repo, commit, tree):
    """Seal: exact original bytes/modes/manifest from the same real native source.

    Compare uncompressed members, not Python/zlib-dependent compressed output.
    The actual transport SHA256 is recorded separately by the approval writer.
    """
    e.full_hash(commit, e.HEX40)
    e.full_hash(tree, e.HEX40)
    try:
        source = exporter.inspect_source(e.checked_path(repo), commit)
        e.require(source['commit'] == commit and source['tree'] == tree, 'engine source commit/tree mismatch')
        _required_inputs(source['selected'])
        _trusted_template(source['payload'][exporter.TEMPLATE])
        expected = exporter.archive_bytes(source)
        inventory, actual = _read_members(path)
        _compare_projection(path, actual, expected)
        return inventory
    except exporter.SourceError as error:
        raise ValueError('engine source refused: ' + str(error)) from None


def validate_bound_archive(path, *, commit, tree, asset):
    """Consumer INTERNAL check; caller MUST authenticate approval before invoking.

    Recompute original Git blob identities and the protected exporter's generated
    manifest/instructions. Excluded inventory and commit/tree existence cannot be
    proved here: they are independently approval-bound, not self-authorizing.
    """
    e.full_hash(commit, e.HEX40)
    e.full_hash(tree, e.HEX40)
    e.validate_asset(asset)
    try:
        inventory, members = _read_members(path)
        e.require(e.asset_record(path, inventory) == asset, 'engine source asset drift')
        e.require(MANIFEST in members, 'engine source manifest missing')
        identity = e.loads(members[MANIFEST][1])
        e.require(identity.get('schema') == exporter.SCHEMA, 'engine source manifest schema mismatch')
        e.require(identity.get('source_commit') == commit and identity.get('source_tree') == tree,
                  'engine source commit/tree mismatch')
        e.require(identity.get('qualification_claimed') is False, 'source export is not qualification')
        records = identity.get('files')
        e.require(isinstance(records, dict) and set(records) == set(members) - {MANIFEST},
                  'engine source manifest inventory mismatch')
        entries, payload = {}, {}
        for name, record in records.items():
            e.require(isinstance(record, dict), 'invalid engine source record')
            mode, data = members[name]
            e.require(type(record.get('bytes')) is int and record['bytes'] == len(data)
                      and record.get('sha256') == exporter.digest(data)
                      and record.get('archive_mode') == format(mode, '04o'), 'engine source file identity mismatch')
            if name in GENERATED:
                e.require(record.get('origin') == 'generated' and record.get('git_blob') is None
                          and record.get('git_mode') is None, 'invalid generated source identity')
            else:
                e.require(record.get('origin') == 'git_blob' and record.get('git_mode') == '100' + format(mode, '03o')
                          and record.get('git_blob') == exporter.git_digest('blob', data), 'engine source Git blob identity mismatch')
                entries[name] = {k: record[k] for k in ('git_blob', 'git_mode', 'bytes')}
                payload[name] = data
        e.require(GENERATED <= set(records), 'generated source instructions missing')
        _required_inputs(payload)
        _trusted_template(payload[exporter.TEMPLATE])
        toolchain = exporter.toml(payload['rust-toolchain.toml']).get('toolchain', {}).get('channel')
        e.require(isinstance(toolchain, str) and toolchain == identity.get('rust_toolchain'), 'source toolchain mismatch')
        projection = identity.get('projection')
        e.require(isinstance(projection, dict) and type(projection.get('excluded_entries')) is int
                  and projection['excluded_entries'] >= 0, 'invalid source projection')
        e.full_hash(projection.get('excluded_inventory_sha256'))
        source = dict(commit=commit, tree=tree, toolchain=toolchain, entries=entries,
                      selected=sorted(payload), payload=payload, findings=[], blockers=[],
                      excluded_entries=projection['excluded_entries'],
                      excluded_inventory_sha256=projection['excluded_inventory_sha256'])
        # archive_bytes rechecks blob/mode/privacy and generates exact metadata.
        # Missing review policy grants NO privacy allowances. A future policy must
        # be implemented in the protected exporter, never as an approval waiver.
        expected = exporter.archive_bytes(source)
        _compare_projection(path, members, expected)
        return identity
    except exporter.SourceError as error:
        raise ValueError('engine source refused: ' + str(error)) from None
