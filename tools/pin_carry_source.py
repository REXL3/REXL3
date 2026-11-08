#!/usr/bin/env python3
"""Add the carried-gate source pins to one registry entry sealed before carry-forward (one time, per entry).

A carried gate whose source is a sealed approval must name an approval pinned in
`runtimes/templates/APPROVALS.json` with its `binary_sha256` and each profile's receipt digest under
`profiles/<profile>/receipt/sha256` (CONTRIBUTING.md, registry). Entries registered before carry-forward have
neither. This tool derives both from the entry's own sealed approval and proves them against its sealed evidence:

  1. the approval bytes are the ones the entry pins (`approval_sha256`, runtime id, source commit and tree);
  2. a private copy of the approval and its release assets, with `evidence.tar.gz` unpacked, passes
     `release_evidence.verify_assets` (every archive, receipt and raw evidence byte, and the bundle identities);
  3. each approved profile's receipt names the approval's `binary_sha256`.

It only adds keys: an entry that already holds a pin keeps it, and a different value is refused. Without
`--write` it prints the entry it would write; with `--write` it rewrites the registry file and prints the new
sha256 for the `TRUSTED_APPROVALS_SHA256` pin. The registry change itself belongs in a
`model/<model>-approval` PR (CONTRIBUTING.md section 3).

    python3 tools/pin_carry_source.py --registry runtimes/templates/APPROVALS.json \\
        --approved runtimes/templates/APPROVED.json --key qwen3.8/qwen3.8-flash-next/v2 \\
        --approval-dir DIR [--engine-dir DIR] [--write]

`DIR` holds `approval.json` and the release assets it names; `--engine-dir` holds the engine release binaries
when the approval takes them from an engine release. Exit 0: pins derived (and written with `--write`).
1: refused.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

try:
    from . import release_evidence as e
except ImportError:
    import release_evidence as e


def derive_pins(approval_dir, *, registry, key, approved_templates, engine_dir=None):
    """Verify the entry's sealed approval with its evidence and return the pins it supports."""
    source = e.checked_path(approval_dir)
    approval = e.trusted_entry(source / 'approval.json', registry, key=key)
    with tempfile.TemporaryDirectory(prefix='pin-carry-source-') as tmp:
        stage = Path(tmp)
        shutil.copyfile(source / 'approval.json', stage / 'approval.json')
        for asset in e.approval_assets(approval):
            shutil.copyfile(e.safe_file(source, asset['archive']), stage / asset['archive'])
        evidence = approval['evidence_asset']
        e.extract_files(stage / evidence['archive'], stage, evidence['members'])
        doc, _ = e.verify_assets(stage / 'approval.json', registry=registry, approved_templates=approved_templates,
                                 engine_dir=engine_dir, key=key)
        binary = doc['identities']['binary_sha256']
        profiles = {}
        for name, selected in doc['profiles'].items():
            receipt = e.load_json(e.safe_file(stage, selected['receipt']['path']))
            e.require(receipt.get('binary_sha256') == binary == selected['identity']['binary_sha256'],
                      'profile %s receipt names another binary than the approval' % name)
            profiles[name] = {'receipt': {'sha256': selected['receipt']['sha256']}}
    return {'binary_sha256': e.full_hash(binary), 'profiles': profiles}


def pinned_entry(entry, pins):
    """``entry`` with the pins added; existing pins must agree."""
    e.require(isinstance(entry, dict), 'registry entry required')
    updated = json.loads(json.dumps(entry))
    if 'binary_sha256' in updated:
        e.require(updated['binary_sha256'] == pins['binary_sha256'], 'registry entry already pins another binary_sha256')
    updated['binary_sha256'] = pins['binary_sha256']
    profiles = updated.setdefault('profiles', {})
    e.require(isinstance(profiles, dict), 'registry entry profiles must be an object')
    for name, pin in pins['profiles'].items():
        current = profiles.get(name, {}).get('receipt', {}).get('sha256')
        e.require(current in (None, pin['receipt']['sha256']), 'registry entry already pins another receipt for ' + name)
        profiles.setdefault(name, {})['receipt'] = dict(pin['receipt'])
    return updated


def main(argv=None):
    parser = argparse.ArgumentParser(prog='pin_carry_source.py', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--registry', required=True)
    parser.add_argument('--approved', required=True, help='template allow-list (runtimes/templates/APPROVED.json)')
    parser.add_argument('--key', required=True, help='<family>/<model>/v<N>')
    parser.add_argument('--approval-dir', required=True)
    parser.add_argument('--engine-dir')
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args(argv)
    try:
        registry_path = Path(args.registry)
        registry = e.load_json(registry_path)
        approved = e.load_json(args.approved)
        e.require(isinstance(approved, dict) and isinstance(approved.get('templates'), dict), 'invalid template allow-list')
        entry = (registry.get('approvals') or {}).get(args.key)
        e.require(isinstance(entry, dict), 'no registry entry ' + args.key)
        pins = derive_pins(args.approval_dir, registry=registry, key=args.key, approved_templates=approved['templates'],
                           engine_dir=Path(args.engine_dir) if args.engine_dir else None)
        updated = pinned_entry(entry, pins)
        if args.write:
            registry['approvals'][args.key] = updated
            registry_path.write_bytes(e.json_bytes(registry))
            print('PINNED %s TRUSTED_APPROVALS_SHA256=%s' % (args.key, e.sha256_file(registry_path)))
        else:
            print(json.dumps({args.key: updated}, indent=2, sort_keys=True))
    except (ValueError, OSError, KeyError, TypeError) as error:
        print('REFUSED: %s' % error, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
