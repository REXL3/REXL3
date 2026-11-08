#!/usr/bin/env python3
"""Classify each manifest model entry's qualification scope for a release (contract section 7).

    python3 tools/release_scope.py classify --repo-root PATH --version vX.Y.Z

The release PR attaches this output so reviewers can see, per model, which tier it is cut
under and the exact commit the tier was assessed against:

  - T0 (untouched scope): the model's family/kernel scope is untouched by the newer engine
    commits, so its prior sealed approval travels as-is in the manifest (`carried_forward: true`
    with `qualified_engine_release_tag`, `qualification_tier`, `scope_diff_base`).
  - T1/T2 (shared-glue smoke / gate-scoped re-run): the required gates re-run on the new engine
    binary, unaffected gate receipts carry forward (`carried_gates`), and a new approval is sealed
    on the new engine build. The manifest itself only marks carried-forward entries; a fresh entry
    is reported as `fresh` and its tier is decided by the model PR's scope.
  - T3: a new model or pack weight/sampling/chat-template change; full re-qualification.

The tool validates the manifest (schema `rexl3.release-manifest.v1`) and reads each model's
sealed approval from the local prod tree to report the true source commit. It never writes or
publishes anything.
"""
import argparse
from pathlib import Path
import sys

try:
    from . import release_evidence as e
    from .release_manifest import validate_manifest
except ImportError:
    import release_evidence as e
    from release_manifest import validate_manifest


def classify(repo_root, version):
    root = Path(repo_root)
    manifest = validate_manifest(e.load_json(root / 'releases' / version / 'manifest.json'), version)
    engine = manifest['engine']
    lines = [f'engine {engine["engine_release_tag"]} dev_commit {engine["dev_commit"]}']
    for item in manifest['models']:
        name = f"{item['family']}/{item['model']} {item['version']}"
        if item.get('carried_forward') is True:
            lines.append(f'{name}: T0 scope_diff_base {item["scope_diff_base"]} '
                         f'engine_release_tag {item["qualified_engine_release_tag"]}')
            continue
        tree = root / 'runtimes' / item['family'] / item['model'] / item['version']
        approval_path = tree / 'approval.json'
        e.require(approval_path.is_file(), 'sealed tree missing: ' + str(tree))
        source = e.load_json(approval_path).get('source_commit')
        e.full_hash(source, e.HEX40)
        lines.append(f'{name}: fresh scope_diff_base {source} engine_release_tag {engine["engine_release_tag"]}')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    classify_parser = sub.add_parser('classify', help='classify a release manifest')
    classify_parser.add_argument('--repo-root', default='.')
    classify_parser.add_argument('--version', required=True)
    args = parser.parse_args(argv)
    try:
        print(classify(args.repo_root, args.version))
    except (ValueError, OSError, KeyError, TypeError) as error:
        print('SCOPE_REFUSED: ' + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
