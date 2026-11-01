#!/usr/bin/env python3
"""Model-branch completeness check (contract section 6).

    python3 tools/check_model_branch.py --repo-root PATH --base REF --head REF \
        [--rc-commit SHA] [--local] [--branch NAME] [--tree-dir PATH]

Prints one line per check (`PASS`/`FAIL name: reason`), exit 0 all pass, 1 any fail, 2 usage error.
--local additionally downloads and digest-checks the dev assets via tools/fetch_dev_assets.py.
Without --local the assets directory must already hold the authenticated dev assets (CI fetches them
with the same secret as verify). The registry, the template allow-list and the pin are read from the
base revision only, never from the head.
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

try:
    from . import release_evidence as e
    from . import model_layout as m
    from . import promote_runtime as p
    from .verify_runtime_tree import verify_tree
except ImportError:
    import release_evidence as e
    import model_layout as m
    import promote_runtime as p
    from verify_runtime_tree import verify_tree

CHECKS = ('branch/model', 'scope', 'MODEL.json', 'LATEST', 'tree', 'source commit', 'capability contract', 'docs and fragment')

ALLOWED_MODEL_PREFIXES = ('runtimes/', 'docs/packs/', 'changes/unreleased/')


class CheckError(ValueError):
    pass


def git(root, *args):
    result = subprocess.run(['git', '--no-replace-objects', '-C', str(root), *args],
                            capture_output=True, check=False)
    if result.returncode:
        raise CheckError('git ' + ' '.join(args) + ' failed')
    return result.stdout


def diff_dirs(base_dir, head_dir):
    """Changed files between two checked-out trees: union of paths present in either, differing bytes."""
    changed = []
    seen = set()
    for root_dir, other_dir in ((base_dir, head_dir), (head_dir, base_dir)):
        for base, dirs, names in os.walk(root_dir):
            rel_base = Path(base).relative_to(root_dir)
            if '.git' in dirs:
                dirs.remove('.git')
            for name in names:
                rel = (rel_base / name).as_posix()
                if rel in seen:
                    continue
                seen.add(rel)
                ours = Path(base) / name
                theirs = Path(other_dir) / rel
                if not theirs.is_file() or ours.read_bytes() != theirs.read_bytes():
                    changed.append(('M' if theirs.is_file() else 'A' if root_dir == head_dir else 'D', rel))
    return sorted(changed)


def git_bytes(root, ref, path):
    out = subprocess.run(['git', '--no-replace-objects', '-C', str(root), 'show', f'{ref}:{path}'],
                         capture_output=True, check=False)
    if out.returncode:
        raise CheckError('missing at ' + ref + ': ' + path)
    return out.stdout


def changed_files(root, base, head):
    out = git(root, 'diff', '-z', '--name-status', base, head)
    items = out.split(b'\0')
    files = []
    for i in range(0, len(items) - 1, 2):
        status = items[i].decode()
        path = items[i + 1].decode('utf-8', 'surrogateescape')
        files.append((status, path))
    return files


def model_folder_changes(files):
    folders = set()
    for _, path in files:
        match = re.fullmatch(r'runtimes/([^/]+)/([^/]+)/(?:v[1-9][0-9]*|MODEL\.json|LATEST)(?:/.*)?', path)
        if match:
            folders.add((match[1], match[2]))
    return folders


def run_checks(root, base, head, *, branch, rc_commit, local, assets_dir, tree_dir, base_tree_dir=None):
    results = []

    def report(name, ok, reason):
        results.append((name, ok, reason))

    def base_bytes(rel):
        if base_tree_dir is not None:
            path = Path(base_tree_dir) / rel
            if not path.is_file():
                raise CheckError('missing at base: ' + rel)
            return path.read_bytes()
        return git_bytes(root, base, rel)

    def head_bytes(rel):
        path = Path(tree_dir) / rel
        if path.is_file():
            return path.read_bytes()
        return git_bytes(root, head, rel)

    base = base.strip()
    head = head.strip()
    if base_tree_dir is None:
        git(root, 'rev-parse', '--verify', base + '^{commit}')
        git(root, 'rev-parse', '--verify', head + '^{commit}')
        files = changed_files(root, base, head)
    else:
        files = diff_dirs(base_tree_dir, tree_dir)

    model = None
    family = None
    if branch and branch.startswith('model/') and not branch.endswith('-approval'):
        model = branch[len('model/'):]
        m.model_name(model)
    if model is None:
        for name in CHECKS:
            report(name, True, 'not a model PR (no-op)')
        return results
    folders = model_folder_changes(files)
    if not folders:
        report('branch/model', False, 'no runtimes/<family>/<model>/ changes')
    elif len(folders) > 1:
        report('branch/model', False, 'more than one model folder changed: ' + ', '.join(sorted(f + '/' + g for f, g in folders)))
        model = None
    else:
        (family, changed_model), = folders
        if changed_model != model:
            report('branch/model', False, f'folder {changed_model} differs from branch model {model}')
            model = None
        else:
            report('branch/model', True, f'{family}/{model}')

    # Scope: section 3 model branch rules.
    scope_bad = []
    if model is not None:
        for status, path in files:
            if path.startswith('runtimes/'):
                if not re.match(rf'runtimes/{re.escape(family)}/{re.escape(model)}/', path):
                    scope_bad.append(path)
            elif path.startswith('docs/packs/'):
                if path != f'docs/packs/{model}.md':
                    scope_bad.append(path)
            elif path.startswith('changes/unreleased/'):
                if path != f'changes/unreleased/{model}.md':
                    scope_bad.append(path)
            else:
                scope_bad.append(path)
    report('scope', not scope_bad, '' if not scope_bad else 'out of scope: ' + ', '.join(scope_bad))

    tree_path = None
    model_doc = None
    approval_doc = None
    if model is not None:
        # MODEL.json check (validity vs its folder)
        try:
            m.validate_model_doc(e.loads(head_bytes(f'runtimes/{family}/{model}/MODEL.json')), family, model)
            report('MODEL.json', True, '')
        except (CheckError, ValueError) as error:
            report('MODEL.json', False, str(error))

        # LATEST check
        try:
            latest = head_bytes(f'runtimes/{family}/{model}/LATEST').decode().strip()
            match = re.fullmatch(r'v([1-9][0-9]*)', latest)
            tree_path = f'runtimes/{family}/{model}/{latest}'
            listing = git(root, 'ls-tree', '--name-only', head, '--', tree_path)
            ok = match is not None and listing.strip() != b''
            report('LATEST', ok, '' if ok else 'LATEST does not name an existing v<N>')
        except CheckError as error:
            report('LATEST', False, str(error))

    # Tree check: the sealed tree passes the verifier against the registry on the base.
    if model is not None and tree_path is not None:
        try:
            registry_raw = e.loads(base_bytes('runtimes/templates/APPROVALS.json'))
            approved_raw = e.loads(base_bytes('runtimes/templates/APPROVED.json'))['templates']
            pin_line = base_bytes('.github/workflows/verify.yml').decode()
            match = re.search(r'^\s*TRUSTED_APPROVALS_SHA256:\s*"?([0-9a-f]{64})"?\s*$', pin_line, re.M)
            if match is None:
                raise CheckError('verify.yml pin line missing on the base')
            e.load_registry(_write_temp(e.json_bytes(registry_raw)), match[1])
            version = int(tree_path.rsplit('/', 1)[1][1:])
            entry = registry_raw.get('approvals', {}).get(m.registry_key(family, model, version))
            if not isinstance(entry, dict):
                raise CheckError('registry entry missing on the base for the model version')
            runtime_id = entry.get('runtime_id')
            if local:
                tmp_assets = Path(tempfile.mkdtemp(prefix='.check-model-assets-'))
                p.download_release(f'approve-{runtime_id}-{version}',
                                   tmp_assets / family / model / f'v{version}',
                                   registry=registry_raw, approved_templates=approved_raw)
                assets_dir = tmp_assets
            if assets_dir is None:
                raise CheckError('no authenticated dev assets (run --local or provide --assets-dir)')
            tree_dir_path = Path(tree_dir) / 'runtimes' / family / model / f'v{version}'
            problems = verify_tree(tree_dir_path, approved_raw, trusted_approvals=registry_raw,
                                   assets_dir=assets_dir, family=family, model=model)
            approval_path = tree_dir_path / 'approval.json'
            approval_doc = e.load_json(approval_path) if approval_path.exists() else None
            report('tree', not problems, '' if not problems else '; '.join(problems))
        except (CheckError, ValueError, OSError) as error:
            report('tree', False, str(error))

    # Source commit check.
    if rc_commit:
        e.full_hash(rc_commit, e.HEX40)
        if approval_doc is not None:
            report('source commit', approval_doc.get('source_commit') == rc_commit,
                   '' if approval_doc.get('source_commit') == rc_commit
                   else 'sealed approval source commit differs from --rc-commit')
        else:
            report('source commit', False, 'sealed approval not available')
    else:
        report('source commit', True, 'not applicable (no --rc-commit)')

    # Capability contract from the sealed profile.
    if model is not None and tree_path is not None:
        try:
            profile_bytes = head_bytes(tree_path + '/profile.json')
            profile = e.loads(profile_bytes)
            models = e.loads(head_bytes(tree_path + '/model.lock.json'))
            problems = capability_problems(profile, model_doc, models=models)
            report('capability contract', not problems, '' if not problems else '; '.join(problems))
        except (CheckError, ValueError) as error:
            report('capability contract', False, str(error))
    else:
        report('capability contract', False, 'no tree to inspect')

    # Docs and fragment must exist and be non-empty at the head.
    if model is not None:
        missing = []
        for doc_path in (f'docs/packs/{model}.md', f'changes/unreleased/{model}.md'):
            try:
                if not head_bytes(doc_path).strip():
                    missing.append(doc_path + ' (empty)')
            except CheckError:
                missing.append(doc_path + ' (missing)')
        report('docs and fragment', not missing, '; '.join(missing))
    else:
        report('docs and fragment', False, 'no model')

    return results


# The engine's `--spec` values (dev crates/spec/src/mode.rs). `auto` lets the arbiter use every drafter
# the family offers (its MTP head or its draft pack) and `mtp` drafts with the MTP head; those are the
# speculative decoding the capability contract requires. `copy` (prompt lookup, no model compute) and
# `none` (serial) are not.
SPEC_MODES = ('auto', 'mtp', 'copy', 'none')
SPECULATIVE_MODES = ('auto', 'mtp')


def speculation_on(argv, *, draft_pack=False):
    """Does a sealed profile argv run speculative decoding, read the way the engine parses it?

    The engine takes `--spec VALUE` (the last occurrence wins; the default is `auto`), `--draft DIR`, and
    `--no-drafts` (no drafting whatever the mode). The contract needs speculation declared in the sealed
    profile, not left to the engine default: `--spec auto|mtp`, `--draft DIR`, or a `draft` role in the
    model lock (``draft_pack``). The effective mode must then be one of SPECULATIVE_MODES with drafting
    on. A `--spec` or `--draft` without a value, or an unknown `--spec` value, fails closed.
    """
    if not isinstance(argv, list) or not all(isinstance(token, str) for token in argv):
        return False
    mode, declared, drafting = 'auto', bool(draft_pack), True
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in ('--spec', '--draft'):
            if index + 1 >= len(argv) or argv[index + 1].startswith('--') or not argv[index + 1]:
                return False
            if token == '--spec':
                if argv[index + 1] not in SPEC_MODES:
                    return False
                mode = argv[index + 1]
            declared = True
            index += 2
            continue
        if token == '--no-drafts':
            drafting = False
        index += 1
    return declared and drafting and mode in SPECULATIVE_MODES


def capability_problems(profile, model_doc, models=None):
    problems = []
    e.require(isinstance(profile, dict) and profile.get('schema') == 'rexl3.runtime-profile.v1'
              and profile.get('state') == 'final', 'sealed runtime profile required')
    profiles = profile.get('profiles')
    if not isinstance(profiles, dict) or not profiles:
        return ['profile has no choices']
    draft_pack = False
    if isinstance(models, dict):
        packs = models.get('packs')
        if isinstance(packs, dict):
            # Only the `draft` role is a drafter. One runtime entry may pin a pack per profile (dev #159),
            # so another non-target role is an alternate target pack, not a draft.
            draft_pack = bool(packs.get('draft'))
    vision_on = False
    for name, choice in profiles.items():
        if not isinstance(choice, dict):
            problems.append(f'profile {name} malformed')
            continue
        caps = choice.get('capabilities')
        if not isinstance(caps, dict):
            problems.append(f'profile {name} capabilities missing')
            continue
        if caps.get('prefix_cache') is not True:
            problems.append(f'profile {name}: prefix cache is not on and proven')
        if not speculation_on(choice.get('argv'), draft_pack=draft_pack):
            problems.append(f'profile {name}: speculative decoding is not on and proven')
        if caps.get('images') is True:
            vision_on = True
    if not vision_on:
        justification = (model_doc or {}).get('pack', {}).get('vision_does_not_fit')
        if not isinstance(justification, str) or not justification.strip():
            problems.append('vision is off and MODEL.json records no does-not-fit justification')
    return problems


def _write_temp(data):
    path = Path(tempfile.mkdtemp(prefix='.check-model-registry-')) / 'registry.json'
    path.write_bytes(data)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--head', required=True)
    parser.add_argument('--rc-commit')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--branch')
    parser.add_argument('--tree-dir', help='checked-out head working tree (default: --repo-root)')
    parser.add_argument('--base-tree-dir', help='checked-out base working tree (default: read base files from git objects)')
    parser.add_argument('--assets-dir', help='authenticated dev assets root (default: REXL3_ASSETS_DIR)')
    args = parser.parse_args(argv)
    if args.tree_dir is None:
        args.tree_dir = args.repo_root
    assets_dir = args.assets_dir
    if assets_dir is None:
        assets_dir = os.environ.get('REXL3_ASSETS_DIR')
    try:
        results = run_checks(args.repo_root, args.base, args.head, branch=args.branch,
                             rc_commit=args.rc_commit, local=args.local,
                             assets_dir=assets_dir, tree_dir=args.tree_dir,
                             base_tree_dir=args.base_tree_dir)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print('FAIL internal: ' + str(error))
        return 1
    failed = 0
    for name, ok, reason in results:
        line = ('PASS' if ok else 'FAIL') + ' ' + name
        if reason:
            line += ': ' + reason
        print(line)
        if not ok:
            failed += 1
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
