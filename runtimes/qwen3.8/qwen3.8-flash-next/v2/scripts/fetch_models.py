#!/usr/bin/env python3
"""Immutable, resumable acquisition shared by setup/start/verify (stdlib only).

Generalizes MiniCPM's fetch_models.sh: only declared roles, full revision and exact
filenames, an isolated HF client, mandatory bytes/SHA AND engine fingerprints.
No repo/path overrides. Interrupted HF .cache files may resume; a corrupt final
payload is never silently replaced. --accept-model-download is NOT installer --yes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import signal
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract import ContractError, DSV_REPO, DSV_REPOS, canonical_digest, load_json, relative_file, safe_file, verify_release

ROOT = Path(__file__).resolve().parents[1]
DISK_RESERVE_BYTES = 1024 * 1024 * 1024
OWNED_TERM_SECONDS = 5
OWNED_KILL_SECONDS = 5


class RuntimeFailure(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def demand(ok, code, message):
    if not ok:
        raise RuntimeFailure(code, message)


def clean_environment(explicit=None):
    """No inherited model/tuning, loader injection, proxy or credential variables."""
    env = {'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8',
           'HOME': str(Path.home()), 'PYTHONDONTWRITEBYTECODE': '1'}
    # Only the caller's selected temporary directory is operational, not tuning.
    if 'TMPDIR' in os.environ:
        env['TMPDIR'] = os.environ['TMPDIR']
    env.update(explicit or {})
    return env


def run_owned(argv, *, env, timeout, stdout=subprocess.PIPE, cancel=None, terminate_timeout=OWNED_TERM_SECONDS):
    """Bounded work plus TERM/KILL cleanup; native up needs its own rollback budget."""
    proc = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=stdout,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True)
    deadline = time.monotonic() + timeout
    try:
        while True:
            demand(cancel is None or not cancel.is_set(), 24, 'OPERATION_INTERRUPTED')
            demand(time.monotonic() < deadline, 24, 'OPERATION_TIMEOUT')
            try:
                output, _ = proc.communicate(timeout=min(.2, max(.01, deadline-time.monotonic())))
                return subprocess.CompletedProcess(argv, proc.returncode, output)
            except subprocess.TimeoutExpired:
                continue
    finally:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                # Drain pipes while a native launcher reports rollback progress.
                proc.communicate(timeout=terminate_timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=OWNED_KILL_SECONDS)
        if proc.stdout:
            proc.stdout.close()


def secure_directory(path, create=True):
    path = Path(os.path.abspath(path))
    cursor = Path(path.anchor)
    private_ancestor = False
    for component in path.parts[1:]:
        cursor /= component
        if create:
            try:
                cursor.mkdir(mode=0o700)
            except FileExistsError:
                pass
        if cursor.exists() or cursor.is_symlink():
            info = cursor.lstat()
            demand(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode), 22, 'UNSAFE_STATE_DIRECTORY')
            # Trusted root-owned sticky temp parents are permitted, but an
            # attacker-writable XDG/state parent may not replace our lock inode.
            trusted_sticky = info.st_uid == 0 and bool(info.st_mode & stat.S_ISVTX)
            demand(not info.st_mode & 0o022 or trusted_sticky or private_ancestor, 22, 'UNSAFE_STATE_PARENT_PERMISSIONS')
            # A private owner-only ancestor makes lower group bits unreachable
            # to other accounts. Never relax the final state-directory policy.
            private_ancestor |= info.st_uid == os.getuid() and not info.st_mode & 0o077
        elif not create:
            return path
    if path.exists():
        info = path.stat()
        demand(info.st_uid == os.getuid() and not info.st_mode & 0o022, 22, 'UNSAFE_STATE_PERMISSIONS')
    return path


def state_root(profile, create=False):
    base = Path(os.environ.get('XDG_DATA_HOME', str(Path.home() / '.local/share')))
    demand(base.is_absolute(), 2, 'XDG_DATA_HOME_MUST_BE_ABSOLUTE')
    return secure_directory(base / 'rexl3' / profile['runtime_id'], create=create)


def private_open(path, mode='a', *, node_lock=False):
    """Never follow a state/log/lock symlink or operate on a device/FIFO."""
    path = Path(path)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0) | os.O_NONBLOCK
    if mode == 'x':
        flags |= os.O_EXCL
    if mode == 'a':
        flags |= os.O_APPEND
    fd = os.open(path, flags, 0o600)
    try:
        info = os.fstat(fd)
        # Node flock files may already be 0644; read permission cannot replace
        # the lock inode. State/log files stay private; no file may be writable
        # by another account and no chmod/unlink is used to 'repair' a lock.
        forbidden = 0o022 if node_lock else 0o077
        demand(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and info.st_nlink == 1
               and not info.st_mode & forbidden, 22, 'UNSAFE_STATE_FILE')
        return os.fdopen(fd, 'a+' if mode == 'a' else 'w+')
    except BaseException:
        os.close(fd)
        raise


@contextmanager
def advisory_lock(path, busy='RUNTIME_BUSY', *, node_lock=False):
    with private_open(path, node_lock=node_lock) as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeFailure(22, busy) from error
        try:
            yield stream
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def lock_held(path):
    if not Path(path).exists():
        return False
    try:
        with advisory_lock(path):
            return False
    except RuntimeFailure as error:
        if str(error) != 'RUNTIME_BUSY':
            raise
        return True


def atomic_json(path, value):
    path = Path(path)
    demand(not path.is_symlink(), 22, 'UNSAFE_STATE_FILE')
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def model_paths(state, models, roles=None):
    roles = set(models['packs']) if roles is None else set(roles)
    return {role: state / 'models' / role / pack['revision'] for role, pack in models['packs'].items() if role in roles}


def inspect_payload(directory, pack, allow_missing=False):
    """Managed WRITE destinations: full hashes, no symlinks or hardlinks.

    External read-only reuse goes through verify_payload instead. Never reuse its
    stable-hardlink permission for this downloader's destination/cache namespace.
    """
    if directory.exists() or directory.is_symlink():
        secure_directory(directory, create=False)
        for entry in directory.rglob('*'):
            rel = entry.relative_to(directory).as_posix()
            info = entry.lstat()
            demand(not stat.S_ISLNK(info.st_mode) and (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)),
                   23, 'MODEL_INTEGRITY: link or special file')
            if entry.is_file():
                # Only official local-dir resumable-download bookkeeping is exempt.
                demand(rel in pack['files'] or rel.startswith('.cache/huggingface/'), 23, 'MODEL_INTEGRITY: unlisted sidecar')
                demand(info.st_nlink == 1, 23, 'MODEL_INTEGRITY: hardlinked payload')
    missing = []
    for rel, item in pack['files'].items():
        relative_file(rel)
        demand(not rel.startswith('.cache/'), 2, 'RESERVED_MODEL_PATH')
        path = directory / rel
        if not path.exists():
            demand(allow_missing, 23, 'MODEL_INTEGRITY: missing payload')
            missing.append(rel)
            continue
        demand(path.is_file() and not path.is_symlink(), 23, 'MODEL_INTEGRITY: non-regular payload')
        from remote_identity import verify_file
        try:
            # Single-link admission through the final hash/readback, not only a
            # preflight lstat that a concurrently created alias can invalidate.
            verify_file(path, item)
        except (ValueError, OSError) as error:
            raise RuntimeFailure(23, 'MODEL_INTEGRITY: corrupt or unsafe payload; preserve and inspect it explicitly, no automatic replacement') from error
    return missing


def isolated_hf(root, state, release):
    descriptor_path = state / 'python-environment.json'
    demand(descriptor_path.is_file() and not descriptor_path.is_symlink(), 20, 'HF_CLIENT_MISSING: run install-deps')
    descriptor = load_json(descriptor_path)
    policy = load_json(safe_file(root, 'dependencies.lock.json'))
    requirement = relative_file(policy['python_requirements'])
    expected = release['files'][requirement]['sha256']
    demand(isinstance(descriptor, dict) and descriptor.get('requirements_sha256') == expected,
           20, 'HF_CLIENT_REQUIREMENTS_MISMATCH: run install-deps')
    envdir = state / ('venv-' + expected)
    hf = envdir / 'bin/hf'
    python = envdir / 'bin/python'
    demand(descriptor.get('hf') == str(hf) and descriptor.get('python') == str(python)
           and hf.is_file() and not hf.is_symlink() and python.is_file()
           and os.access(hf, os.X_OK) and os.access(python, os.X_OK),
           20, 'HF_CLIENT_DESCRIPTOR_INVALID: run install-deps')
    secure_directory(envdir, create=False)
    secure_directory(hf.parent, create=False)
    return hf


def inspect_index(directory, pack):
    """Pinned index bytes must not direct the native loader outside the manifest."""
    name = 'model.safetensors.index.json'
    if name not in pack['files']:
        return
    try:
        index = load_json(safe_file(directory, name))
        weight_map = index.get('weight_map') if isinstance(index, dict) else None
        demand(isinstance(weight_map, dict) and bool(weight_map), 23, 'MODEL_INDEX_INVALID')
        references = set()
        for value in weight_map.values():
            path = relative_file(value)
            demand(path.endswith('.safetensors') and path in pack['files'], 23, 'MODEL_INDEX_INVALID')
            references.add(path)
        if pack['repo'] in DSV_REPOS:
            expected = {path for path in pack['files'] if path.endswith('.safetensors')}
            demand(references == expected, 23, 'MODEL_INDEX_INVALID: DSV shard roster mismatch')
    except (ContractError, OSError, TypeError, ValueError):
        raise RuntimeFailure(23, 'MODEL_INDEX_INVALID') from None


def fingerprint_check(root, models, paths, cancel=None, binary=None):
    """Verify every selected role's fingerprint through the engine's probe.

    The engine names the primary (non-draft) role `target` and the optional draft
    role `draft`; a named lock role is probed exactly the same way as the target.
    """
    primary = next(role for role in paths if role != 'draft')
    argv = [str(binary or safe_file(root, 'bin/rexl3-serve')), '--print-fingerprints', '--model', str(paths[primary])]
    if 'draft' in paths:
        argv += ['--draft', str(paths['draft'])]
    try:
        result = run_owned(argv, env=clean_environment(), timeout=120, cancel=cancel)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeFailure(23, 'MODEL_FINGERPRINT_PROBE_FAILED') from error
    demand(result.returncode == 0, 23, 'MODEL_FINGERPRINT_PROBE_FAILED')
    got = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if fields and fields[0] in ('target', 'draft'):
            demand(len(fields) == 2 and fields[0] not in got, 23, 'MODEL_FINGERPRINT_INVALID')
            got[fields[0]] = fields[1]
    labels = {'target': primary}
    if 'draft' in paths:
        labels['draft'] = 'draft'
    expected = {label: models['packs'][role]['fingerprint'] for label, role in labels.items()}
    demand(got == expected, 23, 'MODEL_FINGERPRINT_MISMATCH')


def prepare_models(root=ROOT, *, accept_download=False, verify_only=False, cancel=None, existing_paths=None, binary=None, roles=None):
    root = Path(root)
    profile, models, release = verify_release(root)
    if roles is None:
        roles = {'target'} | ({'draft'} if 'draft' in models['packs'] else set())
    roles = set(roles)
    primary = roles - {'draft'}
    demand(bool(primary) and len(primary) == 1 and primary <= set(models['packs']),
           2, 'EXISTING_MODEL_ROLES_MISMATCH')
    if existing_paths is not None:
        # No acquisition, chmod, marker, cache, rename or receipt in external trees.
        from remote_identity import absolute_path, verify_file, verify_payload
        demand(isinstance(existing_paths, dict) and set(existing_paths) == roles,
               2, 'EXISTING_MODEL_ROLES_MISMATCH')
        paths = {role: Path(absolute_path(path)) for role, path in existing_paths.items()}
        try:
            selected = Path(binary) if binary else safe_file(root, 'bin/rexl3-serve')
            verify_file(selected, release['files']['bin/rexl3-serve'])
            for role in roles:
                verify_payload(paths[role], models['packs'][role])
                inspect_index(paths[role], models['packs'][role])
            fingerprint_check(root, models, paths, cancel=cancel, binary=selected)
        except (ValueError, OSError) as error:
            from remote_identity import reason_code
            # The code names the failed check and carries no path; anything else stays hidden.
            code = reason_code(str(error)) if isinstance(error, ValueError) else None
            raise RuntimeFailure(23, 'EXISTING_MODEL_IDENTITY_FAILED' + (': ' + code if code else '')) from error
        return paths
    state = state_root(profile, create=True)
    paths = model_paths(state, models, roles)
    with advisory_lock(state / 'models.lock', 'MODEL_SETUP_BUSY'):
        marker = state / 'models.verified.json'
        if marker.exists() or marker.is_symlink():
            demand(not marker.is_symlink(), 23, 'MODEL_INTEGRITY: unsafe verification record')
        missing = {}
        receipts = {}
        for role in roles:
            pack = models['packs'][role]
            for name in pack['files']:
                demand(not name.startswith('-') and not any(char in name for char in '*?['), 2, 'UNSAFE_HF_FILENAME')
            receipt = state / 'model-receipts' / role / (pack['revision'] + '.json')
            receipts[role] = receipt
            if receipt.exists() or receipt.is_symlink():
                demand(not receipt.is_symlink() and load_json(receipt).get('pack_digest') == canonical_digest(pack),
                       23, 'MODEL_INTEGRITY: verified revision manifest changed')
            missing[role] = inspect_payload(paths[role], pack, allow_missing=not verify_only and not receipt.exists())
        count_bytes = sum(models['packs'][role]['files'][name]['bytes'] for role, names in missing.items() for name in names)
        if count_bytes:
            hf = isolated_hf(root, state, release)
            print('MODEL_DOWNLOAD: up to ' + str(count_bytes) + ' bytes into ' + str(state / 'models'), flush=True)
            if not accept_download:
                demand(sys.stdin.isatty(), 20, 'MODEL_DOWNLOAD_APPROVAL_REQUIRED: use --accept-model-download (separate from --yes)')
                demand(input('Download the exact locked model files? [y/N] ').strip().lower() in ('y', 'yes'),
                       20, 'MODEL_DOWNLOAD_DECLINED')
            # Conservative: full remaining files + 1 GiB; resumable fragments may already consume disk.
            demand(shutil.disk_usage(state).free >= count_bytes + DISK_RESERVE_BYTES, 22, 'INSUFFICIENT_MODEL_DISK')
            hf_home = secure_directory(state / 'hf-client')
            env = clean_environment({'HF_HOME': str(hf_home), 'HF_ENDPOINT': 'https://huggingface.co',
                                     'HF_HUB_DISABLE_TELEMETRY': '1', 'HF_HUB_DISABLE_PROGRESS_BARS': '1'})
            token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
            if token:
                env['HF_TOKEN'] = token  # client only: never argv, stdout, state or server environment
            else:
                env['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
            for role, names in missing.items():
                directory = secure_directory(paths[role])
                pack = models['packs'][role]
                for name in names:
                    argv = [str(hf), 'download', pack['repo'], name, '--revision', pack['revision'], '--local-dir', str(directory)]
                    try:
                        result = run_owned(argv, env=env, stdout=subprocess.DEVNULL, timeout=24 * 60 * 60, cancel=cancel)
                    except (OSError, subprocess.SubprocessError) as error:
                        raise RuntimeFailure(23, 'MODEL_DOWNLOAD_FAILED: retry setup to resume') from error
                    demand(result.returncode == 0, 23, 'MODEL_DOWNLOAD_FAILED: retry setup to resume')
                    # Fail immediately rather than downloading further files after a corrupted response.
                    inspect_payload(directory, pack, allow_missing=True)
        for role in roles:
            pack = models['packs'][role]
            inspect_payload(paths[role], pack)
            inspect_index(paths[role], pack)
        fingerprint_check(root, models, paths, cancel=cancel)
        if not verify_only:
            for role, receipt in receipts.items():
                secure_directory(receipt.parent)
                atomic_json(receipt, {'schema': 'rexl3.pack-verified.v1', 'pack_digest': canonical_digest(models['packs'][role])})
            atomic_json(marker, {'schema': 'rexl3.models-verified.v1', 'model_lock_digest': canonical_digest(models)})
    return paths


def main(argv=None, root=ROOT):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accept-model-download', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--pack', default='target', help='lock role to fetch and verify (default: target)')
    options = parser.parse_args(argv)
    try:
        prepare_models(root, accept_download=options.accept_model_download, verify_only=options.verify_only,
                       roles={options.pack} | ({'draft'} if _has_draft(root) else set()))
        print('MODELS_VERIFIED')
        return 0
    except RuntimeFailure as error:
        print(str(error), file=sys.stderr); return error.code
    except (ContractError, KeyError, TypeError, OSError, ValueError):
        print('MODEL_CONTRACT_INVALID', file=sys.stderr); return 2


def _has_draft(root):
    from contract import load_json
    try:
        models = load_json(safe_file(Path(root), 'model.lock.json'))
    except ContractError:
        return False
    return 'draft' in models.get('packs', {})


if __name__ == '__main__':
    raise SystemExit(main())
