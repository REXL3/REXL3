#!/usr/bin/env python3
"""Ubuntu 24.04/ARM64 dependency adapter; no release versions live in code.

Only an externally verified, finalized bundle may enter this installer. The
reviewed dependency lock must include the complete apt solver change set, not
just top-level requests. A new package family needs a reviewed adapter change.
Base packages may be newer than their builder snapshot; native library/driver
versions are deliberately strict. No OS upgrade, downgrade, removal, repository
provisioning, compiler/source fallback, global pip or reboot is performed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid

# -I intentionally omits the script directory. Trust only this bundled directory.
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from contract import ContractError, load_json, relative_file, safe_file, sha256_file, verify_release

LOCK_TIMEOUT_S = 120
SAFE_PATH = '/usr/sbin:/usr/bin:/sbin:/bin'
NATIVE_FIELDS = frozenset({'version', 'commit', 'system_fingerprint', 'nvcc', 'cuda_arch',
                         'kernel_hash', 'cudart', 'cuda_driver', 'cublas'})
# Leaf packages only; no cuda, cuda-toolkit, cuda-drivers, kernel or OS meta-package.
BASE_PACKAGES = frozenset({
    'apt', 'dpkg', 'bash', 'coreutils', 'ca-certificates', 'python3', 'python3-minimal',
    'python3-venv', 'python3.12', 'python3.12-minimal', 'python3.12-venv',
    'python3-pip-whl', 'python3-setuptools-whl', 'libpython3-stdlib',
    'libpython3.12-minimal', 'libpython3.12-stdlib', 'libc6', 'libstdc++6', 'libgcc-s1',
    'libgomp1', 'libnuma1', 'libssl3t64', 'libffi8', 'libbz2-1.0', 'libexpat1',
    'liblzma5', 'libsqlite3-0', 'libreadline8t64', 'libtinfo6', 'zlib1g',
    'libuuid1', 'libcrypt1', 'libgdbm6t64', 'libgdbm-compat4t64', 'readline-common',
    'media-types', 'tzdata', 'openssl', 'curl', 'wget', 'rsync', 'openssh-client',
    'tar', 'gzip', 'unzip', 'xz-utils', 'util-linux', 'pciutils', 'iproute2',
})
NATIVE_PACKAGE = re.compile(r'(?:cuda-cudart-\d+-\d+|libcublas-\d+-\d+|libnccl2|'
                            r'libnvidia-compute-\d+(?:-server)?|nvidia-utils-\d+(?:-server)?)\Z')
COMMAND_PACKAGES = {
    'python3': ('python3',), 'bash': ('bash',), 'dpkg-query': ('dpkg',), 'dpkg': ('dpkg',),
    'apt-get': ('apt',), 'sha256sum': ('coreutils',), 'timeout': ('coreutils',),
    'dirname': ('coreutils',), 'uname': ('coreutils',), 'id': ('coreutils',),
    'curl': ('curl',), 'wget': ('wget',), 'rsync': ('rsync',), 'ssh': ('openssh-client',),
    'scp': ('openssh-client',), 'tar': ('tar',), 'gzip': ('gzip',), 'unzip': ('unzip',),
    'xz': ('xz-utils',), 'flock': ('util-linux',), 'lspci': ('pciutils',), 'ip': ('iproute2',),
}
LIBRARY_PACKAGES = {
    'libc.so.6': ('libc6',), 'libm.so.6': ('libc6',), 'libdl.so.2': ('libc6',),
    'libpthread.so.0': ('libc6',), 'librt.so.1': ('libc6',),
    'libstdc++.so.6': ('libstdc++6',), 'libgcc_s.so.1': ('libgcc-s1',),
    'libgomp.so.1': ('libgomp1',), 'libnuma.so.1': ('libnuma1',),
    'libssl.so.3': ('libssl3t64',), 'libcrypto.so.3': ('libssl3t64',),
    'libnccl.so.2': ('libnccl2',),
}


class DependencyFailure(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def demand(ok, code, message):
    if not ok:
        raise DependencyFailure(code, message)


def clean_environment():
    """Do not forward tokens, proxy credentials, pip config or tuning to probes."""
    result = {'PATH': SAFE_PATH, 'HOME': str(Path.home()), 'LC_ALL': 'C', 'LANG': 'C',
              'PIP_CONFIG_FILE': os.devnull, 'PIP_DISABLE_PIP_VERSION_CHECK': '1',
              'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
              'HF_HUB_DISABLE_TELEMETRY': '1', 'HF_HUB_OFFLINE': '1', 'DO_NOT_TRACK': '1',
              'DEBIAN_FRONTEND': 'noninteractive', 'NEEDRESTART_MODE': 'l'}
    for name in ('XDG_DATA_HOME', 'XDG_CACHE_HOME', 'TMPDIR'):
        if name in os.environ:
            result[name] = os.environ[name]
    return result


OUTPUT_LIMIT = 256 * 1024
STOP_GRACE_S = 2


def _command_failure(kind, argv):
    label = Path(argv[0]).name if argv else 'dependency-command'
    if not re.fullmatch(r'[A-Za-z0-9_.+-]{1,80}', label):
        label = 'dependency-command'
    return DependencyFailure(20, f'{kind}: {label}; inspect locally; see docs/SETUP.md')


@contextmanager
def _deferred_signals(cancel=None):
    """Keep termination from unwinding the installation lock before drain.

    A second Ctrl-C must not bypass cleanup. SIGKILL/host failure cannot be
    caught; Linux service RuntimeMaxSec still bounds the owned transaction.
    Internal control calls share the outer handler but ignore its pending flag.
    """
    if cancel is not None:
        yield cancel
        return
    pending, previous = [], {}
    try:
        def remember(signum, frame):
            if not pending:
                pending.append(signum)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous[signum] = signal.signal(signum, remember)
    except ValueError:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        raise DependencyFailure(20, 'DEPENDENCY_SUPERVISION_REQUIRES_MAIN_THREAD') from None
    try:
        yield pending
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


class _Capture:
    """Drain both pipes concurrently; keep at most OUTPUT_LIMIT bytes per pipe."""
    def __init__(self, process):
        self.selector = selectors.DefaultSelector()
        self.streams = [process.stdout, process.stderr]
        self.buffers = [bytearray(), bytearray()]
        self.exceeded = False
        try:
            for index, stream in enumerate(self.streams):
                if stream is not None:
                    os.set_blocking(stream.fileno(), False)
                    self.selector.register(stream, selectors.EVENT_READ, index)
        except BaseException:
            self.close()
            raise

    @property
    def eof(self):
        return not self.selector.get_map()

    def drain(self, timeout):
        if self.eof:
            time.sleep(timeout)
            return
        for key, _ in self.selector.select(timeout):
            chunk = os.read(key.fd, 65536)
            if not chunk:
                self.selector.unregister(key.fileobj)
                continue
            buffer = self.buffers[key.data]
            room = OUTPUT_LIMIT - len(buffer)
            buffer.extend(chunk[:room])
            self.exceeded |= len(chunk) > room

    def output(self):
        return tuple(None if stream is None else bytes(buffer).decode('utf-8', errors='replace')
                     .replace('\r\n', '\n').replace('\r', '\n')
                     for stream, buffer in zip(self.streams, self.buffers))

    def close(self):
        self.selector.close()
        for stream in self.streams:
            if stream is not None:
                stream.close()


def _raise_after_drain(reason, argv, cancel):
    if cancel and cancel[0] == signal.SIGINT:
        raise KeyboardInterrupt
    if reason:
        raise _command_failure(reason, argv)


def _run_local(argv, *, timeout=900, interactive=False, env=None, cancel=None) -> subprocess.CompletedProcess[str]:
    """Owned transport for fixed control tools and non-Linux CPU fixtures.

    Do not use this as a Linux installer sandbox: a process group cannot own
    setsid/double-fork descendants. run() therefore requires service cgroups on
    Linux. Keep the session leader unreaped while its pipes remain open, so a
    wrapper that already exited cannot hide a still-running pipe-owning child,
    and the process-group ID cannot be reused before our final signal.
    """
    with _deferred_signals(cancel) as pending:
        process = subprocess.Popen(argv, stdin=None if interactive else subprocess.DEVNULL,
                                   stdout=None if interactive else subprocess.PIPE,
                                   stderr=None if interactive else subprocess.PIPE,
                                   env=clean_environment() if env is None else env, shell=False,
                                   start_new_session=not interactive)
        capture, reason, stopping, killed = None, None, None, False
        deadline = time.monotonic() + timeout
        def send(signum):
            try:
                if interactive:  # sudo -v must retain the controlling terminal.
                    process.send_signal(signum)
                else:
                    os.killpg(process.pid, signum)
            except (ProcessLookupError, PermissionError):
                # macOS may report EPERM for a zombie-only group. A denied
                # signal is never proof of exit: still drain pipes and wait.
                pass
        try:
            capture = _Capture(process)
            while True:
                capture.drain(.02)
                now = time.monotonic()
                if pending:
                    reason = reason or 'DEPENDENCY_INTERRUPTED'
                if capture.exceeded:
                    reason = reason or 'DEPENDENCY_OUTPUT_LIMIT'
                if now >= deadline:
                    reason = reason or 'DEPENDENCY_TIMEOUT'
                if reason and stopping is None:
                    send(signal.SIGTERM)
                    stopping = now
                if stopping is not None and not killed and now >= stopping + STOP_GRACE_S:
                    send(signal.SIGKILL)
                    killed = True
                if capture.eof:
                    try:
                        process.wait(timeout=.02)
                        break
                    except subprocess.TimeoutExpired:
                        pass
            output = capture.output()
        except BaseException:
            # No poll()/wait() has reaped the leader on this path. Never signal
            # a remembered PID after reaping it, and never release on timeout
            # while our owned child is still running.
            if process.returncode is None:
                send(signal.SIGKILL)
            process.wait()
            raise
        finally:
            if capture is not None:
                capture.close()
        _raise_after_drain(reason, argv, pending)
        return subprocess.CompletedProcess(argv, process.returncode, *output)


def _manager_environment(privileged):
    env = clean_environment()
    if not privileged:
        # Do not inherit an arbitrary D-Bus address or user runtime directory.
        env['XDG_RUNTIME_DIR'] = '/run/user/' + str(os.getuid())
        env['DBUS_SESSION_BUS_ADDRESS'] = 'unix:path=' + env['XDG_RUNTIME_DIR'] + '/bus'
    return env


def _require_systemd(*, privileged, cancel=None):
    """Ubuntu 24.04 base: systemd >=255, cgroup v2 and a reachable manager.

    A normal login's user manager is required for probes/venv/pip; the system
    manager owns apt. No installation of supervision tools, lingering, sudoers,
    cgroup delegation, or systemwide configuration changes are attempted.
    """
    message = 'DEPENDENCY_SUPERVISION_UNAVAILABLE: require Ubuntu 24.04 systemd/cgroup-v2 and a user login; see docs/SETUP.md'
    demand(sys.platform == 'linux' and Path('/run/systemd/system').is_dir()
           and Path('/sys/fs/cgroup/cgroup.controllers').is_file()
           and Path('/usr/bin/systemd-run').is_file(), 20, message)
    try:
        result = _run_local(['/usr/bin/systemctl', '--system' if privileged else '--user',
                             '--no-ask-password', 'show', '--property=Version', '--value'],
                            timeout=5, env=_manager_environment(privileged), cancel=cancel)
    except (OSError, DependencyFailure):
        raise DependencyFailure(20, message) from None
    match = re.match(r'([0-9]+)(?:[.\s-]|$)', result.stdout or '')
    demand(result.returncode == 0 and match and int(match[1]) >= 255, 20, message)


def _service_command(argv, unit, timeout, *, privileged):
    command = ['/usr/bin/systemd-run', '--system' if privileged else '--user',
               '--no-ask-password', '--quiet', '--wait', '--pipe', '--collect',
               '--service-type=exec', '--expand-environment=no', '--unit=' + unit,
               '--property=ExitType=cgroup', '--property=KillMode=control-group',
               '--property=SendSIGKILL=yes', '--property=Restart=no',
               '--property=TimeoutStartSec=30s', f'--property=TimeoutStopSec={STOP_GRACE_S}s',
               f'--property=RuntimeMaxSec={timeout:g}s']
    if privileged:
        demand(argv[0] in ('apt-get', '/usr/bin/apt-get'), 20, 'PRIVILEGED_COMMAND_REFUSED')
        # Only the distro apt executable is elevated, never candidate Python,
        # a bundled helper, a shell script, or a caller-controlled executable.
        argv = ['/usr/bin/apt-get', *argv[1:]]
        service_env = {'PATH': SAFE_PATH, 'HOME': '/root', 'LC_ALL': 'C', 'LANG': 'C',
                       'DEBIAN_FRONTEND': 'noninteractive', 'NEEDRESTART_MODE': 'l'}
        if os.geteuid() != 0:
            command = ['/usr/bin/sudo', '-n', '--', *command]
    else:
        service_env = clean_environment()
    # systemd's manager environment is NOT the caller's clean environment.
    # Reset it explicitly; do not leak manager tokens/config to pip or apt.
    command += ['--', '/usr/bin/env', '-i', '--',
                *(key + '=' + value for key, value in sorted(service_env.items())), *argv]
    return command, _manager_environment(privileged)


def _unit_empty(unit, *, privileged, allow_missing=True):
    """Read back this exact unit and cgroup; unknown is never proof of drain."""
    try:
        result = _run_local(['/usr/bin/systemctl', '--system' if privileged else '--user',
                             '--no-ask-password', 'show', '--all', '--property=LoadState,ActiveState,ControlGroup',
                             '--', unit], timeout=5, env=_manager_environment(privileged), cancel=[])
        fields = dict(line.split('=', 1) for line in (result.stdout or '').splitlines() if '=' in line)
        if result.returncode not in (0, 1) or set(fields) != {'LoadState', 'ActiveState', 'ControlGroup'}:
            return None
        if fields['LoadState'] == 'not-found':
            return True if allow_missing and fields['ActiveState'] == 'inactive' and not fields['ControlGroup'] else None
        if fields['LoadState'] != 'loaded':
            return None
        if fields['ActiveState'] not in ('inactive', 'failed'):
            return False
        group = fields['ControlGroup']
        if not group:
            return True
        path = Path(group)
        if not path.is_absolute() or '..' in path.parts or path.name != unit:
            return None
        events = Path('/sys/fs/cgroup') / group.lstrip('/') / 'cgroup.events'
        try:
            values = dict(line.split() for line in events.read_text().splitlines())
        except FileNotFoundError:
            return True  # Only after the manager reported this exact unit stopped.
        return {'0': True, '1': False}.get(values.get('populated', ''))
    except (OSError, ValueError, DependencyFailure):
        return None


def _stop_unit(unit, *, privileged):
    command = ['/usr/bin/systemctl', '--system' if privileged else '--user',
               '--no-ask-password', 'stop', '--', unit]
    if privileged and os.geteuid() != 0:
        command = ['/usr/bin/sudo', '-n', '--', *command]
    try:
        result = _run_local(command, timeout=5, env=_manager_environment(privileged), cancel=[])
        return result.returncode == 0
    except (OSError, DependencyFailure):
        return False  # Expired sudo is NOT evidence apt stopped. Keep draining.


def _run_service(argv, *, timeout, privileged):
    with _deferred_signals() as pending:
        _require_systemd(privileged=privileged, cancel=pending)
        _raise_after_drain('DEPENDENCY_INTERRUPTED' if pending else None, argv, pending)
        unit = 'rexl3-dependency-' + uuid.uuid4().hex + '.service'
        command, env = _service_command(argv, unit, timeout, privileged=privileged)
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, env=env, shell=False, start_new_session=True)
        deadline = time.monotonic() + timeout
        capture, reason, fault = None, None, None
        next_stop, warned = 0, False
        try:
            capture = _Capture(process)
        except BaseException as error:
            fault = error
        while True:
            # A fault in capture, status readback or even diagnostic output
            # cannot unwind an admitted transaction. There is deliberately no
            # finally here that returns ownership merely because a client died.
            try:
                if capture is not None and fault is None:
                    capture.drain(.05)
                else:
                    time.sleep(.05)
                now = time.monotonic()
                if pending:
                    reason = reason or 'DEPENDENCY_INTERRUPTED'
                if fault:
                    reason = reason or 'DEPENDENCY_SUPERVISION_FAILED'
                if capture is not None and capture.exceeded:
                    reason = reason or 'DEPENDENCY_OUTPUT_LIMIT'
                if now >= deadline:
                    reason = reason or 'DEPENDENCY_TIMEOUT'
                if reason and now >= next_stop:
                    next_stop = now + 5
                    _stop_unit(unit, privileged=privileged)
                returncode = process.poll()
                if returncode is not None:
                    # Wait for admission to finish before trusting absence. If
                    # the client was killed mid-admission, a missing unit is
                    # ambiguous; hold for administrator/manager recovery.
                    empty = _unit_empty(unit, privileged=privileged, allow_missing=returncode >= 0)
                    if empty is True:
                        if capture is None or capture.eof or fault:
                            break
                        continue  # Finish buffered output without inventing a failure.
                    reason = reason or 'DEPENDENCY_SUPERVISION_FAILED'
                    if not warned:
                        warned = True
                        print('DEPENDENCY_DRAIN_PENDING: ' + unit
                              + '; installer lock retained until unit/cgroup drain is verified; '
                              'ask the host administrator', file=sys.stderr, flush=True)
                    time.sleep(.1)
            except BaseException as error:
                fault = fault or error
        # This is the only exit: the client has been reaped and this exact unit
        # is drained. No sudo/client PID or numeric descendant list is killed.
        if capture is not None:
            capture.close()
        process.wait()
        if isinstance(fault, KeyboardInterrupt):
            raise KeyboardInterrupt
        _raise_after_drain(reason, argv, pending)
        if fault is not None or capture is None:
            raise _command_failure('DEPENDENCY_SUPERVISION_FAILED', argv)
        return subprocess.CompletedProcess(argv, process.returncode, *capture.output())


def _enclosing_dependency_unit():
    """Keep a supervised doctor's nested probes inside its existing cgroup.

    Launching another transient service would move that probe to a sibling
    cgroup and let it outlive the outer doctor's cancellation. Trust kernel
    membership plus manager kill-policy readback, never an environment marker.
    The outer run() retains the lock until this shared cgroup is fully empty.
    """
    message = 'DEPENDENCY_SUPERVISION_UNAVAILABLE: enclosing command ownership cannot be verified'
    try:
        membership = Path('/proc/self/cgroup').read_text()
    except OSError:
        raise DependencyFailure(20, message) from None
    unit = None
    for line in membership.splitlines():
        if line.startswith('0::'):
            match = re.search(r'/(rexl3-dependency-[0-9a-f]{32}\.service)(?:/|$)', line[3:])
            if match:
                unit = match[1]
    if unit is None:
        return None
    try:
        result = _run_local(['/usr/bin/systemctl', '--user', '--no-ask-password', 'show', '--all',
                             '--property=KillMode,ExitType,SendSIGKILL,RuntimeMaxUSec', '--', unit],
                            timeout=5, env=_manager_environment(False))
    except (OSError, DependencyFailure):
        raise DependencyFailure(20, message) from None
    fields = dict(line.split('=', 1) for line in (result.stdout or '').splitlines() if '=' in line)
    demand(result.returncode == 0 and fields.get('KillMode') == 'control-group'
           and fields.get('ExitType') == 'cgroup' and fields.get('SendSIGKILL') == 'yes'
           and fields.get('RuntimeMaxUSec') not in (None, '', '0', '0us', 'infinity'), 20, message)
    return unit


def run(argv, *, check=True, timeout=900, interactive=False):
    """Bounded output, sanitized errors, and drain-before-unwind ownership.

    Linux requires systemd 255/cgroup v2. User commands run as user services;
    mutating apt commands run as system services with only distro tools elevated.
    SIGINT/TERM/HUP and timeouts request unit stop, then retain the install lock
    until the client has exited AND exact unit/cgroup readback proves it empty.
    If stop permission expires, RuntimeMaxSec still owns cleanup. If readback
    fails, wait for recovery/administrator intervention rather than claim timeout
    and release the lock. This is a safety boundary, not a guaranteed return time
    for a broken manager/kernel. No arbitrary PID/process-name killing is used.

    sudo -v alone inherits the terminal. Non-Linux local execution exists for
    CPU development fixtures only; check_platform refuses installation there.
    """
    argv = [str(word) for word in argv]
    demand(bool(argv) and isinstance(timeout, (int, float)) and math.isfinite(timeout)
           and timeout > 0, 20, 'DEPENDENCY_COMMAND_INVALID')
    sudo_validation = argv in (['sudo', '-v'], ['/usr/bin/sudo', '-v'])
    demand(not interactive or sudo_validation, 20, 'INTERACTIVE_COMMAND_REFUSED')
    demand(Path(argv[0]).name != 'sudo' or (interactive and sudo_validation),
           20, 'PRIVILEGED_COMMAND_REFUSED')
    privileged = argv[0] in ('apt-get', '/usr/bin/apt-get') and '-s' not in argv
    try:
        if privileged:
            result = _run_service(argv, timeout=timeout, privileged=True)
        elif sys.platform == 'linux' and not interactive and not _enclosing_dependency_unit():
            result = _run_service(argv, timeout=timeout, privileged=privileged)
        else:
            result = _run_local(argv, timeout=timeout, interactive=interactive)
    except OSError:
        raise _command_failure('DEPENDENCY_COMMAND_UNAVAILABLE', argv) from None
    if check and result.returncode:
        raise _command_failure('DEPENDENCY_COMMAND_FAILED (exit ' + str(result.returncode) + ')', argv)
    return result


def check_inherited_environment():
    forbidden = {'EXTRA_ARGS', 'LD_PRELOAD', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES',
                 'CUDA_HOME', 'CUDA_PATH', 'NVIDIA_VISIBLE_DEVICES'}
    for name in os.environ:
        if name in forbidden or name.startswith(('REXL3_', 'GB_')):
            # Do not print values (or arbitrary caller-supplied key names).
            raise DependencyFailure(21, 'UNSAFE_INHERITED_ENV: unset CPU-stub, GPU, loader and runtime '
                                    'overrides; use only the finalized profile; see docs/SETUP.md')


def check_platform(requirements):
    demand(platform.system() == requirements['system'] == 'Linux', 21, 'UNSUPPORTED_OS: see docs/SETUP.md')
    demand(platform.machine() == requirements['machine'] == 'aarch64', 21, 'UNSUPPORTED_ARCH: see docs/SETUP.md')
    os_release = platform.freedesktop_os_release()  # Parse data; never source a shell file.
    demand(os_release.get('ID') == 'ubuntu' and os_release.get('VERSION_ID') == '24.04', 21,
           'UNSUPPORTED_INSTALLER_PLATFORM: only Ubuntu 24.04 ARM64 has a reviewed adapter')
    demand(tuple(sys.version_info[:2]) >= max((3, 12), tuple(requirements['minimum_python'])), 20,
           'PYTHON_VERSION_UNSUPPORTED: use Ubuntu 24.04 base Python; see docs/SETUP.md')


def reboot_required():
    return Path('/var/run/reboot-required').exists()


def ensure_no_reboot():
    demand(not reboot_required(), 20, 'REBOOT_REQUIRED: host administrator must explicitly restart '
           'the host and rerun; no automatic reboot is performed')


def validate_native_contract(release):
    expected = release.get('native_version_fields')
    demand(isinstance(expected, dict) and set(expected) == NATIVE_FIELDS, 23,
           'NATIVE_VERSION_CONTRACT_INCOMPLETE: capture every build_info.rs --version field')
    for value in expected.values():
        demand(isinstance(value, str) and 0 < len(value) < 256 and '\n' not in value
               and value.lower() not in ('none', 'unknown', 'unavailable') and '-dirty' not in value,
               23, 'NATIVE_VERSION_CONTRACT_INVALID')
    commit = expected['commit']
    demand(re.fullmatch(r'[0-9a-f]{12,40}', commit) and release['production_commit'].startswith(commit),
           23, 'NATIVE_COMMIT_NOT_BOUND_TO_PRODUCTION')
    demand(expected['version'] == release['version'], 23, 'NATIVE_RELEASE_VERSION_MISMATCH')
    demand(re.fullmatch(r'[0-9a-f]{16}', expected['kernel_hash']), 23, 'NATIVE_KERNEL_HASH_INVALID')
    cap = release['platform']['gpu_compute_capability']
    demand(re.fullmatch(r'\d+\.\d+', cap) and expected['cuda_arch'] == 'sm_' + cap.replace('.', ''),
           23, 'NATIVE_CUDA_ARCH_MISMATCH')
    for key in ('nvcc', 'cudart', 'cuda_driver', 'cublas'):
        demand(re.fullmatch(r'\d+(?:\.\d+)+', expected[key]), 23, 'NATIVE_VERSION_FIELD_INVALID: ' + key)


def package_name(name):
    return name.removesuffix(':arm64')


def is_native(package):
    return NATIVE_PACKAGE.fullmatch(package_name(package)) is not None


def is_driver(package):
    return package_name(package).startswith(('nvidia-', 'libnvidia-', 'linux-', 'cuda-drivers'))


def parse_requirements(path):
    """Accept hash-locked wheels only, not pip directives, URLs, includes or sdists.

    pip-compile's backslash continuations and # comments are accepted. Resolve
    markers/extras for the target platform when generating the lock; the runtime
    lock contains the exact complete closure. pip still verifies wheel hashes.
    """
    demand(path.stat().st_size <= 1024 * 1024, 2, 'PYTHON_REQUIREMENTS_TOO_LARGE')
    physical = path.read_text(encoding='utf-8').splitlines()
    records, pending = [], ''
    for raw in physical:
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        continuation = line.endswith('\\')
        pending += (' ' if pending else '') + (line[:-1].strip() if continuation else line)
        if not continuation:
            records.append(pending)
            pending = ''
    demand(not pending and bool(records), 2, 'PYTHON_REQUIREMENTS_INCOMPLETE')
    packages = {}
    for record in records:
        match = re.fullmatch(r'([A-Za-z0-9][A-Za-z0-9_.-]*)==([0-9][A-Za-z0-9.!+_-]*)'
                             r'(?:\s+--hash=sha256:[0-9a-f]{64})+', record)
        if match is None:
            raise DependencyFailure(2, 'PYTHON_REQUIREMENTS_NOT_HASH_LOCKED: only exact hashed wheel pins accepted')
        name = re.sub(r'[-_.]+', '-', match[1]).lower()
        demand(name not in packages, 2, 'PYTHON_REQUIREMENTS_DUPLICATE')
        packages[name] = match[2]
    demand('huggingface-hub' in packages, 2, 'OFFICIAL_HF_CLIENT_PIN_REQUIRED')
    return packages


def validate_policy(root, release):
    policy = load_json(safe_file(root, 'dependencies.lock.json'))
    demand(isinstance(policy, dict) and set(policy) == {'schema', 'apt', 'python_requirements'}
           and policy['schema'] == 'rexl3.dependencies.v1', 2, 'DEPENDENCY_POLICY_INVALID')
    entries = policy['apt']
    demand(isinstance(entries, list) and bool(entries), 2, 'DEPENDENCY_APT_LIST_INVALID')
    names = set()
    for entry in entries:
        demand(isinstance(entry, dict) and set(entry) == {'package', 'version', 'strict_version'}, 2,
               'DEPENDENCY_PACKAGE_RECORD_INVALID')
        package, version = entry['package'], entry['version']
        demand(isinstance(package, str) and re.fullmatch(r'[a-z0-9][a-z0-9+.-]*(?::arm64)?', package),
               2, 'INVALID_PACKAGE_NAME')
        name = package_name(package)
        demand(name not in names, 2, 'DUPLICATE_PACKAGE_RECORD')
        names.add(name)
        demand(name in BASE_PACKAGES or is_native(name), 2, 'UNREVIEWED_PACKAGE_MAPPING')
        demand(isinstance(version, str) and re.fullmatch(r'[0-9][A-Za-z0-9.+:~_-]*', version),
               2, 'INVALID_PACKAGE_VERSION')
        demand(type(entry['strict_version']) is bool, 2, 'DEPENDENCY_VERSION_POLICY_REQUIRED')
        demand(not is_native(name) or entry['strict_version'], 2, 'NATIVE_PACKAGE_REQUIRES_STRICT_VERSION')
    demand({'python3', 'python3-venv', 'ca-certificates', 'bash', 'apt', 'dpkg'} <= names, 2,
           'BOOTSTRAP_PACKAGE_MAPPING_INCOMPLETE')
    demand(any(name.startswith('nvidia-utils-') for name in names), 2,
           'NVIDIA_SMI_INSTALL_MAPPING_MISSING')
    requirements = release['platform']
    for command in requirements['commands']:
        if command == 'hf':
            continue  # Explicitly resolved from the verified isolated descriptor.
        candidates = COMMAND_PACKAGES.get(command, ())
        if command == 'nvidia-smi':
            candidates = tuple(name for name in names if name.startswith('nvidia-utils-'))
        demand(bool(set(candidates) & names), 2, 'COMMAND_INSTALL_MAPPING_MISSING')
    for library in requirements['libraries']:
        candidates = LIBRARY_PACKAGES.get(library, ())
        if re.fullmatch(r'libcudart\.so\.\d+', library):
            candidates = tuple(name for name in names if name.startswith('cuda-cudart-'))
        elif re.fullmatch(r'libcublas(?:Lt)?\.so\.\d+', library):
            candidates = tuple(name for name in names if name.startswith('libcublas-'))
        elif library == 'libcuda.so.1':
            candidates = tuple(name for name in names if name.startswith('libnvidia-compute-'))
        demand(bool(set(candidates) & names), 2, 'LIBRARY_INSTALL_MAPPING_MISSING')
    relative_file(policy['python_requirements'])
    demand(policy['python_requirements'] in release['files'], 23, 'PYTHON_REQUIREMENTS_NOT_IN_RELEASE')
    req = safe_file(root, policy['python_requirements'])
    pins = parse_requirements(req)
    return policy, req, pins


def validated_bundle(root):
    # Shared validator enforces pending state, immutable paths, sizes, modes and hashes.
    for name in ('profile.json', 'model.lock.json', 'release.lock.json', 'dependencies.lock.json'):
        path = Path(root) / name
        demand(path.exists() or path.is_symlink(), 2, 'PROFILE_NOT_READY: this is not a finalized runtime bundle')
    profile, models, release = verify_release(root)
    required = {'profile.json', 'model.lock.json', 'dependencies.lock.json', 'bin/rexl3-serve',
                'scripts/contract.py', 'scripts/bootstrap.sh', 'scripts/install_dependencies.py', 'scripts/doctor.py'}
    demand(required <= set(release['files']), 23, 'RELEASE_INVENTORY_INCOMPLETE')
    validate_native_contract(release)
    policy, req, pins = validate_policy(root, release)
    return profile, models, release, policy, req, pins


def safe_managed_path(path):
    """Reject symlinks in every existing component, including broken links."""
    path = Path(path)
    demand(path.is_absolute() and '..' not in path.parts, 2, 'UNSAFE_STATE_PATH')
    for component in reversed((path, *path.parents)):
        if component.is_symlink():
            raise DependencyFailure(23, 'UNSAFE_STATE_SYMLINK')
    if path.exists():
        info = path.stat()
        demand(stat.S_ISDIR(info.st_mode) or (stat.S_ISREG(info.st_mode) and info.st_nlink == 1),
               23, 'UNSAFE_STATE_FILE_TYPE')
    return path


def state_root(profile):
    base = Path(os.environ.get('XDG_DATA_HOME', str(Path.home() / '.local/share')))
    state = safe_managed_path(base / 'rexl3' / profile['runtime_id'])
    for directory in (base, base / 'rexl3', state):
        if directory.exists():
            ensure_owned(directory)
    return state


def private_makedirs(path):
    """mkdir -p with a private mode on every component this call creates.

    ``Path.mkdir(parents=True, mode=0o700)`` applies the mode to the leaf only: the parents are made
    with the process umask (0o775 under a 002 umask) and the state-root probe then refuses them with
    UNSAFE_STATE_OWNERSHIP_OR_MODE before anything is installed.
    """
    missing = []
    probe = path
    while not probe.exists():
        missing.append(probe)
        if probe.parent == probe:
            break
        probe = probe.parent
    for directory in reversed(missing):
        directory.mkdir(mode=0o700, exist_ok=True)
        os.chmod(directory, 0o700)


def ensure_owned(path):
    info = path.stat()
    demand(info.st_uid == os.getuid() and not info.st_mode & 0o022, 23, 'UNSAFE_STATE_OWNERSHIP_OR_MODE')


@contextmanager
def installation_lock(state):
    safe_managed_path(state)
    private_makedirs(state)
    ensure_owned(state)
    path = safe_managed_path(state / 'install.lock')
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        demand(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_uid == os.getuid(),
               23, 'UNSAFE_INSTALL_LOCK')
        deadline = time.monotonic() + LOCK_TIMEOUT_S
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                demand(time.monotonic() < deadline, 22, 'INSTALL_BUSY: another installer owns this runtime')
                time.sleep(0.2)
        yield
    finally:
        os.close(fd)  # Never unlink: waiting installers must share the same inode.


def installed_version(package):
    result = run(['dpkg-query', '-W', '-f=${Status}\t${Version}', package], check=False, timeout=30)
    demand(result.returncode in (0, 1), 20, 'DPKG_QUERY_FAILED: inspect package database locally')
    if result.returncode == 1 or not result.stdout.startswith('install ok installed\t'):
        return None
    value = result.stdout.removeprefix('install ok installed\t').strip()
    demand(re.fullmatch(r'[0-9][A-Za-z0-9.+:~_-]*', value), 20, 'DPKG_VERSION_READBACK_INVALID')
    return value


def package_status(policy):
    return {entry['package']: installed_version(entry['package']) for entry in policy['apt']}


def require_packages(policy, installed, *, readback=False):
    for entry in policy['apt']:
        got = installed[entry['package']]
        demand(got is not None, 20, 'SYSTEM_DEPENDENCY_READBACK_FAILED' if readback else
               'SYSTEM_DEPENDENCY_MISSING: run ./launch.sh install-deps; see docs/SETUP.md')
        demand(not entry['strict_version'] or got == entry['version'], 20,
               'SYSTEM_DEPENDENCY_VERSION_MISMATCH: strict native policy; review lock instead of relaxing it')


def approve(message, yes=False):
    print(message, flush=True)
    if yes:
        return
    demand(sys.stdin.isatty(), 20, 'APPROVAL_REQUIRED: rerun with --yes for routine installs')
    try:
        answer = input('Continue? [y/N] ').strip().lower()
    except EOFError:
        answer = ''
    demand(answer in ('y', 'yes'), 20, 'INSTALL_DECLINED: no package changes made')


# Enforce signed-repository checks even if host apt.conf weakened defaults. Do not
# provision repositories/keys or permit broad automatic restarts via needrestart.
APT_OPTIONS = ['-o', 'DPkg::Lock::Timeout=60', '-o', 'Acquire::Retries=0',
               '-o', 'Acquire::http::Timeout=30', '-o', 'Acquire::https::Timeout=30',
               '-o', 'APT::Update::Error-Mode=any', '-o', 'APT::Get::AllowUnauthenticated=false',
               '-o', 'Acquire::AllowInsecureRepositories=false',
               '-o', 'Acquire::AllowDowngradeToInsecureRepositories=false']


def review_simulation(text, policy, installed, wanted):
    demand(not re.search(r'^(?:Remv|Purg)\s', text, re.M), 20, 'PACKAGE_REMOVAL_REFUSED')
    entries = {package_name(entry['package']): entry for entry in policy['apt']}
    changes = {}
    for line in text.splitlines():
        if not line.startswith('Inst '):
            continue
        match = re.fullmatch(r'Inst\s+(\S+)(?:\s+\[[^\]]*\])?\s+\(([^\s)]+)[^\n]*\)', line)
        if match is None:
            raise DependencyFailure(20, 'APT_PLAN_UNPARSEABLE')
        name, version = package_name(match[1]), match[2]
        demand(name in entries and name not in changes, 20, 'UNREVIEWED_APT_SOLVER_CHANGE')
        entry = entries[name]
        demand(version == entry['version'], 20, 'APT_PLAN_VERSION_MISMATCH')
        current = installed[entry['package']]
        demand(current is None or (entry['strict_version'] and is_native(name)), 20,
               'BASE_PACKAGE_UPGRADE_REFUSED: compatible installed OS packages are preserved')
        changes[name] = version
    # Configuring an already-unpacked package can execute privileged postinst
    # scripts even without an Inst row. Only configure versions installed by
    # this exact reviewed transaction; outstanding host repairs are not implied.
    configured = set()
    for line in text.splitlines():
        if not line.startswith('Conf '):
            continue
        match = re.fullmatch(r'Conf\s+(\S+)\s+\(([^\s)]+)[^\n]*\)', line)
        if match is None:
            raise DependencyFailure(20, 'APT_CONFIGURATION_PLAN_UNPARSEABLE')
        name, version = package_name(match[1]), match[2]
        demand(name in changes and changes[name] == version and name not in configured,
               20, 'UNREVIEWED_APT_CONFIGURATION')
        configured.add(name)
    demand(changes and {package_name(word.split('=', 1)[0]) for word in wanted} <= set(changes),
           20, 'APT_PINNED_VERSION_UNAVAILABLE_OR_PLAN_INCOMPLETE')
    return changes


def install_system(policy, options):
    installed = package_status(policy)
    wanted = []
    for entry in policy['apt']:
        name, version = entry['package'], entry['version']
        current = installed[name]
        if current is not None and (not entry['strict_version'] or current == version):
            continue
        if current is not None:
            demand(is_native(name), 20, 'BASE_PACKAGE_VERSION_CHANGE_REFUSED')
            older = run(['dpkg', '--compare-versions', version, 'lt', current], check=False, timeout=30)
            demand(older.returncode in (0, 1), 20, 'DPKG_VERSION_COMPARISON_FAILED')
            demand(older.returncode == 1, 20, 'PACKAGE_DOWNGRADE_REFUSED: review qualified native versions')
        wanted.append(name + '=' + version)
    if not wanted:
        return
    approve('Proposed pinned system changes (signed configured repositories only): ' + ', '.join(wanted), options.yes)
    if os.geteuid() != 0:
        demand(shutil.which('sudo', path=SAFE_PATH), 20, 'SUDO_REQUIRED: ask the host administrator; see docs/SETUP.md')
        run(['sudo', '-v'], interactive=True, timeout=120)
    # run() owns elevation and the complete apt cgroup, not a sudo wrapper PID.
    run(['apt-get', *APT_OPTIONS, 'update'])
    simulation = run(['apt-get', *APT_OPTIONS, '-s', 'install', '--no-remove', '--no-install-recommends', *wanted])
    changes = review_simulation(simulation.stdout, policy, installed, wanted)
    if any(is_driver(name) for name in changes):
        demand(options.allow_driver_change, 20, 'DRIVER_CHANGE_APPROVAL_REQUIRED: --yes is not driver approval; '
               'use --allow-driver-change only after reviewing host impact')
        idle = run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], check=False, timeout=30)
        demand(idle.returncode == 0 and not idle.stdout.strip(), 20,
               'DRIVER_CHANGE_REQUIRES_IDLE_VERIFIED_GPU: broken or unavailable driver needs the host administrator')
    approve('Install the reviewed exact package plan; no removals, OS upgrades or reboot.', options.yes)
    # Explicitly pin every solver change too. --no-remove fails if the plan drifts
    # toward removal. No --allow-downgrades or --allow-change-held-packages.
    specs = [name + '=' + version for name, version in sorted(changes.items())]
    run(['apt-get', *APT_OPTIONS, 'install', '-y', '--no-remove',
         '--no-install-recommends', *specs])
    after = package_status(policy)
    require_packages(policy, after, readback=True)
    for name, version in changes.items():
        key = next(entry['package'] for entry in policy['apt'] if package_name(entry['package']) == name)
        demand(after[key] == version, 20, 'SYSTEM_DEPENDENCY_READBACK_VERSION_FAILED')
    ensure_no_reboot()


def environment_paths(state, digest):
    envdir = safe_managed_path(state / ('venv-' + digest))
    return envdir, safe_managed_path(envdir / 'bin/python'), safe_managed_path(envdir / 'bin/hf')


def normalize_tool_modes(envdir):
    """pip's console scripts land 0o775 (its script writer makes them group-writable regardless of umask),
    while [`probe_python_environment`] refuses a group-writable interpreter or client. Normalize the files
    it installs so the private-environment check holds on a stock host; directories are left as created."""
    bindir = envdir / 'bin'
    for entry in sorted(bindir.iterdir()):
        if stat.S_ISREG(entry.lstat().st_mode):
            os.chmod(entry, 0o755)
    # `python -m venv` inherits the caller's umask. Under a 002 umask every directory it makes lands
    # 0o775, and the private-environment probe refuses the interpreter's directory before it looks at
    # any file. Normalize the tree, not just bin/.
    for base, dirs, _files in os.walk(envdir):
        for name in dirs:
            os.chmod(os.path.join(base, name), 0o755)
    os.chmod(envdir, 0o755)


def require_isolated_interpreter(envdir, python):
    output = run([str(python), '-I', '-B', '-c',
                  'import json,sys; print(json.dumps({"prefix":sys.prefix,"base_prefix":sys.base_prefix}))'], timeout=30)
    try:
        context = json.loads(output.stdout)
    except (ValueError, TypeError):
        raise DependencyFailure(20, 'VENV_CONTEXT_READBACK_FAILED') from None
    demand(isinstance(context, dict) and context.get('prefix') == str(envdir)
           and isinstance(context.get('base_prefix'), str) and context['base_prefix'] != str(envdir),
           20, 'VENV_NOT_ISOLATED: refusing to run pip against system Python')


PYTHON_INVENTORY = '''import importlib.metadata, json, re, sys
packages = {}
for distribution in importlib.metadata.distributions():
    name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower()
    if name in packages:
        raise SystemExit(3)
    packages[name] = distribution.version
print(json.dumps({"prefix": sys.prefix, "base_prefix": sys.base_prefix, "packages": packages}))
'''


def probe_python_environment(envdir, python, hf, pins, *, help_probe=False):
    for path in (python, hf):
        safe_managed_path(path)
        demand(path.is_file() and os.access(path, os.X_OK), 20, 'PYTHON_OR_HF_INSTALL_INCOMPLETE')
        ensure_owned(path)
    result = run([str(python), '-I', '-B', '-c', PYTHON_INVENTORY], timeout=30)
    try:
        inventory = json.loads(result.stdout)
    except (ValueError, TypeError):
        raise DependencyFailure(20, 'PYTHON_ENVIRONMENT_READBACK_INVALID') from None
    demand(isinstance(inventory, dict) and inventory.get('prefix') == str(envdir)
           and inventory.get('base_prefix') != str(envdir), 20, 'PYTHON_ENVIRONMENT_NOT_ISOLATED')
    actual = inventory.get('packages')
    if not isinstance(actual, dict) or not all(actual.get(k) == v for k, v in pins.items()):
        raise DependencyFailure(20, 'PYTHON_PACKAGE_VERSION_READBACK_FAILED')
    # venv's distro-owned pip (and on older Python, setuptools) are trusted seed
    # tools, not floating runtime dependencies. Every other distribution is pinned.
    demand(set(actual) <= set(pins) | {'pip', 'setuptools'}, 20, 'UNPINNED_PYTHON_PACKAGE_REFUSED')
    run([str(python), '-I', '-B', '-m', 'pip', '--isolated', '--disable-pip-version-check', 'check'], timeout=60)
    if help_probe:
        run([str(python), '-I', '-B', str(hf), '--help'], timeout=30)


def check_python_environment(state, req, pins):
    digest = sha256_file(req)
    descriptor = safe_managed_path(state / 'python-environment.json')
    demand(descriptor.is_file(), 20, 'PYTHON_ENVIRONMENT_MISSING: run ./launch.sh install-deps')
    ensure_owned(state)
    ensure_owned(descriptor)
    data = load_json(descriptor)
    envdir, python, hf = environment_paths(state, digest)
    expected = {'python': str(python), 'hf': str(hf), 'requirements_sha256': digest}
    demand(data == expected, 23, 'PYTHON_ENVIRONMENT_DESCRIPTOR_MISMATCH')
    marker = safe_managed_path(envdir / '.requirements-sha256')
    demand(marker.is_file() and marker.read_text() == digest, 20, 'PYTHON_ENVIRONMENT_INCOMPLETE')
    probe_python_environment(envdir, python, hf, pins)
    return data


def atomic_write(path, data):
    safe_managed_path(path)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        safe_managed_path(path)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def install_python(state, req, pins, options, *, expected_digest):
    # Freeze already-validated requirements before invoking a tool. pip must not
    # reopen a bundle path that could be changed during an apt/network wait.
    contents = req.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    demand(digest == expected_digest, 23, 'PYTHON_REQUIREMENTS_CHANGED_DURING_INSTALL')
    envdir, python, hf = environment_paths(state, digest)
    descriptor = safe_managed_path(state / 'python-environment.json')
    marker = safe_managed_path(envdir / '.requirements-sha256')
    if descriptor.is_file() and marker.is_file():
        try:
            check_python_environment(state, req, pins)
        except DependencyFailure as error:
            # A previous success marker is not proof that software still exists.
            # Repair only missing/mismatched managed dependencies; descriptor,
            # ownership, path, isolation and unpinned-package failures stay fatal.
            recoverable = {
                'PYTHON_OR_HF_INSTALL_INCOMPLETE',
                'PYTHON_PACKAGE_VERSION_READBACK_FAILED',
                'PYTHON_ENVIRONMENT_INCOMPLETE',
            }
            if str(error) not in recoverable:
                raise
        else:
            return
    approve('Install hash-pinned Python/HF wheels in this runtime\'s isolated user environment.', options.yes)
    snapshot = safe_managed_path(state / ('requirements-' + digest + '.lock'))
    atomic_write(snapshot, contents.decode('utf-8'))
    demand(sha256_file(snapshot) == digest, 23, 'PYTHON_REQUIREMENTS_SNAPSHOT_READBACK_FAILED')
    if envdir.exists():
        ensure_owned(envdir)
    # --copies means managed executable paths are never symlinks to an external
    # executable; do not test system ensurepip (Debian disables invoking it there).
    if not python.exists():
        run([sys.executable, '-I', '-B', '-m', 'venv', '--copies', str(envdir)])
    safe_managed_path(python)
    demand(python.is_file(), 20, 'VENV_CREATION_READBACK_FAILED')
    require_isolated_interpreter(envdir, python)
    run([str(python), '-I', '-B', '-m', 'pip', '--isolated', '--disable-pip-version-check',
         'install', '--index-url', 'https://pypi.org/simple', '--only-binary=:all:',
         '--require-hashes', '--no-deps', '--force-reinstall', '--no-cache-dir', '--no-compile', '-r', str(snapshot)])
    normalize_tool_modes(envdir)
    probe_python_environment(envdir, python, hf, pins, help_probe=True)
    # No completion marker/descriptor before all functional and version probes.
    atomic_write(marker, digest)
    data = {'python': str(python), 'hf': str(hf), 'requirements_sha256': digest}
    atomic_write(descriptor, json.dumps(data, sort_keys=True) + '\n')
    check_python_environment(state, req, pins)


def report_contract_error(error):
    # Contract messages can reference untrusted input. Emit a stable bounded code,
    # not arbitrary values, JSON, exception filenames or package-manager output.
    if 'PROFILE_NOT_READY' in str(error):
        return 'PROFILE_NOT_READY: this is not a finalized runtime bundle'
    return 'CONTRACT_OR_BUNDLE_INVALID: immutable contract validation failed; see docs/SETUP.md'


def main(argv=None, root=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yes', action='store_true')
    parser.add_argument('--allow-driver-change', action='store_true')
    options = parser.parse_args([] if argv is None else argv)
    root = Path(root) if root is not None else ROOT
    try:
        profile, _, release, policy, req, pins = validated_bundle(root)
        release_digest = sha256_file(safe_file(root, 'release.lock.json'))
        check_inherited_environment()
        check_platform(release['platform'])
        ensure_no_reboot()
        state = state_root(profile)
        with installation_lock(state):
            # Another installer may have waited while bundle files changed.
            profile, _, release, policy, req, pins = validated_bundle(root)
            demand(sha256_file(safe_file(root, 'release.lock.json')) == release_digest, 23,
                   'BUNDLE_CHANGED_DURING_INSTALL')
            install_system(policy, options)
            profile, _, release, policy, req, pins = validated_bundle(root)
            demand(sha256_file(safe_file(root, 'release.lock.json')) == release_digest, 23,
                   'BUNDLE_CHANGED_DURING_INSTALL')
            install_python(state, req, pins, options,
                           expected_digest=release['files'][policy['python_requirements']]['sha256'])
            ensure_no_reboot()
            # Keep the lock until fresh preflight finishes. Installation success
            # alone is not startup readiness, and a child failure remains nonzero.
            result = run([sys.executable, '-I', '-B', str(root / 'scripts/doctor.py')],
                         check=False, timeout=180)
            if result.returncode:
                print('PREFLIGHT_FAILED: fresh doctor exit ' + str(result.returncode)
                      + '; run ./launch.sh doctor for details; see docs/SETUP.md', file=sys.stderr)
                return result.returncode if result.returncode > 0 else 128 - result.returncode
        print('DEPENDENCIES_INSTALLED_AND_VERIFIED: fresh doctor passed; model/start readiness remains separate')
        return 0
    except DependencyFailure as error:
        print(str(error), file=sys.stderr)
        return error.code
    except ContractError as error:
        print(report_contract_error(error), file=sys.stderr)
        return 2 if 'PROFILE_NOT_READY' in str(error) else 23
    except (KeyError, TypeError, ValueError, OSError, subprocess.SubprocessError):
        print('DEPENDENCY_VALIDATION_OR_IO_FAILED: no successful installation recorded; see docs/SETUP.md', file=sys.stderr)
        return 20
    except KeyboardInterrupt:
        print('INSTALL_INTERRUPTED: rerun; no server has been started', file=sys.stderr)
        return 130


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
