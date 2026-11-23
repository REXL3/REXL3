#!/usr/bin/env python3
"""Read-only bundle/dependency/native/GPU checks; DOCTOR_PASS is not READY.

No model load, inference, downloads, installs, sudo, state creation, GPU locking,
port binding or server start. setup/start own missing model bytes, available disk,
site paths/ports and workload ownership, which cannot be proven by static checks.
Native --version is intentionally strict for *all* source-defined fields,
including the runtime CUDA driver/cuBLAS versions: newer is not automatically
qualified. Never edit the lock or change drivers to suppress that failure. The NVIDIA
driver packages are judged by the lock's tested floor: a later build of the same branch
passes with DRIVER_NOT_YET_QUALIFIED and `driver_qualification: newer-than-tested`.
A lock with `host_policy: known-working` extends that rule to the other native packages
and to the host-reported CUDA runtime, driver CUDA version and cuBLAS: at or above the
tested floor passes, and anything not among the qualified versions is reported
(`dependency_qualification: not-yet-qualified`). `host_versions` records what was found.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from contract import ContractError, safe_file
import install_dependencies as dependencies

DoctorFailure = dependencies.DependencyFailure
# Must match crates/cluster/src/schema.rs and existing node flock tooling.
# An operator's TMPDIR must never create a separate physical-GPU namespace.
GPU_LOCK_PATH = Path('/tmp/rexl3-gpu.lock')


def gpu_idle():
    """Read-only startup probe; an unavailable/failed probe is never idle."""
    try:
        result = dependencies.run(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
            check=False, timeout=30,
        )
        return result.returncode == 0 and not result.stdout.strip()
    except dependencies.DependencyFailure:
        return False


# Read-only functional probes for the adapter's base-tool mappings. scp has no
# portable version switch: its package mate ssh reports the OpenSSH version.
COMMAND_PROBES = {name: (name, '--version') for name in dependencies.COMMAND_PACKAGES}
COMMAND_PROBES.update({'ssh': ('ssh', '-V'), 'scp': ('ssh', '-V'),
                       'unzip': ('unzip', '-v'), 'ip': ('ip', '-Version')})


def native_identity(root, release, policy=None):
    binary = safe_file(root, 'bin/rexl3-serve')
    dependencies.demand(os.access(binary, os.X_OK), 23, 'NATIVE_BINARY_NOT_EXECUTABLE')
    with binary.open('rb') as stream:
        header = stream.read(64)
    dependencies.demand(len(header) == 64 and header[:6] == b'\x7fELF\x02\x01'
                        and header[18:20] == b'\xb7\0', 23,
                        'NATIVE_BINARY_ARCH_MISMATCH: expected little-endian ELF64 AArch64')
    output = dependencies.run([str(binary), '--version'], timeout=30).stdout
    version = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.split(None, 1)
        dependencies.demand(len(fields) == 2 and fields[0] in dependencies.NATIVE_FIELDS
                            and fields[0] not in version, 23, 'NATIVE_VERSION_OUTPUT_INVALID')
        version[fields[0]] = fields[1]
    dependencies.demand(set(version) == dependencies.NATIVE_FIELDS, 23, 'NATIVE_VERSION_OUTPUT_INCOMPLETE')
    dependencies.demand(version['nvcc'].lower() not in ('none', 'unknown', 'unavailable'),
                        21, 'CPU_STUB_BINARY_REFUSED')
    dependencies.demand('-dirty' not in version['commit'], 23, 'DIRTY_NATIVE_BINARY_REFUSED')
    expected = release['native_version_fields']
    if not dependencies.known_working(policy):
        for name, value in expected.items():
            dependencies.demand(version.get(name) == value, 23,
                                'NATIVE_VERSION_MISMATCH: ' + name + '; exact qualified native versions required')
        return version, []
    try:
        newer = dependencies.native_status(expected, version, True)
    except ValueError as error:
        name = str(error)
        raise DoctorFailure(23, 'NATIVE_VERSION_MISMATCH: ' + name + (
            '; this host reports ' + version[name] + ' and this runtime needs ' + expected[name]
            + ' or a later version of the same major. Run ./launch.sh install-deps to install the qualified CUDA '
            'runtime and cuBLAS, and check that no other CUDA library directory precedes them in the loader cache; '
            'see docs/SETUP.md' if name in dependencies.HOST_NATIVE_FIELDS and name in version else
            '; the fingerprint differs from the sealed one in more than its cuBLAS part. Run the sealed binary '
            'without serving variables in the environment; see docs/SETUP.md' if name == 'system_fingerprint' else
            '; the binary is not the one this release sealed. Download the release again; see docs/SETUP.md')) from None
    dependencies.warn_unqualified_native(expected, version, newer)
    return version, newer


def inspect(root=None):
    root = Path(root) if root is not None else ROOT
    profile, _, release, policy, req, pins = dependencies.validated_bundle(root)
    dependencies.check_inherited_environment()
    dependencies.check_platform(release['platform'])
    dependencies.ensure_no_reboot(policy)
    requirements = release['platform']
    installed = dependencies.package_status(policy)
    dependencies.require_packages(policy, installed)
    dependencies.require_loaded_driver(policy, installed)
    for name in requirements['commands']:
        if name == 'hf':
            continue  # Resolved below, never a globally installed client.
        dependencies.demand(shutil.which(name, path=dependencies.SAFE_PATH) is not None, 20,
                            'DEPENDENCY_MISSING: ' + name + '; run ./launch.sh install-deps; see docs/SETUP.md')
        if name in COMMAND_PROBES:
            dependencies.run(list(COMMAND_PROBES[name]), timeout=30)
    for name in requirements['libraries']:
        try:
            ctypes.CDLL(name)
        except OSError:
            raise DoctorFailure(20, 'LIBRARY_MISSING: ' + name + '; run ./launch.sh install-deps; see docs/SETUP.md') from None
    dependencies.check_python_environment(dependencies.state_root(profile), req, pins)
    identity = native_identity(root, release, policy)
    version, newer_native = identity if isinstance(identity, tuple) else ({}, [])
    gpu = dependencies.run(['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader'], timeout=30).stdout
    rows = gpu.strip().splitlines()
    dependencies.demand(len(rows) == 1 and rows[0].count(',') == 1, 21, 'EXPECTED_ONE_LOCAL_GPU')
    name, capability = [part.strip() for part in rows[0].split(',')]
    dependencies.demand(requirements['gpu_name_contains'] in name
                        and capability == requirements['gpu_compute_capability'], 21,
                        'UNSUPPORTED_GPU: this binary\'s kernels are compiled for compute capability '
                        + str(requirements['gpu_compute_capability']) + ' and the release was qualified on a GPU named '
                        + str(requirements['gpu_name_contains']) + '; run it on that device; see docs/SETUP.md')
    dependencies.warn_untested_driver(policy, installed)
    dependencies.warn_unqualified_packages(policy, installed)
    relaxed = dependencies.known_working(policy)
    unqualified = sorted([dependencies.package_name(entry['package'])
                          for entry in dependencies.unqualified_packages(policy, installed)] + list(newer_native))
    return {'status': 'DOCTOR_PASS', 'runtime_id': profile['runtime_id'],
            'architecture': requirements['machine'],
            'native_version_policy': 'tested-floor' if relaxed else 'exact',
            'driver_package_policy': 'tested-floor',
            'driver_qualification': dependencies.driver_qualification(policy, installed),
            'host_policy': dependencies.HOST_POLICY if relaxed else 'exact',
            # 'not-yet-qualified' names an accepted native package or CUDA library version the
            # release was not qualified on; release qualification never seals from it.
            'dependency_qualification': 'not-yet-qualified' if unqualified else 'tested',
            # What this check ran on: every locked package as installed, and the versions the
            # binary read from the host's CUDA libraries.
            'host_versions': {'packages': dict(sorted(installed.items())),
                              'native': {name: version[name] for name in dependencies.HOST_NATIVE_FIELDS
                                         if name in version},
                              'not_yet_qualified': unqualified,
                              'reboot_pending': dependencies.reboot_required()},
            'note': 'Read-only static checks passed. No inference performed. setup/start must still verify '
                    'model bytes, disk space, permissions, site ports, locks and owned readiness.'}


def main(argv=None, root=None):
    parser = argparse.ArgumentParser(description=__doc__)
    # Called by runtime.main: never accidentally parse the parent's command args.
    parser.parse_args([] if argv is None else argv)
    try:
        print(json.dumps(inspect(root), indent=2))
        return 0
    except DoctorFailure as error:
        print(str(error), file=sys.stderr)
        return error.code
    except ContractError as error:
        print(dependencies.report_contract_error(error), file=sys.stderr)
        return 2 if 'PROFILE_NOT_READY' in str(error) else 23
    except (KeyError, TypeError, ValueError, OSError):
        print('DOCTOR_CONTRACT_OR_PROBE_FAILED: see docs/SETUP.md; no changes made', file=sys.stderr)
        return 20
    except KeyboardInterrupt:
        print('DOCTOR_INTERRUPTED: no changes made', file=sys.stderr)
        return 130


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
