# Runtime setup and recovery

## Supported platform

The initial installer adapter is Ubuntu 24.04 on Linux ARM64. The exact supported
GPU, compute capability, memory requirements, native libraries and executables
are recorded in the assembled `release.lock.json`, not guessed by this template.
A different OS requires its own reviewed/tested installer adapter; do not run apt
commands on macOS, Windows or an unidentified Linux distribution.

The tester path uses a prebuilt, hashed native server. It does not clone the
private development repository, compile Rust/CUDA, use CPU-stub builds or pull
an unpinned latest binary. Supply native `rexl3-cluster` too for cluster profiles.

## Trusted bootstrap boundary

If Python is absent or unusable, the externally verified shell entry point first
checks the fixed Ubuntu 24.04 ARM64 platform and offers a minimal signed `python3`
installation. This stage does not read package names or hooks from unvalidated
JSON. It refuses upgrades, removals, driver/kernel changes, unrelated pending
configuration and any package outside its fixed interpreter closure. It rechecks
Python before executing the full policy validator. Bash/coreutils, a working
apt/dpkg database, configured signed repositories, network access and approved
sudo/root authority remain host prerequisites; an installer cannot safely repair
its own broken OS/package manager.

Linux process supervision additionally requires **systemd 255 or newer, cgroup v2,
and reachable system and normal-login user managers**. The launcher does not
install/change systemd, enable lingering, edit sudoers, or delegate cgroups. Sudo
must authorize the fixed systemd operations used to supervise approved apt
transactions, not just a bare apt executable. Ordinary probes/venv/pip run in
user-owned transient services; apt runs in system-owned transient services. No
candidate Python or bundled shell code is elevated to root.

A deadline/cancellation requests termination of the owned unit and waits for exact
unit/cgroup readback before releasing installation ownership. If the manager is
unreachable or cleanup cannot be verified, the installer deliberately waits for
administrator recovery rather than claim that a mutating transaction stopped.
Do not SIGKILL it to bypass this protection. A forced kill cannot preserve its
Python-held lock, although the independent service deadline remains. Native
Ubuntu transaction/cleanup validation is still required; CPU mocks are not proof
of root-service behavior. `doctor` changes no installed software or runtime data;
its read-only probes may use short-lived supervised services.

## Dependencies

Use `./launch.sh` for the full path. It installs missing supported software and
rechecks the result before downloading or running the model. For diagnosis only,
`./launch.sh doctor` makes no installations. To prepare software alone use
`./launch.sh install-deps`.

System packages are limited to the reviewed, signed repository/package mapping
in `dependencies.lock.json`. Python/HF tooling is isolated per requirements digest
and installed using exact versions, hashes and binary wheels from PyPI. It does
not modify system Python or silently select a floating version. An unavailable
exact package/version is a blocker for the profile/build owner, not permission to
install a different one.

Privilege is requested only when needed. Never paste a password or token into a
support thread. `--yes` before the command permits routine installation; it does
not grant driver-maintenance or reboot authority. Driver changes need separate
explicit consent and verified GPU idleness. Failed installation, failed readback,
reboot-required state and incompatible libraries stop launch.

## Model acquisition

`setup` prepares the locked model without starting inference. Size and destination
are shown before download. `--accept-model-download` provides explicit consent in
a noninteractive setup. Only exact commits and file identities from
`model.lock.json` are accepted. Mutable branch names, a different quantization,
external draft weights for embedded-draft DSV and mismatching payloads are refused.

Private model access, if needed, uses the normal official HF client's environment
credential mechanism. Do not put credentials in profile JSON, command arguments,
URL query strings, screenshots or diagnostics. Do not enable shell tracing.

## Recovery

- **Profile not ready:** ask the profile owner to finish and push the reviewed
  normalized contracts. Do not bypass the gate or copy historical serve flags.
- **Unsupported platform:** use the qualified platform or add a reviewed adapter.
- **Missing/incompatible software:** retry `install-deps` after resolving the named
  network/repository/privilege problem. Do not use broad OS upgrades as a repair.
- **Corrupt model payload:** do not start. Preserve the diagnostic and repair the
  named cache content deliberately; the downloader will not silently replace a
  corrupt verified payload or change the pin.
- **Busy port/GPU/lock:** wait for or coordinate with its owner. Never `pkill` broad
  names or kill an unrelated process to make this launcher pass.
- **Failed startup/smoke:** inspect local logs; report sanitized diagnostics plus
  the candidate/profile identity. Do not attach logs that contain conversations.
- **Stale state:** use the documented guarded `reset-state` operation only after
  proving no owned process remains. Model caches are intentionally preserved.

Single-node startup shares the fixed `/tmp/rexl3-gpu.lock` used by the existing
node tooling, regardless of `TMPDIR`, and checks for active compute processes
before preparation and again before server launch. This coordinates cooperating
operators; it is not permission to evict an unmanaged job.

Remote ranks currently remain refused until their complete pinned payload and
sidecar identity and sanitized environment can be attested by a qualified adapter.
Changing profile JSON alone does not lift that boundary.

Every advertised topology and capability still requires exact-artifact native
qualification plus a clean-host walkthrough before inviting testers.
