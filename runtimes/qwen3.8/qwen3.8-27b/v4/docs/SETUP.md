# Setup, site bindings and provenance

Start with the README's exact-release download and checksum-before-extraction
procedure. Never execute an unverified installer to verify its own trust root.
Keep the archive and sibling `approval.json`; do not assume approval is inside the
archive. A source-only repository checkout is not the runnable binary bundle.

## Supported platform and dependencies

The current adapter targets Ubuntu 24.04 ARM64, NVIDIA GB10 (SM121), Python 3.12+
and the exact native ABI/build requirements in `release.lock.json`. This is not a
portable installer for Windows, macOS, x86 or an unidentified Linux distribution.
There is no CUDA toolkit/compiler requirement for the prebuilt tester path; the
CUDA runtime, driver and cuBLAS libraries still must match. A cluster profile may
also require its locked transport libraries and tools on every rank.

Generated from this release's `dependencies.lock.json` (no guessed version list):

| APT package | Locked version | Version policy |
| --- | --- | --- |
| `apt` | `2.8.3` | minimum (newer accepted) |
| `bash` | `5.2.21-2ubuntu4` | minimum (newer accepted) |
| `ca-certificates` | `20260601~24.04.1` | minimum (newer accepted) |
| `coreutils` | `9.4-3ubuntu6.3` | minimum (newer accepted) |
| `cuda-cudart-13-0` | `13.0.96-1` | exact |
| `curl` | `8.5.0-2ubuntu10.15` | minimum (newer accepted) |
| `dpkg` | `1.22.6ubuntu6.6` | minimum (newer accepted) |
| `gzip` | `1.12-1ubuntu3.2` | minimum (newer accepted) |
| `iproute2` | `6.1.0-1ubuntu6.4` | minimum (newer accepted) |
| `libc6` | `2.39-0ubuntu8.9` | minimum (newer accepted) |
| `libcublas-13-0` | `13.1.1.3-1` | exact |
| `libgcc-s1` | `14.2.0-4ubuntu2~24.04.1` | minimum (newer accepted) |
| `libnuma1` | `2.0.18-1ubuntu0.24.04.1` | minimum (newer accepted) |
| `libnvidia-compute-580` | `580.173.02-0ubuntu0.24.04.1` | exact |
| `libssl3t64` | `3.0.13-0ubuntu3.16` | minimum (newer accepted) |
| `libstdc++6` | `14.2.0-4ubuntu2~24.04.1` | minimum (newer accepted) |
| `nvidia-utils-580` | `580.173.02-0ubuntu0.24.04.1` | exact |
| `pciutils` | `1:3.10.0-2build1` | minimum (newer accepted) |
| `python3` | `3.12.3-0ubuntu2.1` | minimum (newer accepted) |
| `python3-pip-whl` | `24.0+dfsg-1ubuntu1.3+esm3` | minimum (newer accepted) |
| `python3-setuptools-whl` | `68.1.2-2ubuntu1.2` | minimum (newer accepted) |
| `python3-venv` | `3.12.3-0ubuntu2.1` | minimum (newer accepted) |
| `tar` | `1.35+dfsg-3ubuntu0.4` | minimum (newer accepted) |
| `util-linux` | `2.39.3-9ubuntu6.6` | minimum (newer accepted) |

`strict_version: true` is exact equality, including driver/runtime packages.
For base packages marked non-strict, the locked version is a minimum, not an
instruction to downgrade a newer installation. The installer uses only the
reviewed signed package map and validates the solver's whole change set. It does
not add repositories/keys, perform a whole-OS upgrade, remove/downgrade packages,
install compiler meta-packages or fall back to source builds. If an exact native
package is unavailable, stop and request a reviewed policy/release update.
Python/HF dependencies come from exact hash-pinned binary wheels in
`requirements-runtime.lock`, installed in a per-requirements-digest environment;
there is no global pip mutation or ambient HF executable fallback.

Read the actual native/platform requirements before arranging host maintenance:

<!-- runtime-example: platform-inspect -->
```python
import json
from pathlib import Path
release = json.loads(Path('release.lock.json').read_text())
print(json.dumps({key: release[key] for key in
                 ('runtime_id', 'version', 'production_commit', 'production_tree',
                  'platform', 'native_version_fields')}, indent=2))
```

The platform block has no minimum-memory field: how much RAM a profile needs is
in its pack sheet and the release's native evidence, not in a doctor check.
Large mapped expert payloads, KV and prefix caches have different memory/storage
costs. Use the release's topology-specific native evidence and a quiet device;
never substitute weight-file GiB for required resident RAM. `doctor` compares all
native `--version` fields, including runtime driver/cuBLAS, exactly against the
lock. An arbitrary newer driver is not automatically qualified.

## Bootstrap trust and consent boundaries

`./launch.sh --yes install-deps` performs dependency preparation only.
`./launch.sh doctor` is read-only diagnosis (no installation, model load or
inference). Both `setup` and `start` run dependency bootstrap before doing their
model work on a single node or rank verification for attested TP. Bootstrap acts
on the controller host, not on remote ranks. Global `--yes` and
`--allow-driver-change` must precede the command;
`--accept-model-download`, `--profile` and `--site` follow `setup`/`start`.
`./launch.sh` with no arguments defaults to `start`.

If Python is missing/unusable, the verified Bash bootstrap checks Ubuntu 24.04
ARM64 and offers only its fixed minimal signed Python interpreter closure before
reading package policy in Python. It refuses upgrades, removals, drivers/kernel
changes, unrelated pending package configuration and packages outside that
closure. Bash/coreutils, functioning apt/dpkg, configured signed repositories,
network access and approved sudo authority must already exist. An installer cannot
safely repair its own broken base OS.

Process supervision requires **systemd 255 or newer, cgroup v2, and reachable system
and normal-login user managers**. The login user manager must work for the actual
launch user, including when using a noninteractive session. The installer does not
enable lingering, edit sudoers, install systemd or delegate cgroups. Approved apt
operations run in system-owned transient services; ordinary probes/venv/pip use
user-owned transient services. Sudo must authorize the fixed supervision
operations, not only a bare apt command. No candidate Python or bundled script is
elevated to root.

A cancelled/timed-out install requests termination and waits for exact unit/cgroup
drain readback before releasing ownership. An unreachable manager or unknown
cleanup state is an administrator-recovery condition, not permission to SIGKILL
the controller or launch another transaction. The independent service deadline is
not proof of completed cleanup. Read-only doctor probes may use short-lived
supervised services without changing installed software or runtime data.

- `--yes`: routine dependency installation consent only.
- `--accept-model-download`: separate explicit single-node managed weight-download
  consent. It never authorizes downloads/repair into an external model directory
  or provisions attested TP ranks.
- `--allow-driver-change`: separate driver-maintenance permission, still subject
  to the reviewed policy and verified GPU idleness; never a bypass for mismatching
  native versions. Coordinate with the GPU owner before using it.
- Reboot: never automatic. A reboot-required state stops launch; the administrator
  chooses whether/when to reboot, then repeats doctor.

In noninteractive use, missing consent or unauthorized sudo fails instead of
prompting indefinitely. Never paste a password into a script, tool call or support
thread; authorize sudo directly with the host administrator in a normal terminal.

## Model identity, disk and reuse

Locked payload total: **17788106147 bytes** (16.57 GiB). For an empty single-node managed cache, the current downloader requires at least **18861847971 free bytes** (payload plus its 1073741824-byte / 1 GiB reserve). For a partial download it checks missing-file bytes plus the same reserve; existing fragments already consume disk. This is a download floor, not a RAM requirement or a promise of peak disk use. Allow additional space for the archive, extraction, Python wheels/environment, logs and rollback copies. Attested two-node profiles instead require the full payload pre-staged on each rank; this managed-download floor does not enable remote acquisition. Full hash verification rereads every locked file; neither setup nor startup has a promised duration.

The lock's file inventory is authoritative, including non-weight sidecars. Model
identity is repository **and exact commit**, file sizes/hashes, index and native
fingerprint—not a directory name. Setup verifies the entire payload on reuse;
a prior receipt does not excuse current verification. It refuses changed,
extra/unlisted payload content, unsafe links and corrupt previously verified
content instead of silently overwriting it. A managed interrupted download can
resume at the same pin; a hash mismatch requires deliberate operator repair.

Single-node managed paths are under
`${XDG_DATA_HOME:-$HOME/.local/share}/rexl3/<runtime-id>/models/<role>/<revision>/`.
No payload is stored in the sealed bundle. For a **single-node** profile without
an external `model_dir` site binding, reuse means that verified managed cache, not
an arbitrary external path or HF snapshot symlink. A declared `model_dir` binding
allows single-node external reuse. For **attested two-node TP**, external paths
are mandatory instead: `nodes.leader.model_dir` and `nodes.follower.model_dir`.
Each rank needs the entire locked payload, not half the shards; neither path is a
managed download destination.

All external model/binary sources are a **read-only trust boundary**: their files,
directories and ancestor paths must satisfy the runtime's ownership, permission,
link and identity checks. Missing, unsafe or invalid content fails; the launcher
does not download into, chmod or repair the external source. Do not change
permissions, rename shards or remove sidecars from somebody else's directory
merely to satisfy this runtime. Admission never repairs or replaces these sources.

Read-only admission trusts root and the current UID. World write is refused;
only a root-owned sticky **ancestor** retains the temporary-directory exception,
never the model root. `0775` directories / `0664` files can be safe when their
owning group is demonstrably the current owner's private primary local group:
matching account/group names, no other primary or supplementary member UID,
and stable root-owned account authority. Name/GID equality or empty `gr_mem`
alone is not proof. There is no trusted-GID override. The helper freezes and
cross-checks `/etc/passwd`, `/etc/group`, `/etc/nsswitch.conf`, NSS enumeration and
per-account supplementary memberships, then reads them back after admission.
Supported NSS policies are plain `files` or `files systemd` (also for explicit
`initgroups`). Supplemental userdb records, unknown providers, active dynamic-user
state/container mappings, NSS disagreements and ambiguous records fail closed.
The built-in `io.systemd.DynamicUser` socket alone is supported only with no extra
NSS accounts/groups; its reserved GID range cannot authorize a private group.

The supported access model is Linux local ext2/3/4, XFS, Btrfs or tmpfs with
verifiable POSIX access ACLs. ACLs are read on held no-follow descriptors,
validated (version, structure, tags, IDs, mask and mode agreement), and evaluated
for **effective** writes. Harmless read/execute grants and mask-disabled writes
are allowed; foreign effective writes, unknown access mechanisms/filesystems
(including NFS/SMB/FUSE/overlay) and failed ACL reads are not assumed safe.
Default ACLs affect future creation, not existing read-only payload access.

Locked read-only model payloads may be regular **stable hardlinks**: the complete
SHA-256 and device/inode/link-count/size/mtime/ctime/access identity are verified
without visiting or changing aliases. Binary/dependency verification and every
managed downloader/write destination retain single-link admission. Symlinks are
never admitted. Ancestor rechecks bind ownership, mode and ACL rather than ctime
alone, since unrelated sibling creation may change directory ctime.

This is point-in-time verification, not a filesystem lease or hostile root/same-UID
isolation. Keep the admitted tree, aliases and account authority unchanged while
serving. Privileged change-and-restore, previously opened writer descriptors and
credentials retained after group removal are outside this baseline; stronger
isolation requires a separately designed OS boundary, not a permission bypass.

## Choose a profile and site, not new tuning

| Profile | Declared site fields | Site requirement | Model preparation |
| --- | --- | --- | --- |
| `qwen38-single-262k` | `binary_dir`, `http_host`, `http_port`, `model_dir` | optional; omitted site uses profile defaults / managed cache | typed `model_dir` binding; read-only verification |

For a **single-node** profile, an omitted site or empty overlay `{}` selects
profile defaults and the managed cache. An `attested-v1` cluster profile instead
has a **required two-node site**; omitted/empty sites fail with exit 2 and
`EXPLICIT_TWO_NODE_SITE_REQUIRED`. Only fields declared by the selected profile
**and implemented by its launcher** are accepted. Unknown, unused, unsafe or
wrongly typed fields are refusals. `site.example.json` is a cluster starting point,
not a single-node overlay or a way to discover hosts/enable unsupported behavior.

For a single-node profile declaring HTTP fields, a site may contain a literal
loopback `http_host` and integer `http_port`. A model-reuse site additionally
requires the `model_dir` field in the table. This example builds that reuse overlay
from an operator-selected existing absolute directory; it refuses undeclared
bindings rather than implying that historical launchers support them:

<!-- runtime-example: existing-model-site -->
```python
import json, os
from pathlib import Path
root = Path(os.environ['RUNTIME_ROOT'])
profile = json.loads((root / 'profile.json').read_text())
choice = profile['profiles'][os.environ['PROFILE']]
if choice['kind'] != 'single':
    raise SystemExit('This example is single-node only; use the required two-node site for attested TP')
if 'model_dir' not in choice['site_fields']:
    raise SystemExit('Selected profile does not declare model_dir; use its managed cache')
model = Path(os.environ['MODEL_DIR'])
if not model.is_absolute() or not model.is_dir() or '..' in model.parts:
    raise SystemExit('MODEL_DIR must be an existing absolute directory')
site = Path(os.environ['WORK']) / 'site.json'
with site.open('x', encoding='utf-8') as stream:
    json.dump({'model_dir': str(model)}, stream, indent=2)
site.chmod(0o600)
print(site)
```

```bash
export SITE="$WORK/site.json"
SITE_ARGS=(--site "$SITE")
./launch.sh print-command --profile "$PROFILE" "${SITE_ARGS[@]}"
./launch.sh --yes setup --profile "$PROFILE" "${SITE_ARGS[@]}"
./launch.sh verify --profile "$PROFILE" "${SITE_ARGS[@]}"
```

`MODEL_DIR` must be set by the operator before the Python example. This checks
only the site's shape/path; setup/verify perform identity verification. Reuse
requires the matching launcher implementation, not just an allowlisted field.
Keep this site file stable for start/status/logs/stop; changing it changes instance
identity. Export the same `SITE` for the README client: it uses real `print-command`
resolution for the address/port, not the profile's unresolved defaults. `API_BASE`
is needed only for a deliberate alternate route such as a matching private tunnel.

### Required two-node site and pre-staged ranks (`attested-v1`)

1. Select the attested TP profile from the generated table. Copy
   `site.example.json` to a **new private file outside the sealed bundle**, replace
   its placeholder addresses/paths with operator-approved values, and export its
   absolute filename as `SITE`. Each of `nodes.leader` and `nodes.follower` needs
   a literal IPv4 `address` and `fabric_address`, SSH `user`, absolute `model_dir` and
   absolute `binary_dir`. The HTTP API binds to the leader's `address`, not the
   profile's loopback default. The site must also state `tp_port`, `fabric_interface`
   (the network interface carrying the TP traffic on both ranks) and `hca` (a
   non-empty list of RoCE device:port rails such as `["mlx5_0:1"]`; list every rail
   you want NCCL to use, for example `["mlx5_0:1", "mlx5_1:1"]` for two). Omitting
   any of them is refused (`SITE_TP_PORT_REQUIRED`, `SITE_FABRIC_INTERFACE_REQUIRED`,
   `SITE_HCA_REQUIRED`): the sealed cluster file's example network is never used as
   a default, because NCCL would wait on a device your host does not have. The
   example file ships `REPLACE_WITH_...` names for the interface and both rails,
   and an unedited copy is refused (`SITE_FABRIC_INTERFACE_PLACEHOLDER`,
   `SITE_HCA_PLACEHOLDER`). Find your names with `ip -br link` and `ibdev2netdev`
   (or `rdma link`). Keep the normalized native cluster
   configuration named by the profile sealed; the site supplies placement only.
2. On **each rank**, have the exact release's `rexl3-serve` **and** `rexl3-cluster`
   binaries pre-staged in its `binary_dir`. The binary hashes and native version
   fields must match `release.lock.json`. Pre-stage the **complete** locked model
   revision in its `model_dir`, including all payload shards, indexes and sidecars.
   Different paths per rank are supported; different bytes/revisions are not.
3. Arrange the locked dependencies, matching native libraries, Python and
   qualified transport on each rank through separately authorized host preparation.
   Controller `install-deps` is not a remote installer. Arrange noninteractive SSH
   authentication and already-trusted host keys for the selected users/addresses;
   the adapter requires strict host-key verification and does not forward agents,
   credentials or arbitrary inherited environment/loader settings to rank processes.
4. Resolve, then verify. `print-command` is read-only **local configuration
   resolution**, not an SSH/payload/dependency check. `setup` and `verify` perform
   read-only rank identity verification over SSH; `setup` also bootstraps local
   controller dependencies. There is **no automatic remote download** or repair.
   Even `setup --accept-model-download` still only verifies the pre-staged ranks.
   `SETUP_VERIFIED` is not inference readiness; start and check owned readiness
   separately after verification succeeds.

<!-- runtime-example: cluster-verify -->
```bash
: "${SITE:?Set the reviewed two-node site absolute filename first}"
export SITE
SITE_ARGS=(--site "$SITE")
./launch.sh print-command --profile "$PROFILE" "${SITE_ARGS[@]}"
./launch.sh --yes setup --profile "$PROFILE" "${SITE_ARGS[@]}"
./launch.sh verify --profile "$PROFILE" "${SITE_ARGS[@]}"
```

Do not copy historical cluster JSON, add unreviewed environment files, edit
sealed knobs or bypass the controller to start ranks manually. Older adapters
without `attested-v1` do not acquire remote support from a profile/documentation
change. Port/SSH reachability alone does not prove identity or TP readiness. No
topology or full-context/image qualification is transferred from another model
revision or archive; declared support still requires the exact release receipts.

## Files, state and corresponding source

| Item | Role |
| --- | --- |
| `launch.sh`, `scripts/` | Verified shared launcher, installer, doctor, model preparation and lifecycle |
| `bin/` | Prebuilt native executables; cluster controller when declared |
| `profile.json`, sealed configuration files | Serving choices, capability declarations, timeouts and allowed site fields |
| `model.lock.json` | Exact pack revision, complete payload hashes/sizes and license provenance |
| `dependencies.lock.json`, `requirements-runtime.lock` | System package policy and exact hashed Python wheels |
| `source.manifest.json`, `TEMPLATE.json` | Generated-source inventory and recorded template identity; not new approval |
| `release.lock.json` | Native/platform/build identity and immutable bundle inventory |
| `approval.json` beside the archive | External exact-archive approval/evidence, not assumed in the tarball |
| `KNOWN_ISSUES.md`, `docs/`, README agent checklist | Limitations and operator/agent procedures |
| `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`, `sbom/` | License and dependency attribution |

For source auditing/redistribution, obtain the matching corresponding source from
the exact production release's accessible source artifact/link. The publisher
must provide it without requiring private dev access; this requirement is not an
assertion that an unnamed source asset already exists. Select its exact filename
and checksum from the release's source record, verify before extraction, and
compare its recorded commit/tree to `release.lock.json`. Never guess an asset
name or use a wildcard that might select the runtime archive instead. Do not
build engine source merely to operate the prebuilt bundle.

Update/rollback procedure:

1. Preserve the current archive, sibling approval, profile/site selection and
   sanitized diagnostics. Keep local conversations/logs private.
2. Stop using the current bundle with its original profile/site. Verify cleanup;
   status 24 is expected for stopped, but an ownership error is not stopped.
3. Verify/extract the selected replacement or previous release in a new directory.
   Do not mutate an in-use bundle or copy its verification records to a new pin.
4. Reuse only an identity-matching payload. Resolve stale instance identity with
   guarded `reset-state` only after all previous owned processes are absent; do
   not delete the state tree, environment or weights as an upgrade shortcut.
5. Repeat install-deps → doctor → setup/verify → print-command → start → bounded
   owned readiness → enabled API examples. A rollback is a tested launch of the
   old exact artifact, not just a file copy. Never claim success from HTTP 200 alone.
