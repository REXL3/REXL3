# Troubleshooting the runtime candidate

Use `./launch.sh doctor` for read-only diagnosis. Use `install-deps` to repair
supported missing software, or `setup` to prepare the locked model without
inference. Do not bypass a failed check to produce a green screenshot.

| Failure | Required action |
| --- | --- |
| `PROFILE_NOT_READY` | The source handoff or assembled bundle is incomplete. Have the profile owner supply final reviewed data and real native build metadata. |
| Unsupported platform | Use the qualified Ubuntu 24.04 ARM64 host. Do not install distro-specific packages on another OS. |
| Missing or incompatible dependencies | Read the approved package policy, approve only the named repair, then rerun `install-deps`. An unavailable exact version requires a reviewed policy update and requalification. |
| Approval refused or missing sudo | Stop and ask the host administrator. Do not run the entire launcher as root. |
| Driver maintenance or reboot required | Arrange explicit maintenance with the GPU owner. Routine `--yes` does not grant this authority; the launcher never reboots. |
| Changed file, binary, mode, config or requirements | Replace the candidate from its independently verified archive, or have the owner create/requalify a new one. Do not edit the inventory hashes to accept corruption. |
| Model download refused or offline | Give separate download consent only after reviewing size/destination. Restore access and rerun setup at the same pin. Never substitute a different quantization or revision. |
| Existing payload hash mismatch | Preserve diagnostics and repair only the named managed cache through an explicit operator action. The downloader refuses silent replacement of corrupt existing content. |
| Busy device/port/lock | Coordinate with its owner. A healthy existing listener cannot satisfy this invocation's readiness and is never killed for convenience. |
| Readiness, identity or smoke failure | Keep local logs, use sanitized diagnostics, and fix the named problem. Cleanup applies only to owned processes. |
| Stale or reused PID | Do not signal it. Inspect ownership and use guarded reset only after the previous owner is absent. Models, logs and user configuration are preserved. |
| Dependency supervision unavailable or cleanup unknown | Use the supported systemd 255+/cgroup-v2 host with a working login user manager. If a transaction was admitted, recover its exact unit with the administrator; the installer retains ownership until verified drain. Never bypass this by force-killing the controller. |
| Remote cluster refused | This template does not yet admit remote ranks. A qualified adapter must bind complete remote payload/sidecar identity and sanitized environment before multi-node release. Profile JSON alone does not lift this restriction. |

`logs` is a local operator command and can contain sensitive data. Default
`diagnostics` deliberately does not package raw logs, prompts, replies, full
environments, credential files or SSH material. Review a report before sharing.

For the final runtime's specific limitations and known failures, read
`KNOWN_ISSUES.md`, supplied by the profile owner. Lack of known-issues data is not
permission to manufacture a blanket pass. Model capability claims and exactness
still require native evidence from the same artifact/profile/model identities.
