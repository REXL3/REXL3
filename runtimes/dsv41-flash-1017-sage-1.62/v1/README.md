# REXL3 runtime bundle

This directory is a **shared source template**, not a finalized model release.
An assembled candidate also has `profile.json`, `model.lock.json`,
`dependencies.lock.json`, `requirements-runtime.lock`, `release.lock.json` and
native executables under `bin/`. Without those inputs, launch fails closed.
The final profile and native evidence are supplied by the hardware-capable agent;
this template contains no guessed model parameters or qualification results.

## Quickstart for a finalized candidate

1. Verify the externally supplied archive checksum and extract into a new directory.
2. Run **`./launch.sh`** as your normal user (not with sudo).
3. Approve required software installation and, separately, the model download.
4. Wait for owned-process readiness and a successful actual chat smoke response.

Missing supported software is installed before inference. The installer uses an
isolated, hash-locked Python environment and the reviewed system package map. It
never upgrades the whole OS or silently changes drivers. A required reboot is
reported, never performed. Running the launcher as root is refused for `start`.

The declared default profile is used. Do not set `EXTRA_ARGS`, override the model
repository/revision, or add arbitrary tuning variables: final profiles are data,
and changing their behavior requires new review and qualification.

## Commands

```bash
./launch.sh --help
./launch.sh doctor
./launch.sh --yes install-deps
./launch.sh setup --accept-model-download
./launch.sh print-command
./launch.sh start
./launch.sh status
./launch.sh logs
./launch.sh stop
./launch.sh verify
./launch.sh diagnostics
```

Global installer flags precede the command. `--yes` approves routine dependency
installation only; it does **not** approve downloading hundreds of GiB, driver
replacement or reboot. `--accept-model-download` is the separate explicit model
acquisition consent. `doctor` is read-only. Standalone `install-deps` does not
acquire models or start a server.

Single-node `start` stays in the foreground. Keep the terminal open; Ctrl-C stops
only its owned child. Use another terminal for `status`, `logs` or `stop`. A busy
port or lock is an error, not permission to attach to or kill someone else's job.
A failed readiness/smoke check cleans up only the processes this invocation owns.

Cluster handlers reuse the shipped `rexl3-cluster` controller and its locks, but
this version refuses remote ranks pending a qualified adapter for remote payload
attestation and environment scrubbing. Only local resolved configurations have CPU
fixture coverage. Do not advertise TP from this branch. Single-node HTTP is
loopback-only; site overlays support declared HTTP host/port fields only. Unsupported
site substitutions are refused, not silently ignored.

See `docs/SETUP.md` and the candidate's pinned profile/platform contracts for the
actual supported hardware, software versions and capabilities. A CPU fixture pass
or a `state: final` profile alone is **not** tester-release qualification.

## State and feedback

State is kept under `${XDG_DATA_HOME:-$HOME/.local/share}/rexl3/<runtime-id>/`.
Model payloads are outside this source tree. Verification reads the full pinned
payload; initial verification of a large model can take significant time.
`reset-state` is guarded and must not delete downloaded models or another job.
Diagnostics use an allowlist; raw prompts, responses, logs, environment values,
credentials and SSH keys are not included. Review any report before sharing it.

## License

`LICENSE` and `NOTICE` are the GNU Affero General Public License v3.0 (`AGPL-3.0-only`) and cover
everything in this bundle that the REXL3 authors wrote: the launch and contract scripts, the serve
profile, the evaluator and the documentation. The engine binaries in `bin/` are built from the same
licence. Model weights are not in this bundle: `model.lock.json` names the pack and its revision,
and that pack's own license governs the weights. Third-party notices travel with the bundle in
`THIRD_PARTY_NOTICES.md` and `sbom/`.
