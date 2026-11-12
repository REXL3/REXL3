//! `gb-cluster`: the operator side of a tensor-parallel group of R ranks (docs/CLUSTER_OPERATIONS.md).
//!
//! One JSON cluster file ([`schema`]) describes the nodes, the rank map of each group, the ports, the NCCL pins and the shared serve
//! flags. From it the launcher derives, for rank r of R, the exact remote script ([`script`]) and runs it over a [`transport::Transport`]
//! (ssh, or a fake in the tests): [`check`] is the pre-flight, [`ops`] is `up`, `down` and `status`, [`supervise`] is the group-aware
//! restart policy. Everything is std + serde_json: no ssh library, no libc crate (one `signal` declaration in [`signal`]).
//!
//! The runtime's own contract (docs/NEMOTRON_TP.md sections 5.5, 5.7, 12.4) is what this crate automates: the same binary and the same
//! plan-affecting flags and environment on every rank, `NCCL_MAX_NCHANNELS=4`, rank 0 first or last within `--tp-connect-timeout`, GPU
//! locks in node-list order, SIGTERM on rank 0 for the graceful stop, kill by PID, never `pkill -f`, and nodes the owner has ruled out
//! are refused by name before a byte goes over ssh.

pub mod check;
pub mod http;
pub mod ops;
pub mod schema;
pub mod script;
pub mod signal;
pub mod supervise;
pub mod transport;

/// Exit codes of `rexl3-cluster` itself (the supervisor's give-up codes follow sysexits.h so that a systemd unit can name them).
pub mod exit {
    /// Done: the check passed, the group is up, the group was stopped, the supervisor was stopped by the operator.
    pub const OK: i32 = 0;
    /// The operation failed (a check FAILed, a rank did not come up, a forced kill was needed, the group is not up).
    pub const FAILED: i32 = 1;
    /// A usage error or an invalid cluster file (nothing was run).
    pub const USAGE: i32 = 2;
    /// The supervisor gave up because restarting cannot help: a refused boot twice in a row (EX_CONFIG).
    pub const GAVE_UP_REFUSED: i32 = 78;
    /// The supervisor gave up because the group failed too often in the window (EX_TEMPFAIL).
    pub const GAVE_UP_RATE: i32 = 75;
}
