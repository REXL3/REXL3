//! How a script reaches a node: `ssh user@address bash -s` with the script on stdin (or a local `bash -s` for a node marked `local`).
//! The trait is the seam the tests use: a fake transport answers with canned output.

use crate::schema::{Cluster, Node};
use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Exec {
    pub status: i32,
    pub stdout: String,
    pub stderr: String,
}

pub trait Transport {
    /// Run `script` on `node` through `bash -s` and return its output; `Err` is a failure to run at all (ssh missing, a timeout).
    fn run(&self, node: &Node, script: &str, timeout: Duration) -> Result<Exec, String>;
}

/// The real transport: the ssh options of the cluster file, then `user@address bash -s`.
pub struct SshTransport {
    pub ssh_options: Vec<String>,
}

impl SshTransport {
    pub fn new(c: &Cluster) -> SshTransport {
        SshTransport { ssh_options: c.ssh_options.clone() }
    }
}

/// The command line of a node's session, for `--dry-run` and for the error messages.
pub fn session_words(options: &[String], node: &Node) -> Vec<String> {
    if node.local {
        vec!["bash".into(), "-s".into()]
    } else {
        let mut w = vec!["ssh".to_string()];
        w.extend(options.iter().cloned());
        w.push(format!("{}@{}", node.user, node.address));
        w.push("bash".into());
        w.push("-s".into());
        w
    }
}

/// `--dry-run` rendering of one session: the command line and the script it would be fed.
pub fn describe(options: &[String], node: &Node, what: &str, script: &str) -> String {
    let mut s = format!("### {}   # node {}: {what}\n", session_words(options, node).join(" "), node.name);
    s.push_str(script);
    if !script.ends_with('\n') {
        s.push('\n');
    }
    s.push_str("### end\n");
    s
}

impl Transport for SshTransport {
    fn run(&self, node: &Node, script: &str, timeout: Duration) -> Result<Exec, String> {
        let w = session_words(&self.ssh_options, node);
        let mut child = Command::new(&w[0])
            .args(&w[1..])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("cannot run `{}`: {e}", w[0]))?;
        let mut stdin = child.stdin.take().ok_or("no stdin")?;
        let script = script.to_string();
        // a thread feeds the script so that a remote that never reads cannot block the poll below
        let feeder = std::thread::spawn(move || {
            let _ = stdin.write_all(script.as_bytes());
        });
        let mut out = child.stdout.take().ok_or("no stdout")?;
        let mut err = child.stderr.take().ok_or("no stderr")?;
        let t_out = std::thread::spawn(move || {
            let mut b = Vec::new();
            let _ = out.read_to_end(&mut b);
            b
        });
        let t_err = std::thread::spawn(move || {
            let mut b = Vec::new();
            let _ = err.read_to_end(&mut b);
            b
        });
        let deadline = Instant::now() + timeout;
        let status = loop {
            match child.try_wait() {
                Ok(Some(st)) => break st,
                Ok(None) => {
                    if Instant::now() >= deadline {
                        let _ = child.kill();
                        let _ = child.wait();
                        return Err(format!("`{}` did not finish within {timeout:?} (node {})", w.join(" "), node.name));
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(e) => return Err(format!("waiting for `{}`: {e}", w[0])),
            }
        };
        let _ = feeder.join();
        let stdout = String::from_utf8_lossy(&t_out.join().unwrap_or_default()).into_owned();
        let stderr = String::from_utf8_lossy(&t_err.join().unwrap_or_default()).into_owned();
        Ok(Exec { status: status.code().unwrap_or(-1), stdout, stderr })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(local: bool) -> Node {
        Node {
            name: "N1".into(),
            number: Some(1),
            address: "192.0.2.1".into(),
            fabric_address: "192.0.2.1".into(),
            user: "u".into(),
            gpu_lock: "/tmp/l".into(),
            expected_hostname: None,
            local,
            disabled: false,
            reason: None,
            env: Default::default(),
            flags: vec![],
        }
    }

    #[test]
    fn a_local_node_is_bash_and_a_remote_one_is_ssh() {
        let o = vec!["-o".to_string(), "BatchMode=yes".to_string()];
        assert_eq!(session_words(&o, &node(true)).join(" "), "bash -s");
        assert_eq!(session_words(&o, &node(false)).join(" "), "ssh -o BatchMode=yes u@192.0.2.1 bash -s");
    }

    #[test]
    fn a_local_session_runs_the_script_and_reports_status_and_output() {
        let t = SshTransport { ssh_options: vec![] };
        let r = t.run(&node(true), "echo hello\necho oops >&2\nexit 7\n", Duration::from_secs(10)).unwrap();
        assert_eq!(r.status, 7);
        assert_eq!(r.stdout.trim(), "hello");
        assert_eq!(r.stderr.trim(), "oops");
    }

    #[test]
    fn a_session_that_does_not_finish_is_killed() {
        let t = SshTransport { ssh_options: vec![] };
        let e = t.run(&node(true), "sleep 30\n", Duration::from_millis(300)).unwrap_err();
        assert!(e.contains("did not finish"), "{e}");
    }
}
