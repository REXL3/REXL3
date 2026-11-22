use gb_cluster::schema::Node;
use gb_cluster::transport::{SshTransport, Transport};
use std::time::Duration;

fn local_node() -> Node {
    Node {
        name: "local-test".into(),
        number: Some(1),
        address: "192.0.2.1".into(),
        fabric_address: "192.0.2.1".into(),
        user: "test".into(),
        gpu_lock: "/tmp/rexl3-test-lock".into(),
        expected_hostname: None,
        local: true,
        disabled: false,
        reason: None,
        env: Default::default(),
        flags: vec![],
    }
}

#[test]
fn timeout_returns_partial_stdout_and_stderr() {
    let transport = SshTransport { ssh_options: vec![] };
    let error = transport
        .run(&local_node(), "echo remote-progress\necho remote-diagnostic >&2\nsleep 30\n", Duration::from_millis(300))
        .unwrap_err();

    assert!(error.contains("did not finish"), "{error}");
    assert!(error.contains("stdout tail: remote-progress"), "{error}");
    assert!(error.contains("stderr tail: remote-diagnostic"), "{error}");
}
