use gb_server::preparation::{PreparationGate, RequestWork};
use std::time::Duration;

#[tokio::test]
async fn shutdown_refuses_new_jobs_but_waits_for_actual_worker_credit() {
    let gate = PreparationGate::new(Default::default()).unwrap();
    let work = RequestWork::new(Duration::from_secs(5)).with_shutdown(gate.shutdown_token());
    let permit = gate.acquire(4096, 1024, &work).await.unwrap();
    assert_eq!(gate.jobs(), 1);
    assert_eq!(gate.active_bytes(), 4096);
    gate.shutdown();
    assert_eq!(work.check().unwrap_err().status, 503);
    let independent = RequestWork::new(Duration::from_secs(5));
    assert!(gate.acquire(4096, 1024, &independent).await.is_err());
    assert_eq!(gate.jobs(), 1, "shutdown must not pretend the active owner ended");
    assert!(tokio::time::timeout(Duration::from_millis(20), gate.wait_idle()).await.is_err());
    drop(permit);
    tokio::time::timeout(Duration::from_secs(1), gate.wait_idle()).await.unwrap();
    assert_eq!(gate.active_bytes(), 0);
    assert_eq!(gate.retained_bytes(), 0);
}
