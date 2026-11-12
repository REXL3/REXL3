use gb_server::preparation::{PreparationGate, PreparationLimits, RequestWork};
use std::time::Duration;
#[tokio::test]
async fn weighted_credit_and_backlog_survive_future_drop() {
    let gate = PreparationGate::new(PreparationLimits {
        bytes: 100,
        retained_bytes: 100,
        workers: 2,
        backlog: 2,
        wait: Duration::from_millis(40),
        deadline: Duration::from_millis(100),
    })
    .unwrap();
    let work = RequestWork::new(Duration::from_secs(5));
    let p = gate.acquire(100, 40, &work).await.unwrap();
    let g = gate.clone();
    let w = work.clone();
    let waiting = tokio::spawn(async move { g.acquire(1, 40, &w).await });
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert_eq!(gate.active_bytes(), 100);
    assert!(gate.acquire(1, 40, &work).await.is_err());
    assert!(waiting.await.unwrap().is_err());
    let (tx, rx) = std::sync::mpsc::channel();
    let worker = tokio::task::spawn_blocking(move || {
        rx.recv().unwrap();
        drop(p);
    });
    drop(work);
    assert_eq!(gate.active_bytes(), 100);
    tx.send(()).unwrap();
    worker.await.unwrap();
    assert_eq!(gate.active_bytes(), 0);
}
#[test]
fn deadline_and_cancel_are_cooperative() {
    let w = RequestWork::new(Duration::ZERO);
    assert_eq!(w.check().unwrap_err().status, 504);
    assert_eq!(RequestWork::new(Duration::MAX).check().unwrap_err().status, 504);
    let w = RequestWork::new(Duration::from_secs(5));
    w.cancel();
    assert_eq!(w.check().unwrap_err().status, 499);
}

#[tokio::test]
async fn cancelled_join_handle_keeps_inputs_charged_until_worker_drop() {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    struct Input {
        gate: Arc<PreparationGate>,
        dropped: Arc<AtomicBool>,
    }
    impl Drop for Input {
        fn drop(&mut self) {
            assert_eq!(self.gate.retained_bytes(), 40);
            assert_eq!(self.gate.active_bytes(), 100);
            self.dropped.store(true, Ordering::SeqCst);
        }
    }
    let gate =
        PreparationGate::new(PreparationLimits { bytes: 100, retained_bytes: 100, workers: 1, backlog: 1, ..Default::default() })
            .unwrap();
    let work = RequestWork::new(Duration::from_secs(5));
    let permit = gate.acquire(100, 40, &work).await.unwrap();
    let dropped = Arc::new(AtomicBool::new(false));
    let job = gb_server::preparation::PreparationJob::new(Input { gate: gate.clone(), dropped: dropped.clone() }, permit);
    let (tx, rx) = std::sync::mpsc::channel();
    let (entered, started) = std::sync::mpsc::channel();
    let worker = tokio::task::spawn_blocking(move || {
        let _input = job.inputs();
        entered.send(()).unwrap();
        rx.recv_timeout(Duration::from_secs(2)).unwrap();
    });
    started.recv_timeout(Duration::from_secs(1)).unwrap();
    drop(worker); // Detached blocking work is NOT cancelled.
    assert_eq!(gate.retained_bytes(), 40);
    assert!(!dropped.load(Ordering::SeqCst));
    tx.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        while gate.jobs() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(dropped.load(Ordering::SeqCst));
    assert_eq!(gate.active_bytes(), 0);
    assert_eq!(gate.retained_bytes(), 0);
}
