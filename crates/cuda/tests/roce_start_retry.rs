use gb_cuda::roce::start_with_rollback;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

#[test]
fn failed_start_is_retryable_and_success_is_idempotent() {
    let started = AtomicBool::new(false);
    let lock = Mutex::new(());
    let mut attempts = 0;
    let mut call = || {
        start_with_rollback(&started, &lock, || {
            attempts += 1;
            if attempts == 1 {
                Err("injected proxy startup failure".into())
            } else {
                Ok(())
            }
        })
    };

    assert_eq!(call().unwrap_err(), "injected proxy startup failure");
    assert!(!started.load(Ordering::Acquire));
    call().unwrap();
    assert!(started.load(Ordering::Acquire));
    call().unwrap();
    assert_eq!(attempts, 2, "a repeated successful start must not invoke the proxy again");
}

#[test]
fn isolation_failure_does_not_publish_started() {
    let started = AtomicBool::new(false);
    let lock = Mutex::new(());
    let result = start_with_rollback(&started, &lock, || Err("injected isolation failure".into()));
    assert_eq!(result.unwrap_err(), "injected isolation failure");
    assert!(!started.load(Ordering::Acquire));
}
