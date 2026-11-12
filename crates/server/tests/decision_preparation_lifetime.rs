use gb_server::budget::{Limits, RequestBudget, RequestLease};
use std::sync::Arc;

#[tokio::test]
async fn finished_preparation_releases_cpu_slot_without_releasing_retained_body_credit() {
    let budget = RequestBudget::new(Limits::default()).unwrap();
    let mut raw = budget.lease();
    raw.grow_to(1024).unwrap();
    let lease = Arc::new(RequestLease::new(raw, budget.prepare().unwrap()));
    let second = budget.prepare().unwrap();
    assert!(budget.prepare().is_err());
    lease.finish_preparation();
    let unrelated_text = budget.prepare().expect("native decisions wait must not monopolize CPU preparation");
    assert_eq!(budget.used(), 1024, "retained body accounting remains active");
    lease.finish_preparation(); // idempotent, no double return
    assert!(budget.prepare().is_err());
    drop(unrelated_text);
    drop(second);
    drop(lease);
    assert_eq!(budget.used(), 0);
}

#[tokio::test]
async fn decision_credit_is_independent_and_follows_last_owner() {
    let budget = RequestBudget::new(Limits::default()).unwrap();
    let a = Arc::new(budget.decision_admitted().unwrap());
    let worker = a.clone();
    let b = budget.decision_admitted().unwrap();
    let c = budget.decision_admitted().unwrap();
    let d = budget.decision_admitted().unwrap();
    assert!(budget.decision_admitted().is_err());
    drop(a);
    assert!(budget.decision_admitted().is_err(), "detached preparation worker still owns credit");
    assert!(budget.prepare().is_ok(), "native score capacity is not the shared CPU gate");
    drop(worker);
    assert!(budget.decision_admitted().is_ok());
    drop((b, c, d));
}
