//! Hard message AND retained-byte bounds. The GPU producer only uses `send` (try-send).
//! Failure has a separate one-cell signal: a full data queue cannot hide a truncated reply.
use crate::runtime::{FailKind, SeqError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Notify, OwnedSemaphorePermit, Semaphore};

pub const CAPACITY: usize = 64;
pub const BYTE_CAPACITY: usize = 1 << 20;
pub const MAX_MESSAGE_BYTES: usize = 256 << 10;
pub trait Message: Send {
    fn retained_bytes(&self) -> usize;
    fn failure(e: SeqError) -> Self;
}
pub type FailureHook = Arc<dyn Fn(&SeqError) + Send + Sync>;
struct Shared {
    failure_hook: Option<FailureHook>,
    bytes: Arc<Semaphore>,
    failure: Mutex<Option<SeqError>>,
    wake: Notify,
}
struct Envelope<T> {
    value: T,
    _credit: OwnedSemaphorePermit,
}
pub struct Sender<T> {
    tx: mpsc::Sender<Envelope<T>>,
    shared: Arc<Shared>,
}
pub struct Receiver<T> {
    rx: mpsc::Receiver<Envelope<T>>,
    shared: Arc<Shared>,
    ended: bool,
    // Credit follows the consumer-held value until its next demand (or receiver drop), including SSE body polling.
    held: Option<OwnedSemaphorePermit>,
}
impl<T> Clone for Sender<T> {
    fn clone(&self) -> Self {
        Self { tx: self.tx.clone(), shared: self.shared.clone() }
    }
}
pub fn channel<T: Message>() -> (Sender<T>, Receiver<T>) {
    channel_with_failure(None)
}
pub fn channel_with_failure<T: Message>(failure_hook: Option<FailureHook>) -> (Sender<T>, Receiver<T>) {
    let (tx, rx) = mpsc::channel(CAPACITY);
    let shared = Arc::new(Shared {
        bytes: Arc::new(Semaphore::new(BYTE_CAPACITY)),
        failure: Mutex::new(None),
        wake: Notify::new(),
        failure_hook,
    });
    (Sender { tx, shared: shared.clone() }, Receiver { rx, shared, ended: false, held: None })
}
fn overflow() -> SeqError {
    SeqError { kind: FailKind::Backpressure, message: "the bounded response backlog or write-stall budget was exceeded".into() }
}
impl<T: Message> Sender<T> {
    pub fn has_failure_hook(&self) -> bool {
        self.shared.failure_hook.is_some()
    }
    pub fn fail(&self, e: SeqError) {
        let mut f = self.shared.failure.lock().unwrap_or_else(|e| e.into_inner());
        if f.is_none() {
            if let Some(hook) = &self.shared.failure_hook {
                hook(&e);
            }
            *f = Some(e);
        }
        self.shared.wake.notify_one();
    }
    pub fn is_closed(&self) -> bool {
        self.tx.is_closed() || self.shared.failure.lock().unwrap_or_else(|e| e.into_inner()).is_some()
    }
    pub async fn closed(&self) {
        if !self.is_closed() {
            self.tx.closed().await;
        }
    }
    /// Never waits on the engine thread. Overflow cancels the sequence, NOT one token.
    pub fn send(&self, value: T) -> Result<(), SeqError> {
        let n = value.retained_bytes().max(1);
        if self.is_closed() {
            return Err(overflow());
        }
        let credit = if n <= MAX_MESSAGE_BYTES { self.shared.bytes.clone().try_acquire_many_owned(n as u32).ok() } else { None };
        if let Some(credit) = credit {
            if self.tx.try_send(Envelope { value, _credit: credit }).is_ok() {
                return Ok(());
            }
        }
        let e = overflow();
        self.fail(e.clone());
        Err(e)
    }
    /// Async producers propagate response-body demand; at most ONE capped value waits per producer.
    pub async fn send_wait(&self, value: T, deadline: Instant, stall: Duration) -> Result<(), SeqError> {
        let n = value.retained_bytes().max(1);
        if n > MAX_MESSAGE_BYTES || self.is_closed() {
            let e = overflow();
            self.fail(e.clone());
            return Err(e);
        }
        if Instant::now() >= deadline {
            let e = SeqError { kind: FailKind::RequestTimeout, message: "the request deadline expired".into() };
            self.fail(e.clone());
            return Err(e);
        }
        let at = deadline.min(Instant::now() + stall);
        // Fast readers never allocate a timer or consume async cooperative scheduling budget merely to send an available slot.
        let result = if let Ok(credit) = self.shared.bytes.clone().try_acquire_many_owned(n as u32) {
            match self.tx.try_send(Envelope { value, _credit: credit }) {
                Ok(()) => return Ok(()),
                Err(mpsc::error::TrySendError::Full(envelope)) => {
                    tokio::time::timeout_at(at.into(), async { self.tx.send(envelope).await.map_err(|_| ()) }).await
                }
                Err(mpsc::error::TrySendError::Closed(_)) => Ok(Err(())),
            }
        } else {
            tokio::time::timeout_at(at.into(), async {
                let credit = self.shared.bytes.clone().acquire_many_owned(n as u32).await.map_err(|_| ())?;
                self.tx.send(Envelope { value, _credit: credit }).await.map_err(|_| ())
            })
            .await
        };
        if matches!(result, Ok(Ok(()))) {
            return Ok(());
        }
        let e = if Instant::now() >= deadline {
            SeqError { kind: FailKind::RequestTimeout, message: "the request deadline expired".into() }
        } else {
            overflow()
        };
        self.fail(e.clone());
        Err(e)
    }
}
impl<T: Message> Receiver<T> {
    pub async fn recv(&mut self) -> Option<T> {
        self.held.take();
        if self.ended {
            return None;
        }
        loop {
            let wake = self.shared.wake.notified();
            let error = self.shared.failure.lock().unwrap_or_else(|e| e.into_inner()).clone();
            if let Some(e) = error {
                self.rx.close();
                while self.rx.try_recv().is_ok() {}
                self.ended = true;
                return Some(T::failure(e));
            }
            tokio::select! {
                biased;
                _ = wake => {},
                item = self.rx.recv() => return item.map(|e| { self.held = Some(e._credit); e.value }),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Debug, PartialEq)]
    enum Item {
        Data(Vec<u8>),
        Error(FailKind),
    }
    impl Message for Item {
        fn retained_bytes(&self) -> usize {
            match self {
                Self::Data(v) => v.capacity(),
                Self::Error(_) => 1,
            }
        }
        fn failure(e: SeqError) -> Self {
            Self::Error(e.kind)
        }
    }
    #[tokio::test]
    async fn count_overflow_never_silently_drops_or_finishes() {
        let (tx, mut rx) = channel();
        for _ in 0..CAPACITY {
            tx.send(Item::Data(vec![1])).unwrap();
        }
        assert!(tx.send(Item::Data(vec![2])).is_err());
        assert_eq!(rx.recv().await, Some(Item::Error(FailKind::Backpressure)));
        assert_eq!(rx.recv().await, None);
        assert_eq!(tx.shared.bytes.available_permits(), BYTE_CAPACITY);
    }
    #[tokio::test]
    async fn speculative_batches_are_charged_by_bytes_not_message_count() {
        let (tx, mut rx) = channel();
        for _ in 0..4 {
            tx.send(Item::Data(vec![1; MAX_MESSAGE_BYTES])).unwrap();
        }
        assert!(tx.send(Item::Data(vec![2])).is_err());
        assert_eq!(rx.recv().await, Some(Item::Error(FailKind::Backpressure)));
        assert_eq!(tx.shared.bytes.available_permits(), BYTE_CAPACITY);
    }
    #[tokio::test]
    async fn async_write_stall_fails_and_releases_credits() {
        let (tx, mut rx) = channel();
        for _ in 0..CAPACITY {
            tx.send(Item::Data(vec![1])).unwrap();
        }
        assert!(tx
            .send_wait(Item::Data(vec![2]), Instant::now() + Duration::from_secs(1), Duration::from_millis(10))
            .await
            .is_err());
        assert_eq!(rx.recv().await, Some(Item::Error(FailKind::Backpressure)));
        assert_eq!(tx.shared.bytes.available_permits(), BYTE_CAPACITY);
    }
    #[tokio::test]
    async fn consumed_and_dropped_values_return_all_credits() {
        let (tx, mut rx) = channel();
        tx.send(Item::Data(vec![3; 1024])).unwrap();
        assert_eq!(rx.recv().await, Some(Item::Data(vec![3; 1024])));
        assert_eq!(tx.shared.bytes.available_permits(), BYTE_CAPACITY - 1024, "consumer-held value remains charged");
        tx.send(Item::Data(vec![3; 1024])).unwrap();
        drop(rx);
        assert_eq!(tx.shared.bytes.available_permits(), BYTE_CAPACITY);
        assert!(tx.is_closed());
    }
}
