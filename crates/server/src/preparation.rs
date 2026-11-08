//! Bounded, weighted CPU preparation admission. Every lease moves into the blocking worker.
use crate::api::ApiError;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::time::{Duration, Instant};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Clone, Copy, Debug)]
pub struct PreparationLimits {
    pub bytes: usize,
    pub retained_bytes: usize,
    pub workers: usize,
    /// Total admitted active + waiting jobs, not an unlimited semaphore FIFO.
    pub backlog: usize,
    pub wait: Duration,
    pub deadline: Duration,
}
impl Default for PreparationLimits {
    fn default() -> Self {
        Self {
            bytes: 2 << 30,
            retained_bytes: 128 << 20,
            workers: 2,
            backlog: 64,
            wait: Duration::from_secs(300),
            deadline: Duration::from_secs(1800),
        }
    }
}
#[derive(Clone, Debug)]
pub struct RequestWork {
    cancelled: Arc<AtomicBool>,
    stopping: Option<Arc<AtomicBool>>,
    pub deadline: Instant,
}
impl RequestWork {
    pub fn new(timeout: Duration) -> Self {
        let now = Instant::now();
        Self { cancelled: Arc::new(AtomicBool::new(false)), stopping: None, deadline: now.checked_add(timeout).unwrap_or(now) }
    }
    pub fn with_shutdown(mut self, stopping: Arc<AtomicBool>) -> Self {
        self.stopping = Some(stopping);
        self
    }
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
    pub fn check(&self) -> Result<(), ApiError> {
        if self.stopping.as_ref().is_some_and(|f| f.load(Ordering::Acquire)) {
            return Err(ApiError::status(503, "server_shutting_down", "image preparation is shutting down"));
        }
        if Instant::now() >= self.deadline {
            return Err(timeout_error());
        }
        if self.cancelled.load(Ordering::Acquire) {
            return Err(ApiError::status(499, "client_closed_request", "request cancelled"));
        }
        Ok(())
    }
    pub fn remaining(&self) -> Duration {
        self.deadline.saturating_duration_since(Instant::now())
    }
}
/// Request-future cancellation is advisory: actual TCP behavior is tested, never assumed.
pub struct CancelOnDrop(pub RequestWork);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}
pub fn timeout_error() -> ApiError {
    ApiError::status(504, "request_timeout", "request preparation deadline exceeded")
}
fn overload() -> ApiError {
    ApiError::status(503, "server_overloaded", "image preparation capacity exhausted").retry_after(5)
}
#[derive(Debug)]
pub struct PreparationGate {
    pub limits: PreparationLimits,
    stopping: Arc<AtomicBool>,
    memory: Arc<Semaphore>,
    retained: Arc<Semaphore>,
    workers: Arc<Semaphore>,
    backlog: Arc<Semaphore>,
    active: Arc<AtomicU64>,
    pub wait_count: AtomicU64,
    pub wait_max_micros: AtomicU64,
    pub rejected: AtomicU64,
    pub processed: AtomicU64,
    pub cancelled: AtomicU64,
    pub timed_out: AtomicU64,
}
/// Fields drop CPU/retained credits before a new job can enter the backlog.
pub struct PreparationPermit {
    memory: OwnedSemaphorePermit,
    _retained: OwnedSemaphorePermit,
    _workers: OwnedSemaphorePermit,
    _backlog: OwnedSemaphorePermit,
    active: Arc<AtomicU64>,
}
/// Input owners precede the permit, so captured URLs/bodies are destroyed before returning retained-byte credit.
pub struct PreparationJob<T> {
    inputs: T,
    _permit: PreparationPermit,
}
impl<T> PreparationJob<T> {
    pub fn new(inputs: T, permit: PreparationPermit) -> Self {
        Self { inputs, _permit: permit }
    }
    pub fn inputs(&self) -> &T {
        &self.inputs
    }
}
impl Drop for PreparationPermit {
    fn drop(&mut self) {
        self.active.fetch_sub(self.memory.num_permits() as u64, Ordering::AcqRel);
    }
}
impl PreparationGate {
    pub fn new(limits: PreparationLimits) -> Result<Arc<Self>, String> {
        if limits.bytes == 0
            || limits.retained_bytes == 0
            || limits.workers == 0
            || limits.backlog == 0
            || limits.bytes > u32::MAX as usize
            || limits.retained_bytes > u32::MAX as usize
            || limits.bytes > Semaphore::MAX_PERMITS
            || limits.retained_bytes > Semaphore::MAX_PERMITS
            || limits.wait.is_zero()
            || limits.deadline.is_zero()
            || limits.deadline > Duration::from_secs(86400)
            || limits.workers > Semaphore::MAX_PERMITS
            || limits.backlog > Semaphore::MAX_PERMITS
        {
            return Err("invalid preparation capacity".into());
        }
        Ok(Arc::new(Self {
            limits,
            stopping: Arc::new(AtomicBool::new(false)),
            memory: Arc::new(Semaphore::new(limits.bytes)),
            retained: Arc::new(Semaphore::new(limits.retained_bytes)),
            workers: Arc::new(Semaphore::new(limits.workers)),
            backlog: Arc::new(Semaphore::new(limits.backlog)),
            active: Arc::new(AtomicU64::new(0)),
            wait_count: AtomicU64::new(0),
            wait_max_micros: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
            processed: AtomicU64::new(0),
            cancelled: AtomicU64::new(0),
            timed_out: AtomicU64::new(0),
        }))
    }
    pub fn shutdown_token(&self) -> Arc<AtomicBool> {
        self.stopping.clone()
    }
    pub fn shutdown(&self) {
        self.stopping.store(true, Ordering::Release);
        self.workers.close();
        self.memory.close();
        self.retained.close();
        self.backlog.close();
    }
    pub async fn wait_idle(&self) {
        while self.jobs() != 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }
    pub fn active_bytes(&self) -> u64 {
        self.active.load(Ordering::Acquire)
    }
    pub fn retained_bytes(&self) -> usize {
        self.limits.retained_bytes - self.retained.available_permits()
    }
    pub fn jobs(&self) -> usize {
        self.limits.backlog - self.backlog.available_permits()
    }
    pub async fn acquire(&self, bytes: usize, retained: usize, work: &RequestWork) -> Result<PreparationPermit, ApiError> {
        work.check()?;
        let attempt = async {
            if bytes > self.limits.bytes || retained > self.limits.retained_bytes {
                return Err(overload());
            }
            let backlog = self.backlog.clone().try_acquire_owned().map_err(|_| overload())?;
            // Charge all waiting retained input BEFORE entering an asynchronous wait. No partial/min credit.
            let retained = self.retained.clone().try_acquire_many_owned(retained as u32).map_err(|_| overload())?;
            let workers = self.workers.clone().acquire_owned().await.map_err(|_| overload())?;
            let memory = self.memory.clone().acquire_many_owned(bytes as u32).await.map_err(|_| overload())?;
            work.check()?;
            self.active.fetch_add(bytes as u64, Ordering::AcqRel);
            Ok(PreparationPermit {
                memory,
                _retained: retained,
                _workers: workers,
                _backlog: backlog,
                active: self.active.clone(),
            })
        };
        let t0 = Instant::now();
        self.wait_count.fetch_add(1, Ordering::Relaxed);
        let result = tokio::time::timeout(self.limits.wait.min(work.remaining()), attempt).await.unwrap_or_else(|_| {
            if work.remaining().is_zero() {
                Err(timeout_error())
            } else {
                Err(overload())
            }
        });
        self.wait_max_micros.fetch_max(t0.elapsed().as_micros().min(u64::MAX as u128) as u64, Ordering::Relaxed);
        if result.is_err() {
            self.rejected.fetch_add(1, Ordering::Relaxed);
        }
        result
    }
}
/// Conservative decoder/metadata/resample reserve plus family-token-sized patch/float/output space.
/// One image is processed at a time per job; retained encoded images are charged separately.
pub fn image_work_bytes(tokens: usize) -> Result<usize, ApiError> {
    tokens
        .checked_mul(32 << 10)
        .and_then(|n| n.checked_add(1 << 30))
        .ok_or_else(|| ApiError::bad("image preparation size overflow"))
}
