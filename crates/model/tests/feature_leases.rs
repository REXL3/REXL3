use gb_model::serve::{feature_ref, FeatureBudget, LeasedFeatures};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
struct Payload(Arc<AtomicBool>, Arc<FeatureBudget>);
impl Drop for Payload {
    fn drop(&mut self) {
        assert_eq!(self.1.free(), 0, "feature allocation destructor runs while its lease is still held");
        self.0.store(true, Ordering::SeqCst);
    }
}
#[test]
fn lease_follows_last_feature_owner_and_errors_return_credit() {
    let b = FeatureBudget::new(100);
    let dropped = Arc::new(AtomicBool::new(false));
    let f: Arc<dyn std::any::Any + Send + Sync> =
        Arc::new(LeasedFeatures::new(Payload(dropped.clone(), b.clone()), b.acquire(100).unwrap()));
    assert!(feature_ref::<Payload>(&*f).is_some());
    let queued = f.clone();
    drop(f);
    assert_eq!(b.free(), 0);
    assert!(b.acquire(1).is_none());
    drop(queued);
    assert!(dropped.load(Ordering::SeqCst));
    assert_eq!(b.free(), 100);
    let p = b.acquire(100).unwrap();
    drop(p); // encode error before publishing
    assert_eq!(b.free(), 100);
    assert!(b.acquire(u64::MAX).is_none());
    assert_eq!(b.free(), 100);
}
#[test]
fn legacy_unleased_feature_api_is_preserved() {
    assert_eq!(feature_ref::<u64>(&42u64), Some(&42));
    let old = gb_model::feature_budget::FeatureBudget::new(12);
    let _: Arc<FeatureBudget> = old;
}

#[test]
fn service_rechecks_cancellation_before_encoder_launch_and_keeps_legacy_encode() {
    use gb_model::error::ModelError;
    use gb_model::serve::{EncodedImage, ImageEncoder, VisionService};
    use std::sync::atomic::AtomicUsize;
    struct Mock(Arc<AtomicUsize>);
    impl ImageEncoder for Mock {
        fn encode(&mut self, _: &gb_vision::Preprocessed) -> Result<EncodedImage, ModelError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(EncodedImage { tokens: 1, merged: (1, 1), features: Arc::new(()) })
        }
    }
    let calls = Arc::new(AtomicUsize::new(0));
    let service = VisionService::new(Box::new(Mock(calls.clone())), 1);
    let pre = gb_vision::Preprocessed { size: (32, 32), grid: (1, 2, 2), patches: vec![], pixel_hash: 1, patch_dim: 1 };
    assert!(service.encode_checked(&pre, || Err(ModelError::Request("cancelled before launch".into()))).is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(service.encode(&pre).unwrap().tokens, 1);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
