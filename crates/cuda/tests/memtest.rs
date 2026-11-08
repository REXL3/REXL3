#[test]
fn mem_info_reports_something_sane() {
    if !gb_cuda::device_available() {
        return eprintln!("no CUDA device: skipping");
    }
    let (free, total) = gb_cuda::rt::mem_info().unwrap();
    eprintln!("cudaMemGetInfo: free {:.2} GiB of {:.2} GiB", free as f64 / 2f64.powi(30), total as f64 / 2f64.powi(30));
    assert!(total > 0 && free <= total);
}
