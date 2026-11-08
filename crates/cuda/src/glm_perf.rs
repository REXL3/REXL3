//! Small kernels of the GLM decode-performance work (`csrc/glm_perf.cu`) that are not part of a model's arithmetic.

use crate::rt::{CudaError, DevBuf, Stream};

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_map_ids(ids: *mut i32, map: *const i32, n: i32, cols: i32, stream: *mut c_void) -> i32;
    }
}

/// `ids[i] = map[ids[i]]` for the first `n` entries (i32 `[n]` and `[cols]`): the column a restricted head chose becomes its token id. An index outside
/// `0..cols` is left alone.
pub fn map_ids(ids: &mut DevBuf, map: &DevBuf, n: usize, cols: usize, st: &Stream) -> Result<(), CudaError> {
    if ids.bytes() < n * 4 || map.bytes() < cols * 4 || i32::try_from(n).is_err() || i32::try_from(cols).is_err() {
        return Err(CudaError(1));
    }
    // SAFETY: the extents were checked above; the launch is asynchronous on `st`
    CudaError::check(unsafe { ffi::gb_map_ids(ids.ptr() as *mut i32, map.ptr() as *const i32, n as i32, cols as i32, st.raw()) })
}
