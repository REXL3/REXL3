//! CUDA graphs of the decode path: capture a sequence of launches on a stream once, replay it for new inputs.
//!
//! A graph bakes in every kernel argument that is not read from device memory: pointers, row counts, grid sizes. What a replay may change
//! is whatever the kernels READ from device buffers the capture named (token ids, positions, slots, page tables, keep counts), so a caller
//! keeps its per-call inputs in fixed device buffers, writes them before the replay, and keys its graphs by every host-side scalar that
//! reaches a launcher (see `GlmModel::graph_key` for the model's list).
//!
//! **Capture is local and never fatal.** [`Graph::capture`] runs the caller's launches on a stream in thread-local capture mode (an illegal
//! call from this thread, such as a synchronous copy, an allocation or a stream sync, fails the capture instead of racing it), always ends the
//! capture, destroys what was captured on failure and clears the thread's last-error state, and returns a [`CaptureError`]: the launches ran
//! nowhere, the device is as it was, and the caller runs the same work eagerly. [`CudaError::is_fatal`] treats every code it does not know as
//! fatal, which is right for a launch or a sync (a sticky error) and wrong for the capture's own refusals (the 900-series codes), so a capture
//! failure is classified here: [`CaptureError::fatal`] is `Some` only for a code that really is sticky (a device fault seen while capturing).

use crate::rt::{CudaError, Event, Stream};
use core::ffi::c_void;

#[cfg(not(gb_no_cuda))]
mod ffi {
    use core::ffi::c_void;
    extern "C" {
        pub fn cudaStreamBeginCapture(s: *mut c_void, mode: i32) -> i32;
        pub fn cudaStreamEndCapture(s: *mut c_void, g: *mut *mut c_void) -> i32;
        pub fn cudaGraphInstantiate(exec: *mut *mut c_void, g: *mut c_void, flags: u64) -> i32;
        pub fn cudaGraphLaunch(exec: *mut c_void, s: *mut c_void) -> i32;
        pub fn cudaGraphExecDestroy(exec: *mut c_void) -> i32;
        pub fn cudaGraphDestroy(g: *mut c_void) -> i32;
        pub fn cudaGetLastError() -> i32;
        pub fn cudaStreamWaitEvent(s: *mut c_void, e: *mut c_void, flags: u32) -> i32;
    }
}

/// `cudaStreamCaptureModeThreadLocal`: unsafe calls made by THIS thread while a capture is open are refused; other threads are not affected.
#[cfg(not(gb_no_cuda))]
const CAPTURE_THREAD_LOCAL: i32 = 1;

/// The codes of `driver_types.h` that only spoil a capture (or the launch of a graph), as opposed to the device: the stream-capture family
/// (900-909), the graph exec update refusal (910), and the argument errors a launcher returns for one call.
pub fn is_capture_code(code: i32) -> bool {
    matches!(code, 1 | 9 | 801 | 900..=910)
}

/// A code that leaves the process unusable even though it was seen while capturing (a sticky device fault).
pub fn is_sticky(code: i32) -> bool {
    !is_capture_code(code) && CudaError(code).is_fatal()
}

/// Why a capture produced no graph. The launches ran nowhere and nothing is left behind; the caller runs the same work eagerly.
#[derive(Debug)]
pub enum CaptureError<E> {
    /// The CUDA runtime refused to begin, end or instantiate the capture, or invalidated it (a call that cannot be captured was made on the
    /// stream or by this thread). The code is the runtime's.
    Refused(CudaError),
    /// The captured closure itself failed with its own error (a launcher's argument error, a model error); nothing was captured.
    Closure(E),
}

impl<E> CaptureError<E> {
    /// The sticky device fault behind a refusal, if that is what it was: the one case a caller must treat as fatal.
    pub fn fatal(&self) -> Option<&CudaError> {
        match self {
            CaptureError::Refused(e) if is_sticky(e.0) => Some(e),
            _ => None,
        }
    }
}

impl<E: std::fmt::Display> std::fmt::Display for CaptureError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CaptureError::Refused(e) => write!(f, "graph capture refused: {e}"),
            CaptureError::Closure(e) => write!(f, "graph capture: a launch failed: {e}"),
        }
    }
}

/// An instantiated graph, owned: destroyed when dropped. Replayable many times; one stream at a time.
pub struct Graph {
    exec: *mut c_void,
}

// An opaque runtime handle; the model that owns it drives one stream from one thread.
unsafe impl Send for Graph {}

/// An open capture on a stream (see [`Graph::begin`]). Ends the capture when dropped without being finished (a panic or an early return in the
/// caller), so a stream never stays in capture mode.
pub struct Capture {
    stream: *mut c_void,
    done: bool,
}

impl Graph {
    /// Start capturing what is enqueued on `stream` (thread-local mode). The caller enqueues asynchronous work on `stream` only (no synchronous
    /// copy, no allocation, no host wait, nothing read from the host that changes between replays) and then calls [`Capture::finish`] with the
    /// outcome of its launches. The work does not execute while it is captured; [`Graph::launch`] runs it.
    ///
    /// `stream` should be created with [`Stream::new_nonblocking`]: a blocking stream orders with the legacy default stream, which another
    /// thread's synchronous copy would then drag into the capture.
    pub fn begin(stream: &Stream) -> Result<Capture, CudaError> {
        #[cfg(not(gb_no_cuda))]
        unsafe {
            let rc = ffi::cudaStreamBeginCapture(stream.raw(), CAPTURE_THREAD_LOCAL);
            if rc != 0 {
                ffi::cudaGetLastError();
                return Err(CudaError(rc));
            }
            Ok(Capture { stream: stream.raw(), done: false })
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = stream;
            Err(CudaError(-1))
        }
    }

    /// Capture what `launches` enqueues on `stream` (see [`Graph::begin`]); `launches` runs once. On `Err` nothing ran and nothing is left behind.
    pub fn capture<E>(stream: &Stream, launches: impl FnOnce() -> Result<(), E>) -> Result<Graph, CaptureError<E>> {
        let cap = Graph::begin(stream).map_err(CaptureError::Refused)?;
        cap.finish(launches())
    }

    /// Replay on `stream` (asynchronous). The error is the launch's own: classify it like any launch (`CudaError::is_fatal`).
    pub fn launch(&self, stream: &Stream) -> Result<(), CudaError> {
        #[cfg(not(gb_no_cuda))]
        unsafe {
            CudaError::check(ffi::cudaGraphLaunch(self.exec, stream.raw()))
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = stream;
            Err(CudaError(-1))
        }
    }
}

impl Capture {
    /// End the capture and instantiate it. `ran` is what the captured launches returned: on `Err` (or when the runtime invalidated the capture)
    /// the capture is ended and destroyed, the thread's last-error state is cleared, and nothing is left behind.
    pub fn finish<E>(mut self, ran: Result<(), E>) -> Result<Graph, CaptureError<E>> {
        self.done = true;
        #[cfg(not(gb_no_cuda))]
        unsafe {
            // end the capture whatever happened, so the stream leaves capture mode
            let mut g: *mut c_void = core::ptr::null_mut();
            let end = ffi::cudaStreamEndCapture(self.stream, &mut g);
            let discard = |g: *mut c_void| {
                if !g.is_null() {
                    ffi::cudaGraphDestroy(g);
                }
                ffi::cudaGetLastError();
            };
            if let Err(e) = ran {
                discard(g);
                return Err(CaptureError::Closure(e));
            }
            if end != 0 || g.is_null() {
                discard(g);
                return Err(CaptureError::Refused(CudaError(if end != 0 { end } else { 901 })));
            }
            let mut exec: *mut c_void = core::ptr::null_mut();
            let inst = ffi::cudaGraphInstantiate(&mut exec, g, 0);
            ffi::cudaGraphDestroy(g);
            if inst != 0 || exec.is_null() {
                ffi::cudaGetLastError();
                return Err(CaptureError::Refused(CudaError(if inst != 0 { inst } else { 1 })));
            }
            Ok(Graph { exec })
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = ran;
            Err(CaptureError::Refused(CudaError(-1)))
        }
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        #[cfg(not(gb_no_cuda))]
        if !self.done {
            unsafe {
                let mut g: *mut c_void = core::ptr::null_mut();
                ffi::cudaStreamEndCapture(self.stream, &mut g);
                if !g.is_null() {
                    ffi::cudaGraphDestroy(g);
                }
                ffi::cudaGetLastError();
            }
        }
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        #[cfg(not(gb_no_cuda))]
        unsafe {
            ffi::cudaGraphExecDestroy(self.exec);
        }
    }
}

mod spin_ffi {
    crate::launchers! {
        pub fn gb_spin(cycles: i64, stream: *mut core::ffi::c_void) -> i32;
    }
}

/// Keep `stream` busy for `cycles` clock cycles (one thread; for the mutation tests of the stream joins: a side branch made slow on purpose).
pub fn spin(stream: &Stream, cycles: i64) -> Result<(), CudaError> {
    CudaError::check(unsafe { spin_ffi::gb_spin(cycles, stream.raw()) })
}

/// Fork a captured stream: make `side` part of the capture that is open on `main`, ordered after everything `main` has recorded so far. `event`
/// is recorded on `main` and waited for by `side`. Outside a capture this is the ordinary event dependency.
pub fn fork(main: &Stream, side: &Stream, event: &Event) -> Result<(), CudaError> {
    event.record(main)?;
    wait(side, event)
}

/// Join `side` back into `main`: later work on `main` runs after everything `side` has been given (every captured side stream MUST be joined
/// before the capture ends, or the capture fails with an unjoined-origin error).
pub fn join(main: &Stream, side: &Stream, event: &Event) -> Result<(), CudaError> {
    event.record(side)?;
    wait(main, event)
}

pub fn wait(stream: &Stream, event: &Event) -> Result<(), CudaError> {
    #[cfg(not(gb_no_cuda))]
    unsafe {
        CudaError::check(ffi::cudaStreamWaitEvent(stream.raw(), event.raw(), 0))
    }
    #[cfg(gb_no_cuda)]
    {
        let _ = (stream, event);
        Err(CudaError(-1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_captures_own_refusals_are_not_fatal_and_a_device_fault_is() {
        // the 900-series: unsupported, invalidated, merge, unmatched, unjoined, isolation, implicit, wrong thread; and the exec update refusal
        for code in [900, 901, 902, 903, 904, 905, 906, 908, 910, 801, 1, 9] {
            assert!(is_capture_code(code), "{code}");
            assert!(!is_sticky(code), "{code} spoils the capture, not the engine");
            assert!(CaptureError::<String>::Refused(CudaError(code)).fatal().is_none());
        }
        // sticky device errors stay fatal
        for code in [700, 710, 714, 715, 719, 999] {
            assert!(!is_capture_code(code), "{code}");
            assert!(is_sticky(code), "{code}");
            assert!(CaptureError::<String>::Refused(CudaError(code)).fatal().is_some(), "{code} is the device's");
        }
        // the closure's own error is the caller's to classify
        assert!(CaptureError::Closure("a launcher failed").fatal().is_none());
    }
}
