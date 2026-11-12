//! The core's device context (stream, cuBLAS, cuSOLVER, the tile encoder) and its row-major fp32 matrices, with
//! wrappers for the kernels in `csrc/quant_core.cu`.

use crate::exl3_trellis::{EncodeError, Encoder};
use crate::gpu::cublas::{self, MathMode, Op};
use crate::gpu::{cusolver, GpuError};
use gb_cuda::rt::{DevBuf, Stream};
use std::fmt;

/// cuBLAS workspace bytes (the size PyTorch hands cuBLAS on this device; GEMM kernel selection depends on it).
pub const CUBLAS_WORKSPACE_BYTES: usize = 32 << 20;

#[derive(Debug)]
pub enum QuantError {
    Gpu(GpuError),
    Encode(EncodeError),
    Shape(String),
    Cholesky(String),
}

impl fmt::Display for QuantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantError::Gpu(e) => write!(f, "{e}"),
            QuantError::Encode(e) => write!(f, "encode: {e:?}"),
            QuantError::Shape(s) => write!(f, "shape: {s}"),
            QuantError::Cholesky(s) => write!(f, "cholesky: {s}"),
        }
    }
}

impl std::error::Error for QuantError {}

impl From<GpuError> for QuantError {
    fn from(e: GpuError) -> Self {
        QuantError::Gpu(e)
    }
}
impl From<gb_cuda::rt::CudaError> for QuantError {
    fn from(e: gb_cuda::rt::CudaError) -> Self {
        QuantError::Gpu(e.into())
    }
}
impl From<EncodeError> for QuantError {
    fn from(e: EncodeError) -> Self {
        QuantError::Encode(e)
    }
}

pub type Result<T> = std::result::Result<T, QuantError>;

mod ffi {
    use core::ffi::c_void;
    crate::gpu::foreign! { 100;
        pub fn rq_scale(x: *mut f32, rows: i64, cols: i64, ld: i64, v: *const f32, mode: i32, stream: *mut c_void) -> i32;
        pub fn rq_binary(
            out: *mut f32, ldo: i64, a: *const f32, lda: i64, b: *const f32, ldb: i64, rows: i64, cols: i64, op: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn rq_scalar(out: *mut f32, x: *const f32, n: i64, a: f32, op: i32, stream: *mut c_void) -> i32;
        pub fn rq_copy2d(dst: *mut f32, ldd: i64, src: *const f32, lds: i64, rows: i64, cols: i64, stream: *mut c_void) -> i32;
        pub fn rq_transpose(dst: *mut f32, src: *const f32, rows: i64, cols: i64, stream: *mut c_void) -> i32;
        pub fn rq_tril_from_colmajor(dst: *mut f32, src: *const f32, n: i64, stream: *mut c_void) -> i32;
        pub fn rq_diag(x: *mut f32, n: i64, ld: i64, v: f32, op: i32, stream: *mut c_void) -> i32;
        pub fn rq_zero_diag_blocks(x: *mut f32, n: i64, block: i32, stream: *mut c_void) -> i32;
        pub fn rq_diag_blocks_colmajor(out: *mut f32, src: *const f32, n: i64, stream: *mut c_void) -> i32;
        pub fn rq_identity_blocks(out: *mut f32, blocks: i64, stream: *mut c_void) -> i32;
        pub fn rq_rows_to_tiles(tiles: *mut f32, rows: *const f32, ld: i64, n_tiles: i64, stream: *mut c_void) -> i32;
        pub fn rq_tiles_to_rows(rows: *mut f32, ld: i64, tiles: *const f32, n_tiles: i64, stream: *mut c_void) -> i32;
        pub fn rq_gather_tiles(
            out: *mut f32, w: *const f32, n: i64, ta: *const i32, tc: *const i32, count: i64, stream: *mut c_void,
        ) -> i32;
        pub fn rq_gather_rows(dst: *mut f32, src: *const f32, idx: *const i32, rows: i64, cols: i64, stream: *mut c_void) -> i32;
        pub fn rq_gather_sym(dst: *mut f32, src: *const f32, idx: *const i32, n: i64, stream: *mut c_void) -> i32;
        pub fn rq_gather_blocks_i16(
            dst: *mut i16, src: *const i16, idx: *const i32, blocks: i64, width: i64, stream: *mut c_void,
        ) -> i32;
        pub fn rq_scaled_copies(out: *mut f32, x: *const f32, a: *const f32, n: i64, sets: i64, stream: *mut c_void) -> i32;
        pub fn rq_reduce(
            out: *mut f64, a: *const f32, b: *const f32, outputs: i64, len: i64, step: i64, ostride: i64, kind: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn rq_reduce_all(out: *mut f64, part: *mut f64, a: *const f32, b: *const f32, n: i64, kind: i32, stream: *mut c_void) -> i32;
        pub fn rq_tile_sumsq(out: *mut f64, w: *const f32, k: i64, n: i64, stream: *mut c_void) -> i32;
        pub fn rq_gs_err(
            out: *mut f64, dec: *const f32, t: *const f32, inv_s: *const f32, sets: i64, len: i64, stream: *mut c_void,
        ) -> i32;
    }
}

fn ck(rc: i32) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else if cfg!(gb_quant_no_cuda) {
        Err(GpuError::NoCuda.into())
    } else {
        Err(GpuError::Cuda(rc).into())
    }
}

/// The device context of the core: one stream, the cuBLAS and cuSOLVER handles bound to it, and the tile encoder.
pub struct Gpu {
    pub stream: Stream,
    pub blas: cublas::Handle,
    pub solver: cusolver::Handle,
    pub encoder: Encoder,
    pub had: Mat,
    _workspace: DevBuf,
}

impl Gpu {
    pub fn new(mode: MathMode) -> Result<Gpu> {
        let stream = Stream::new()?;
        let blas = cublas::Handle::new(&stream, mode)?;
        let workspace = DevBuf::alloc(CUBLAS_WORKSPACE_BYTES)?;
        unsafe { blas.set_workspace(workspace.ptr(), CUBLAS_WORKSPACE_BYTES)? };
        let solver = cusolver::Handle::new(&stream)?;
        let encoder = Encoder::new()?;
        let had = Mat::from_host(&super::hadamard::matrix(), 128, 128)?;
        Ok(Gpu { stream, blas, solver, encoder, had, _workspace: workspace })
    }

    pub fn s(&self) -> *mut core::ffi::c_void {
        self.stream.raw()
    }

    pub fn sync(&self) -> Result<()> {
        Ok(self.stream.sync()?)
    }

    /// Row-major C (m x n) = alpha A (m x k) B (k x n) + beta C, as PyTorch issues it to column-major cuBLAS:
    /// `sgemm(N, N, n, m, k, B, A, C)`.
    ///
    /// # Safety
    /// Device pointers with the given leading dimensions.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sgemm(
        &self,
        ta: Op,
        tb: Op,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: *const f32,
        lda: usize,
        b: *const f32,
        ldb: usize,
        beta: f32,
        c: *mut f32,
        ldc: usize,
    ) -> Result<()> {
        if k == 0 {
            return Ok(());
        }
        unsafe {
            self.blas.sgemm(ta, tb, m as i32, n as i32, k as i32, alpha, a, lda as i32, b, ldb as i32, beta, c, ldc as i32)?
        };
        Ok(())
    }
}

/// A row-major fp32 matrix on the device.
#[derive(Debug)]
pub struct Mat {
    pub buf: DevBuf,
    pub rows: usize,
    pub cols: usize,
}

impl Mat {
    pub fn alloc(rows: usize, cols: usize) -> Result<Mat> {
        Ok(Mat { buf: DevBuf::alloc(rows * cols * 4)?, rows, cols })
    }
    pub fn zeros(rows: usize, cols: usize) -> Result<Mat> {
        Ok(Mat { buf: DevBuf::zeroed(rows * cols * 4)?, rows, cols })
    }
    pub fn from_host(v: &[f32], rows: usize, cols: usize) -> Result<Mat> {
        if v.len() != rows * cols {
            return Err(QuantError::Shape(format!("{} values for a {rows} x {cols} matrix", v.len())));
        }
        Ok(Mat { buf: DevBuf::from_slice(v)?, rows, cols })
    }
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn ptr(&self) -> *mut f32 {
        self.buf.ptr() as *mut f32
    }
    /// The address of element (i, j).
    pub fn at(&self, i: usize, j: usize) -> *mut f32 {
        unsafe { self.ptr().add(i * self.cols + j) }
    }
    pub fn to_host(&self, g: &Gpu) -> Result<Vec<f32>> {
        g.sync()?;
        let mut v = vec![0f32; self.len()];
        self.buf.download(&mut v)?;
        Ok(v)
    }
    pub fn clone_on(&self, g: &Gpu) -> Result<Mat> {
        let mut m = Mat::alloc(self.rows, self.cols)?;
        m.buf.copy_from(&self.buf, self.len() * 4, &g.stream)?;
        Ok(m)
    }
}

/// A device copy of host values (indices, scales).
pub fn upload<T: Copy>(v: &[T]) -> Result<DevBuf> {
    Ok(DevBuf::from_slice(v)?)
}

pub fn download<T: Copy + Default>(g: &Gpu, b: &DevBuf, n: usize) -> Result<Vec<T>> {
    g.sync()?;
    let mut v = vec![T::default(); n];
    b.download_range(0, &mut v)?;
    Ok(v)
}

/// How [`scale`] applies the vector.
#[derive(Clone, Copy)]
pub enum Scale {
    MulCols = 0,
    MulRows = 1,
    DivCols = 2,
    DivRows = 3,
}

/// x[i, j] (op) v[j] or v[i] over a (rows x cols) block with leading dimension ld; `v` is a device vector.
pub(crate) fn scale(g: &Gpu, x: *mut f32, rows: usize, cols: usize, ld: usize, v: &DevBuf, mode: Scale) -> Result<()> {
    ck(unsafe { ffi::rq_scale(x, rows as i64, cols as i64, ld as i64, v.ptr() as *const f32, mode as i32, g.s()) })
}

pub fn scale_mat(g: &Gpu, x: &mut Mat, v: &[f32], mode: Scale) -> Result<()> {
    let d = upload(v)?;
    scale(g, x.ptr(), x.rows, x.cols, x.cols, &d, mode)?;
    g.sync()
}

#[derive(Clone, Copy)]
pub enum Bin {
    Sub = 0,
    Add = 1,
    Mul = 2,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn binary(
    g: &Gpu,
    out: *mut f32,
    ldo: usize,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    rows: usize,
    cols: usize,
    op: Bin,
) -> Result<()> {
    ck(unsafe { ffi::rq_binary(out, ldo as i64, a, lda as i64, b, ldb as i64, rows as i64, cols as i64, op as i32, g.s()) })
}

/// out = x * a (in place when out == x).
pub(crate) fn mul_scalar(g: &Gpu, out: *mut f32, x: *const f32, n: usize, a: f32) -> Result<()> {
    ck(unsafe { ffi::rq_scalar(out, x, n as i64, a, 0, g.s()) })
}

pub(crate) fn copy2d(g: &Gpu, dst: *mut f32, ldd: usize, src: *const f32, lds: usize, rows: usize, cols: usize) -> Result<()> {
    ck(unsafe { ffi::rq_copy2d(dst, ldd as i64, src, lds as i64, rows as i64, cols as i64, g.s()) })
}

/// The transpose of x (its column-major copy).
pub fn transpose(g: &Gpu, x: &Mat) -> Result<Mat> {
    let out = Mat::alloc(x.cols, x.rows)?;
    ck(unsafe { ffi::rq_transpose(out.ptr(), x.ptr(), x.rows as i64, x.cols as i64, g.s()) })?;
    Ok(out)
}

pub fn tril_from_colmajor(g: &Gpu, cm: &Mat) -> Result<Mat> {
    let out = Mat::alloc(cm.rows, cm.cols)?;
    ck(unsafe { ffi::rq_tril_from_colmajor(out.ptr(), cm.ptr(), cm.rows as i64, g.s()) })?;
    Ok(out)
}

#[derive(Clone, Copy)]
pub enum Diag {
    Add = 0,
    Sub = 1,
}

pub fn diag(g: &Gpu, x: &mut Mat, v: f32, op: Diag) -> Result<()> {
    ck(unsafe { ffi::rq_diag(x.ptr(), x.rows as i64, x.cols as i64, v, op as i32, g.s()) })
}

pub fn zero_diag_blocks(g: &Gpu, x: &mut Mat, block: usize) -> Result<()> {
    ck(unsafe { ffi::rq_zero_diag_blocks(x.ptr(), x.rows as i64, block as i32, g.s()) })
}

/// The 16 x 16 diagonal blocks of x, each stored column-major (blocks x 256).
pub fn diag_blocks_colmajor(g: &Gpu, x: &Mat) -> Result<Mat> {
    let out = Mat::alloc(x.rows / 16, 256)?;
    ck(unsafe { ffi::rq_diag_blocks_colmajor(out.ptr(), x.ptr(), x.rows as i64, g.s()) })?;
    Ok(out)
}

pub fn identity_blocks(g: &Gpu, blocks: usize) -> Result<Mat> {
    let out = Mat::alloc(blocks, 256)?;
    ck(unsafe { ffi::rq_identity_blocks(out.ptr(), blocks as i64, g.s()) })?;
    Ok(out)
}

pub(crate) fn rows_to_tiles(g: &Gpu, tiles: &DevBuf, rows: *const f32, ld: usize, n_tiles: usize) -> Result<()> {
    ck(unsafe { ffi::rq_rows_to_tiles(tiles.ptr() as *mut f32, rows, ld as i64, n_tiles as i64, g.s()) })
}

pub(crate) fn tiles_to_rows(g: &Gpu, rows: *mut f32, ld: usize, tiles: &DevBuf, n_tiles: usize) -> Result<()> {
    ck(unsafe { ffi::rq_tiles_to_rows(rows, ld as i64, tiles.ptr() as *const f32, n_tiles as i64, g.s()) })
}

pub fn gather_tiles(g: &Gpu, out: &DevBuf, w: &Mat, ta: &DevBuf, tc: &DevBuf, count: usize) -> Result<()> {
    ck(unsafe {
        ffi::rq_gather_tiles(
            out.ptr() as *mut f32,
            w.ptr(),
            w.cols as i64,
            ta.ptr() as *const i32,
            tc.ptr() as *const i32,
            count as i64,
            g.s(),
        )
    })
}

pub fn gather_rows(g: &Gpu, src: &Mat, idx: &DevBuf) -> Result<Mat> {
    let out = Mat::alloc(src.rows, src.cols)?;
    ck(unsafe { ffi::rq_gather_rows(out.ptr(), src.ptr(), idx.ptr() as *const i32, src.rows as i64, src.cols as i64, g.s()) })?;
    Ok(out)
}

/// dst[i, :] = src[idx[i], :] for `rows` rows of `cols` values (a new buffer).
pub(crate) fn gather_rows_raw(g: &Gpu, src: *const f32, idx: &DevBuf, rows: usize, cols: usize) -> Result<DevBuf> {
    let out = DevBuf::alloc(rows.max(1) * cols * 4)?;
    ck(unsafe { ffi::rq_gather_rows(out.ptr() as *mut f32, src, idx.ptr() as *const i32, rows as i64, cols as i64, g.s()) })?;
    Ok(out)
}

pub fn gather_sym(g: &Gpu, src: &Mat, idx: &DevBuf) -> Result<Mat> {
    let out = Mat::alloc(src.rows, src.cols)?;
    ck(unsafe { ffi::rq_gather_sym(out.ptr(), src.ptr(), idx.ptr() as *const i32, src.rows as i64, g.s()) })?;
    Ok(out)
}

pub fn gather_blocks_i16(g: &Gpu, dst: &DevBuf, src: &DevBuf, idx: &DevBuf, blocks: usize, width: usize) -> Result<()> {
    ck(unsafe {
        ffi::rq_gather_blocks_i16(
            dst.ptr() as *mut i16,
            src.ptr() as *const i16,
            idx.ptr() as *const i32,
            blocks as i64,
            width as i64,
            g.s(),
        )
    })
}

pub fn scaled_copies(g: &Gpu, out: &DevBuf, x: &DevBuf, a: &DevBuf, n: usize, sets: usize) -> Result<()> {
    ck(unsafe {
        ffi::rq_scaled_copies(out.ptr() as *mut f32, x.ptr() as *const f32, a.ptr() as *const f32, n as i64, sets as i64, g.s())
    })
}

/// What a reduction sums (in fp64).
#[derive(Clone, Copy)]
pub enum Red {
    /// x^2, squared in fp64
    SumSq = 0,
    /// a * b, the product rounded to fp32 first
    Dot = 1,
    Sum = 2,
}

/// `outputs` fp64 sums, output o over `len` elements at `o * ostride + i * step` (to the host).
#[allow(clippy::too_many_arguments)]
pub(crate) fn reduce(
    g: &Gpu,
    a: *const f32,
    b: *const f32,
    outputs: usize,
    len: usize,
    step: usize,
    ostride: usize,
    kind: Red,
) -> Result<Vec<f64>> {
    let out = DevBuf::alloc(outputs.max(1) * 8)?;
    ck(unsafe {
        ffi::rq_reduce(out.ptr() as *mut f64, a, b, outputs as i64, len as i64, step as i64, ostride as i64, kind as i32, g.s())
    })?;
    download(g, &out, outputs)
}

/// One fp64 sum over n contiguous values.
pub(crate) fn reduce_all(g: &Gpu, a: *const f32, b: *const f32, n: usize, kind: Red) -> Result<f64> {
    let parts = n.div_ceil(1 << 16).max(1);
    let part = DevBuf::alloc(parts * 8)?;
    let out = DevBuf::alloc(8)?;
    ck(unsafe { ffi::rq_reduce_all(out.ptr() as *mut f64, part.ptr() as *mut f64, a, b, n as i64, kind as i32, g.s()) })?;
    Ok(download::<f64>(g, &out, 1)?[0])
}

pub fn tile_sumsq(g: &Gpu, w: &Mat) -> Result<Vec<f64>> {
    let tiles = (w.rows / 16) * (w.cols / 16);
    let out = DevBuf::alloc(tiles.max(1) * 8)?;
    ck(unsafe { ffi::rq_tile_sumsq(out.ptr() as *mut f64, w.ptr(), w.rows as i64, w.cols as i64, g.s()) })?;
    download(g, &out, tiles)
}

pub fn gs_err(g: &Gpu, dec: &DevBuf, t: &DevBuf, inv_s: &DevBuf, sets: usize, len: usize) -> Result<Vec<f64>> {
    let out = DevBuf::alloc(sets.max(1) * 8)?;
    ck(unsafe {
        ffi::rq_gs_err(
            out.ptr() as *mut f64,
            dec.ptr() as *const f32,
            t.ptr() as *const f32,
            inv_s.ptr() as *const f32,
            sets as i64,
            len as i64,
            g.s(),
        )
    })?;
    download(g, &out, sets)
}

/// f32(sum / n): the mean of an fp64 sum, rounded once.
pub fn mean32(sum: f64, n: usize) -> f32 {
    (sum / n as f64) as f32
}
