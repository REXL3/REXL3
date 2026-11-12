//! Load the mapped pack onto the device: allocate the plan's regions and copy every item, through its `Xform`. The embedding is
//! a sidecar, never a region: the kernels read it in place from a read-only mapping of its shard where the GPU reads pageable memory
//! ([`embed_in_place`]), and from a device copy otherwise.

use super::bind::LayoutError;
use super::config::BailingConfig;
use super::inventory::PackInventory;
use super::plan::LoadPlan;
use super::weights::{BailingWeights, Xform};
use crate::error::ModelError;
use crate::loadctx::LoadMode;
use gb_cuda::loader::{Loader, Options, Report, Xform as CopyXform};
use gb_cuda::mla_gated::pack_kv_b;
use gb_cuda::rt::DevBuf;
use gb_exl3::loadstats::{self, Component, Stage};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The loaded device image of a pack: one buffer per region, in plan order.
pub struct Regions {
    /// Region id -> device buffer.
    pub bufs: BTreeMap<u32, DevBuf>,
}

impl Regions {
    pub fn bytes(&self) -> u64 {
        self.bufs.values().map(|b| b.bytes() as u64).sum()
    }
    pub fn get(&self, id: u32) -> Option<&DevBuf> {
        self.bufs.get(&id)
    }
}

/// The embedding sidecar: the shard's `model.word_embeddings.weight` (bf16 `[vocab, hidden]`), never a region of the plan. The
/// step's rows come either from a read-only mapping of the shard that the host gathers from ([`Self::map_in_place`], [`gather_rows`],
/// where the GPU reads pageable memory, see [`embed_in_place`]) or from a device copy ([`Self::read_all_bf16`]) that a kernel gathers
/// from; rows are also read on demand by `row`.
pub struct EmbeddingSidecar {
    pub file: std::fs::File,
    /// The shard that holds the table (the file `file` is open on).
    pub path: PathBuf,
    /// Byte offset of the table in the shard.
    pub offset: u64,
    pub vocab: usize,
    pub hidden: usize,
}

/// Whether the embedding table is read in place from its shard rather than copied to device memory: the GPU must read pageable
/// memory, file mappings included, through the CPU's page tables (`cudaDevAttrPageableMemoryAccess`, true on the GB10). Probed once
/// per process, so the plan's post-allocation reserve (`engine::post_alloc_reserve_bytes`) and the allocation in `BailingModel::new`
/// always agree.
pub fn embed_in_place() -> bool {
    gb_cuda::rt::pageable_memory_access()
}

impl EmbeddingSidecar {
    /// The table of tensor `t`, which lives in shard `files[t.src.file]` of `dir` at byte offset `t.src.offset`.
    pub fn open(
        dir: &Path,
        files: &[String],
        t: &super::weights::PlainTensor,
        vocab: usize,
        hidden: usize,
    ) -> Result<Self, ModelError> {
        let shard = files
            .get(usize::from(t.src.file))
            .ok_or_else(|| ModelError::Config(format!("{}: shard {} of {}", t.name, t.src.file, files.len())))?;
        let path = dir.join(shard);
        let file = std::fs::File::open(&path).map_err(|e| ModelError::Config(format!("{}: {e}", path.display())))?;
        let s = Self { file, path, offset: t.src.offset, vocab, hidden };
        let len = s.file.metadata().map_err(|e| ModelError::Config(format!("{}: {e}", s.path.display())))?.len();
        if s.offset.checked_add(s.bytes() as u64).is_none_or(|end| end > len) {
            return Err(ModelError::Config(format!(
                "{}: the embedding table ({} bytes at {}) runs past the end of {} ({len} bytes)",
                t.name,
                s.bytes(),
                s.offset,
                s.path.display()
            )));
        }
        Ok(s)
    }

    /// Bytes of the table: `vocab * hidden` bf16.
    pub fn bytes(&self) -> usize {
        self.vocab * self.hidden * 2
    }

    /// The table as it is read in place: a read-only shared mapping of its shard, viewed as a buffer (the bytes are the file's,
    /// exactly the ones [`Self::read_all_bf16`] reads). Nothing is copied and no device memory is taken: the pages are the shard's
    /// page cache, which Linux counts as reclaimable, faulted in by the rows that are read. Only where [`embed_in_place`] holds.
    pub fn map_in_place(&self) -> Result<DevBuf, ModelError> {
        let block = gb_cuda::rt::Block::map_file_readonly(&self.path, self.offset, self.bytes())
            .map_err(|e| ModelError::Config(format!("mapping the embedding table of {} failed ({e})", self.path.display())))?;
        Ok(DevBuf::view(&std::sync::Arc::new(block), 0, self.bytes()))
    }

    /// Row `i` as bf16 u16 bits.
    pub fn row(&self, i: usize) -> Result<Vec<u16>, ModelError> {
        use std::io::{Read, Seek, SeekFrom};
        let mut file = self.file.try_clone().map_err(|e| ModelError::Config(format!("embedding clone: {e}")))?;
        file.seek(SeekFrom::Start(self.offset + i as u64 * self.hidden as u64 * 2))
            .map_err(|e| ModelError::Config(format!("embedding seek: {e}")))?;
        let mut out = vec![0u16; self.hidden];
        let n = out.len() * 2;
        let mut bytes = vec![0u8; n];
        file.read_exact(&mut bytes).map_err(|e| ModelError::Config(format!("embedding read: {e}")))?;
        for (o, c) in out.iter_mut().zip(bytes.chunks_exact(2)) {
            *o = u16::from_le_bytes([c[0], c[1]]);
        }
        Ok(out)
    }

    /// All `vocab * hidden` rows as bf16 u16 bits, in one sequential read (a ~1.2 GiB read for Ling-3.1).
    pub fn read_all_bf16(&self) -> Result<Vec<u16>, ModelError> {
        use std::io::{Read, Seek, SeekFrom};
        let mut file = self.file.try_clone().map_err(|e| ModelError::Config(format!("embedding clone: {e}")))?;
        file.seek(SeekFrom::Start(self.offset)).map_err(|e| ModelError::Config(format!("embedding seek: {e}")))?;
        let total = self.vocab * self.hidden;
        let mut out = vec![0u16; total];
        const CHUNK: usize = 32 << 20;
        let mut bytes = vec![0u8; CHUNK];
        let mut remain = total * 2;
        let mut at = 0usize;
        while remain > 0 {
            let take = remain.min(CHUNK);
            file.read_exact(&mut bytes[..take]).map_err(|e| ModelError::Config(format!("embedding read: {e}")))?;
            for i in 0..take / 2 {
                out[at + i] = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
            }
            at += take / 2;
            remain -= take;
        }
        Ok(out)
    }
}

/// Row gathers of at most this many rows are not announced to the kernel (`gather_rows`): their pages are read one by one.
const WILLNEED_MIN_ROWS: usize = 8;

/// Rows `ids` of a row-major table of `row_bytes`-byte rows that starts at `base`, written to `out` back to back (`ids.len() * row_bytes`
/// bytes); an id outside `0..vocab` gives a row of zeros (what the `gb_embed` kernel writes). A pure copy: every byte of `out` is a byte of
/// the table. A batch of more than [`WILLNEED_MIN_ROWS`] rows first asks the kernel (`MADV_WILLNEED`) to bring in the pages of all its rows
/// at once, so the reads of a cold shard overlap instead of being taken one fault at a time.
///
/// # Safety
/// `base` must be readable for `vocab * row_bytes` bytes for the duration of the call.
pub unsafe fn gather_rows(base: *const u8, vocab: usize, row_bytes: usize, ids: &[i32], out: &mut [u8]) {
    assert_eq!(out.len(), ids.len() * row_bytes, "the output holds one row per id");
    let row_of = |id: i32| (id >= 0 && (id as usize) < vocab).then(|| base.add(id as usize * row_bytes));
    #[cfg(target_os = "linux")]
    if ids.len() > WILLNEED_MIN_ROWS {
        let page = libc::sysconf(libc::_SC_PAGESIZE).max(4096) as usize;
        for &id in ids {
            if let Some(src) = row_of(id) {
                let start = src as usize / page * page;
                let end = (src as usize + row_bytes).div_ceil(page) * page;
                libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_WILLNEED);
            }
        }
    }
    for (dst, &id) in out.chunks_exact_mut(row_bytes).zip(ids) {
        match row_of(id) {
            Some(src) => std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), row_bytes),
            None => dst.fill(0),
        }
    }
}

/// What a load does about threads: the pipelined loader's options (workers, staging slots, span size, `O_DIRECT` mode).
#[derive(Clone, Debug)]
pub struct LoadOptions {
    pub loader: Options,
}

impl LoadOptions {
    /// The loader's options from the environment (`REXL3_LOAD_WORKERS`, ...); the default is 16 reader threads, as for GLM and DeepSeek:
    /// the pack is on one NVMe drive whose throughput grows with the number of threads reading it (1.1 GB/s at 1, 7.5 GB/s at 16:
    /// `docs/GLM_LOAD.md`).
    pub fn from_env() -> Self {
        let mut loader = Options::from_env();
        if std::env::var("REXL3_LOAD_WORKERS").is_err() {
            loader.workers = 16;
        }
        Self { loader }
    }
}

/// The bytes a plan item puts on the device for the `src` bytes it reads from its shard, under `x` (little endian, as the shard
/// stores them and the device wants them). Pure: the serial reference runs it on the calling thread, the pipelined loader on a
/// loader thread, the tests on synthetic bytes.
pub fn convert(x: Xform, src: &[u8]) -> Result<Vec<u8>, String> {
    match x {
        Xform::None => Ok(src.to_vec()),
        Xform::Bf16ToF32 => {
            if src.len() % 2 != 0 {
                return Err(format!("a bf16 tensor of {} bytes", src.len()));
            }
            // a bf16 value is the high half of the f32 with the same value
            Ok(src.chunks_exact(2).flat_map(|c| ((u16::from_le_bytes([c[0], c[1]]) as u32) << 16).to_le_bytes()).collect())
        }
        Xform::PackKvB => {
            if src.len() % 2 != 0 {
                return Err(format!("a kv_b_proj of {} bytes", src.len()));
            }
            // rows are the fp16 u16 bits (the pack stores fp16)
            let rows: Vec<u16> = src.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            // checkpoint kv_b_proj.weight is fp16; the MLA kernels run in bf16 (MlaGatedWeights.dtype = BF16)
            let (uk, uvt) = pack_kv_b(&rows, gb_cuda::exl3_linear::Dtype::F16, gb_cuda::exl3_linear::Dtype::BF16);
            Ok([&uk, &uvt].into_iter().flat_map(|v| v.iter().flat_map(|x| x.to_le_bytes())).collect())
        }
        // the fused `dst` receives the three parts item by item; the plan never holds this arm as a whole item
        Xform::CatQkvConv => Err("internal: CatQkvConv as a whole item".into()),
    }
}

/// `len` bytes of `path` from byte `off`.
fn read_range(path: &Path, off: u64, len: usize) -> std::io::Result<Vec<u8>> {
    use std::os::unix::fs::FileExt;
    let _t = loadstats::timer(Stage::Read, len as u64);
    let f = std::fs::File::open(path)?;
    let mut buf = vec![0u8; len];
    f.read_exact_at(&mut buf, off)?;
    Ok(buf)
}

/// Allocate the plan's device regions, zeroed (every byte a plan item does not write, alignment padding included, stays zero).
fn allocate(plan: &LoadPlan) -> Result<Regions, ModelError> {
    let mut bufs = BTreeMap::new();
    for r in plan.device_regions() {
        let b = DevBuf::zeroed_view(r.bytes as usize)
            .map_err(|e| ModelError::Config(format!("cudaMalloc {} ({} bytes): {e}", r.kind, r.bytes)))?;
        bufs.insert(r.id, b);
    }
    Ok(Regions { bufs })
}

fn region_view(regions: &Regions, item: &super::plan::LoadItem) -> Result<DevBuf, ModelError> {
    let b = regions
        .bufs
        .get(&item.region)
        .ok_or_else(|| ModelError::Config(format!("item {}: unknown region {}", item.name, item.region)))?;
    Ok(b.subview(item.dst as usize, item.dst_bytes as usize))
}

/// The serial reference (`REXL3_LOAD=serial`): each item is read whole from its shard, converted and uploaded on its own. The pipelined
/// load below is compared with it (`tests/ling31_load_pipeline.rs`), so the two always put the same bytes on the device.
fn load_serial(
    dir: &Path,
    plan: &LoadPlan,
    regions: &Regions,
    progress: &mut impl FnMut(usize, usize),
) -> Result<(), ModelError> {
    let total = plan.items.len();
    for (i, item) in plan.items.iter().enumerate() {
        let mut dst = region_view(regions, item)?;
        let src = read_range(&dir.join(&plan.files[item.file as usize]), item.offset, item.bytes as usize)
            .map_err(|e| ModelError::Config(format!("read {}: {e}", item.name)))?;
        let bytes = match item.xform {
            Xform::None => src,
            x => convert(x, &src).map_err(|e| ModelError::Config(format!("{}: {e}", item.name)))?,
        };
        dst.upload(&bytes).map_err(|e| ModelError::Config(format!("upload {}: {e}", item.name)))?;
        progress(i + 1, total);
    }
    Ok(())
}

/// The pipelined load: every item is declared to the loader (a plain copy straight from its file range, a conversion as bytes a loader
/// thread computes), which reads the shards with a few readers over large spans (`O_DIRECT` for the spans the page cache does not hold)
/// and copies each span to the device while the next is read. Returns when every copy has completed on the device.
fn load_pipelined(dir: &Path, plan: &LoadPlan, regions: &Regions, o: &Options) -> Result<Report, ModelError> {
    let mut loader = Loader::new();
    let files: Vec<usize> = plan.files.iter().map(|f| loader.file(&dir.join(f))).collect();
    let _scope = loadstats::scoped_component(Component::Target);
    for item in &plan.items {
        let dst = region_view(regions, item)?;
        match item.xform {
            Xform::None => {
                loader.copy_file(&dst, files[item.file as usize], item.offset, item.bytes as usize, CopyXform::None)?
            }
            x => {
                let path = dir.join(&plan.files[item.file as usize]);
                let (off, len, name) = (item.offset, item.bytes as usize, item.name.clone());
                loader.copy_compute(&dst, item.dst_bytes as usize, move || {
                    let src = read_range(&path, off, len)?;
                    let _t = loadstats::timer(Stage::Convert, len as u64);
                    convert(x, &src).map_err(|e| std::io::Error::other(format!("{name}: {e}")))
                })?
            }
        }
    }
    Ok(loader.run(o)?)
}

/// Load `plan` from `dir`: allocate the device regions and copy every item. `REXL3_LOAD=serial` selects the serial reference, anything
/// else the pipelined loader (whose [`Report`] is returned). `progress` is called with (done, total) items by the serial reference, and
/// once with (total, total) after the pipelined load.
pub fn load_plan(
    dir: &Path,
    plan: &LoadPlan,
    progress: &mut impl FnMut(usize, usize),
) -> Result<(Regions, Option<Report>), ModelError> {
    load_plan_with(dir, plan, LoadMode::from_env(), &LoadOptions::from_env().loader, progress)
}

/// [`load_plan`] with the mode and the loader's options given.
pub fn load_plan_with(
    dir: &Path,
    plan: &LoadPlan,
    mode: LoadMode,
    o: &Options,
    progress: &mut impl FnMut(usize, usize),
) -> Result<(Regions, Option<Report>), ModelError> {
    let regions = allocate(plan)?;
    let report = match mode {
        LoadMode::Serial => {
            load_serial(dir, plan, &regions, progress)?;
            None
        }
        LoadMode::Pipeline => {
            let r = load_pipelined(dir, plan, &regions, o)?;
            progress(plan.items.len(), plan.items.len());
            Some(r)
        }
    };
    Ok((regions, report))
}

/// The loaded device weights: regions plus the sidecar embedding.
pub struct Loaded {
    pub regions: Regions,
    pub embedding: EmbeddingSidecar,
    /// What the pipelined loader did (`None` after the serial reference).
    pub report: Option<Report>,
}

pub fn load_pack(dir: &Path, pack: &BailingPack, progress: &mut impl FnMut(usize, usize)) -> Result<Loaded, ModelError> {
    let (regions, report) = load_plan(dir, &pack.plan, progress)?;
    let embedding =
        EmbeddingSidecar::open(dir, &pack.plan.files, &pack.weights.embed_tokens, pack.config.vocab, pack.config.hidden)?;
    Ok(Loaded { regions, embedding, report })
}

/// A mapped pack (config + typed weights + plan).
pub struct BailingPack {
    pub config: BailingConfig,
    pub weights: BailingWeights,
    pub plan: LoadPlan,
}

impl BailingPack {
    pub fn from_inventory(inv: &PackInventory, config: BailingConfig) -> Result<Self, LayoutError> {
        let weights = BailingWeights::map(inv, &config)?;
        let plan = weights
            .load_plan()
            .map_err(|e| LayoutError { problems: vec![crate::families::bailing_moe_v3::bind::Problem::HeaderIssue(e)] })?;
        Ok(Self { config, weights, plan })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::bailing_moe_v3::weights::{Dst, PlainDtype, PlainTensor, Src};
    use gb_exl3::budget::Category;

    /// A shard whose table starts mid-page and 12 bytes past a 16-byte boundary, as the real pack's does (offset 3,106,828 in shard
    /// 1: 2,060 into its page), with other bytes before and after it.
    const OFFSET: u64 = 3 * 4096 + 2060;
    const VOCAB: usize = 300;
    const HIDDEN: usize = 64;

    fn table() -> Vec<u16> {
        (0..VOCAB * HIDDEN).map(|i| (i as u32).wrapping_mul(2_654_435_761).rotate_left(7) as u16).collect()
    }

    fn shard(tag: &str, tail: usize) -> (PathBuf, Vec<String>, PlainTensor) {
        let dir = std::env::temp_dir().join(format!("rexl3-ling-embed-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut bytes: Vec<u8> = (0..OFFSET).map(|i| (i % 251) as u8).collect();
        for v in table() {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes.extend((0..tail).map(|i| (i % 13) as u8));
        let files = vec!["model-00001-of-00002.safetensors".to_string(), "model-00002-of-00002.safetensors".to_string()];
        std::fs::write(dir.join(&files[1]), &bytes).unwrap();
        let t = PlainTensor {
            name: "model.word_embeddings.weight".into(),
            dtype: PlainDtype::Bf16,
            shape: vec![VOCAB, HIDDEN],
            bytes: (VOCAB * HIDDEN * 2) as u64,
            xform: Xform::None,
            dst_bytes: (VOCAB * HIDDEN * 2) as u64,
            src: Src { file: 1, offset: OFFSET },
            dst: Dst { region: u32::MAX, offset: 0 },
            category: Category::Embedding,
        };
        (dir, files, t)
    }

    #[test]
    fn the_table_read_in_place_is_the_bytes_the_device_copy_uploads() {
        let (dir, files, t) = shard("same", 777);
        let e = EmbeddingSidecar::open(&dir, &files, &t, VOCAB, HIDDEN).unwrap();
        assert_eq!(e.path, dir.join(&files[1]), "the table is read from the shard the inventory names");
        // what the device copy uploads (`DevBuf::from_slice` of these u16s, little-endian on the GB10)
        let copy = e.read_all_bf16().unwrap();
        assert_eq!(copy, table());
        let copy_bytes: Vec<u8> = copy.iter().flat_map(|v| v.to_le_bytes()).collect();
        // what the kernels read in place: the buffer's address is the mapping's (the GPU reads that address through the CPU's page
        // tables), and the bytes there are the copy's, byte for byte
        let m = e.map_in_place().unwrap();
        assert_eq!(m.bytes(), VOCAB * HIDDEN * 2);
        let block = m.block().expect("a view of the mapping");
        let host = block.host_ptr_at(0);
        assert_eq!(m.ptr() as usize, host as usize);
        assert_eq!(host as usize % 16, 12, "unaligned like the real pack's table: `gb_embed` takes its scalar path");
        // SAFETY: the mapping holds `m.bytes()` readable bytes for as long as `block` lives
        let mapped = unsafe { std::slice::from_raw_parts(host as *const u8, m.bytes()) };
        assert_eq!(mapped, &copy_bytes[..]);
        // and a row read on demand is the same row
        for i in [0, 1, VOCAB / 2, VOCAB - 1] {
            assert_eq!(e.row(i).unwrap(), copy[i * HIDDEN..(i + 1) * HIDDEN].to_vec());
        }
        drop(m);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn gathered_rows_are_the_tables_rows_and_a_bad_id_is_a_zero_row() {
        let (dir, files, t) = shard("gather", 777);
        let e = EmbeddingSidecar::open(&dir, &files, &t, VOCAB, HIDDEN).unwrap();
        let m = e.map_in_place().unwrap();
        let base = m.block().expect("a view of the mapping").host_ptr_at(0).cast_const();
        let tab = table();
        let row_bytes = HIDDEN * 2;
        // bad ids: negative, one past the end, far past it
        let bad = [-1, VOCAB as i32, i32::MAX, i32::MIN];
        // the short batch reads page by page; the long one (repeats, both ends, bad ids among them) is announced first
        let short: Vec<i32> = vec![0, 7, VOCAB as i32 - 1, 7, -1];
        let long: Vec<i32> = (0..40).map(|i| (i * 37 % VOCAB) as i32).chain([0, VOCAB as i32 - 1, 0]).chain(bad).collect();
        assert!(short.len() <= WILLNEED_MIN_ROWS && long.len() > WILLNEED_MIN_ROWS);
        for ids in [short, long, Vec::new()] {
            let mut out = vec![0xAAu8; ids.len() * row_bytes];
            // SAFETY: the mapping holds VOCAB * HIDDEN bf16 values
            unsafe { gather_rows(base, VOCAB, row_bytes, &ids, &mut out) };
            for (r, &id) in ids.iter().enumerate() {
                let got: Vec<u16> =
                    out[r * row_bytes..(r + 1) * row_bytes].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                if (0..VOCAB as i32).contains(&id) {
                    assert_eq!(got, tab[id as usize * HIDDEN..(id as usize + 1) * HIDDEN].to_vec(), "row of id {id}");
                    assert_eq!(got, e.row(id as usize).unwrap(), "the on-demand read of id {id}");
                } else {
                    assert!(got.iter().all(|&v| v == 0), "id {id} is not in the table: a zero row");
                }
            }
        }
        drop(m);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_table_that_runs_past_its_shard_is_refused_by_name() {
        let (dir, files, mut t) = shard("short", 0);
        assert!(EmbeddingSidecar::open(&dir, &files, &t, VOCAB, HIDDEN).is_ok(), "a table that ends with the file fits");
        let e = EmbeddingSidecar::open(&dir, &files, &t, VOCAB + 1, HIDDEN).err().expect("one row too many");
        assert!(e.to_string().contains("runs past the end"), "{e}");
        t.src.file = 2;
        let e = EmbeddingSidecar::open(&dir, &files, &t, VOCAB, HIDDEN).err().expect("no such shard");
        assert!(e.to_string().contains("shard 2 of 2"), "{e}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
