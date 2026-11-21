//! Speed projection: a control vector on the hyper-connection streams of every steered layer
//! (`legion serve --speed-projection`, ported from Strata's `--experimental-speed-projection`).
//!
//! After a steered layer's FFN output is combined into the streams, every stream `h` of every
//! token loses one direction: `h -= s · (h · v) · v`, where `v` is the layer's unit direction
//! and `s` its norm ([`Loaded`] keeps one row per layer, `v` and `s` apart, as Strata's
//! `cvec_upload` tables do). The vector package describes its direction as a *refusal*
//! direction: with the projection on, the model declines far fewer requests, and removing
//! refusals removes a safety behaviour - you are responsible for what the model writes with
//! it on.
//!
//! The vector is a `controlvector` GGUF (llama.cpp's format): `direction.<l>` tensors of
//! `n_embd` F32 values, layer 0 unused. The bundled one (`data/speed-projection/`, see its
//! README) was made from Qwen3.8-Flash-Next activations and steers layers 4-44;
//! `--speed-projection <FILE>` loads another and steers every layer it carries.
//!
//! The projection is a graph node ([`project`], the `cvec_project` [`ExternalOp`]) built only
//! where it applies, so an unloaded vector costs nothing and unsteered layers run untouched.
//! On a GPU the tables are device memory uploaded per device at graph-build time
//! (`legion_cvec_ensure`) and passed to the kernel as plain pointers, so CUDA graph captures
//! replay without touching them and no launch ever allocates; the CPU body reads the host
//! tables. Both bodies compute the same formula; they agree to rounding, not bit for bit,
//! like the other Legion kernels against their references.

#[cfg(feature = "cuda")]
use std::os::raw::c_void;
use std::sync::{Arc, Once, RwLock};

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};
use crate::model::core::ModelError;

#[cfg(test)]
mod tests;

/// The bundled Qwen3.8-Flash-Next vector (its README stands beside it in the repo).
pub const BUNDLED: &[u8] = include_bytes!(
    "../../../data/speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf"
);

/// The layers Strata's `--experimental-speed-projection on` profile steers with the bundled
/// vector (inclusive).
pub const BUNDLED_RANGE: (usize, usize) = (4, 44);

/// One loaded vector: per-layer unit directions and their scales, host memory.
#[derive(Debug)]
struct Loaded {
    /// `n_embd` the vector was made for (2560 for Qwen3.8-Flash-Next).
    n_embd: usize,
    /// Rows of `dir` and `s`: one past the highest direction the file carried.
    n_layers: usize,
    /// `n_layers · n_embd`: layer `l`'s unit direction, zeros where unsteered.
    dir: Vec<f32>,
    /// `n_layers`: layer `l`'s direction norm, 0 where unsteered.
    s: Vec<f32>,
    /// `n_layers`: whether the request range and the file both cover the layer.
    steered: Vec<bool>,
    /// What to log: the file the vector came from and what it steers.
    summary: String,
}

static LOADED: RwLock<Option<Arc<Loaded>>> = RwLock::new(None);

/// Whether a width mismatch was already reported (after it the vector stops steering, so the
/// per-pass check costs nothing).
static WIDTH_CHECK: Once = Once::new();

static CVEC_PROJECT: ExternalOp = ExternalOp {
    name: "cvec_project",
    cpu: cpu_project,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_project),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

// --- the file ---------------------------------------------------------------------------------

/// A parsed vector file, before normalization.
#[derive(Debug)]
struct Parsed {
    model_hint: Option<String>,
    n_embd: usize,
    /// Layer -> direction; the file's layer 0, if any, is dropped (llama.cpp's loader has no
    /// vector there).
    dirs: std::collections::BTreeMap<usize, Vec<f32>>,
}

/// A cursor over a GGUF's bytes (little endian, as the host is).
struct Cur<'a> {
    b: &'a [u8],
    o: usize,
}

impl Cur<'_> {
    fn u32(&mut self) -> Result<u32, String> {
        let w = self.take(4)?;
        Ok(u32::from_le_bytes(w.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, String> {
        let w = self.take(8)?;
        Ok(u64::from_le_bytes(w.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String, String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8(self.take(n)?.to_vec()).map_err(|_| "invalid UTF-8".to_string())?)
    }

    fn take(&mut self, n: usize) -> Result<&[u8], String> {
        let s = self.b.get(self.o..self.o + n).ok_or("truncated")?;
        self.o += n;
        Ok(s)
    }

    /// Walks past one value of GGUF type `t`, keeping it when it is a string.
    fn value(&mut self, t: u32) -> Result<Option<String>, String> {
        Ok(match t {
            0 | 1 | 7 => {
                self.take(1)?;
                None
            }
            2 | 3 => {
                self.take(2)?;
                None
            }
            4 | 5 | 6 => {
                self.take(4)?;
                None
            }
            8 => Some(self.string()?),
            9 => {
                let et = self.u32()?;
                let n = self.u64()?;
                for _ in 0..n {
                    self.value(et)?;
                }
                None
            }
            10 | 11 | 12 => {
                self.take(8)?;
                None
            }
            t => return Err(format!("unknown GGUF value type {t}")),
        })
    }
}

/// Reads a `controlvector` GGUF (llama.cpp's format, version 2 or 3): the architecture and
/// hint, the alignment, and the F32 `direction.<l>` tensors' values.
fn parse(bytes: &[u8], name: &str) -> Result<Parsed, String> {
    let r = |m: String| Err(format!("{name}: {m}"));
    let mut c = Cur { b: bytes, o: 0 };
    if c.take(4).ok() != Some(b"GGUF".as_slice()) {
        return r("not a GGUF file".into());
    }
    let version = c.u32()?;
    if version != 2 && version != 3 {
        return r(format!("GGUF version {version}"));
    }
    let (n_tensors, n_kv) = (c.u64()?, c.u64()?);
    let mut arch = None;
    let mut model_hint = None;
    let mut alignment: u64 = 32;
    for _ in 0..n_kv {
        let key = c.string()?;
        let t = c.u32()?;
        match key.as_str() {
            "general.architecture" => arch = c.value(t)?,
            "controlvector.model_hint" => model_hint = c.value(t)?,
            // The one integer key read for its value: tensor data starts aligned past the table.
            "general.alignment" if t == 4 => alignment = c.u32()? as u64,
            _ => {
                c.value(t)?;
            }
        }
    }
    if arch.as_deref() != Some("controlvector") {
        return r("not a control vector GGUF (general.architecture is not 'controlvector')".into());
    }
    if alignment == 0 || !alignment.is_power_of_two() {
        return r(format!("general.alignment is {alignment}"));
    }
    // The tensor infos: keep the F32 `direction.<l>` rows, walk past everything else. Their
    // offsets count from the data section, which starts `alignment`-aligned past the whole
    // table - so the rows are read only after the table has been walked.
    let mut infos: Vec<(usize, u64)> = Vec::new();
    let mut n_embd = 0usize;
    for _ in 0..n_tensors {
        let key = c.string()?;
        let n_dims = c.u32()? as usize;
        if n_dims > 4 {
            return r(format!("{key}: {n_dims} dimensions"));
        }
        let mut ne = [1u64; 4];
        for d in ne.iter_mut().take(n_dims) {
            *d = c.u64()?;
        }
        let ty = c.u32()?;
        let off = c.u64()?;
        let Some(layer) = key
            .strip_prefix("direction.")
            .map(|l| l.parse::<usize>().ok())
            .flatten()
        else {
            continue;
        };
        if layer == 0 {
            continue;
        }
        if ty != ffi::ggml_type_GGML_TYPE_F32 {
            return r(format!("{key} must be F32"));
        }
        let e = ne[0] as usize;
        if ne[1..].iter().any(|&d| d != 1) {
            return r(format!("{key} must be a vector"));
        }
        if n_embd == 0 {
            n_embd = e;
        } else if e != n_embd {
            return r(format!("{key} is {e} wide, the vector is {n_embd}"));
        }
        infos.push((layer, off));
    }
    let data_start =
        c.o.checked_add(alignment as usize - 1)
            .ok_or("size overflow")?
            & !(alignment as usize - 1);
    let mut dirs: std::collections::BTreeMap<usize, Vec<f32>> = Default::default();
    for (layer, off) in infos {
        let len = n_embd.checked_mul(4).ok_or("size overflow")?;
        let at = data_start
            .checked_add(off as usize)
            .ok_or("size overflow")?;
        let v =
            c.b.get(at..at.checked_add(len).ok_or("truncated")?)
                .ok_or("truncated")?;
        dirs.insert(
            layer,
            v.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        );
    }
    if dirs.is_empty() {
        return r("no direction.<layer> tensors".into());
    }
    if n_embd == 0 || n_embd > i32::MAX as usize {
        return r(format!("n_embd is {n_embd}"));
    }
    Ok(Parsed {
        model_hint,
        n_embd,
        dirs,
    })
}

/// Loads a vector from `bytes` (the bundled file or one the user named), steering the layers
/// of `range` the file covers, and reports what it steers. With CUDA built in the tables are
/// handed to the CUDA side, which uploads them per device at first use ([`project`]).
pub fn load(bytes: &[u8], name: &str, range: (usize, usize)) -> Result<String, String> {
    let p = parse(bytes, name)?;
    if let Some(hint) = &p.model_hint
        && hint != "qwen4exp"
    {
        tracing::warn!("speed projection: {name} was made for '{hint}', not qwen4exp");
    }
    let loaded = prepare(&p, name, range)?;
    #[cfg(feature = "cuda")]
    if unsafe {
        legion_cvec_store(
            loaded.dir.as_ptr(),
            loaded.s.as_ptr(),
            loaded.n_layers as i32,
            loaded.n_embd as i64,
        )
    } != 0
    {
        return Err("speed projection: the CUDA side rejected the tables".into());
    }
    let summary = loaded.summary.clone();
    *LOADED.write().unwrap() = Some(Arc::new(loaded));
    Ok(summary)
}

/// The tables [`parse`] finds, normalized and clamped to `range` (what the file carries
/// outside it is dropped): the unit direction and its norm per layer, as Strata's loader
/// makes them, so the kernel applies `h -= s · (h · v) · v`.
fn prepare(p: &Parsed, name: &str, range: (usize, usize)) -> Result<Loaded, String> {
    let (first, want_last) = (range.0.max(1), range.1.max(1));
    let file_last = *p.dirs.keys().next_back().expect("non-empty");
    let n_layers = file_last + 1;
    // The range that takes effect: what was asked for, clamped to what the file carries.
    let last = want_last.min(file_last);
    let mut dir = vec![0f32; n_layers * p.n_embd];
    let mut s = vec![0f32; n_layers];
    let mut steered = vec![false; n_layers];
    let mut n_steered = 0usize;
    for (&l, d) in &p.dirs {
        if l < first || l > last {
            continue;
        }
        let mut nrm = 0f64;
        for &x in d {
            nrm += (x as f64) * (x as f64);
        }
        let nrm = nrm.sqrt();
        if !(nrm > 0.0) || !nrm.is_finite() {
            continue;
        }
        s[l] = nrm as f32;
        let row = &mut dir[l * p.n_embd..(l + 1) * p.n_embd];
        for (o, &x) in row.iter_mut().zip(d) {
            *o = x / nrm as f32;
        }
        steered[l] = true;
        n_steered += 1;
    }
    if n_steered == 0 {
        return Err(format!(
            "{name} has no direction usable in layers {first}..{} (it carries {}..{})",
            want_last.min(file_last),
            p.dirs.keys().next().copied().unwrap_or(0),
            file_last
        ));
    }
    Ok(Loaded {
        n_embd: p.n_embd,
        n_layers,
        dir,
        s,
        steered,
        summary: format!("{name}, layers {first}..{last} ({n_steered} steered, per-layer)"),
    })
}

// --- the state --------------------------------------------------------------------------------

/// Whether a vector is loaded and may still steer ([`project`] checks the width once).
pub fn loaded() -> bool {
    LOADED.read().unwrap().is_some()
}

/// The log line naming the loaded vector.
pub fn summary() -> String {
    LOADED
        .read()
        .unwrap()
        .as_ref()
        .map(|l| l.summary.clone())
        .unwrap_or_default()
}

/// Whether layer `l`'s write is followed by the projection.
fn steers(l: &Loaded, il: usize) -> bool {
    l.steered.get(il).copied().unwrap_or(false)
}

/// Whether the main loop must fold layer `il`'s block output into the streams now (before the
/// projection) rather than leave it for the next mixer. `false` with no vector loaded, so an
/// unloaded model runs exactly as before.
pub(super) fn steered(il: usize) -> bool {
    LOADED
        .read()
        .unwrap()
        .as_ref()
        .is_some_and(|l| steers(l, il))
}

/// Reports a width mismatch once and drops the vector: a model of another width runs stock.
fn drop_mismatch(l: &Loaded, n_embd: usize) -> bool {
    if l.n_embd == n_embd {
        return true;
    }
    WIDTH_CHECK.call_once(|| {
        tracing::warn!(
            "speed projection off: {} is not a {}-wide model's vector",
            l.summary,
            n_embd
        );
    });
    *LOADED.write().unwrap() = None;
    false
}

// --- the op -----------------------------------------------------------------------------------

/// Whether [`project`] can take `res`: F32 `[e, hc, n]`, contiguous.
fn fits(res: Tensor<'_>) -> bool {
    let [e, hc, n, one] = res.ne();
    let nb = res.nb();
    one == 1
        && res.ty() == types::F32
        && nb[0] == 4
        && nb[1] == (4 * e) as usize
        && nb[2] == (4 * e * hc) as usize
        && e <= i32::MAX as i64
        && hc <= i32::MAX as i64
        && n <= i32::MAX as i64
}

/// Projects the hyper-connection streams `res` (`F32 [e, hc, n]`) after steered layer `il`:
/// every stream of every token loses its component along the layer's direction,
/// `h -= s · (h · v) · v`. `w` is a weight of the stage that runs `res`, for its device.
/// Without a vector or a steered layer: `res` unchanged, node-free.
pub(super) fn project<'a>(
    ctx: &'a Context,
    il: usize,
    res: Tensor<'a>,
    w: Tensor<'_>,
) -> Result<Tensor<'a>, ModelError> {
    // A clone so the read lock is released before any `drop_mismatch` (which writes it).
    let Some(l) = LOADED.read().unwrap().clone() else {
        return Ok(res);
    };
    if !steers(&l, il) {
        return Ok(res);
    }
    if l.n_embd != res.ne()[0] as usize {
        // A model of another width: report once and run stock.
        drop_mismatch(&l, res.ne()[0] as usize);
        return Ok(res);
    }
    if !fits(res) {
        return Err(ModelError::Unsupported(format!(
            "speed projection: layer {il} streams {:?} are not contiguous F32 [e, hc, n]",
            res.ne()
        )));
    }
    #[cfg(feature = "cuda")]
    let mut devp: i32 = -1;
    #[cfg(feature = "cuda")]
    if let Some(dev) = w.cuda_device() {
        // Upload the tables now, at graph-build time: the launch later only reads them.
        let e = unsafe { legion_cvec_ensure(dev as i32) };
        if e != 0 {
            return Err(ModelError::Layout(format!(
                "speed projection: cannot upload the vector to CUDA device {dev} (error {e})"
            )));
        }
        devp = dev as i32 + 1;
    }
    #[cfg(not(feature = "cuda"))]
    let (devp, _) = (-1i32, &w);
    // (A host `res` runs the CPU body even with CUDA built in: it is the CPU backend's node.)
    Ok(ctx.external_with(
        &CVEC_PROJECT,
        types::F32,
        res.ne(),
        &[res],
        &[il as i32, devp],
    ))
}

/// The host tables of steered layer `il` (the CPU body; the CUDA side keeps its own).
fn tables(il: usize) -> Option<(Arc<Loaded>, usize)> {
    let l = LOADED.read().unwrap().clone()?;
    steers(&l, il).then(|| (l, il))
}

unsafe fn cpu_project(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `project`: src[0] is F32 contiguous [e, hc, n] host memory and
    // dst its projected copy of the same shape.
    unsafe {
        let res = &*dst.src[0];
        let (e, hc, n) = (res.ne[0] as usize, res.ne[1] as usize, res.ne[2] as usize);
        let x = res.data as *const f32;
        let y = dst.data as *mut f32;
        let Some((l, il)) = tables(crate::ggml::external_params(dst)[0] as usize) else {
            // The vector was dropped between building and running (a width mismatch seen
            // late): the stock model, the streams copied through.
            for i in (ith..e * hc * n).step_by(nth) {
                *y.add(i) = *x.add(i);
            }
            return;
        };
        let v = &l.dir[il * l.n_embd..(il + 1) * l.n_embd];
        let s = l.s[il];
        for row in (ith..hc * n).step_by(nth) {
            let (h, t) = (row % hc, row / hc);
            // A stream of one token, contiguous: `e` values at `(t · hc + h) · e`.
            let base = (t * hc + h) * e;
            let mut dot = 0f64;
            for i in 0..e {
                dot += f64::from(*x.add(base + i)) * f64::from(v[i]);
            }
            let k = s * dot as f32;
            for i in 0..e {
                *y.add(base + i) = *x.add(base + i) - k * v[i];
            }
        }
    }
}

/// The kernel's parameters (src/cuda/cvec.cu): plain pointers and sizes; the device tables
/// are picked by `dev`, and the launcher never allocates.
#[cfg(feature = "cuda")]
#[repr(C)]
struct CvecProj {
    res: *const f32,
    dst: *mut f32,
    e: i64,
    hc: i64,
    n: i64,
    layer: i32,
    dev: i32,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    /// Keeps a host copy of the tables for [`legion_cvec_ensure`] to upload per device.
    fn legion_cvec_store(dir: *const f32, s: *const f32, n_layers: i32, n_embd: i64) -> i32;
    /// Uploads the tables to device `dev` if it lacks them; a CUDA error code.
    fn legion_cvec_ensure(dev: i32) -> i32;
    /// Launches the projection kernel on `stream`.
    fn legion_cvec_project_launch(p: *const CvecProj, stream: *mut c_void) -> i32;
}

#[cfg(feature = "cuda")]
unsafe fn cuda_project(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `project`: F32 contiguous device streams of the stream's device.
    unsafe {
        let res = &*dst.src[0];
        let p = crate::ggml::external_params(dst);
        let args = CvecProj {
            res: res.data as *const f32,
            dst: dst.data as *mut f32,
            e: res.ne[0],
            hc: res.ne[1],
            n: res.ne[2],
            layer: p[0],
            dev: p[1] - 1,
        };
        match legion_cvec_project_launch(&args, stream) {
            0 => Ok(()),
            e => Err(format!("cvec_project: error {e}")),
        }
    }
}
