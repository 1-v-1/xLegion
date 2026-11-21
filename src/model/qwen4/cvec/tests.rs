//! The speed projection: the file's format against the parser, the tables against their
//! definition, the CPU op against the formula `h -= s · (h · v) · v`.
//!
//! Only [`cpu_op_projects_the_streams`] touches the process-wide vector state (the CPU graph
//! runs the same body serving does); the rest works through [`super::parse`] /
//! [`super::prepare`], which are pure. Synthetic vectors use a width (8) and layers no other
//! unit test's model runs, so a leaked load could not steer anyone else's graph either.

use super::*;
use crate::ggml::{Backend, ComputeAllocator};

/// One test string of directions: layer -> values (built sorted, as the parser stores them).
fn dirs(layers: &[(usize, Vec<f32>)]) -> std::collections::BTreeMap<usize, Vec<f32>> {
    layers.iter().cloned().collect()
}

fn string_bytes(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u64).to_le_bytes());
    out.extend(s.as_bytes());
}

fn kv_str(out: &mut Vec<u8>, key: &str, v: &str) {
    string_bytes(out, key);
    out.extend(8u32.to_le_bytes()); // GGUF_TYPE_STRING
    string_bytes(out, v);
}

/// A `controlvector` GGUF in memory, the shape [`parse`] reads: `dirs` as F32 `direction.<l>`
/// tensors of one dimension, data starting `alignment`-aligned past the table. One array KV
/// rides along, for the parser's walk past types it keeps out of the way.
fn control_gguf(
    arch: &str,
    hint: &str,
    alignment: u64,
    dirs: &std::collections::BTreeMap<usize, Vec<f32>>,
) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend((dirs.len() as u64).to_le_bytes());
    let n_kv: u64 = 3; // architecture, hint, and one array
    out.extend(n_kv.to_le_bytes());
    kv_str(&mut out, "general.architecture", arch);
    kv_str(&mut out, "controlvector.model_hint", hint);
    string_bytes(&mut out, "general.name");
    out.extend(9u32.to_le_bytes()); // GGUF_TYPE_ARRAY of strings
    out.extend(8u32.to_le_bytes());
    out.extend(1u64.to_le_bytes());
    string_bytes(&mut out, "a test vector");
    let e = dirs.values().next().map(|d| d.len()).unwrap_or(0) as u64;
    for (l, _) in dirs {
        string_bytes(&mut out, &format!("direction.{l}"));
        out.extend(1u32.to_le_bytes()); // one dimension
        out.extend(e.to_le_bytes());
        out.extend(0u32.to_le_bytes()); // F32
        let idx: usize = dirs.keys().position(|&k| k == *l).expect("sorted");
        out.extend(((idx as u64) * e * 4).to_le_bytes()); // offset, relative to the data start
    }
    let pad = (alignment as usize - out.len() % alignment as usize) % alignment as usize;
    out.resize(out.len() + pad, 0);
    for d in dirs.values() {
        for &x in d {
            out.extend(x.to_le_bytes());
        }
    }
    out
}

#[test]
fn parse_walks_the_format() {
    let d = dirs(&[(1, vec![1.0, 2.0, 2.0]), (3, vec![0.0, 0.0, 1.0])]);
    let bytes = control_gguf("controlvector", "qwen4exp", 32, &d);
    let p = parse(&bytes, "test").unwrap();
    assert_eq!(p.n_embd, 3);
    assert_eq!(p.model_hint.as_deref(), Some("qwen4exp"));
    assert_eq!(p.dirs, d, "data read at the aligned offsets");

    // A file that is not a control vector, and a truncated one, are rejected by name.
    let other = control_gguf("llama", "qwen4exp", 32, &d);
    assert!(
        parse(&other, "other")
            .unwrap_err()
            .contains("controlvector")
    );
    assert!(
        parse(&bytes[..bytes.len() - 8], "cut")
            .unwrap_err()
            .contains("truncated")
    );
    assert!(parse(b"no", "no").unwrap_err().contains("GGUF"));
}

#[test]
fn prepare_normalizes_and_clamps() {
    // [3, 4] is 5 long; the zero layer can steer nothing.
    let d = dirs(&[
        (1, vec![3.0, 4.0]),
        (2, vec![0.0, 0.0]),
        (3, vec![0.0, 1.0]),
    ]);
    let p = parse(&control_gguf("controlvector", "qwen4exp", 32, &d), "t").unwrap();
    let l = prepare(&p, "t", (1, 9)).unwrap();
    assert_eq!((l.n_embd, l.n_layers), (3, 4));
    assert_eq!(l.steered, vec![false, true, false, true]);
    assert_eq!(l.s[1], 5.0);
    assert_eq!(&l.dir[3..5], &[0.6, 0.8]);
    assert_eq!(l.s[3], 1.0);
    assert_eq!(l.summary, "t, layers 1..3 (2 steered, per-layer)");

    // A range cuts the file down; one past what it carries is an error, not a silent no-op.
    let l = prepare(&p, "t", (3, 9)).unwrap();
    assert_eq!(l.steered, vec![false, false, false, true]);
    assert_eq!(l.summary, "t, layers 3..3 (1 steered, per-layer)");
    assert!(
        prepare(&p, "t", (2, 2))
            .unwrap_err()
            .contains("no direction usable")
    );
}

#[test]
fn the_bundled_vector_is_what_its_package_says() {
    let p = parse(BUNDLED, "bundled").expect("the bundled file parses");
    assert_eq!(p.n_embd, 2560, "Qwen3.8-Flash-Next is 2560 wide");
    assert_eq!(p.model_hint.as_deref(), Some("qwen4exp"));
    let layers: Vec<usize> = p.dirs.keys().copied().collect();
    assert_eq!(layers.first(), Some(&1));
    assert_eq!(layers.last(), Some(&47));
    let l = prepare(&p, "bundled", BUNDLED_RANGE).unwrap();
    // Strata's profile steers 4..44 and reports 41 steered layers.
    let n = l.steered.iter().filter(|&&s| s).count();
    assert_eq!(n, 41, "{n} steered, expected 41");
    assert!(steers(&l, 4) && steers(&l, 44));
    assert!(!steers(&l, 3) && !steers(&l, 45));
    // The package's directions are unit: the norms stay near 1, and every row stays unit.
    for layer in 4..=44 {
        let v = &l.dir[layer * 2560..(layer + 1) * 2560];
        let mut nrm = 0f64;
        for &x in v {
            nrm += (x as f64) * (x as f64);
        }
        assert!(
            (nrm.sqrt() - 1.0).abs() < 1e-4,
            "layer {layer} row is not unit ({nrm})"
        );
        assert!(
            (l.s[layer] - 1.0).abs() < 1e-4,
            "layer {layer} norm {}",
            l.s[layer]
        );
    }
}

/// The CPU op against the formula, on a graph built the way `project` builds one: streams of
/// `[e, hc, n]`, one steered layer, run on several threads (rows split between them). The
/// process-wide state this test loads is cleared at its end.
#[test]
fn cpu_op_projects_the_streams() {
    let (e, hc, n) = (8usize, 2usize, 3usize);
    // Layers 1..3 (1 and 3 exercise the copy-through and the untouched tensors around the
    // steered one); the steered layer 2 takes a direction of norm 2 along an axis mix.
    let raw = vec![0.4f32, -0.8, 1.2, 0.1, -0.5, 0.9, 0.2, 1.1];
    let d = dirs(&[
        (1, (0..e).map(|i| 0.25 * i as f32).collect()),
        (2, raw.clone()),
        (3, (0..e).map(|i| 1.0 - 0.1 * i as f32).collect()),
    ]);
    let bytes = control_gguf("controlvector", "qwen4exp", 32, &d);
    load(&bytes, "test", (2, 2)).unwrap(); // only layer 2 steered
    assert!(steered(2));
    assert!(!steered(1) && !steered(3));

    let backend = Backend::cpu().unwrap();
    let ctx = Context::new(16, 16);
    let res = ctx
        .new_tensor(types::F32, &[e as i64, hc as i64, n as i64])
        .as_input();
    let mut x = Vec::with_capacity(e * hc * n);
    for i in 0..e * hc * n {
        x.push(((i * i * 7 + i * 13) as f32 / 97.0).sin() * 3.0 - 1.0);
    }
    // A weight to name the device by; host memory, so the node runs the CPU body.
    let w = ctx.new_tensor(types::F32, &[4]);
    let y = project(&ctx, 2, res, w).unwrap();
    // An unsteered layer hands the same tensor back: no node, no cost.
    assert_eq!(project(&ctx, 1, res, w).unwrap().raw(), res.raw());

    let y = ctx.cont(y).as_output();
    let mut graph = ctx.new_graph(32);
    graph.expand(y);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    res.set_f32(&x);
    let mut backend = backend;
    backend.set_threads(3);
    backend.compute(&graph).unwrap();
    let out = floats(&y.bytes());
    // The reference straight from the file's values: unit v, s its norm, h -= s · (h · v) · v.
    let mut nrm = 0f64;
    for &v in &raw {
        nrm += (v as f64) * (v as f64);
    }
    let nrm = nrm.sqrt() as f32;
    let unit: Vec<f32> = raw.iter().map(|v| v / nrm).collect();
    for t in 0..n {
        for h in 0..hc {
            let base = (t * hc + h) * e;
            let mut dot = 0f64;
            for i in 0..e {
                dot += f64::from(x[base + i]) * f64::from(unit[i]);
            }
            let k = nrm * dot as f32;
            for i in 0..e {
                let want = x[base + i] - k * unit[i];
                assert!(
                    (out[base + i] - want).abs() <= 1e-5 * (1.0 + want.abs()),
                    "stream {h} token {t} value {i}: {} vs {want}",
                    out[base + i]
                );
            }
        }
    }

    // A vector of another model's width reports once and steps aside (and with it, this
    // test's state, which is exactly how the stack runs stock again).
    assert!(!drop_mismatch(
        &Loaded {
            n_embd: 8,
            n_layers: 1,
            dir: vec![],
            s: vec![],
            steered: vec![true],
            summary: "t".into(),
        },
        7
    ));
    assert!(!loaded(), "the mismatch dropped the vector");
    assert!(!steered(2), "cleared: the stack runs stock again");
}

fn floats(b: &[u8]) -> Vec<f32> {
    b.chunks(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect()
}
