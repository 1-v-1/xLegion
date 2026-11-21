# speed-projection (EXPERIMENTAL, off by default)

`Qwen3.8-Flash-Next-experimental-speed-projection.gguf` is a 480 KB control vector for
Qwen3.8-Flash-Next: one direction of 2,560 F32 values per layer (`direction.1` ..
`direction.47`, llama.cpp's `controlvector` GGUF format; the norms are near 1 but not exact,
and the loader normalizes). The JSON beside it is the configuration its authors selected and
their scores, as the Strata package shipped them.

**What it does.** `legion serve --speed-projection on` removes each direction from every
hyper-connection stream after the FFN write of layers 4-44 (`h -= s · (h · v) · v` with `v`
the unit direction and `s` its norm, per layer) - the same operation as Strata's
`--experimental-speed-projection` / llama.cpp's `--cvec-mode project --cvec-dir per-layer`.
The package's own documentation describes the vector as a **refusal-direction projection**:
the model declines far fewer requests (it reports 1 of 50 vs 50 of 50 on its test set), and
removing refusals removes a safety behaviour - you are responsible for what the model writes
with it on. It also shifts the model's predictions slightly on ordinary text (the package
scores: KL mean 0.186, p99 3.14 against its baseline). In Legion it costs the step one
kernel per steered layer; the rest of the model runs untouched.

**Turning it on.** `legion serve --model <qwen4exp.gguf> --speed-projection on` - the
bundled file and layers 4-44, Strata's profile. A path loads another `controlvector` GGUF
and steers every layer it carries. See [serving.md](../../docs/serving.md), "Kernel and
load switches".

**Origin and license.** Copied from Strata
([github.com/Niko1221/Strata](https://github.com/Niko1221/Strata),
`terminal/data/experimental-speed-projection/`), which describes it as a local package of a
published vector whose original filenames contain "refusal" (the bytes are the same), made
from Qwen3.8-Flash-Next activations - so the **Qwen Community License 1.0** of the model
applies to it. Made for the original Qwen3.8-Flash-Next (not Swift 1.5). Measured against
the base model `unsloth/Qwen3.8-Flash-Next-GGUF` UD-Q2_K_XL (shard 1
`a4f3b21e77353999829f2f767e9ac21ce9c71d29a74f2cc9eda48c9bf23c8b86`); the directions' layer
width (2,560) is what a loaded model's streams are checked against, so other qwen4exp
checkpoints of the same width steer too, with their own behaviour change.
