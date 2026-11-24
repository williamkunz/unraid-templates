# Strata for Unraid

An Unraid Community Applications template for [Strata](https://github.com/Niko1221/Strata), which runs large mixture-of-experts models on an NVIDIA GPU plus system RAM: Qwen3.8-Flash-Next and its variants, the smaller Qwen3.6-35B-A3B, Inkling-2-Mini for cards with less memory, and Tencent Hy4-Flash. It serves an OpenAI- and Anthropic-compatible API, a chat page and a live monitor.

The template runs Strata's official image, `ghcr.io/niko1221/strata:latest-cuda13`. Report template or Unraid problems in this repository's [issues](https://github.com/williamkunz/unraid-templates/issues), and problems with Strata itself to the [Strata project](https://github.com/Niko1221/Strata/issues).

## Install

Search for **Strata** in the Apps tab. Before installing you need:

- the **Nvidia Driver plugin** on driver 580 or newer, and an RTX 20/30/40/50 card. For a GTX 10, P100 or V100, or a driver from 575 to 579, change the Repository tag to `latest-cuda12`.
- 32 GB of RAM or more for the Qwen models; more RAM allows a larger, better model ([Strata's model guide](https://github.com/Niko1221/Strata/blob/main/docs/MODELS.md)). Inkling-2-Mini runs on much less, Hy4-Flash wants 64 GB or more.
- room for the model on an SSD pool, mapped as a direct pool path (`/mnt/cache/...`), not `/mnt/user`

### Choosing a model

Leave **Model** and **Model family** on their blank first choice and Strata picks one for your server; the log says which and why. To choose yourself, set both.

The model names are compression levels: how the model's weights are squeezed into fewer bits. RAM in the tables below is the server's total RAM.

#### Reading the model names

A name is built from parts. `UD-Q5_K_XL`, for example, is Unsloth's file (`UD-`) at about 5 bits per weight (`Q5`), in the K-quant format (`_K`), in its largest mix (`_XL`).

| Part | Examples | What it means |
| --- | --- | --- |
| `Q` and a number | `Q2_0`, `UD-Q4_K_XL`, `Q8_0` | Quantized: the weights are stored in fewer bits. The number is roughly the bits per weight. Fewer bits mean a smaller download and less RAM, and lose a little more quality: 2 is small, 8 is almost the full model. |
| `IQ` and a number | `IQ2_XS`, `IQ3_S` | An i-quant: weights are stored as patterns picked from a fixed table, which keeps more quality at very low bit counts but takes more work to compute, so these are slower than a plain format of the same size. |
| `_0` | `Q2_0`, `Q8_0` | The simplest format: one scale for each small block of weights. Fast to compute. |
| `_K` | `UD-Q2_K_XL`, `UD-Q5_K_XL` | A K-quant: blocks are grouped into larger blocks with their own scales, a middle ground between quality and size. |
| `XXS`, `XS`, `S`, `M` | `IQ3_XXS`, `IQ3_S`, `UD-IQ1_M` | Size within the same bit count, from extra small to medium. A larger one keeps a little more precision. |
| `XL` | `UD-Q4_K_XL`, `UD-Q6_K_XL` | Unsloth's largest mix: the most important layers are kept above the name's bit count. |
| `UD-` | `UD-IQ2_XS`, `UD-Q5_K_XL` | Unsloth Dynamic: Unsloth's files, which choose the format layer by layer instead of using one for the whole model. |
| `NVFP4` | `NVFP4` | NVIDIA's 4-bit floating-point format. RTX 50 cards compute it directly; other NVIDIA cards run it at 8-bit precision. |
| `-NVIDIA`, `-QAD` | `NVFP4-NVIDIA`, `NVFP4-QAD` | `-NVIDIA` is NVIDIA's own 4-bit release, converted for Strata. `-QAD` is NVIDIA's model retrained to work well at 4 bits (quantization-aware distillation). |
| `Gyro-S`, `Gyro-M` | | agentionai's Gyro format, which compresses the model's experts in its own way; S is small, M is medium. |

The number is only a guide: `IQ3_S` is about 3.5 bits, and the Coder's `IQ1_M` is stored like `IQ3_S` but with half of the model's experts removed, so it is small for another reason.

#### Sizes by model

**Qwen3.8-Flash-Next**, the main model, for chat, code and tools. Model family `qwen`:

| Model | What it is | Download | RAM |
| --- | --- | ---: | ---: |
| `Q2_0` | the fastest, good quality | 66 GB | 48 GB |
| `IQ2_XS` | a little better, nearly as fast | 68 GB | 48 GB |
| `IQ3_XXS` | great quality, slower | 76 GB | 60 GB |
| `IQ3_S` | the best quality, matches the full model; the slowest | 84 GB | 64 GB |

**Other versions of Qwen3.8:**

| Model family | Model | What it is | Download | RAM |
| --- | --- | --- | ---: | ---: |
| `swift` | `IQ2_XS`, `IQ3_XXS`, `IQ3_S` (the `qwen` sizes except `Q2_0`) | UkisAI's fine-tune that thinks for less time before it answers | as `qwen` | as `qwen` |
| `coder` | `IQ1_M` | for code only, with half the model removed | 58 GB | 32 GB |
| `gyro` | `Gyro-S` | the least memory, lower quality | 59 GB | about 28 GB of RAM and graphics memory together |
| `gyro` | `Gyro-M` | about `IQ3_XXS` quality in less memory, 16–21% slower than `Q2_0` | 92 GB | about 35 GB together |
| `unsloth` | `UD-IQ1_M` | 1-bit, the smallest Unsloth file | 68 GB | ~49 GB |
| `unsloth` | `UD-Q2_0` | 2-bit | 71 GB | ~51 GB |
| `unsloth` | `UD-IQ2_XS` | 2-bit i-quant | 76 GB | ~56 GB |
| `unsloth` | `UD-Q2_K_XL` | 2-bit with some 3-bit experts | 80 GB | ~59 GB |
| `unsloth` | `UD-IQ3_XXS` | 3-bit i-quant | 84 GB | ~63 GB |
| `unsloth` | `UD-Q3_K_XL` | 3-bit with some 4-bit experts | 90 GB | ~69 GB |
| `unsloth` | `UD-IQ4_XS` | 4-bit, better than `IQ3_S` | 94 GB | 72 GB |
| `unsloth` | `UD-Q4_K_XL` | 4-bit | 111 GB | 88 GB |
| `unsloth` | `UD-Q5_K_XL` | 5-bit, close to `Q8_0` | 158 GB | 108 GB |
| `unsloth` | `UD-Q6_K_XL` | 6-bit | 169 GB | 118 GB |
| `unsloth` | `Q8_0` | 8-bit, the reference quality | 188 GB | 136 GB |
| `nvfp4` | `NVFP4`, `NVFP4-NVIDIA`, `NVFP4-QAD` | NVIDIA's 4-bit format, fastest on RTX 50 cards; `NVFP4` has the best quality of the three | 102–128 GB | 80 GB |

The `unsloth` RAM figures keep the whole model in memory. With less RAM (from 48 GB, for the 1- to 4-bit sizes) they still run, reading part of the model from the SSD, which is much slower. The 1- to 3-bit sizes are Unsloth's files of 11-04; their RAM figures are worked out from Strata's rule (all the experts plus 16 GB), not published by Strata.

**Qwen3.6-35B-A3B**, a smaller model for servers with less RAM. Model family `qwen36`:

| Model | What it is | Download | RAM Strata plans for |
| --- | --- | ---: | ---: |
| `UD-Q4_K_XL` | 4-bit, the smallest | 22.9 GB | 32 GB |
| `UD-Q6_K_XL` | 6-bit, closer to the full model | 32.6 GB | 44 GB |
| `Q8_0` | 8-bit, the closest to the full model | 37.8 GB | 52 GB |

These three names are also `unsloth` sizes of Qwen3.8, so set Model family to `qwen36` to get Qwen3.6. Qwen3.6 has 35 billion parameters, 3 billion of them used per token. Vision adds 0.9 GB to the download.

The RAM figures are what Strata plans for, not a promise. The engine keeps as many of the model's experts as fit on the graphics card and computes the rest on the CPU, from system RAM. On a 12 or 16 GB card most experts stay in RAM, so free RAM after Unraid and your other containers has to cover them; if it does not, the start fails with `no host memory for the experts`. Strata measured `UD-Q4_K_XL` decoding at about 60 tokens per second on an RTX 3060 12 GB with most experts on the CPU, and about 118 on a 32 GB Radeon AI PRO R9700. [Strata's Qwen3.6 guide](https://github.com/Niko1221/Strata/blob/main/docs/QWEN36.md) has the full measurements.

**Inkling-2-Mini**, from Thinking Machines: small and fast, and the one to pick when the card is the limit rather than the RAM. Model family `inkling-2-mini`. It reads pictures with a 0.64 GB file and drafts with DSpark.

Strata's own `.strata` builds:

| Model | What it is | Download | Typical card |
| --- | --- | ---: | --- |
| `SQ-10` | compact and the fastest | 10.5 GB | a 12 GB card, or a smaller one with the rest in RAM |
| `SQ-15` | the best balance | 14.3 GB | a 16 GB card |
| `SQ-20` | nearly the full model | 19.3 GB | a 24 GB card |

Unsloth's files for the same model, for cards between those sizes:

| Model | Download |
| --- | ---: |
| `UD-IQ1_S` | 7.8 GB |
| `UD-IQ1_M` | 8.0 GB |
| `UD-IQ2_XXS` | 8.5 GB |
| `UD-Q2_K_XL` | 9.1 GB |
| `UD-IQ3_XXS` | 11.7 GB |
| `UD-Q3_K_XL` | 13.6 GB |
| `UD-IQ4_XS` | 15.7 GB |
| `UD-Q4_K_M` | 16.6 GB |
| `UD-Q4_K_XL` | 17.6 GB |
| `UD-Q8_K_XL` | 22.9 GB |

DSpark, its drafter, is decided by a measurement: at the first start Strata times the model with it and without it, and keeps it only where it is at least 10% faster. That makes the first start of a new setup a few minutes longer. Code, tool calls and repeated text come out 1.2x to 2.4x faster with it; plain chat is a few percent slower. Context goes up to 256k.

**Tencent Hy4-Flash**, a large model in Strata's own format. Model family `hy4-flash`. It reads pictures with a 0.93 GB file and drafts with DFly2.

| Model | What it is | Download | Lookup shard | RAM Strata plans for |
| --- | --- | ---: | ---: | --- |
| `SQ-64` | the smallest, and the flagship size | 62.0 GB | 67.6 GB | 64 GB and a card of 8 GB or more |
| `SQ-96` | closer to the full model | 89.6 GB | 67.6 GB | 96 GB |
| `SQ-128` | the closest to the full model | 113.7 GB | 67.6 GB | 128 GB |

Every size reads the same 67.6 GB lookup shard from the Data folder while it runs, so the folder needs the model and the shard together: 130 GB for `SQ-64`, up to 181 GB for `SQ-128`. That makes a direct pool path on an NVMe drive matter more here than anywhere else. Context goes up to 128k.

**The `UD-` names belong to several families.** `UD-Q4_K_XL`, for example, is a size of Qwen3.8, of Qwen3.6 and of Inkling-2-Mini. Set Model family as well as Model, or Strata picks for you.

**DeepSeek is no longer offered.** Strata stopped running DeepSeek models after 0.2.13. A container set to a DeepSeek model now stops at the start and says so: choose a Qwen model instead. The DeepSeek files (191–274 GB) stay in the Data folder until you delete them.

### First start

The first start downloads the model and prepares it. Later starts go straight to serving. The template's Overview explains each setting, and Strata's [Docker guide](https://github.com/Niko1221/Strata/blob/main/docs/DOCKER.md) covers them in full.

An API key is always on. If you leave it empty, the first start makes one; show it with:

```bash
docker exec Strata strata launch --show-key --preset docker
```

## Troubleshooting

### The model loads, then the start stops with "layer 0 never rang"

```
a window failed: verify: layer 0 never rang (operation not supported on global/shared address space)
```

This is a bug in **engine 0.1.66**, the engine that shipped with Strata **0.2.20**. Four of its kernels stage weights into shared memory and then read them back with an instruction that only addresses global memory, so the first warm-up window after the model loads faults. It stops every CUDA card from sm_80 up — RTX 30, 40 and 50 — on any model whose hyper-connection weights are `Q8_0`, Gyro included. Strata's maintainer confirmed it in [#6713](https://github.com/Niko1221/Strata/issues/6713).

**The fix is to update the container.** Strata 0.2.21 and newer carry engine 0.1.67, which fixes it; `latest-cuda13` points there. On the Docker tab, *Check for Updates*, then *Force update* on the container. The engine lives in the image, so this replaces it without touching the models or tuning in your Data folder.

If you want to stay on 0.2.20, put **both** of these in **Engine settings** (`STRATA_SET`), space separated:

```
STRATA_HC_Q8_TC=0 STRATA_HC_Q8_BF16TC=0
```

Both are needed: with only one, the other set of kernels keeps faulting. Strata measured the cost on an RTX 4090 as no change to decoding speed, −2% on long prompts and +22% on short cold ones.

On an RTX 30-series card, if the message comes back after the update but says `an illegal memory access was encountered` instead, see [#7019](https://github.com/Niko1221/Strata/issues/7019) and use the two settings above.

### The log is full of `driverInitFileInfo ... result=11`

Noise from the NVIDIA driver starting up inside the container. It is not a failure on its own; read the lines below it.

## Updates

Strata is in rapid development and changes often, so expect this template to change along with it: settings get added, renamed or removed as new releases come out. Installed apps keep their saved copy of the template, so a new setting only appears after you reinstall or add it by hand.

The template follows `latest-cuda13`, which Strata's maintainer moves to each new release when its image is published. Unraid shows the update on the Docker tab. To stay on one release, change the tag to a version such as `0.2.21-cuda13`.

The engine is part of the image, so updating the container is what updates the engine; `strata update` inside it is not the way. A new engine can make an old tuning result less useful, so after a big jump it is worth one start with **Tuning** `quick` and **Tune again** `1`, then setting Tune again back to `0`.

## Credits and licence

[Strata](https://github.com/Niko1221/Strata) and its image are by Niko1221 and the Strata contributors, under the MIT licence. The icon is Strata's own. This repository is MIT licensed; see [LICENSE](../LICENSE).
