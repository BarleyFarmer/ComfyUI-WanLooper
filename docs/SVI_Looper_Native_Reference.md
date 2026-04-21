# SVI Looper Native — Settings Reference

## Node: SVI Looper Native (`SVILooperNative`)

Multi-segment SVI video looper using WAN 2.2 dual high/low noise models with temporal continuity via `prev_samples`.

---

## Required Inputs

### `model_high` / `model_low` (MODEL)
The high-noise and low-noise WAN 2.2 I2V models. These should arrive fully patched from upstream:

**Expected chain per model:**
```
UnetLoaderGGUF → SVI Pro LoRA → LightX2V LoRA → ModelSamplingSD3 (shift) → model_high/low
```

Shift values: 3.5 is a good starting point for both. Range 3.0–4.0. Higher values increase noise structure but can cause artifacts with distilled LoRAs above 4.0.

### `vae` (VAE)
`wan_2.1_vae.safetensors`. Used for both encoding anchor frames (image→latent) and decoding generated latents (latent→frames).

### `clip` (CLIP)
`umt5_xxl_fp8_e4m3fn_scaled.safetensors`. Used for encoding per-segment text prompts.

### `clip_vision` (CLIP_VISION)
`clip_vision_h.safetensors`. Currently reserved for future use (Phase 3). Connect it but it is not actively used in the SVI conditioning path — image conditioning comes through VAE-encoded `anchor_samples` instead.

### `start_image` (IMAGE)
The first frame of the video. Resize to target resolution upstream (e.g., via `ImageResizeKJv2`). This image seeds segment 1's generation and establishes the reference for color drift correction.

### `width` / `height` (INT)
Output resolution. Must be multiples of 16. Default: 480×640 (portrait 3:4).

Common portrait resolutions:
- 480×640 (base)
- 512×688
- 672×896
- 960×1280

### `steps` (INT, default 8)
Total denoising steps per segment. With LightX2V distillation LoRAs, 8–10 steps is the sweet spot. Without distillation, use 20+.

### `split_step` (INT, default 3)
The step index where the high-noise model hands off to the low-noise model. Steps 0 through `split_step - 1` use the high model; steps `split_step` through `steps - 1` use the low model.

**How it works:**
- `BasicScheduler` generates the full sigma schedule from the high model
- `SplitSigmas` splits at this step index
- High model gets `sigmas[:split_step+1]` (includes endpoints)
- Low model gets `sigmas[split_step:]` (continues from split point)

**Tuning:** With 8 total steps, split at 2–4. Higher split = more steps on high model (more structural work), fewer on low (less refinement). Split at 3 is a good starting point.

### `cfg` (FLOAT, default 1.0)
Classifier-free guidance scale. With SVI Pro LoRAs, keep at 1.0. Only increase (3.0–5.0) if running without SVI Pro.

### `sampler_name` (combo)
The sampling algorithm. Common choices:
- `euler` — fast, reliable, works well with LightX2V at low step counts
- `dpmpp_2m_sde` — higher quality at higher step counts (20+), not ideal for 8 steps

### `scheduler` (combo)
Sigma schedule distribution. Common choices:
- `simple` — linear sigma spacing, works with LightX2V
- `beta57` — optimized for WAN 2.2 (alpha=0.5, beta=0.7), requires RES4LYF. Better quality but only available if RES4LYF loads before this node registers
- `normal` — Gaussian-weighted spacing

### `seed` (INT)
Base seed. Incremented by +1 for each subsequent segment.

---

## Segment Transition Settings

### `overlap` (INT, default 3)
Number of frames to crossfade between consecutive segments. The tail of segment N and the head of segment N+1 are blended together over this many frames.

- `0` = hard cut (no blending)
- `3` = subtle transition (recommended)
- `5` = longer blend (may show ghosting if segments diverge significantly)

**How it works with trimming:** Each segment is first trimmed to the anchor point (see `anchor_frame_offset`). Then the overlap zone is blended between the trimmed tail of the previous segment and the head of the current segment. The previous segment's saved file is re-written with its last `overlap` frames removed, and the current segment's saved file starts with the blended frames.

### `anchor_frame_offset` (INT, default -5)
Which frame from the end of each segment to extract as the anchor (start image for the next segment). Negative values count from the end.

**How it works:**
```
49-frame segment, anchor_frame_offset = -5:
  Frame 0 ─────────────────────── Frame 48
                              ↑
                         Frame 44 = anchor
                              ↓
              Segment trimmed to frames 0-44 (45 frames)
              Anchor frame → color corrected → seeds next segment
```

**Why not use the last frame?** The final frames of a WAN generation tend to have:
- Motion blur / deformation
- Quality drop-off
- Temporal artifacts

Pulling from a few frames earlier gives a cleaner, more stable anchor. Values of -3 to -7 work well. More negative = earlier frame = more stable but more content trimmed from each segment.

**Frame count math for a 3-segment run:**
```
Segment 1: 49 frames generated → trimmed to 45 (anchor at frame 44)
Segment 2: 49 frames generated → trimmed to 45
Segment 3: 49 frames generated → trimmed to 45

With overlap=3:
Segment 1: 45 - 3 = 42 frames saved (tail removed for blend)
Segment 2: 3 (blended) + 42 (unique) = 45 saved, then - 3 for next overlap = 42 saved
Segment 3: 3 (blended) + 42 (unique) = 45 saved (last segment, no tail removal)

Total: 42 + 42 + 45 = 129 frames
```

### `overlap_mode` (combo, default "ease_in_out")
Blending curve for the overlap zone:
- `ease_in_out` — smooth S-curve (3t² - 2t³). Recommended. Natural-looking transition.
- `linear_blend` — straight linear interpolation. Simpler but can look mechanical.
- `filmic_crossfade` — blends in linear light space (gamma 2.2). Better color accuracy during blend.
- `cut` — no blending at all, just hard concatenation.

### `overlap_side` (combo, default "source")
Which segment fades during the blend:
- `source` — previous segment fades out, new segment fades in. This is the natural forward-flow direction. **Use this.**
- `new_images` — reversed. Rarely useful.

---

## SVI-Specific Settings

### `color_correction` (BOOLEAN, default True)
Normalizes each anchor frame's color statistics (per-channel mean and std) back to the original start image's statistics. Prevents progressive color drift (sepia shift, desaturation) across many segments.

**How it works:**
1. Before the loop starts, compute `ref_mean` and `ref_std` from the original `start_image`
2. After each anchor extraction, normalize: `anchor = ((anchor - anchor_mean) / anchor_std) * ref_std + ref_mean`
3. Clamp to [0, 1]

Negligible performance cost (milliseconds per frame). Disable if you want the video's color to evolve naturally over segments (e.g., sunset scene where lighting intentionally shifts).

---

## Optional Inputs

### `positive_prompt` / `negative_prompt` (STRING, forceInput)
Global fallback prompts. If a `SVI Loop Config` node has an empty prompt, the global `positive_prompt` is used. The `negative_prompt` applies to all segments.

### `loop_1` through `loop_10` (SVI_LOOP_CONFIG)
Per-segment configuration from `SVI Loop Config` companion nodes. Only connected configs are executed — unconnected slots are skipped.

---

## Outputs

### `full_video` (IMAGE)
All segments concatenated with overlap blending applied. Ready to feed into `VHS_VideoCombine` or similar.

### `last_anchor` (IMAGE)
The anchor frame extracted from the final segment. Can be used as `start_image` for a subsequent looper node or for inspection.

### `used_prompts` (STRING)
Log of which prompt was used for each segment, with frame counts. Connect to `ShowText` for visibility.

---

## Node: SVI Loop Config (`LoopConfigSVI`)

Companion node — one per segment.

### `prompt` (STRING, multiline)
The positive prompt for this segment. If empty, falls back to the global `positive_prompt` on the main looper node.

### `frames` (INT, default 49)
Number of frames to generate for this segment. 49 is standard for WAN 2.2 (maps to 13 temporal latent slots at stride 4).

### `model_high` / `model_low` (MODEL, optional)
Per-segment model overrides. Wire any LoRA stack upstream and feed the result here. If not connected, uses the global models from the main looper node.

**Important:** The override model must include the full chain — SVI Pro LoRA + LightX2V + shift + your creative LoRA. Use Set/Get nodes to branch from the main model chain after shift is applied, then add your per-segment LoRA on top.

```
Main chain: GGUF → SVI Pro → LightX2V → Shift → Set_model_high → [main looper input]
                                                       ↓
Per-segment: Get_model_high → Creative LoRA → SVI Loop Config model_high
```
