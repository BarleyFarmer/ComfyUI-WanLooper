# SVILooper — Claude Code Build Brief
**Last updated: 2026-04-12**

---

## File Locations

| Purpose | Path |
|---------|------|
| Dev working copy | `E:\AI_Studio\comfy-node-dev\ComfyUI-SVILooper\nodes_svi.py` |
| Install target | `E:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-SVILooper\nodes_svi.py` |
| Init file | `E:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-SVILooper\__init__.py` |

**Workflow:** Always edit the dev copy first, then copy to install. Verify with grep before copying.

Do NOT modify:
- `nodes.py`
- `nodes_highlow.py`

---

## Architecture History (what was tried and why it was abandoned)

| Approach | Status | Reason abandoned |
|----------|--------|-----------------|
| SVILooper (monolithic, IAMCCS + Clownshark, graph-wired) | Abandoned | Beige/noise output — IAMCCS_WanImageMotion called from Python produced bad conditioning |
| SVILoopPrep + SVILoopFinish (hybrid split nodes) | Abandoned | Same IAMCCS issue; ComfyUI DAG prevents stateful loops in graph anyway |
| SVILooperCK (monolithic, Clownshark internal loop) | Abandoned | Same IAMCCS issue; also Clownshark loading complexity |
| **SVILooperKSA (monolithic, KSamplerAdvanced internal loop)** | **ACTIVE — working** | Uses WanImageToVideo + CLIPVisionEncode (same path as I2VLooperHL) |

**Root cause of all earlier failures:** `IAMCCS_WanImageMotion`, when called from Python (not ComfyUI graph), produced beige/uniform output regardless of sampler. The fix was to drop IAMCCS entirely and use the proven `WanImageToVideo` + `CLIPVisionEncode` conditioning path from `I2VLooperHL`.

---

## Current Nodes in nodes_svi.py

### NODE_CLASS_MAPPINGS (current)

```python
NODE_CLASS_MAPPINGS = {
    "SVILoopConfig": SVILoopConfig,   # unchanged from original — kept for reference
    "SVILoopPrep":   SVILoopPrep,     # hybrid node — no longer primary workflow
    "SVILoopFinish": SVILoopFinish,   # hybrid node — no longer primary workflow
    "SVILooperCK":   SVILooperCK,     # Clownshark monolithic — not recommended
    "SVILooperKSA":  SVILooperKSA,    # KSamplerAdvanced monolithic — PRIMARY
}
```

---

## PRIMARY NODE: SVILooperKSA

### Category
`video/svi_looper`

### Outputs
| Name | Type | Notes |
|------|------|-------|
| full_video | IMAGE | All segments concatenated |
| last_anchor | IMAGE | Anchor frame from final segment |
| used_prompts | STRING | Log of which prompt was used per loop |

### Required Inputs
| Name | Type | Default | Notes |
|------|------|---------|-------|
| model_high | MODEL | | Global fallback high-noise model |
| model_low | MODEL | | Global fallback low-noise model |
| vae | VAE | | |
| clip | CLIP | | |
| clip_vision | CLIP_VISION | | Required for CLIPVisionEncode |
| start_image | IMAGE | | First segment's start frame |
| positive | CONDITIONING | | Global positive (fallback if no loop prompt) |
| negative | CONDITIONING | | Global negative |
| width | INT | 832 | step 16 |
| height | INT | 480 | step 16 |
| frames | INT | 49 | Frames per segment (before anchor trim) |
| num_loops | INT | 1 | 1–10 segments |
| overlap | INT | 0 | Frame overlap between segments (ImageBatchExtendWithOverlap) |
| anchor_frame_offset | INT | -5 | Frames from end to extract anchor. e.g. -5 = 5th-from-last |
| seed | INT | 0 | Base seed, incremented +1 per loop |
| steps | INT | 8 | Total denoising steps |
| split_step | INT | 4 | Step at which high model hands off to low model |
| cfg | FLOAT | 1.0 | CFG scale |
| sampler_name | ENUM | | From comfy.samplers.KSampler.SAMPLERS |
| scheduler | ENUM | | From comfy.samplers.KSampler.SCHEDULERS |

### Optional Inputs
| Name | Type | Notes |
|------|------|-------|
| loop_prompts | STRING multiline | One line per segment. Blank = use global positive. Fewer lines than num_loops → last line repeats |
| model_high_1 … model_high_10 | MODEL | Per-segment high model. Connect LoraLoaderModelOnly output. Falls back to global model_high if not connected |
| model_low_1 … model_low_10 | MODEL | Per-segment low model. Falls back to global model_low if not connected |

---

## Per-Loop Internal Flow

### 1. Model selection
```python
high_model = seg_high[loop_id] if seg_high[loop_id] is not None else model_high
low_model  = seg_low[loop_id]  if seg_low[loop_id]  is not None else model_low
```
`seg_high` and `seg_low` are 1-indexed lists built from the optional model inputs.

### 2. Prompt encoding
- If `loop_prompts` line N is non-empty: `clip.tokenize()` → `clip.encode_from_tokens(return_pooled=True)` → `[[cond, {"pooled_output": pooled}]]`
- Else: use global `positive` conditioning

### 3. Conditioning — CLIPVisionEncode + WanImageToVideo
```python
from comfy_extras.nodes_wan import WanImageToVideo

cv_result  = nodes.CLIPVisionEncode().encode(clip_vision, current_start_image, "none")
i2v_result = WanImageToVideo().execute(
    positive=c_pos, negative=negative, vae=vae,
    width=width, height=height, length=frames, batch_size=1,
    start_image=current_start_image, clip_vision_output=cv_out,
)
pos_cond, neg_cond, latent_dict = i2v_result[0], i2v_result[1], i2v_result[2]
```
**This is the proven path — identical to I2VLooperHL. Do not replace with IAMCCS_WanImageMotion.**

### 4. Two-pass KSamplerAdvanced
```python
ksampler = nodes.KSamplerAdvanced()

# Pass 1 — high model
high_out = ksampler.sample(
    model=high_model, add_noise="enable", noise_seed=loop_seed,
    steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
    positive=pos_cond, negative=neg_cond, latent_image=latent_dict,
    start_at_step=0, end_at_step=split_step, return_with_leftover_noise="enable",
)

# Pass 2 — low model
low_out = ksampler.sample(
    model=low_model, add_noise="disable", noise_seed=0,
    steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
    positive=pos_cond, negative=neg_cond,
    latent_image={"samples": high_out[0]["samples"]},
    start_at_step=split_step, end_at_step=1000, return_with_leftover_noise="disable",
)
sampled = low_out[0]["samples"]
```

### 5. VAE decode
```python
decoded = vae.decode(sampled)
while decoded.ndim > 4:
    decoded = decoded.squeeze(0)
# Resize to width x height if mismatch
```

### 6. Anchor frame extraction
```python
anchor_frame_idx = max(0, min(actual_frames - 1, actual_frames + anchor_frame_offset))
raw_anchor = _GetImageRange.imagesfrombatch(images=decoded, start_index=anchor_frame_idx, num_frames=1)[0]
```

### 7. Color drift correction
Each VAE encode/decode cycle shifts per-channel mean+std slightly. Without correction this accumulates into progressive desaturation/sepia over multiple segments.

```python
# Computed once before the loop from the resized start_image:
ref_mean = current_start_image.float().mean(dim=(0,1,2), keepdim=True)
ref_std  = current_start_image.float().std(dim=(0,1,2),  keepdim=True).clamp(min=1e-5)

# Per loop, normalize anchor frame statistics back to reference:
anchor_f = raw_anchor.float()
a_mean   = anchor_f.mean(dim=(0,1,2), keepdim=True)
a_std    = anchor_f.std(dim=(0,1,2),  keepdim=True).clamp(min=1e-5)
anchor_normalized = ((anchor_f - a_mean) / a_std) * ref_std + ref_mean
anchor_normalized = anchor_normalized.clamp(0.0, 1.0).to(raw_anchor.dtype)
current_start_image = anchor_normalized  # feeds next loop's WanImageToVideo
```

### 8. Segment trimming (prevents transition time-jump)
Without trimming: segment 1 saves frames 0–48, but segment 2 starts from frame 44's content → visible ~5-frame backward jump at the cut.
With trimming: segment 1 saves only frames 0–44, segment 2 continues cleanly.

```python
decoded_trimmed = decoded[:anchor_frame_idx + 1]
```

### 9. Overlap stitching
```python
overlap_result = _ImageBatchExtend.imagesfrombatch(
    source_images=prev_decoded, new_images=decoded_trimmed,
    overlap=overlap, overlap_side="source", overlap_mode="ease_in_out",
)
segment_frames = overlap_result[0]
```
Skip if `loop_idx == 0` or `overlap == 0` or `prev_decoded is None`.

### 10. Save segment
```python
seg_path = os.path.join(segment_dir, f"segment_{loop_id:03d}.pt")
torch.save(segment_frames.cpu(), seg_path)
```
`prev_decoded = decoded_trimmed` (trimmed, for next loop's overlap source)

### 11. Cleanup
```python
comfy.model_management.soft_empty_cache()
torch.cuda.empty_cache()
gc.collect()
```

### Final assembly
```python
for sp in segment_paths:
    seg = torch.load(sp, map_location="cpu", weights_only=True)
    full_video = seg if full_video is None else torch.cat([full_video, seg], dim=0)
shutil.rmtree(segment_dir)
```

---

## External Dependencies (loaded at module import time)

| Object | Source | How loaded |
|--------|--------|-----------|
| `_IAMCCS_WanImageMotion` | `IAMCCS-nodes/iamccs_wan_svipro_motion.py` | `_load_class()` — kept for SVILoopPrep/Finish nodes but NOT used in SVILooperKSA |
| `_GetImageRange` | `comfyui-kjnodes/nodes/image_nodes.py` | `_load_package_class()` |
| `_ImageBatchExtend` | `comfyui-kjnodes/nodes/image_nodes.py` | `_load_package_class()` |
| `_CKSampler`, `_CKChain` | `RES4LYF/beta/samplers.py` | `_load_clownshark()` with lazy-load fallback — used only by SVILooperCK |
| `WanImageToVideo` | `comfy_extras.nodes_wan` | Imported inside `generate()` via `from comfy_extras.nodes_wan import WanImageToVideo` |
| `nodes.KSamplerAdvanced` | ComfyUI core | `import nodes as _comfy_nodes` at top of file |
| `nodes.CLIPVisionEncode` | ComfyUI core | Same `_comfy_nodes` |

---

## Recommended Graph Wiring for SVILooperKSA

```
UnetLoaderGGUF ──→ LoraLoaderModelOnly (SVI lora) ──→ ModelSamplingSD3 ──→ model_high (global)
                                                     ├──→ LoraLoaderModelOnly (seg1 lora) ──→ model_high_1
                                                     ├──→ LoraLoaderModelOnly (seg2 lora) ──→ model_high_2
                                                     └──→ LoraLoaderModelOnly (seg3 lora) ──→ model_high_3

UnetLoaderGGUF ──→ LoraLoaderModelOnly (SVI lora) ──→ ModelSamplingSD3 ──→ model_low  (global)
                                                     ├──→ LoraLoaderModelOnly (seg1 lora) ──→ model_low_1
                                                     └──→ ...

CLIPLoader           ──→ clip
VAELoader            ──→ vae
CLIPVisionModelLoader ──→ clip_vision
LoadImage            ──→ start_image
CLIPTextEncode       ──→ positive
CLIPTextEncode       ──→ negative
```

**SVI LoRAs** (constant across all segments) are applied before `ModelSamplingSD3` in the graph.
**Per-segment creative LoRAs** go into `model_high_N` / `model_low_N` optional inputs.

---

## Important Rules

1. **Never use IAMCCS_WanImageMotion inside SVILooperKSA** — it produces beige/uniform output when called from Python. Use WanImageToVideo + CLIPVisionEncode only.
2. Always edit dev copy first, then copy to install. Never edit install directly.
3. ComfyUI must be restarted after any `nodes_svi.py` change.
4. If the node's INPUT_TYPES schema changes, the old node instance in a workflow must be deleted and re-added fresh (stale widget_values cause validation errors).
5. `beta57` scheduler is available as a string at runtime via RES4LYF monkey-patch.
6. Keep all `print()` logging at each major step for console visibility.
7. The `_comfy_nodes` alias (`import nodes as _comfy_nodes`) is used to avoid shadowing ComfyUI's `nodes` module.
