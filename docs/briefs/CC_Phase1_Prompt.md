# Phase 1 Build: I2VLooperHL_v2

## Context
You are working on ComfyUI custom nodes in `E:\AI_Studio\comfy-node-dev\ComfyUI-SVILooper\`.
The install target is `E:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-SVILooper\`.

**Do NOT modify** `nodes.py`, `nodes_highlow.py`, or `nodes_svi.py`. Create a new file: `nodes_looper_v2.py`.

After creating the file, update `__init__.py` to import from `nodes_looper_v2` alongside the existing imports.

## What this node does

A multi-segment I2V video looper using dual high-noise/low-noise WAN 2.2 models with KSamplerAdvanced two-pass sampling. Each segment uses the anchor frame from the previous segment as its start image. Segments are stitched with overlap blending to eliminate hard cuts.

## Reference: Working sampling code

The sampling approach in `nodes_highlow.py` → `I2VLooperHL.generate()` is **proven working**. The new node reuses that exact sampling logic (WanImageToVideo + KSamplerAdvanced two-pass). Read `nodes_highlow.py` to understand the sampling pattern. Do not change the sampling approach.

## Node architecture: 2 classes

### 1. `LoopConfigV2` (companion config node)

**Registered as:** `"LoopConfigV2"` → display name `"Loop Config v2"`
**Category:** `"video/looper_v2"`

Each instance represents one segment's settings. Users connect 1–10 of these to the main looper node.

**Required inputs:**
- `prompt` — STRING, multiline, default ""
- `frames` — INT, default 49, min 1, max 200

**Optional inputs:**
- `lora_high` — combo from `folder_paths.get_filename_list("loras")` with "None" prepended, default "None"
- `lora_high_strength` — FLOAT, default 1.0, min 0.0, max 2.0, step 0.05
- `lora_low` — combo from loras list with "None" prepended, default "None"
- `lora_low_strength` — FLOAT, default 1.0, min 0.0, max 2.0, step 0.05

**Output:** `("LOOP_CONFIG_V2",)` named `"loop_config"`

**FUNCTION:** `build` — returns a dict with all the above values.

### 2. `I2VLooperV2` (main looper node)

**Registered as:** `"I2VLooperV2"` → display name `"I2V Looper v2"`
**Category:** `"video/looper_v2"`

**Required inputs:**
- `model_high` — MODEL
- `model_low` — MODEL
- `vae` — VAE
- `clip` — CLIP
- `clip_vision` — CLIP_VISION
- `start_image` — IMAGE
- `width` — INT, default 480, min 16, max 4096, step 16
- `height` — INT, default 640, min 16, max 4096, step 16
- `steps` — INT, default 8, min 1, max 100
- `split_step` — INT, default 3, min 1, max 50, tooltip "Step where high-noise model hands off to low-noise model"
- `cfg` — FLOAT, default 1.0, min 0.0, max 30.0, step 0.1
- `sampler_name` — combo: `comfy.samplers.KSampler.SAMPLERS`
- `scheduler` — combo: `comfy.samplers.KSampler.SCHEDULERS`
- `seed` — INT, default 0, min 0, max 0xffffffffffffffff
- `overlap` — INT, default 5, min 0, max 20, step 1, tooltip "Overlap frames for blending between segments (0 = hard cut)"
- `anchor_frame_offset` — INT, default -5, min -50, max -1, step 1, tooltip "Frames from end to extract anchor. -5 = 5th from last"

**Optional inputs:**
- `positive_prompt` — STRING, forceInput True, tooltip "Global fallback positive prompt"
- `negative_prompt` — STRING, forceInput True, tooltip "Global negative prompt"
- `loop_1` through `loop_10` — LOOP_CONFIG_V2, each optional

**Outputs:**
- `full_video` — IMAGE (all segments concatenated with overlap blending)
- `last_anchor` — IMAGE (the anchor frame from the final segment)
- `used_prompts` — STRING (log of prompts used per segment)

**FUNCTION:** `generate`

## Internal flow of `generate()`

### Setup
1. Count connected loop configs (skip None). If 0, return a single black frame and empty string.
2. Resize `start_image` to `width × height` using `comfy.utils.common_upscale` (bilinear, center crop). Store as `current_start_image`.
3. Compute reference color statistics from `current_start_image` for drift correction:
   ```python
   ref_mean = current_start_image.float().mean(dim=(0,1,2), keepdim=True)
   ref_std = current_start_image.float().std(dim=(0,1,2), keepdim=True).clamp(min=1e-5)
   ```
4. Create a temp directory for segment .pt files.
5. Set `current_seed = seed`.

### Per-segment loop (iterate over loop_1..loop_10, skip None)

For each connected `LoopConfigV2`:

**a) Extract config:**
```python
loop_prompt = config["prompt"].strip() or positive_prompt or ""
loop_frames = config["frames"] if config["frames"] > 0 else 49
lora_high_name = config.get("lora_high")  # None if "None"
lora_high_str = config.get("lora_high_strength", 1.0)
lora_low_name = config.get("lora_low")
lora_low_str = config.get("lora_low_strength", 1.0)
```

**b) Encode text conditioning:**
```python
tokens_pos = clip.tokenize(loop_prompt)
cond_pos, pooled_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True)
c_pos = [[cond_pos, {"pooled_output": pooled_pos}]]

tokens_neg = clip.tokenize(negative_prompt)
cond_neg, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)
c_neg = [[cond_neg, {"pooled_output": pooled_neg}]]
```

**c) CLIP Vision encode + WanImageToVideo conditioning:**
```python
from comfy_extras.nodes_wan import WanImageToVideo

cv_result = nodes.CLIPVisionEncode().encode(clip_vision, current_start_image, "none")
cv_out = cv_result[0]

i2v_result = WanImageToVideo().execute(
    positive=c_pos, negative=c_neg, vae=vae,
    width=width, height=height, length=loop_frames, batch_size=1,
    start_image=current_start_image, clip_vision_output=cv_out,
)
pos_cond, neg_cond, latent_dict = i2v_result[0], i2v_result[1], i2v_result[2]
```

**d) Apply per-segment LoRAs (if specified):**
```python
high_model = model_high
low_model = model_low

if lora_high_name:
    high_model, _ = self._load_lora(model_high, clip, lora_high_name, lora_high_str)
if lora_low_name:
    low_model, _ = self._load_lora(model_low, clip, lora_low_name, lora_low_str)
```
Copy the `_load_lora` helper from `I2VLooperHL` in `nodes_highlow.py`.

**e) Two-pass KSamplerAdvanced sampling:**
```python
ksampler = nodes.KSamplerAdvanced()

# Pass 1: High-noise model
high_out = ksampler.sample(
    model=high_model, add_noise="enable", noise_seed=current_seed,
    steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
    positive=pos_cond, negative=neg_cond, latent_image=latent_dict,
    start_at_step=0, end_at_step=split_step, return_with_leftover_noise="enable",
)

# Cache cleanup between passes
comfy.model_management.soft_empty_cache()
torch.cuda.empty_cache()
gc.collect()

# Pass 2: Low-noise model
low_out = ksampler.sample(
    model=low_model, add_noise="disable", noise_seed=0,
    steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
    positive=pos_cond, negative=neg_cond,
    latent_image={"samples": high_out[0]["samples"]},
    start_at_step=split_step, end_at_step=1000, return_with_leftover_noise="disable",
)
sampled = low_out[0]["samples"]
```

**f) VAE decode:**
```python
decoded = vae.decode(sampled)
while decoded.ndim > 4:
    decoded = decoded.squeeze(0)
if decoded.shape[1] != height or decoded.shape[2] != width:
    decoded = comfy.utils.common_upscale(
        decoded.movedim(-1, 1), width, height, "bilinear", "center"
    ).movedim(1, -1)
```

**g) Anchor frame extraction:**
```python
actual_frames = decoded.shape[0]
anchor_idx = max(0, min(actual_frames - 1, actual_frames + anchor_frame_offset))
raw_anchor = decoded[anchor_idx:anchor_idx+1].clone()
```

**h) Color drift correction on anchor:**
```python
anchor_f = raw_anchor.float()
a_mean = anchor_f.mean(dim=(0,1,2), keepdim=True)
a_std = anchor_f.std(dim=(0,1,2), keepdim=True).clamp(min=1e-5)
anchor_normalized = ((anchor_f - a_mean) / a_std) * ref_std + ref_mean
anchor_normalized = anchor_normalized.clamp(0.0, 1.0).to(raw_anchor.dtype)
current_start_image = anchor_normalized  # feeds next segment
```

**i) Trim segment to anchor point:**
```python
decoded_trimmed = decoded[:anchor_idx + 1]
```

**j) Overlap stitching (segments 2+):**

If `loop_idx > 0` and `overlap > 0` and `prev_decoded is not None`:

Implement the overlap blend inline (do NOT import from KJNodes — we want zero external dependencies for this logic). Use `ease_in_out` blending:

```python
prefix = prev_decoded[:-overlap]
blend_src = prev_decoded[-overlap:]
blend_dst = decoded_trimmed[:overlap]
suffix = decoded_trimmed[overlap:]

t = torch.linspace(0, 1, overlap + 2, device=blend_src.device, dtype=blend_src.dtype)[1:-1]
eased_t = 3 * t * t - 2 * t * t * t
eased_t = eased_t.view(-1, 1, 1, 1)
blended = (1 - eased_t) * blend_src + eased_t * blend_dst

segment_frames = torch.cat([prefix, blended, suffix], dim=0)
```

If first segment or overlap=0: `segment_frames = decoded_trimmed`

**k) Save segment to temp file:**
```python
seg_path = os.path.join(segment_dir, f"segment_{loop_id:03d}.pt")
torch.save(segment_frames.cpu(), seg_path)
segment_paths.append(seg_path)
```

**l) Update state for next iteration:**
```python
prev_decoded = decoded_trimmed  # NOT segment_frames — the un-blended trimmed frames
current_seed += 1
```

**m) Cleanup:**
```python
if lora_high_name and high_model is not model_high:
    del high_model
if lora_low_name and low_model is not model_low:
    del low_model
del decoded, sampled
comfy.model_management.soft_empty_cache()
torch.cuda.empty_cache()
gc.collect()
```

### Final assembly

```python
full_video = None
for seg_path in segment_paths:
    seg = torch.load(seg_path, map_location="cpu", weights_only=True)
    full_video = seg if full_video is None else torch.cat([full_video, seg], dim=0)
    del seg
    gc.collect()

shutil.rmtree(segment_dir, ignore_errors=True)

last_anchor = current_start_image  # the final anchor frame
```

Return `(full_video, last_anchor, "\n".join(used_prompts_log))`

## Console logging

Print clear progress at each step, prefixed with `[I2VLooperV2]`:
```
[I2VLooperV2] ── Loop 1/3 ──────────────────────────
[I2VLooperV2] Loop 1: prompt = "a woman walks..."
[I2VLooperV2] Loop 1: 49 frames, seed=12345
[I2VLooperV2] Loop 1: CLIP Vision encoding...
[I2VLooperV2] Loop 1: WanImageToVideo conditioning...
[I2VLooperV2] Loop 1: sampling HIGH (steps 0-3)
[I2VLooperV2] Loop 1: sampling LOW (steps 3-end)
[I2VLooperV2] Loop 1: VAE decode → 49 frames
[I2VLooperV2] Loop 1: anchor at frame 44
[I2VLooperV2] Loop 1: color-normalized (mean 0.360→0.427)
[I2VLooperV2] Loop 1: trimmed to 45 frames
[I2VLooperV2] Loop 1: saved segment (45 frames)
[I2VLooperV2] Loop 2: overlap stitching (5 frames, ease_in_out)
```

## __init__.py update

Add to `__init__.py`:
```python
from .nodes_looper_v2 import NODE_CLASS_MAPPINGS as _NCM_V2, NODE_DISPLAY_NAME_MAPPINGS as _NDM_V2
```

And merge into the combined dicts:
```python
NODE_CLASS_MAPPINGS = {**_NCM_BASE, **_NCM_HL, **_NCM_SVI, **_NCM_V2}
NODE_DISPLAY_NAME_MAPPINGS = {**_NDM_BASE, **_NDM_HL, **_NDM_SVI, **_NDM_V2}
```

## NODE_CLASS_MAPPINGS for nodes_looper_v2.py

```python
NODE_CLASS_MAPPINGS = {
    "LoopConfigV2": LoopConfigV2,
    "I2VLooperV2": I2VLooperV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoopConfigV2": "Loop Config v2",
    "I2VLooperV2": "I2V Looper v2",
}
```

## After creating the file

1. Show me the full diff / file listing
2. Copy `nodes_looper_v2.py` and updated `__init__.py` to the install target at `E:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-SVILooper\`
3. Verify with grep that both files are in place
