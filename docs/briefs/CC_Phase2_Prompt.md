# Phase 2 Build: SVILooperNative

## Context
You are working on ComfyUI custom nodes in `E:\AI_Studio\comfy-node-dev\ComfyUI-SVILooper\`.
The install target is `E:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-SVILooper\`.

**Do NOT modify** `nodes.py`, `nodes_highlow.py`, `nodes_svi.py`, or `nodes_looper_v2.py`. Create a new file: `nodes_svi_v2.py`.

After creating the file, update `__init__.py` to import from `nodes_svi_v2` alongside the existing imports.

## Reference files to read first

Before writing any code, read these files to understand the patterns:
1. `nodes_looper_v2.py` — Phase 1 looper (overlap stitching, anchor extraction, color correction, segment save/load, console logging). Reuse all of this logic.
2. `nodes_highlow.py` — the `_load_lora` helper pattern.

## What this node does

A multi-segment SVI (Sequential Video Inference) looper using WAN 2.2 dual high/low noise models. Unlike Phase 1's I2VLooperV2 which uses standard WanImageToVideo conditioning, this node uses WanImageToVideoSVIPro from KJNodes which passes `prev_samples` (latent from the previous segment) to give the model temporal continuity. This eliminates the ghosting at segment transitions that Phase 1 suffers from.

The sampling pipeline uses SamplerCustomAdvanced with SplitSigmas and ScheduledCFGGuidance guiders — matching the architecture of the proven native SVI workflow.

## Node architecture: 2 classes

### 1. `LoopConfigSVI` (companion config node)

**Registered as:** `"LoopConfigSVI"` → display name `"SVI Loop Config"`
**Category:** `"video/svi_looper_v2"`

Each instance represents one segment's settings.

**Required inputs:**
- `prompt` — STRING, multiline, default ""
- `frames` — INT, default 49, min 1, max 200

**Optional inputs:**
- `model_high` — MODEL (overrides global model_high for this segment — user wires any LoRA stack they want upstream)
- `model_low` — MODEL (overrides global model_low for this segment)

**Output:** `("SVI_LOOP_CONFIG",)` named `"loop_config"`

**FUNCTION:** `build` — returns a dict:
```python
{
    "prompt": prompt,
    "frames": frames,
    "model_high": model_high,  # None if not connected
    "model_low": model_low,    # None if not connected
}
```

### 2. `SVILooperNative` (main looper node)

**Registered as:** `"SVILooperNative"` → display name `"SVI Looper Native"`
**Category:** `"video/svi_looper_v2"`

**Required inputs:**
- `model_high` — MODEL (global high-noise model, SVI Pro + LightX2V + shift applied upstream)
- `model_low` — MODEL (global low-noise model)
- `vae` — VAE
- `clip` — CLIP
- `clip_vision` — CLIP_VISION
- `start_image` — IMAGE
- `width` — INT, default 480, min 16, max 4096, step 16
- `height` — INT, default 640, min 16, max 4096, step 16
- `steps` — INT, default 8, min 1, max 100
- `split_step` — INT, default 3, min 1, max 50, tooltip "Sigma split point — high model runs steps 0 to split_step, low model runs split_step to end"
- `cfg` — FLOAT, default 1.0, min 0.0, max 30.0, step 0.1
- `sampler_name` — combo: `comfy.samplers.KSampler.SAMPLERS`
- `scheduler` — combo: `comfy.samplers.KSampler.SCHEDULERS`
- `seed` — INT, default 0, min 0, max 0xffffffffffffffff
- `overlap` — INT, default 3, min 0, max 20, step 1
- `anchor_frame_offset` — INT, default -5, min -50, max -1, step 1
- `overlap_mode` — combo: ["linear_blend", "ease_in_out", "filmic_crossfade", "cut"], default "ease_in_out"
- `overlap_side` — combo: ["source", "new_images"], default "source"
- `motion_latent_count` — INT, default 1, min 0, max 4, tooltip "Temporal latent slots from prev segment to carry forward for SVI continuity. 0 = no motion context."
- `color_correction` — BOOLEAN, default True, tooltip "Normalize anchor frame color statistics to match original start image"

**Optional inputs:**
- `positive_prompt` — STRING, forceInput True, tooltip "Global fallback positive prompt"
- `negative_prompt` — STRING, forceInput True, tooltip "Global negative prompt"
- `loop_1` through `loop_10` — SVI_LOOP_CONFIG, each optional

**Outputs:**
- `full_video` — IMAGE
- `last_anchor` — IMAGE
- `used_prompts` — STRING

**FUNCTION:** `generate`

## Imports

At the top of `nodes_svi_v2.py`, use this import strategy:

```python
import torch
import gc
import os
import tempfile
import shutil

import comfy.model_management
import comfy.utils
import comfy.samplers
import comfy.sd
import comfy.latent_formats
import folder_paths
import nodes as _comfy_nodes
import node_helpers

# ComfyUI core custom sampler components
from comfy_extras.nodes_custom_sampler import (
    Noise_RandomNoise,
    Noise_EmptyNoise,
    BasicScheduler,
    SplitSigmas,
    SamplerCustomAdvanced,
    KSamplerSelect,
)

# KJNodes imports (required dependency)
_WanImageToVideoSVIPro = None
_ScheduledCFGGuidance = None

def _load_kjnodes():
    global _WanImageToVideoSVIPro, _ScheduledCFGGuidance
    if _WanImageToVideoSVIPro is not None:
        return True
    try:
        # Try direct import first
        from comfyui_kjnodes.nodes.nodes import WanImageToVideoSVIPro, ScheduledCFGGuidance
        _WanImageToVideoSVIPro = WanImageToVideoSVIPro
        _ScheduledCFGGuidance = ScheduledCFGGuidance
        return True
    except ImportError:
        pass
    try:
        # Fallback: load from file path
        import importlib.util
        nodes_path = os.path.join(
            folder_paths.get_folder_paths("custom_nodes")[0] if hasattr(folder_paths, "get_folder_paths") else os.path.join(os.path.dirname(os.path.dirname(__file__))),
            "comfyui-kjnodes", "nodes", "nodes.py"
        )
        if not os.path.exists(nodes_path):
            # Try alternate path
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            nodes_path = os.path.join(base, "comfyui-kjnodes", "nodes", "nodes.py")
        spec = importlib.util.spec_from_file_location("kjnodes_nodes", nodes_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _WanImageToVideoSVIPro = getattr(mod, "WanImageToVideoSVIPro")
        _ScheduledCFGGuidance = getattr(mod, "ScheduledCFGGuidance")
        return True
    except Exception as e:
        print(f"[SVILooperNative] ERROR: Could not load KJNodes dependencies: {e}")
        return False
```

**IMPORTANT:** Before using `_WanImageToVideoSVIPro` or `_ScheduledCFGGuidance`, always check they loaded. The `generate()` method should call `_load_kjnodes()` at the start and raise a clear error if it fails.

**ALSO IMPORTANT:** The exact class names for the noise nodes in `comfy_extras.nodes_custom_sampler` may differ from what I've listed. Read the actual file at `E:\ComfyUI\ComfyUI\comfy_extras\nodes_custom_sampler.py` to verify the correct class names for:
- The random noise class (might be `Noise_RandomNoise` or `RandomNoise`)
- The empty/disable noise class (might be `Noise_EmptyNoise` or `DisableNoise`)
- `BasicScheduler`
- `SplitSigmas`
- `SamplerCustomAdvanced`
- `KSamplerSelect`

Check the actual method signatures by reading the source — don't assume parameter names.

Similarly, verify `ScheduledCFGGuidance` method signature by reading the KJNodes source at `E:\ComfyUI\ComfyUI\custom_nodes\comfyui-kjnodes\nodes\nodes.py`. The confirmed signature is:
```python
get_guider(self, model, cfg, positive, negative, start_percent, end_percent) -> (GUIDER,)
```

And verify `WanImageToVideoSVIPro` execute signature. The confirmed signature is:
```python
execute(cls, positive, negative, length, motion_latent_count, anchor_samples, prev_samples=None)
```
Return is `(positive, negative, latent_dict)`.

## Internal flow of `generate()`

### Setup
1. Call `_load_kjnodes()` — raise RuntimeError if it fails.
2. Count connected loop configs. If 0, return black frame + empty string.
3. Resize `start_image` to `width × height` using `comfy.utils.common_upscale`.
4. If `color_correction` is True, compute reference stats:
   ```python
   ref_mean = current_start_image.float().mean(dim=(0,1,2), keepdim=True)
   ref_std = current_start_image.float().std(dim=(0,1,2), keepdim=True).clamp(min=1e-5)
   ```
5. Create temp directory. Set `current_seed = seed`.
6. Initialize `prev_latent_samples = None` — this carries latent context between segments.

### Per-segment loop

For each connected `LoopConfigSVI`:

**a) Extract config:**
```python
loop_prompt = config["prompt"].strip() or positive_prompt or ""
loop_frames = config["frames"] if config["frames"] > 0 else 49
high_model = config.get("model_high") or model_high
low_model = config.get("model_low") or model_low
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

**c) CLIP Vision encode (still needed for WanImageToVideoSVIPro context):**
```python
cv_result = _comfy_nodes.CLIPVisionEncode().encode(clip_vision, current_start_image, "none")
cv_out = cv_result[0]
```

Wait — actually, check whether `WanImageToVideoSVIPro` uses clip_vision_output. Looking at the node interface (Image 2 from earlier), the inputs are: positive, negative, anchor_samples, prev_samples, length, motion_latent_count. There is NO clip_vision input. So CLIP Vision encoding may not be needed for the SVI path. However, the positive conditioning from CLIPTextEncode might need clip_vision_output injected separately — check how your native SVI workflow handles this.

In your native workflow, the positive conditioning comes from a CLIPTextEncode connected to WanImageToVideoSVIPro's positive input. There's no separate CLIPVisionEncode node in the SVI conditioning chain — the SVI approach uses the VAE-encoded anchor_samples instead of CLIP vision for image conditioning.

**So skip CLIP Vision encoding entirely.** The SVI Pro conditioning path does not use it. The image information comes through `anchor_samples` (VAE-encoded latent).

**c) VAE encode start image → anchor_samples:**
```python
print(f"[SVILooperNative] Segment {loop_id} | VAE encode anchor...")
# Need to encode the start image to latent space for SVI conditioning
# The image needs to be in BCHW format for VAE encode
img_for_vae = current_start_image  # shape [1, H, W, 3]
anchor_latent = vae.encode(img_for_vae[:, :, :, :3])  # returns [B, C, T, H, W] or [B, C, H, W]
# Ensure 5D for video latent format: [B, C, T, H, W]
if anchor_latent.ndim == 4:
    anchor_latent = anchor_latent.unsqueeze(2)  # add temporal dim
anchor_samples = {"samples": anchor_latent}
```

**IMPORTANT:** The VAE encode approach above may not produce the correct format. Check how your native SVI workflow encodes the anchor — it uses a `VAEEncode` node which takes `pixels` and `vae`. The ComfyUI VAEEncode node does:
```python
pixels = pixels[:,:,:,:3]  # drop alpha
t = vae.encode(pixels)
return ({"samples": t},)
```
Use the same pattern. The VAE's `.encode()` method handles the format conversion internally.

**d) SVI conditioning:**
```python
print(f"[SVILooperNative] Segment {loop_id} | SVI conditioning (prev_samples={'yes' if prev_latent_samples else 'None'}, motion={motion_latent_count})")

svipro = _WanImageToVideoSVIPro()
svipro_result = svipro.execute(
    positive=c_pos,
    negative=c_neg,
    length=loop_frames,
    motion_latent_count=motion_latent_count,
    anchor_samples=anchor_samples,
    prev_samples=prev_latent_samples,  # None for first segment
)
pos_cond = svipro_result[0]
neg_cond = svipro_result[1]
latent_dict = svipro_result[2]
```

**e) Build sigma schedule and split:**
```python
split_percent = split_step / steps
print(f"[SVILooperNative] Segment {loop_id} | Sigmas: {steps} total → split at {split_step} (high={split_step}, low={steps - split_step})")

scheduler_node = BasicScheduler()
sigmas_full = scheduler_node.get_sigmas(high_model, scheduler, steps, 1.0)
# sigmas_full is a tuple, first element is the sigmas tensor

splitter = SplitSigmas()
sigmas_split = splitter.get_sigmas(sigmas_full[0], split_step)
sigmas_high = sigmas_split[0]  # first portion
sigmas_low = sigmas_split[1]   # second portion
```

**f) Build sampler object:**
```python
sampler_select = KSamplerSelect()
sampler_obj = sampler_select.get_sampler(sampler_name)
# sampler_obj is a tuple, first element is the sampler
```

**g) Build noise objects:**
```python
noise_gen = Noise_RandomNoise()
noise = noise_gen.get_noise(current_seed)
# noise is a tuple, first element is the noise object

empty_noise_gen = Noise_EmptyNoise()
no_noise = empty_noise_gen.get_noise()
# no_noise is a tuple, first element is the empty noise object
```

**h) Build guiders:**
```python
guider_builder = _ScheduledCFGGuidance()

guider_high_result = guider_builder.get_guider(
    model=high_model,
    cfg=cfg,
    positive=pos_cond,
    negative=neg_cond,
    start_percent=0.0,
    end_percent=split_percent,
)
guider_high = guider_high_result[0]

guider_low_result = guider_builder.get_guider(
    model=low_model,
    cfg=cfg,
    positive=pos_cond,
    negative=neg_cond,
    start_percent=split_percent,
    end_percent=1.0,
)
guider_low = guider_low_result[0]
```

**IMPORTANT:** Verify the exact return types by reading the ScheduledCFGGuidance source. It likely returns a tuple where [0] is the GUIDER object.

**i) Two-pass SamplerCustomAdvanced sampling:**
```python
sampler_adv = SamplerCustomAdvanced()

# Pass 1: High-noise
print(f"[SVILooperNative] Segment {loop_id} | ▶ HIGH NOISE PASS ({split_step} steps)")
high_result = sampler_adv.sample(
    noise[0],       # noise object
    guider_high,    # guider
    sampler_obj[0], # sampler
    sigmas_high,    # sigmas
    latent_dict,    # latent
)
# high_result is a tuple: (output_latent, denoised_latent)

comfy.model_management.soft_empty_cache()
torch.cuda.empty_cache()
gc.collect()

# Pass 2: Low-noise
print(f"[SVILooperNative] Segment {loop_id} | ▶ LOW NOISE PASS ({steps - split_step} steps)")
low_result = sampler_adv.sample(
    no_noise[0],                                    # no noise for second pass
    guider_low,                                     # guider
    sampler_obj[0],                                 # sampler
    sigmas_low,                                     # sigmas
    {"samples": high_result[0]["samples"]},          # latent from high pass
)
```

**IMPORTANT:** The exact return format of `SamplerCustomAdvanced.sample()` needs verification. Read the source at `E:\ComfyUI\ComfyUI\comfy_extras\nodes_custom_sampler.py` to confirm. It likely returns `({"samples": tensor}, {"samples": denoised_tensor})` or similar.

**j) Save prev_samples for next segment's SVI conditioning:**
```python
prev_latent_samples = {"samples": low_result[0]["samples"].clone()}
```

**k) VAE decode:**
```python
sampled = low_result[0]["samples"]
print(f"[SVILooperNative] Segment {loop_id} | VAE decode...")
decoded = vae.decode(sampled)
while decoded.ndim > 4:
    decoded = decoded.squeeze(0)
if decoded.shape[1] != height or decoded.shape[2] != width:
    decoded = comfy.utils.common_upscale(
        decoded.movedim(-1, 1), width, height, "bilinear", "center"
    ).movedim(1, -1)
print(f"[SVILooperNative] Segment {loop_id} | VAE decode → {decoded.shape[0]} frames")
```

**l) Anchor frame extraction:**
```python
actual_frames = decoded.shape[0]
anchor_idx = max(0, min(actual_frames - 1, actual_frames + anchor_frame_offset))
raw_anchor = decoded[anchor_idx:anchor_idx+1].clone()
print(f"[SVILooperNative] Segment {loop_id} | Anchor at frame {anchor_idx}")
```

**m) Color drift correction (if enabled):**
```python
if color_correction:
    anchor_f = raw_anchor.float()
    a_mean = anchor_f.mean(dim=(0,1,2), keepdim=True)
    a_std = anchor_f.std(dim=(0,1,2), keepdim=True).clamp(min=1e-5)
    anchor_normalized = ((anchor_f - a_mean) / a_std) * ref_std + ref_mean
    anchor_normalized = anchor_normalized.clamp(0.0, 1.0).to(raw_anchor.dtype)
    current_start_image = anchor_normalized
    print(f"[SVILooperNative] Segment {loop_id} | Color corrected (mean {a_mean.mean():.3f}→{ref_mean.mean():.3f})")
else:
    current_start_image = raw_anchor
```

**n) Trim, overlap stitch, save — identical to Phase 1:**

Copy the trimming, overlap stitching (with all 4 blend modes and overlap_side support), and segment save logic from `nodes_looper_v2.py`. This includes:
- `decoded_trimmed = decoded[:anchor_idx + 1]`
- Overlap blending with the previous segment (re-save prev segment with tail removed)
- Save current segment to temp .pt file
- Update `prev_decoded = decoded_trimmed`

**o) Cleanup per segment:**
```python
del decoded, sampled
comfy.model_management.soft_empty_cache()
torch.cuda.empty_cache()
gc.collect()
current_seed += 1
```

### Final assembly

Same as Phase 1 — load all .pt segments, concatenate, clean up temp dir.

```python
full_video = None
for seg_path in segment_paths:
    seg = torch.load(seg_path, map_location="cpu", weights_only=True)
    full_video = seg if full_video is None else torch.cat([full_video, seg], dim=0)
    del seg
    gc.collect()

shutil.rmtree(segment_dir, ignore_errors=True)

last_anchor = current_start_image
if full_video is None:
    full_video = torch.zeros((1, height, width, 3))

print(f"[SVILooperNative] Done. Total frames: {full_video.shape[0]}")
return (full_video, last_anchor, "\n".join(used_prompts_log))
```

## Console logging format

Use this exact prefix and format for all prints:
```
[SVILooperNative] ═══ Segment 1/3 ═══════════════════
[SVILooperNative] Segment 1 | Prompt: "a woman walks through..."
[SVILooperNative] Segment 1 | 49 frames, seed=12345
[SVILooperNative] Segment 1 | VAE encode anchor...
[SVILooperNative] Segment 1 | SVI conditioning (prev_samples=None, motion=1)
[SVILooperNative] Segment 1 | Sigmas: 8 total → split at 3 (high=3, low=5)
[SVILooperNative] Segment 1 | ▶ HIGH NOISE PASS (3 steps)
[SVILooperNative] Segment 1 | ▶ LOW NOISE PASS (5 steps)
[SVILooperNative] Segment 1 | VAE decode → 49 frames
[SVILooperNative] Segment 1 | Anchor at frame 44
[SVILooperNative] Segment 1 | Color corrected (mean 0.360→0.427)
[SVILooperNative] Segment 1 | Trimmed to 45 frames
[SVILooperNative] Segment 1 | Saved → 45 frames
```

## NODE_CLASS_MAPPINGS

```python
NODE_CLASS_MAPPINGS = {
    "LoopConfigSVI": LoopConfigSVI,
    "SVILooperNative": SVILooperNative,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoopConfigSVI": "SVI Loop Config",
    "SVILooperNative": "SVI Looper Native",
}
```

## __init__.py update

Add:
```python
from .nodes_svi_v2 import NODE_CLASS_MAPPINGS as _NCM_SVIV2, NODE_DISPLAY_NAME_MAPPINGS as _NDM_SVIV2
```

Merge into combined dicts:
```python
NODE_CLASS_MAPPINGS = {**_NCM_BASE, **_NCM_HL, **_NCM_SVI, **_NCM_V2, **_NCM_SVIV2}
NODE_DISPLAY_NAME_MAPPINGS = {**_NDM_BASE, **_NDM_HL, **_NDM_SVI, **_NDM_V2, **_NDM_SVIV2}
```

## Critical implementation notes

1. **Read the actual source files** for `comfy_extras/nodes_custom_sampler.py` and `comfyui-kjnodes/nodes/nodes.py` before writing any code. Verify all class names, method names, parameter names, and return types. Do not assume — check.

2. **WanImageToVideoSVIPro does NOT use CLIP Vision.** Image conditioning comes through `anchor_samples` (VAE-encoded latent). Do not add CLIPVisionEncode to the SVI conditioning path.

3. **The `clip_vision` input is still on the node** for forward compatibility (Phase 3 may need it for IAMCCS WanImageMotion). For now it's accepted but not used in the SVI path. Add a comment noting this.

4. **prev_latent_samples** must persist across loop iterations. Initialize to `None` before the loop. After each segment's sampling, update it with the cloned latent output. Pass it to `WanImageToVideoSVIPro` for segments 2+.

5. **The VAE encode for anchor_samples** needs to produce a video-format latent [B, C, T, H, W]. A single image VAE encode may produce [B, C, H, W]. Check what format `WanImageToVideoSVIPro` expects and add an unsqueeze if needed.

## After creating the file

1. Show me a summary of what was created
2. Copy `nodes_svi_v2.py` and updated `__init__.py` to the install target
3. Verify with grep that both files are in the install location
4. List any assumptions you made about method signatures that you verified from source
