# SVILooper Hybrid Architecture Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the monolithic `SVILooper` node with `SVILoopPrep` + `SVILoopFinish`, exposing Clownshark samplers and WaveSpeedAIPredictor as user-wired ComfyUI graph nodes between them.

**Architecture:** `SVILoopPrep` handles LoRA application, anchor/prev-sample setup, prompt encoding, and WanImageMotion conditioning — outputting conditioned latents and LoRA-patched models to the graph. `SVILoopFinish` handles VAE decode, overlap stitching, anchor frame extraction, segment disk save, LoRA cleanup, and final video assembly. State flows between loops via an `SVI_LOOP_STATE` dict. The Wavespeed-refined anchor for loop N is wired from `SVILoopFinish_N.anchor_frame → WaveSpeedAIPredictor → SVILoopPrep_N+1.refined_anchor` (non-cyclic DAG).

**Tech Stack:** Python 3.10+, PyTorch, ComfyUI node API (`comfy.sd`, `comfy.utils`, `comfy.model_management`, `folder_paths`), existing `_load_class`/`_load_package_class` helpers, IAMCCS_WanImageMotion, GetImageRangeFromBatch, ImageBatchExtendWithOverlap.

---

## Architectural Correction vs. Brief

The brief specifies `refined_anchor` as a required **input** to `SVILoopFinish` AND `anchor_frame` as an **output** of `SVILoopFinish`. This creates a cycle:

```
SVILoopFinish.anchor_frame → WaveSpeedAIPredictor → SVILoopFinish.refined_anchor
```

ComfyUI graphs are DAGs — cycles are invalid. The correct non-cyclic wiring routes the refined anchor **forward**, into the next loop's `SVILoopPrep`:

```
SVILoopFinish_N.anchor_frame → WaveSpeedAIPredictor_N → SVILoopPrep_{N+1}.refined_anchor
```

All other logic from the brief is implemented as specified.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `E:\AI_Studio\comfy-node-dev\ComfyUI-SVILooper\nodes_svi.py` | **Rewrite** | Keeps all imports/loaders; keeps `SVILoopConfig`; removes `SVILooper`; adds `SVILoopPrep`, `SVILoopFinish`; updates `NODE_CLASS_MAPPINGS` |
| `E:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-SVILooper\nodes_svi.py` | **Copy** | Deployment target |
| `E:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-SVILooper\__init__.py` | **No change** | Already merges NCM from nodes_svi dynamically |

`__init__.py` is **not modified** — it already does `from .nodes_svi import NODE_CLASS_MAPPINGS as _NCM_SVI` which will automatically pick up the new mappings.

---

## SVI_LOOP_STATE Dict Schema

`SVI_LOOP_STATE` is a plain Python dict. ComfyUI treats unknown type name strings as passthrough. No class registration needed.

```python
{
    # Anchor/sampling carry-forward
    "anchor_samples":    dict | None,   # {"samples": tensor} VAE-encoded anchor for WanImageMotion
    "prev_samples":      dict | None,   # {"samples": tensor} raw latent from Clownshark output
    "prev_decoded":      Tensor | None, # [B,H,W,C] decoded frames, source for next loop's overlap

    # Segment accumulation
    "segment_paths":     list[str],     # ordered list of .pt file paths
    "segment_dir":       str,           # tempfile.mkdtemp path

    # LoRA-patched models (for Clownshark use and cleanup)
    "model_high_patched":  object,      # LoRA-patched or original model_high
    "model_low_patched":   object,      # LoRA-patched or original model_low
    "model_high_original": object,      # original model_high reference
    "model_low_original":  object,      # original model_low reference
    "lora_high_name":      str | None,
    "lora_low_name":       str | None,

    # Loop metadata
    "loop_index":          int,
    "frames":              int,
    "anchor_frame_offset": int,
    "anchor_override":     Tensor | None,  # [B,H,W,C] or None
    "width":               int,
    "height":              int,
    "vae":                 object,

    # Prompt log (accumulated across loops)
    "used_prompts_log":    list[str],
}
```

---

## Per-Loop Graph Wiring (Reference)

### Loop 0
```
start_image → SVILoopPrep_0 (no prev_loop_state, no refined_anchor)
SVILoopConfig_0 → SVILoopPrep_0
SVILoopPrep_0.positive_out / negative_out / latent_out → ClownsharKSampler_Beta_0
SVILoopPrep_0.model_high_out → ClownsharKSampler_Beta_0
SVILoopPrep_0.model_low_out → ClownsharkChainsampler_Beta_0
SVILoopPrep_0.loop_state → SVILoopFinish_0
ClownsharKSampler_Beta_0 → ClownsharkChainsampler_Beta_0 → SVILoopFinish_0.sampled_latent
SVILoopFinish_0.anchor_frame → WaveSpeedAIPredictor_0
```

### Loop N > 0
```
SVILoopConfig_N → SVILoopPrep_N
SVILoopFinish_{N-1}.next_loop_state → SVILoopPrep_N.prev_loop_state
WaveSpeedAIPredictor_{N-1} → SVILoopPrep_N.refined_anchor
... (same sampler wiring as loop 0) ...
SVILoopFinish_N (is_final_loop=True on last loop)
```

---

## Task 1: Write SVILoopPrep Class

**Files:**
- Modify: `E:\AI_Studio\comfy-node-dev\ComfyUI-SVILooper\nodes_svi.py` (replace SVILooper class with SVILoopPrep + SVILoopFinish)

- [ ] **Step 1: Replace the SVILooper class block with SVILoopPrep**

Remove everything from `# ---------------------------------------------------------------------------\n# NODE 2: SVILooper` through the end of the `SVILooper` class (line ~251–619). Replace with the SVILoopPrep class:

```python
# ---------------------------------------------------------------------------
# NODE 2: SVILoopPrep
# ---------------------------------------------------------------------------

class SVILoopPrep:
    """
    Per-loop preparation node.
    Handles LoRA application, anchor/prev-sample setup, prompt encoding,
    and WanImageMotion conditioning. Outputs conditioned latents and
    LoRA-patched models for the user-wired Clownshark sampler pair.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_high":  ("MODEL",),
                "model_low":   ("MODEL",),
                "vae":         ("VAE",),
                "clip":        ("CLIP",),
                "start_image": ("IMAGE",),
                "positive":    ("CONDITIONING",),
                "negative":    ("CONDITIONING",),
                "width":  ("INT", {"default": 672, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 896, "min": 16, "max": 4096, "step": 16}),
                "loop_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9,
                    "tooltip": "0-based index of this loop segment",
                }),
                "loop_config": ("SVI_LOOP",),
            },
            "optional": {
                "prev_loop_state": ("SVI_LOOP_STATE",),
                "refined_anchor":  ("IMAGE", {
                    "tooltip": "Wavespeed-refined anchor from SVILoopFinish of the previous loop. "
                               "Required for loop_index > 0 unless anchor_override is set in loop_config.",
                }),
            },
        }

    RETURN_TYPES  = ("CONDITIONING", "CONDITIONING", "LATENT", "MODEL", "MODEL", "SVI_LOOP_STATE")
    RETURN_NAMES  = ("positive_out", "negative_out", "latent_out",
                     "model_high_out", "model_low_out", "loop_state")
    FUNCTION      = "prepare"
    CATEGORY      = "video/svi_looper"

    def prepare(self, model_high, model_low, vae, clip, start_image, positive, negative,
                width, height, loop_index, loop_config,
                prev_loop_state=None, refined_anchor=None):

        if _IAMCCS_WanImageMotion is None:
            raise RuntimeError("[SVILoopPrep] IAMCCS_WanImageMotion not loaded")

        # ── Extract loop_config fields ───────────────────────────────────────
        prompt             = loop_config.get("prompt", "")
        frames             = loop_config.get("frames", 49)
        lora_high_name     = loop_config.get("lora_high")
        lora_high_strength = loop_config.get("lora_high_strength", 0.8)
        lora_low_name      = loop_config.get("lora_low")
        lora_low_strength  = loop_config.get("lora_low_strength", 0.8)
        anchor_frame_offset = loop_config.get("anchor_frame_offset", -5)
        anchor_override    = loop_config.get("anchor_override")
        loop_id = loop_index + 1

        print(f"[SVILoopPrep] Loop {loop_id}: starting prepare ({frames} frames, loop_index={loop_index})")

        # ── LoRA application ─────────────────────────────────────────────────
        high_model = model_high
        low_model  = model_low

        if lora_high_name:
            lora_path = folder_paths.get_full_path("loras", lora_high_name)
            lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
            high_model, _ = comfy.sd.load_lora_for_models(
                high_model, clip, lora_data, lora_high_strength, 0.0)
            del lora_data
            print(f"[SVILoopPrep] Loop {loop_id}: applied LoRA {lora_high_name} "
                  f"to high model (strength={lora_high_strength})")

        if lora_low_name:
            lora_path = folder_paths.get_full_path("loras", lora_low_name)
            lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
            low_model, _ = comfy.sd.load_lora_for_models(
                low_model, clip, lora_data, lora_low_strength, 0.0)
            del lora_data
            print(f"[SVILoopPrep] Loop {loop_id}: applied LoRA {lora_low_name} "
                  f"to low model (strength={lora_low_strength})")

        # ── Anchor samples and state ─────────────────────────────────────────
        if loop_index == 0:
            # Loop 0: encode start_image as the initial anchor
            img_bchw = start_image.permute(0, 3, 1, 2)
            img_bchw = comfy.utils.common_upscale(img_bchw, width, height, "lanczos", "disabled")
            anchor_image = img_bchw.permute(0, 2, 3, 1)
            t = vae.encode(anchor_image)
            anchor_samples = {"samples": t}
            prev_samples   = None
            prev_decoded   = None
            segment_paths  = []
            segment_dir    = tempfile.mkdtemp(prefix="svi_looper_")
            used_prompts_log = []
            print(f"[SVILoopPrep] Loop {loop_id}: encoded start_image. segment_dir={segment_dir}")
        else:
            # Loop N>0: anchor comes from Wavespeed-refined IMAGE (or anchor_override)
            if prev_loop_state is None:
                raise RuntimeError(
                    f"[SVILoopPrep] Loop {loop_id}: prev_loop_state required for loop_index > 0")
            prev_samples     = prev_loop_state["prev_samples"]
            prev_decoded     = prev_loop_state["prev_decoded"]
            segment_paths    = prev_loop_state["segment_paths"]
            segment_dir      = prev_loop_state["segment_dir"]
            used_prompts_log = list(prev_loop_state.get("used_prompts_log", []))

            if anchor_override is not None:
                # anchor_override from loop_config takes priority over refined_anchor
                ao_bchw = anchor_override.permute(0, 3, 1, 2)
                ao_bchw = comfy.utils.common_upscale(ao_bchw, width, height, "lanczos", "disabled")
                anchor_image = ao_bchw.permute(0, 2, 3, 1)
                t = vae.encode(anchor_image)
                anchor_samples = {"samples": t}
                print(f"[SVILoopPrep] Loop {loop_id}: using anchor_override from loop_config")
            elif refined_anchor is not None:
                # Wavespeed output from previous SVILoopFinish
                t = vae.encode(refined_anchor)
                anchor_samples = {"samples": t}
                print(f"[SVILoopPrep] Loop {loop_id}: VAE encoded refined_anchor from Wavespeed")
            else:
                raise RuntimeError(
                    f"[SVILoopPrep] Loop {loop_id}: loop_index > 0 requires either "
                    f"refined_anchor input or anchor_override in loop_config")

        # ── Prompt encoding ──────────────────────────────────────────────────
        if prompt.strip():
            tokens = clip.tokenize(prompt)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            c_pos = [[cond, {"pooled_output": pooled}]]
            used_prompts_log.append(f"Loop {loop_id}: {prompt}")
            print(f"[SVILoopPrep] Loop {loop_id}: encoded loop prompt")
        else:
            c_pos = positive
            used_prompts_log.append(f"Loop {loop_id}: (using default positive)")
            print(f"[SVILoopPrep] Loop {loop_id}: using default positive conditioning")

        # ── WanImageMotion conditioning ──────────────────────────────────────
        motion_val = 1.15 if loop_index == 0 else 1.3
        use_prev   = loop_index > 0

        print(f"[SVILoopPrep] Loop {loop_id}: calling IAMCCS_WanImageMotion.apply() "
              f"(motion={motion_val}, use_prev={use_prev})")
        positive_out, negative_out, latent_out = _IAMCCS_WanImageMotion.apply(
            positive             = c_pos,
            negative             = negative,
            anchor_samples       = anchor_samples,
            prev_samples         = prev_samples,
            length               = frames,
            motion_latent_count  = 1,
            motion               = motion_val,
            motion_mode          = "motion_only (prev_samples)",
            add_reference_latents = False,
            latent_precision     = "fp32",
            vram_profile         = "normal",
            include_padding_in_motion = False,
            safety_preset        = "safe" if loop_index == 0 else "safer",
            lock_start_slots     = 1,
            diagnostic_log       = False,
            use_prev_samples     = use_prev,
            latent_refresh       = 0.0,
            delta_max            = 0.0,
        )
        print(f"[SVILoopPrep] Loop {loop_id}: WanImageMotion done")

        # ── Build loop_state ─────────────────────────────────────────────────
        loop_state = {
            "model_high_patched":  high_model,
            "model_low_patched":   low_model,
            "model_high_original": model_high,
            "model_low_original":  model_low,
            "lora_high_name":      lora_high_name,
            "lora_low_name":       lora_low_name,
            "loop_index":          loop_index,
            "frames":              frames,
            "anchor_frame_offset": anchor_frame_offset,
            "anchor_override":     anchor_override,
            "width":               width,
            "height":              height,
            "vae":                 vae,
            "anchor_samples":      anchor_samples,
            "prev_samples":        prev_samples,
            "prev_decoded":        prev_decoded,
            "segment_paths":       segment_paths,
            "segment_dir":         segment_dir,
            "used_prompts_log":    used_prompts_log,
        }

        print(f"[SVILoopPrep] Loop {loop_id}: prepare complete")
        return (positive_out, negative_out, latent_out, high_model, low_model, loop_state)
```

- [ ] **Step 2: Verify the class looks correct in the file**

```bash
grep -n "class SVILoopPrep\|def prepare\|RETURN_TYPES\|RETURN_NAMES\|FUNCTION\|CATEGORY" \
  "E:/AI_Studio/comfy-node-dev/ComfyUI-SVILooper/nodes_svi.py"
```

Expected: lines for `class SVILoopPrep`, `def prepare`, and the correct RETURN_TYPES/NAMES.

---

## Task 2: Write SVILoopFinish Class

**Files:**
- Modify: `E:\AI_Studio\comfy-node-dev\ComfyUI-SVILooper\nodes_svi.py` (add after SVILoopPrep)

- [ ] **Step 1: Add SVILoopFinish class after SVILoopPrep**

```python
# ---------------------------------------------------------------------------
# NODE 3: SVILoopFinish
# ---------------------------------------------------------------------------

class SVILoopFinish:
    """
    Per-loop finish node.
    Handles VAE decode, overlap stitching, anchor frame extraction,
    segment disk save, LoRA cleanup, and final video assembly.

    anchor_frame output → WaveSpeedAIPredictor → next SVILoopPrep.refined_anchor
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampled_latent": ("LATENT",),
                "loop_state":     ("SVI_LOOP_STATE",),
                "overlap": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 50,
                    "tooltip": "Frame overlap for ImageBatchExtendWithOverlap",
                }),
                "is_final_loop": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Set True on the last loop to trigger full video assembly",
                }),
            },
        }

    RETURN_TYPES  = ("LATENT", "IMAGE", "IMAGE", "SVI_LOOP_STATE", "IMAGE", "STRING")
    RETURN_NAMES  = ("prev_samples", "anchor_frame", "segment_frames",
                     "next_loop_state", "full_video_so_far", "used_prompts")
    FUNCTION      = "finish"
    CATEGORY      = "video/svi_looper"

    def finish(self, sampled_latent, loop_state, overlap, is_final_loop):

        if _GetImageRange is None:
            raise RuntimeError("[SVILoopFinish] GetImageRangeFromBatch not loaded")
        if _ImageBatchExtend is None:
            raise RuntimeError("[SVILoopFinish] ImageBatchExtendWithOverlap not loaded")

        # ── Unpack loop_state ────────────────────────────────────────────────
        vae                  = loop_state["vae"]
        width                = loop_state["width"]
        height               = loop_state["height"]
        frames               = loop_state["frames"]
        anchor_frame_offset  = loop_state["anchor_frame_offset"]
        anchor_override      = loop_state["anchor_override"]
        model_high_patched   = loop_state["model_high_patched"]
        model_low_patched    = loop_state["model_low_patched"]
        model_high_original  = loop_state["model_high_original"]
        model_low_original   = loop_state["model_low_original"]
        lora_high_name       = loop_state["lora_high_name"]
        lora_low_name        = loop_state["lora_low_name"]
        loop_index           = loop_state["loop_index"]
        prev_decoded         = loop_state["prev_decoded"]
        segment_paths        = loop_state["segment_paths"]
        segment_dir          = loop_state["segment_dir"]
        used_prompts_log     = loop_state.get("used_prompts_log", [])
        loop_id = loop_index + 1

        print(f"[SVILoopFinish] Loop {loop_id}: finishing")

        # ── 1. VAE decode ────────────────────────────────────────────────────
        print(f"[SVILoopFinish] Loop {loop_id}: VAE decoding")
        decoded = vae.decode(sampled_latent["samples"])
        while decoded.ndim > 4:
            decoded = decoded.squeeze(0)
        if decoded.shape[1] != height or decoded.shape[2] != width:
            d_bchw = decoded.permute(0, 3, 1, 2)
            d_bchw = comfy.utils.common_upscale(d_bchw, width, height, "lanczos", "disabled")
            decoded = d_bchw.permute(0, 2, 3, 1)

        # ── 2. Overlap stitching ─────────────────────────────────────────────
        if loop_index == 0 or overlap == 0 or prev_decoded is None:
            segment_frames = decoded
            print(f"[SVILoopFinish] Loop {loop_id}: no overlap "
                  f"(loop_index={loop_index}, overlap={overlap})")
        else:
            print(f"[SVILoopFinish] Loop {loop_id}: "
                  f"applying ImageBatchExtendWithOverlap (overlap={overlap})")
            overlap_result = _ImageBatchExtend.imagesfrombatch(
                source_images = prev_decoded,
                new_images    = decoded,
                overlap       = overlap,
                overlap_side  = "source",
                overlap_mode  = "ease_in_out",
            )
            segment_frames = overlap_result[0]

        # ── 3. Anchor frame extraction ────────────────────────────────────────
        actual_frames    = decoded.shape[0]
        anchor_frame_idx = max(0, min(actual_frames - 1, actual_frames + anchor_frame_offset))
        print(f"[SVILoopFinish] Loop {loop_id}: extracting anchor at frame {anchor_frame_idx}")
        anchor_result  = _GetImageRange.imagesfrombatch(
            images       = decoded,
            start_index  = anchor_frame_idx,
            num_frames   = 1,
        )
        anchor_frame = anchor_result[0]   # [1,H,W,C] → user wires to WaveSpeedAIPredictor

        # ── 4. Save segment to disk ──────────────────────────────────────────
        seg_path = os.path.join(segment_dir, f"segment_{loop_id:03d}.pt")
        torch.save(segment_frames.cpu(), seg_path)
        new_segment_paths = list(segment_paths) + [seg_path]
        print(f"[SVILoopFinish] Loop {loop_id}: saved segment to {seg_path}")

        # ── 5. Build next_loop_state ─────────────────────────────────────────
        # anchor_samples is intentionally absent — SVILoopPrep for the next loop
        # will build it from the refined_anchor (WaveSpeedAIPredictor output).
        next_loop_state = {
            "anchor_samples":      None,          # filled by next SVILoopPrep
            "prev_samples":        sampled_latent,
            "prev_decoded":        decoded,
            "segment_paths":       new_segment_paths,
            "segment_dir":         segment_dir,
            "used_prompts_log":    used_prompts_log,
            # Pass through everything else that SVILoopPrep doesn't reset
            "model_high_patched":  model_high_original,
            "model_low_patched":   model_low_original,
            "model_high_original": model_high_original,
            "model_low_original":  model_low_original,
            "lora_high_name":      None,
            "lora_low_name":       None,
            "loop_index":          loop_index,
            "frames":              frames,
            "anchor_frame_offset": anchor_frame_offset,
            "anchor_override":     None,
            "width":               width,
            "height":              height,
            "vae":                 vae,
        }

        # ── 6. LoRA cleanup ──────────────────────────────────────────────────
        if lora_high_name and model_high_patched is not model_high_original:
            del model_high_patched
            print(f"[SVILoopFinish] Loop {loop_id}: cleaned up patched high model")
        if lora_low_name and model_low_patched is not model_low_original:
            del model_low_patched
            print(f"[SVILoopFinish] Loop {loop_id}: cleaned up patched low model")
        comfy.model_management.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # ── 7. Full video assembly (final loop only) ─────────────────────────
        used_prompts_str = "\n".join(used_prompts_log)

        if is_final_loop:
            print("[SVILoopFinish] Final loop — assembling full video from all segments")
            full_video = None
            for sp in new_segment_paths:
                seg = torch.load(sp, map_location="cpu", weights_only=True)
                full_video = seg if full_video is None else torch.cat([full_video, seg], dim=0)
                del seg
                gc.collect()
            if full_video is None:
                full_video = torch.zeros((1, height, width, 3))
            try:
                shutil.rmtree(segment_dir)
            except Exception:
                pass
            print(f"[SVILoopFinish] Done. Total frames: {full_video.shape[0]}")
        else:
            full_video = segment_frames   # preview: just this segment

        print(f"[SVILoopFinish] Loop {loop_id}: finish complete")
        return (sampled_latent, anchor_frame, segment_frames,
                next_loop_state, full_video, used_prompts_str)
```

- [ ] **Step 2: Verify the class in the file**

```bash
grep -n "class SVILoopFinish\|def finish\|RETURN_TYPES\|RETURN_NAMES" \
  "E:/AI_Studio/comfy-node-dev/ComfyUI-SVILooper/nodes_svi.py"
```

Expected: lines for `class SVILoopFinish`, `def finish`, and correct RETURN_TYPES.

---

## Task 3: Update NODE_CLASS_MAPPINGS and Remove Unused Imports

**Files:**
- Modify: `E:\AI_Studio\comfy-node-dev\ComfyUI-SVILooper\nodes_svi.py`

- [ ] **Step 1: Replace the node registration block**

Find the existing block:
```python
NODE_CLASS_MAPPINGS = {
    "SVILooper": SVILooper,
    "SVILoopConfig": SVILoopConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SVILooper": "SVI Looper",
    "SVILoopConfig": "SVI Loop Config",
}
```

Replace with:
```python
# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "SVILoopConfig": SVILoopConfig,
    "SVILoopPrep":   SVILoopPrep,
    "SVILoopFinish": SVILoopFinish,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SVILoopConfig": "SVI Loop Config",
    "SVILoopPrep":   "SVI Loop Prep",
    "SVILoopFinish": "SVI Loop Finish",
}
```

- [ ] **Step 2: Remove unused top-level imports**

The new architecture no longer calls WaveSpeedAIPredictor, ClownsharKSampler, or ClownsharkChainsampler internally. The `requests`, `io`, and `PIL.Image` imports are no longer needed (they were only used for the URL-download in the old Wavespeed call). Remove these lines:

```python
import io
import requests
```

```python
from PIL import Image
```

Also remove or comment out the three loader blocks for `_ClownsharKSampler`, `_ClownsharkChain`, and `_WaveSpeedAIPredictor` — they load external classes that are no longer called internally:

```python
# These are now external ComfyUI graph nodes — not called internally.
# Loader blocks removed to avoid unnecessary import overhead and failure noise.
```

Keep `_GetImageRange` and `_ImageBatchExtend` loaders — still called inside SVILoopFinish.
Keep `_IAMCCS_WanImageMotion` loader — still called inside SVILoopPrep.

- [ ] **Step 3: Verify file is syntactically valid**

```bash
python -c "import ast; ast.parse(open('E:/AI_Studio/comfy-node-dev/ComfyUI-SVILooper/nodes_svi.py').read()); print('OK')"
```

Expected output: `OK`

---

## Task 4: Copy to ComfyUI Deployment Target

**Files:**
- Copy: `nodes_svi.py` → `E:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-SVILooper\nodes_svi.py`

- [ ] **Step 1: Copy nodes_svi.py**

```bash
cp "E:/AI_Studio/comfy-node-dev/ComfyUI-SVILooper/nodes_svi.py" \
   "E:/ComfyUI/ComfyUI/custom_nodes/ComfyUI-SVILooper/nodes_svi.py"
```

- [ ] **Step 2: Verify __init__.py needs no changes**

```bash
cat "E:/ComfyUI/ComfyUI/custom_nodes/ComfyUI-SVILooper/__init__.py"
```

Expected content (unchanged):
```python
from .nodes import NODE_CLASS_MAPPINGS as _NCM_BASE, NODE_DISPLAY_NAME_MAPPINGS as _NDM_BASE
from .nodes_highlow import NODE_CLASS_MAPPINGS as _NCM_HL, NODE_DISPLAY_NAME_MAPPINGS as _NDM_HL
from .nodes_svi import NODE_CLASS_MAPPINGS as _NCM_SVI, NODE_DISPLAY_NAME_MAPPINGS as _NDM_SVI

NODE_CLASS_MAPPINGS = {**_NCM_BASE, **_NCM_HL, **_NCM_SVI}
NODE_DISPLAY_NAME_MAPPINGS = {**_NDM_BASE, **_NDM_HL, **_NDM_SVI}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
```

`__init__.py` automatically picks up the new mappings. **No changes needed.**

- [ ] **Step 3: Verify the deployed file exports the three new node keys**

```bash
python -c "
import sys
sys.path.insert(0, 'E:/ComfyUI/ComfyUI')
# Quick key check without running ComfyUI
import ast, re
src = open('E:/ComfyUI/ComfyUI/custom_nodes/ComfyUI-SVILooper/nodes_svi.py').read()
for key in ['SVILoopConfig', 'SVILoopPrep', 'SVILoopFinish']:
    assert key in src, f'Missing: {key}'
    print(f'OK: {key}')
assert 'SVILooper' not in re.findall(r'\"(SVI\w+)\"', src), 'SVILooper still present in mappings'
print('All checks passed')
"
```

---

## Self-Review Checklist

### Spec Coverage

| Spec requirement | Covered by |
|---|---|
| SVILoopConfig unchanged | Task 1 (not touched) |
| SVILooper removed | Task 3 (removed from mappings) |
| SVILoopPrep: LoRA application (model-only, clip=0.0) | Task 1, `load_lora_for_models(..., strength, 0.0)` |
| SVILoopPrep: anchor samples for loop 0 from start_image | Task 1, `loop_index == 0` branch |
| SVILoopPrep: anchor samples for loop N>0 from refined_anchor | Task 1, `refined_anchor` optional input |
| SVILoopPrep: anchor_override priority over refined_anchor | Task 1, `anchor_override is not None` branch |
| SVILoopPrep: prompt encoding with fallback to global positive | Task 1, `prompt.strip()` branch |
| SVILoopPrep: WanImageMotion with exact param list | Task 1, full kwarg call |
| SVILoopPrep: motion=1.15/loop0, 1.3/others | Task 1, `motion_val` |
| SVILoopPrep: use_prev_samples=False/loop0, True/others | Task 1, `use_prev` |
| SVILoopPrep: safety_preset safe/safer | Task 1, conditional |
| SVILoopPrep: 6-output signature | Task 1, RETURN_TYPES/NAMES |
| SVILoopFinish: VAE decode + squeeze + resize | Task 2 |
| SVILoopFinish: overlap stitching via ImageBatchExtendWithOverlap | Task 2 |
| SVILoopFinish: anchor frame extraction via GetImageRangeFromBatch | Task 2 |
| SVILoopFinish: anchor_frame output for WaveSpeedAIPredictor | Task 2, output slot 1 |
| SVILoopFinish: segment disk save with torch.save | Task 2 |
| SVILoopFinish: next_loop_state build | Task 2 |
| SVILoopFinish: LoRA cleanup + cache flush | Task 2 |
| SVILoopFinish: full video assembly on is_final_loop | Task 2 |
| SVILoopFinish: temp dir cleanup on is_final_loop | Task 2, `shutil.rmtree` |
| SVILoopFinish: full_video_so_far = segment_frames when not final | Task 2 |
| NODE_CLASS_MAPPINGS updated | Task 3 |
| NODE_DISPLAY_NAME_MAPPINGS updated | Task 3 |
| SVILooper removed from mappings | Task 3 |
| Copy to ComfyUI deployment | Task 4 |
| __init__.py: no change needed | Task 4, Step 2 |
| print() logging at each major step | Tasks 1–2, every major operation |
| Cyclic graph issue corrected | refined_anchor → SVILoopPrep (not SVILoopFinish) |

### Architectural Deviation from Brief

The brief spec (Step 3, SVILoopFinish) defines `refined_anchor` as a **required input** to `SVILoopFinish`. This creates an impossible cycle in a ComfyUI DAG. This plan implements the correct non-cyclic design where:
- `SVILoopFinish` outputs `anchor_frame` (for WaveSpeedAIPredictor)  
- `SVILoopPrep` of the **next loop** receives `refined_anchor` as an optional input

All other logic, types, and field names match the brief exactly.
