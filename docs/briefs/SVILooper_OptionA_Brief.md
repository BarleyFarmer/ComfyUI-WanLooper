# SVILooper Option A — Hybrid Architecture Rebuild Brief

## Working Directory
`E:\AI_Studio\comfy-node-dev\ComfyUI-SVILooper\`

## Output Target
`E:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-SVILooper\nodes_svi.py`
`E:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-SVILooper\__init__.py`

---

## STEP 1 — READ FIRST

Read the current file in full before writing anything:
`E:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-SVILooper\nodes_svi.py`

Understand the existing loop state management, anchor/prev_samples flow, Wavespeed refine logic, segment disk save/load, and overlap logic. The new architecture reuses all of this logic but splits it across three nodes instead of one.

Also re-read the IAMCCS_WanImageMotion signature from:
`E:\ComfyUI\ComfyUI\custom_nodes\IAMCCS-nodes\`

---

## STEP 2 — NEW ARCHITECTURE OVERVIEW

Replace the single `SVILooper` node with three nodes:

```
SVILoopConfig     — unchanged, keep exactly as-is
SVILoopPrep       — replaces the front half of SVILooper.generate()
SVILoopFinish     — replaces the back half of SVILooper.generate()
```

The Clownshark sampler pair, ClownOptions, sampler selectors, and WaveSpeedAIPredictor all live in the ComfyUI graph between SVILoopPrep and SVILoopFinish. The user wires them.

Per-loop flow in the graph:

```
SVILoopConfig ──→ SVILoopPrep ──→ ClownsharKSampler_Beta ──→ ClownsharkChainsampler_Beta ──→ SVILoopFinish
                                                                                                    │
                                                              WaveSpeedAIPredictor ←── (anchor frame extracted inside Finish, exposed as IMAGE output)
                                                                      │
                                                              SVILoopFinish (receives refined anchor back)
                                                                      │
                                                              ──→ next loop's SVILoopPrep
```

---

## STEP 3 — NODE SPECIFICATIONS

---

### NODE: SVILoopConfig
**No changes.** Keep exactly as currently written.

---

### NODE: SVILoopPrep

This node handles loop state input, LoRA application, and WanImageMotion conditioning. It is called once per loop segment.

```python
CATEGORY = "video/svi_looper"
RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "SVI_LOOP_STATE")
RETURN_NAMES = ("positive_out", "negative_out", "latent_out", "loop_state")
FUNCTION = "prepare"
```

**INPUT_TYPES required:**
| Name | Type | Default | Notes |
|------|------|---------|-------|
| model_high | MODEL | | HN model |
| model_low | MODEL | | LN model |
| vae | VAE | | |
| clip | CLIP | | |
| start_image | IMAGE | | Used only for loop_idx == 0 |
| positive | CONDITIONING | | Global positive, used as fallback |
| negative | CONDITIONING | | Global negative |
| width | INT | 672 | min 16, max 4096, step 16 |
| height | INT | 896 | min 16, max 4096, step 16 |
| loop_index | INT | 0 | min 0, max 9, tooltip: "0-based index of this loop segment" |
| loop_config | SVI_LOOP | | From SVILoopConfig node |

**INPUT_TYPES optional:**
| Name | Type | Notes |
|------|------|-------|
| prev_loop_state | SVI_LOOP_STATE | None for loop 0. Output of previous SVILoopFinish. |

**Internal logic — prepare():**

1. Extract from `loop_config`: prompt, frames, lora_high, lora_high_strength, lora_low, lora_low_strength, anchor_frame_offset, anchor_override

2. Apply per-loop LoRAs (model-only, clip strength 0.0):
   - Apply lora_high to model_high if set
   - Apply lora_low to model_low if set
   - Store patched models in loop_state for SVILoopFinish cleanup

3. Determine anchor_samples and prev_samples:
   - If loop_index == 0:
     - anchor_image = start_image resized to width x height
     - anchor_samples = VAE encode anchor_image
     - prev_samples = None
   - Else:
     - Extract from prev_loop_state: anchor_samples, prev_samples
     - (These were set by the previous SVILoopFinish)

4. Prompt encoding:
   - If loop_config prompt non-empty: re-encode via clip → c_pos
   - Else: use passed-in positive conditioning

5. Call IAMCCS_WanImageMotion with exact signature from source:
   - positive = c_pos
   - negative = negative
   - anchor_samples = anchor_samples
   - prev_samples = prev_samples
   - length = frames from loop_config
   - motion_latent_count = 1
   - motion = 1.15 if loop_index == 0 else 1.3
   - motion_mode = "motion_only (prev_samples)"
   - add_reference_latents = False
   - latent_precision = "fp32"
   - vram_profile = "normal"
   - include_padding_in_motion = False
   - safety_preset = "safe" if loop_index == 0 else "safer"
   - lock_start_slots = 1
   - diagnostic_log = False
   - use_prev_samples = False if loop_index == 0 else True
   - latent_refresh = 0.0
   - delta_max = 0.0

6. Build loop_state dict containing:
   - model_high_patched (LoRA-patched or original)
   - model_low_patched (LoRA-patched or original)
   - model_high_original (reference for cleanup)
   - model_low_original (reference for cleanup)
   - lora_high_name, lora_low_name (for cleanup flags)
   - loop_index
   - frames (from loop_config)
   - anchor_frame_offset (from loop_config)
   - anchor_override (from loop_config, may be None)
   - width, height
   - vae reference
   - prev_decoded = None (will be set by SVILoopFinish of previous loop — for loop 0 this is None)
   - Carry prev_decoded from prev_loop_state if loop_index > 0

7. Return (positive_out, negative_out, latent_out, loop_state)

**Note:** `model_high_patched` and `model_low_patched` from loop_state are what the user connects to ClownsharKSampler_Beta and ClownsharkChainsampler_Beta respectively in the graph. These need to be exposed. Add two additional outputs:

```python
RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "MODEL", "MODEL", "SVI_LOOP_STATE")
RETURN_NAMES = ("positive_out", "negative_out", "latent_out", "model_high_out", "model_low_out", "loop_state")
```

`model_high_out` and `model_low_out` are the LoRA-patched models for use with the Clownshark samplers.

---

### NODE: SVILoopFinish

This node handles VAE decode, overlap stitching, anchor frame extraction, optional Wavespeed refine passthrough, segment disk save, and state handoff to the next loop.

```python
CATEGORY = "video/svi_looper"
RETURN_TYPES = ("LATENT", "IMAGE", "IMAGE", "SVI_LOOP_STATE", "IMAGE", "STRING")
RETURN_NAMES = ("prev_samples", "anchor_frame", "segment_frames", "next_loop_state", "full_video_so_far", "used_prompts")
FUNCTION = "finish"
```

**INPUT_TYPES required:**
| Name | Type | Notes |
|------|------|-------|
| sampled_latent | LATENT | Output of ClownsharkChainsampler_Beta |
| loop_state | SVI_LOOP_STATE | From SVILoopPrep of this same loop |
| refined_anchor | IMAGE | Output of WaveSpeedAIPredictor — or any IMAGE if anchor_override is set. If anchor_override was set in loop_config, this input is still required but its value is ignored internally. |
| overlap | INT | default 0, min 0, max 50 |
| is_final_loop | BOOLEAN | default False, tooltip: "Set True on the last loop to trigger full video assembly" |

**INPUT_TYPES optional:**
| Name | Type | Notes |
|------|------|-------|
| prev_loop_state | SVI_LOOP_STATE | None for loop 0. Needed to access previous segment frames for overlap. |

**Internal logic — finish():**

1. Extract from loop_state: vae, width, height, frames, anchor_frame_offset, anchor_override, model_high_patched, model_low_patched, model_high_original, model_low_original, lora_high_name, lora_low_name, loop_index, prev_decoded

2. VAE decode sampled_latent["samples"]
   - Squeeze extra dims
   - Resize to width x height if mismatch
   - Result: decoded (IMAGE tensor)

3. Overlap stitching:
   - If loop_index == 0 or overlap == 0 or prev_decoded is None:
     - segment_frames = decoded
   - Else:
     - Call ImageBatchExtendWithOverlap:
       - source_images = prev_decoded
       - new_images = decoded
       - overlap = overlap
       - overlap_side = "source"
       - overlap_mode = "ease_in_out"
     - segment_frames = result

4. Anchor frame extraction:
   - If anchor_override is not None (was set in loop_config):
     - extracted_anchor = anchor_override (resize to width x height)
     - Skip Wavespeed — refined_anchor input is ignored
     - next_anchor_samples = VAE encode extracted_anchor
     - anchor_out = extracted_anchor
   - Else:
     - anchor_frame_idx = max(0, frames + anchor_frame_offset)
     - Use GetImageRangeFromBatch to extract single frame at anchor_frame_idx from decoded
     - anchor_out = extracted frame (this is what the user feeds into WaveSpeedAIPredictor in the graph)
     - After Wavespeed runs externally, refined_anchor comes back in as input
     - next_anchor_samples = VAE encode refined_anchor
     - anchor_out = extracted frame (pre-refine, for the WaveSpeed node connection)

   **Important:** anchor_frame output goes OUT to WaveSpeedAIPredictor in the graph. refined_anchor comes back IN. These are separate inputs/outputs.

5. Save segment to disk:
   - Use existing torch.save pattern to temp dir
   - Store seg_path in next_loop_state

6. Build next_loop_state:
   - anchor_samples = next_anchor_samples (VAE-encoded refined anchor)
   - prev_samples = sampled_latent (raw latent dict, for next loop's WanImageMotion)
   - prev_decoded = decoded (pre-overlap frames, for next loop's overlap source)
   - segment_paths = prev segment_paths + [this seg_path]
   - Carry segment_dir from prev_loop_state or create new one if loop_index == 0

7. Cleanup LoRA-patched models:
   - If lora_high_name and model_high_patched is not model_high_original: del model_high_patched
   - If lora_low_name and model_low_patched is not model_low_original: del model_low_patched
   - soft_empty_cache(), empty_cache(), gc.collect()

8. Full video assembly (only if is_final_loop == True):
   - Load all segment .pt files from next_loop_state["segment_paths"]
   - torch.cat along dim 0
   - Delete temp dir
   - full_video_so_far = assembled video
   Else:
   - full_video_so_far = segment_frames (just this segment, for preview)

9. Return (prev_samples, anchor_frame, segment_frames, next_loop_state, full_video_so_far, used_prompts_log)

---

### LOOP STATE TYPE

`SVI_LOOP_STATE` is a plain Python dict. No special class needed. ComfyUI treats unknown string type names as passthrough — this is fine.

Contents passed between nodes:
```python
{
    "anchor_samples": LATENT dict or None,
    "prev_samples": LATENT dict or None,
    "prev_decoded": IMAGE tensor or None,
    "segment_paths": [list of str],
    "segment_dir": str (temp dir path),
    "model_high_patched": MODEL,
    "model_low_patched": MODEL,
    "model_high_original": MODEL,
    "model_low_original": MODEL,
    "lora_high_name": str or None,
    "lora_low_name": str or None,
    "loop_index": int,
    "frames": int,
    "anchor_frame_offset": int,
    "anchor_override": IMAGE tensor or None,
    "width": int,
    "height": int,
    "vae": VAE object,
}
```

---

## STEP 4 — GRAPH WIRING PER LOOP (for documentation/reference)

For each loop segment N, the graph looks like:

```
SVILoopConfig_N ──────────────────────────────────────────→ SVILoopPrep_N
prev_SVILoopFinish.next_loop_state ───────────────────────→ SVILoopPrep_N (prev_loop_state)
                                                             SVILoopPrep_N.model_high_out ──→ ClownsharKSampler_Beta_N
                                                             SVILoopPrep_N.model_low_out ───→ ClownsharkChainsampler_Beta_N
                                                             SVILoopPrep_N.positive_out ────→ ClownsharKSampler_Beta_N
                                                             SVILoopPrep_N.negative_out ────→ ClownsharKSampler_Beta_N
                                                             SVILoopPrep_N.latent_out ──────→ ClownsharKSampler_Beta_N
                                                             ClownsharKSampler_Beta_N ──────→ ClownsharkChainsampler_Beta_N
                                                             ClownsharkChainsampler_Beta_N ─→ SVILoopFinish_N (sampled_latent)
                                                             SVILoopPrep_N.loop_state ──────→ SVILoopFinish_N
                                                             SVILoopFinish_N.anchor_frame ──→ WaveSpeedAIPredictor_N
                                                             WaveSpeedAIPredictor_N ────────→ SVILoopFinish_N (refined_anchor)
                                                             SVILoopFinish_N.next_loop_state → SVILoopPrep_N+1 (prev_loop_state)
                                                             SVILoopFinish_N.prev_samples ──→ (not needed externally, carried in state)
```

Global nodes shared across all loops (wired to each SVILoopPrep):
- UnetLoaderGGUF × 2 → ModelSamplingSD3 × 2 → model_high, model_low
- CLIPLoader → clip
- VAELoader → vae
- CLIPTextEncode positive/negative
- LoadImage → start_image (only used by loop 0's SVILoopPrep)

ClownOptions_ExtraOptions_Beta wired to each ClownsharKSampler_Beta and ClownsharkChainsampler_Beta.

---

## STEP 5 — NODE REGISTRATION

```python
NODE_CLASS_MAPPINGS = {
    "SVILoopConfig": SVILoopConfig,
    "SVILoopPrep": SVILoopPrep,
    "SVILoopFinish": SVILoopFinish,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SVILoopConfig": "SVI Loop Config",
    "SVILoopPrep": "SVI Loop Prep",
    "SVILoopFinish": "SVI Loop Finish",
}
```

Remove `SVILooper` entirely — it is replaced by `SVILoopPrep` + `SVILoopFinish`.

Update `__init__.py` to reflect the new mappings. Ensure nodes.py and nodes_highlow.py mappings are still merged in.

---

## STEP 6 — IMPORTANT RULES

1. Read `nodes_svi.py` fully before writing — reuse all existing import helpers (_ensure_package, _load_package_class) exactly as-is
2. Do not change `SVILoopConfig`
3. Remove `SVILooper` class entirely
4. `SVI_LOOP_STATE` is just a string type name — no special registration needed
5. Model-only LoRA: pass strength for model, 0.0 for clip
6. `beta57` is hardcoded as scheduler string inside SVILoopPrep — it is NOT an external input
7. Keep all print() logging at each major step
8. Write working files to `E:\AI_Studio\comfy-node-dev\ComfyUI-SVILooper\nodes_svi.py` first
9. Then copy to `E:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-SVILooper\nodes_svi.py` and `__init__.py`
10. Do not modify `nodes.py` or `nodes_highlow.py`
