# SVILooper Project Brief
**Date:** April 11, 2026  
**Status:** Active Development — First Test Run Complete, Fixing WanImageMotion Padding Issue

---

## Project Origin

Barley runs a complex 8-segment WAN 2.2 SVI video generation workflow (`SVI_refined_anchors_4-10-26`) using ComfyUI. The workflow uses Clownshark samplers (RES4LYF), IAMCCS WanImageMotion for SVI conditioning, Wavespeed API for anchor frame refinement, and ImageBatchExtendWithOverlap for segment stitching. While producing good results, the workflow is extremely cumbersome to manage — 8 manually chained subgraphs with Set/Get nodes everywhere, making parameter adjustments painful.

The goal: replace the manual multi-segment chain with a looping node architecture that handles segment iteration internally while keeping key elements (Clownshark samplers, Wavespeed refine, overlap) visible and controllable in the graph.

---

## Development Path

### Phase 1 — NativeLooper Evaluation
Discovered `masteroleary/ComfyUI-LooperNode` (brand new, Claude-assisted, 0 stars). Evaluated, fixed two bugs:
- `WanImageToVideo.execute()` called as static method → fixed to instance call
- Missing `NODE_CLASS_MAPPINGS` for high/low nodes → added

Tested `I2V Looper High/Low` node with WAN 2.2 GGUF Q5_K_M. Results: good image quality, good prompt adherence, no drift observed, ping-pong artifact present (expected without overlap), no SVI conditioning.

Conclusion: promising architecture but wrong sampler stack (standard KSamplerAdvanced) and no SVI. Decision: fork and rebuild.

### Phase 2 — SVILooper v1 (Monolithic)
Forked to `ComfyUI-SVILooper`. Wrote CC brief to replace:
- `KSamplerAdvanced` → `ClownsharKSampler_Beta` + `ClownsharkChainsampler_Beta`
- `WanImageToVideo` → `IAMCCS_WanImageMotion`
- Add Wavespeed anchor refinement loop
- Add `ImageBatchExtendWithOverlap` between segments
- Add anchor frame extraction via `ImageFromBatch+`
- Add per-loop anchor override option

CC built `SVILooper` (monolithic) + `SVILoopConfig`. Fixed import issues (`_ensure_package` / `_load_package_class` pattern for loading external node classes without relative import failures). Fixed `motion_mode` string (no "first segment" mode exists — `prev_samples=None` + `use_prev_samples=False` handles loop 0 behavior). Loaded clean.

Problem identified: all key nodes (Clownshark samplers, Wavespeed) hidden inside monolithic node — no graph visibility or control.

### Phase 3 — SVILooper Option A (Hybrid Architecture)
Rebuilt into three nodes:
- **`SVILoopConfig`** — per-loop data container (unchanged from v1)
- **`SVILoopPrep`** — front half: LoRA application, anchor/prev_samples setup, prompt encoding, WanImageMotion conditioning. Outputs: `positive_out`, `negative_out`, `latent_out`, `model_high_out`, `model_low_out`, `loop_state`
- **`SVILoopFinish`** — back half: VAE decode, overlap stitching, anchor frame extraction, segment disk save, state handoff. Outputs: `prev_samples`, `anchor_frame`, `segment_frames`, `next_loop_state`, `full_video_so_far`, `used_prompts`

Key DAG fix: Wavespeed refine sits between loops, not within a loop:
```
SVILoopFinish_N.anchor_frame → WaveSpeedAIPredictor → SVILoopPrep_N+1.refined_anchor
```
This avoids the cycle that would occur if refined_anchor fed back into the same Finish node.

Loop state (`SVI_LOOP_STATE` dict) carries: anchor_samples, prev_samples, prev_decoded, segment_paths, segment_dir, patched models, LoRA references, loop metadata.

Loaded clean:
```
[SVILooper] Loaded IAMCCS_WanImageMotion
[SVILooper] Loaded GetImageRangeFromBatch  
[SVILooper] Loaded ImageBatchExtendWithOverlap
```

### Phase 4 — Test Workflow + First Run
CC built `svi_looper_test_v2.json`. Issues found and fixed:
- `ClownsharKSampler_Beta` widget type errors: `control_after_generate` and `sampler_mode` swapped positions, boolean `True` in string slot → fixed in v2.2/v2.3
- Workflow versioning confusion during fixes — established v2.1 as stable base, v2.3 as fixed version

First run completed end-to-end. Console confirmed all steps executed. However output was garbage texture.

**Root cause identified:** WanImageMotion warning:
```
⚠️ WARNING: motion_range is EMPTY (no frames will be modified). 
include_padding_in_motion=False with no prev_samples means no motion applied.
```

Fix applied (CC, one line):
```python
# Before
include_padding_in_motion = False,
# After  
include_padding_in_motion = loop_index == 0,
```
`loop_index == 0` evaluates to `True` for first segment (needs padding), `False` for extensions (has prev_samples).

**Status at handoff:** Awaiting rerun after restart with padding fix applied.

---

## Hardware & Environment

- **Local:** Windows 11, RTX 3060 Ti, 8GB VRAM, 32GB RAM
- **Models:** WAN 2.2 I2V A14B Q5_K_M GGUF (High Noise + Low Noise split)
- **VAE:** `wan_2.1_vae.safetensors`
- **Text encoder:** `umt5_xxl_fp8_e4m3fn_scaled.safetensors`
- **CLIP Vision:** `clip_vision_h.safetensors`
- **ComfyUI:** `E:\ComfyUI\ComfyUI\`
- **Dev folder:** `E:\AI_Studio\comfy-node-dev\ComfyUI-SVILooper\`
- **Node pack:** `E:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-SVILooper\`
- **Launch flags:** `--novram --disable-smart-memory --disable-pinned-memory --use-sage-attention --preview-method taesd`

---

## Current Node Architecture

### SVILoopConfig
```
INPUTS (required): prompt, frames, lora_high, lora_high_strength, lora_low, lora_low_strength, anchor_frame_offset
INPUTS (optional): anchor_override (IMAGE)
OUTPUTS: loop_config (SVI_LOOP)
```

### SVILoopPrep
```
INPUTS (required): model_high, model_low, vae, clip, start_image, positive, negative, 
                   width, height, loop_index, loop_config, cs_sampler_primary, cs_sampler_chain
INPUTS (optional): prev_loop_state (SVI_LOOP_STATE), refined_anchor (IMAGE)
OUTPUTS: positive_out, negative_out, latent_out, model_high_out, model_low_out, loop_state
```

### SVILoopFinish
```
INPUTS (required): sampled_latent, loop_state, overlap, is_final_loop
INPUTS (optional): prev_loop_state (SVI_LOOP_STATE)
OUTPUTS: prev_samples, anchor_frame, segment_frames, next_loop_state, full_video_so_far, used_prompts
```

---

## Per-Loop Graph Pattern

```
SVILoopConfig_N ──────────────────────────────────── SVILoopPrep_N
prev_SVILoopFinish.next_loop_state ────────────────── SVILoopPrep_N.prev_loop_state
WaveSpeedAIPredictor_{N-1} ────────────────────────── SVILoopPrep_N.refined_anchor

SVILoopPrep_N.model_high_out ──── ClownsharKSampler_Beta_N
SVILoopPrep_N.model_low_out ───── ClownsharkChainsampler_Beta_N
SVILoopPrep_N.positive_out ─────► ClownsharKSampler_Beta_N
SVILoopPrep_N.negative_out ─────► ClownsharKSampler_Beta_N
SVILoopPrep_N.latent_out ───────► ClownsharKSampler_Beta_N
ClownsharKSampler_Beta_N.output ► ClownsharkChainsampler_Beta_N.latent_image
ClownsharkChainsampler_Beta_N.output ► SVILoopFinish_N.sampled_latent
SVILoopPrep_N.loop_state ───────► SVILoopFinish_N.loop_state

SVILoopFinish_N.anchor_frame ───► WaveSpeedAIPredictor_N ──► SVILoopPrep_N+1.refined_anchor
SVILoopFinish_N.next_loop_state ► SVILoopPrep_N+1.prev_loop_state

(last loop only) SVILoopFinish.full_video_so_far ──► VHS_VideoCombine
```

---

## Sampler Configuration (Current Baseline)

| Parameter | Value |
|-----------|-------|
| Primary sampler | `linear/euler` (testing) / `multistep/res_2m` (SVI baseline) |
| Chain sampler | `exponential/res_2s` |
| Scheduler | `beta57` (alpha=0.5, beta=0.7, RES4LYF monkey-patch) |
| Steps | 8 |
| Split step | 3–4 |
| CFG | 1.0 |
| Shift (HN) | 4.0 |
| Shift (LN) | 4.0 |
| ClownOptions | `default_dtype=float32` + `skip_final_model_call` |

---

## Key Technical Decisions & Learnings

1. **External node loading:** Use `_ensure_package` + `_load_package_class` with `importlib.util.spec_from_file_location` and absolute paths. Relative imports fail when ComfyUI loads files as top-level modules.

2. **WanImageMotion loop 0:** No special "first segment" mode exists. Use `motion_only (prev_samples)` for all loops. Loop 0 behavior controlled by `prev_samples=None` + `use_prev_samples=False` + `include_padding_in_motion=True`.

3. **DAG cycle avoidance:** Wavespeed refined anchor feeds into the *next* loop's Prep, not back into the current loop's Finish. N feeds N+1.

4. **LoRA application:** Model-only — pass `strength` for model, `0.0` for clip in `load_lora_for_models`.

5. **Segment persistence:** `torch.save` / `torch.load` to temp dir between loops to free VRAM. Final assembly on `is_final_loop=True`.

6. **Clownshark widget order:** `control_after_generate` sits at position 8 (after seed), `sampler_mode` at position 9. Boolean vs string type mismatch in position 9 caused `AttributeError: 'bool' object has no attribute 'startswith'`.

7. **CC model choice:** Use Opus for precision work (exact filenames, type-accurate JSON, source signature matching). Sonnet sufficient for reasoning/planning tasks.

---

## Files

| File | Location |
|------|----------|
| `nodes_svi.py` | `E:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-SVILooper\` |
| `nodes.py` (NativeLooper, unchanged) | same |
| `nodes_highlow.py` (I2VLooperHL, unchanged) | same |
| `__init__.py` | same |
| Dev copies | `E:\AI_Studio\comfy-node-dev\ComfyUI-SVILooper\` |
| Test workflow | `svi_looper_test_v2.3.json` (stable base with Clownshark fixes) |
| CC Brief (original) | `SVILooper_CC_Brief.md` |
| CC Brief (Option A) | `SVILooper_OptionA_Brief.md` |

---

## Immediate Next Steps

1. Confirm clean output after `include_padding_in_motion = loop_index == 0` fix
2. Validate with `multistep/res_2m` + `beta57` (proven SVI baseline)
3. Test 2-loop chain with Wavespeed anchor passing between loops
4. Validate overlap stitching between segments
5. Build production workflow with 4–8 loops matching original SVI chain capability
6. Consider exposing `motion`, `safety_preset`, `motion_latent_count` as per-loop config options in SVILoopConfig
