# Codex Takeover Progress Report
**Original date:** April 17, 2026  
**Cleaned up:** April 28, 2026  
**Scope:** Current handoff summary after the Claude / Claude Code development era  
**Repo:** `ComfyUI-WanLooper`

---

## Status Note

This report started as a takeover progress memo while the repo still contained active SVI and KSA development threads. The current committed package has since narrowed to the SVI path only.

For current behavior, trust these sources first:

- [`../../README.md`](../../README.md)
- [`../SVI_Looper_Native_Reference.md`](../SVI_Looper_Native_Reference.md)
- `nodes_wan_v2.py`

Older KSA, IAMCCS, Clownshark, prep/finish, and original looper repair notes are preserved as project history. They should not be read as active shipping architecture unless a future development thread explicitly reopens them.

---

## Executive Summary

At takeover, the repo contained several overlapping architectural eras:

- the original fork lineage from `I2VLooperHL`
- Claude / Claude Code-era rebuild prompts and experiments
- hybrid prep/finish and Clownshark-based plans
- a native SVI path that was trying to reproduce a proven KJNodes workflow inside one custom node

The project has now been simplified around one release path:

- **`Wan Loop Config SVI`** — per-segment prompt, frame count, optional per-segment model overrides, and optional per-segment anchor image override
- **`Wan Looper SVI`** — the SVI loop driver that manages anchor choice, prompt encoding, sampling, decoding, overlap stitching, and final assembly

The active package exports only those two SVI nodes, plus backward-compatible aliases for existing workflow JSONs:

- `LoopConfigSVI`
- `SVILooperNative`

---

## Current Shipping Shape

The repo now centers on `nodes_wan_v2.py`.

Current node behavior:

- up to 10 connected `WAN_LOOP_CONFIG` inputs
- no separate `num_loops`; active segment count is derived from connected configs
- per-segment prompts and frame counts
- optional per-segment `model_high` / `model_low` overrides
- optional per-segment `anchor_image` override
- explicit `anchor_mode`, `stitch_mode`, `seed_mode`, `overlap`, `startup_trim`, and `anchor_frame_offset` policies
- KJNodes-backed SVI conditioning and overlap calls
- final outputs: `full_video`, `last_extracted_anchor`, and `All Segment Prompts`

The current seed modes are:

- `fixed`
- `randomize`

`increment_per_segment` was removed after audit because it caused compounding identity drift over longer chains.

---

## Important Corrections Since Takeover

### KJNodes Dependency Pivot

The earlier native path tried to reproduce SVI behavior inline. That was useful for understanding the architecture, but repeated tests showed it was not the right release direction.

The looper now calls the real KJNodes-backed pieces:

- `WanImageToVideoSVIPro`
- `ScheduledCFGGuidance`
- `ImageBatchExtendWithOverlap`

That makes the custom node responsible for loop orchestration while leaving the proven SVI conditioning and overlap behavior to KJNodes.

### Cross-Platform Loader Hardening

The dependency loader was made more robust for local Windows and RunPod/Linux use:

- discovers `custom_nodes` from `folder_paths`
- detects KJNodes folders instead of relying on a hardcoded path
- can resolve already-loaded KJ classes from `sys.modules`
- rejects false positives that are not real classes
- emits useful diagnostics if dependencies cannot be loaded

### Seed Audit

The April 18 audit found that `increment_per_segment` was the main source of longer-chain identity drift compared with the chained-subgraph reference workflow.

The reference workflow effectively used a shared noise trajectory. The looper matched that behavior when using `seed_mode=fixed`, so `fixed` became the default and `increment_per_segment` was removed.

### Overlap Re-Save Fix

The audit also found a segment re-save bug when `overlap > 0`: the previous segment could be re-saved from a raw pre-blend tensor, clobbering blend frames from the previous seam.

The fix reloads the previous segment from disk before trimming its overlap tail, preserving already-blended head frames. After the fix, `overlap=5` and `startup_trim=5` compose correctly.

---

## Current Known Behavior

The SVI path is credible as the release candidate, but it is still pre-release and intentionally documented with caveats.

Known characteristics:

- brightness drift can accumulate across long chains
- VAE encode/decode loss can soften fine detail over repeated segments
- `startup_trim=5` tested well with `overlap=5`, but the UI default remains `startup_trim=0`
- `fixed_initial` is the conservative default anchor mode
- `dynamic_every_segment` and `keyframe_schedule` remain available for experiments that need rolling or scheduled anchors

These behaviors are described in the README and the SVI native reference.

---

## KSA Status

KSA work is no longer part of the current shipping package.

The older KSA notes are still useful as history because they explain the path not taken:

- preserving the original `WanImageToVideo + CLIPVisionEncode + KSamplerAdvanced` feel
- trying to improve hard-cut seams with overlap helpers
- simplifying an earlier god-node interface into config-plus-driver nodes

That work is now deferred. It should not be described as an active second release path in current docs.

---

## Workflow and Example Progress

Current committed example:

- `workflows/ComfyUI-WanLooper_example_workflow.json`
- `workflows/ComfyUI-WanLooper_example_workflow.png`
- `examples/example_start_image.png`
- `examples/2026-04-19_WanLooper_testing_00001.mp4`

Current work in progress:

- `workflows/ComfyUI-WanLooper_example_workflow_advanced_v3.1.json`

The advanced workflow is intended to become the second example workflow after cleanup and rename, likely:

- `workflows/ComfyUI-WanLooper_example_workflow_advanced.json`

Before publishing it, remove testing-specific naming and any environment-specific output metadata that does not belong in a portable example workflow.

---

## Recommended Next Work

The useful next sequence is:

1. keep `nodes_wan_v2.py` stable unless workflow cleanup exposes a real bug
2. finish and rename the advanced example workflow
3. update README workflow listings to include the advanced workflow
4. make historical docs clearly point to the current SVI-only docs
5. avoid reviving KSA language in release-facing documentation unless KSA development explicitly resumes

Current mental model:

- **Shipping path:** `Wan Looper SVI`
- **Historical/deferred path:** KSA and older looper repair work
- **Immediate packaging task:** advanced workflow cleanup and docs alignment
