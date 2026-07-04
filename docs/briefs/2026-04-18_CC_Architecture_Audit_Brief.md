# CC Architecture Audit Brief — Wan Looper SVI
**Date:** April 18, 2026
**Scope:** Investigate suspected architectural divergence between `WanLooperNative` and the reference chained-subgraph SVI Pro workflow
**Repo:** `ComfyUI-WanLooper`
**Working file:** `nodes_wan_v2.py`
**Companion report:** `2026-04-18_Session_Findings_Claude_Chat.md`

---

## Context

This brief is a handoff from a long Claude.ai chat session on April 18, 2026 that produced more data than one chat window could carry. The session alternated between running tests on the looper, comparing against a reference chained-subgraph workflow, and diagnosing the results. Claude made several overconfident diagnoses that had to be walked back — treat anything in the session findings report as "provisional, to be verified" rather than settled.

The bottom line at handoff: **the looper produces visibly worse mid-chain content quality than a structurally equivalent chained-subgraph workflow, at the same seed**. That shouldn't happen if the looper is doing exactly what the reference does. Something in the looper's execution path is diverging from what native node execution would produce, and the goal of this CC session is to find it.

The user (Barley) is no longer in "ship this next week" mode. The project is a hobby experiment with no deadlines. Correctness and understanding matter more than closing tickets.

## State of the looper as of handoff

### What is believed to work

- The KJNodes loader and resolution logic. Loads `WanImageToVideoSVIPro`, `ImageBatchExtendWithOverlap`, and `ScheduledCFGGuidance` correctly from the installed KJNodes pack.
- The overall segment loop structure (per-segment config reading, prompt routing via `PromptSegmentSelector`, anchor mode selection, VAE encode/decode cycle).
- Per-segment seed assignment logic under `seed_mode=increment_per_segment` and `seed_mode=fixed`. Console output confirms seeds are being assigned as intended.
- `startup_trim` as a mechanism for dropping segment-opening transient frames. Produces clean seams in isolation.

### What has a known bug

**The `overlap > 0` re-save interaction.** At the start of each segment N+1's iteration, the looper overwrites segment N's disk save using `prev_decoded[:-overlap]` where `prev_decoded` holds the **raw `decoded_for_stitch` content**, not the previously-saved `segment_frames` (which contained the blend). Consequences:

1. The blend frames written at segment N's disk head by the KJ overlap call get clobbered by the next segment's re-save.
2. Only the final segment retains its blend, because no subsequent segment exists to overwrite it.
3. When `overlap > 0` and `startup_trim > 0` are both set, each seam ends up being a hard cut between `seg_N_decoded[last_non-trimmed_frame]` and `seg_N+1_decoded[post-startup-trim_frame]` — a 10-frame time-skip with no blend bridging it.

Empirical evidence: `overlap=0, trim=5` produced seam diffs in the 4-7 range. `overlap=5, trim=5` produced seam diffs in the 15-17 range at the same frame positions. The overlap path is actively worse than no overlap, not better.

Relevant code region in `nodes_wan_v2.py` lines 882-922. Full trace in the session findings report.

### What is suspected but unverified

**The looper produces worse mid-chain content quality than the reference workflow at equivalent positions, even controlling for seed and settings.** Specifically, at around mid-segment-3 in a 6-segment run:

- Looper output (test `2026-04-18_WanLooper_testing_00005.mp4`, frame 182): tattoo on the dancer's forearm has degraded into chaotic dark scribbles with green color bleeding. Frame 220: the extended arm has smeared into a shapeless mass with no anatomical articulation.
- Reference output (chained-subgraph workflow, same seed 41169, same prompts, same LoRA stack, same 6×81f/6step/split-3 settings, saved as PNG sequence): frames 182/220 equivalent show legitimate motion blur but anatomically coherent bodies. Tattoo remains a coherent shape. Arm has clear bicep/forearm articulation. Late-seg-6 frames still have readable necklace chain links and clean facial geometry.

Claude's best current hypothesis: something in the inlined SVIPro / ScheduledCFGGuidance / SamplerCustomAdvanced call path in the looper is not quite matching what native node execution does. This would produce clean output on seg 1 (where there's no prior context to drift from) and progressively degrading output as each segment's faulty `prev_samples` chain compounds.

**Caveats on this hypothesis:**

1. The reference workflow uses a shared noise tensor across all SamplerCustomAdvanced instances (one `RandomNoise` node feeding all 6 SCAs). The looper with `seed_mode=increment_per_segment` rolls a fresh seed per segment (41169, 41170, ..., 41174). This is a real difference that could explain quality divergence without any architectural bug.
2. The reference workflow's brightness drift trajectory is opposite to the looper's. Reference goes darker (~18 units) over 6 segments; looper goes brighter (~29 units) over the same span. Both drift, but in different directions. Could be seed-related or could be architectural.
3. Claude analyzed some reference frames upscaled to 640×960 by DaVinci, and the "cleaner rendering" observation may have been partially a smoothing artifact. Native 480×640 comparison is pending.

## The next test — the seed_mode=fixed diagnostic

Before auditing the inlined call path, run one controlled test to rule out the seed trajectory as the cause.

### Test setup

All settings identical to test `2026-04-18_WanLooper_testing_00005.mp4` (overlap=0, trim=5 config), **except change `seed_mode` from `increment_per_segment` to `fixed`**. This makes every segment use seed 41169, matching the reference workflow's shared-noise behavior.

Full settings:
- 6 loop configs × 81 frames each, dance-arc prompts via `PromptSegmentSelector`
- 6 steps, split_step 3, cfg 1.0, sampler euler, scheduler simple
- initial_seed 41169
- **seed_mode: fixed** ← the change
- overlap: 0
- startup_trim: 5
- overlap_mode: linear_blend (no-op at overlap=0 but keep for consistency)
- overlap_side: source
- anchor_mode: fixed_initial
- stitch_mode: workflow_style
- anchor_frame_offset: -5
- color_correction: false
- Same dancer image as all previous tests
- Models: Wan 2.2 I2V A14B Q5_K_M GGUF (HN + LN)
- LoRAs at strength 1.0: SVI Pro v2 HIGH/LOW, LightX2V HIGH (1030 rank 64), LightX2V LOW (old 480p rank 64)

### What to compare

After the run completes, compare the new test output to the overlap=0 baseline (`2026-04-18_WanLooper_testing_00005.mp4`) and to the reference workflow PNG output, at these specific frames:

- Frame 182 (mid-seg-3): does the tattoo still break down into scribbles?
- Frame 220 (mid-seg-3): does the extended arm still smear into a shapeless mass?
- Late seg 6 frames (around frames 430-460 in the 461-frame output): does late-chain content quality match what the reference produces?

### Interpretation

**If the seed_mode=fixed test produces content quality matching the reference workflow**, the looper's architecture is sound. The earlier "bad frames at 182/220" were seed-draw bad luck, not architectural bugs. Next steps become: (a) decide whether fixed seed should be the default, (b) document clearly, (c) fix the overlap re-save bug separately, (d) move on to drift mitigation testing.

**If the seed_mode=fixed test still produces scribble-texture tattoos and shapeless arms at mid-seg-3**, the looper has a real architectural divergence from the reference. Proceed to the inlined call path audit below.

## Architectural audit — if seed_mode=fixed doesn't fix it

This section is contingent on the test above showing remaining quality issues. Do not start this work unless the diagnostic confirms it is needed.

The looper inlines three things that would otherwise be native node calls:

### 1. `_wan_pro_condition` (inlined from KJ's V3 `WanImageToVideoSVIPro`)

Location: `nodes_wan_v2.py` around lines 292+ (function definition) and called at the conditioning step for each segment.

The reference workflow uses `WanImageToVideoSVIPro` via its native node execution path. The looper previously called `WanImageToVideoSVIPro.execute(...)` directly but was refactored to inline the logic because the V3 API doesn't always cleanly translate to direct `.execute()` calls.

**What to audit:**
- Compare the inlined Python against KJ's actual V3 implementation in the installed KJNodes pack (local path: `E:\ComfyUI\ComfyUI\custom_nodes\comfyui-kjnodes`, likely in `nodes/model_optimization_nodes.py` or similar). Look specifically at how `motion_latent_count` interacts with `prev_samples`, how `concat_latent_image` and `concat_mask` are constructed, and whether there's any dtype or shape normalization that the native path does but the inline doesn't.
- In particular, verify the mask construction. The looper creates a mask that zeroes out the first latent slot; confirm this matches what the native node does for both the `prev_samples=None` case (seg 1) and the `prev_samples=something` case (segs 2+).
- Verify the `image_cond_latent` concatenation order matches native: anchor first, then motion latent. Verify the dtype cast is identical.

### 2. `Guider_ScheduledCFG` (inlined from KJ's `ScheduledCFGGuidance`)

Location: `nodes_wan_v2.py` line 229+.

The reference uses `ScheduledCFGGuidance.get_guider(...)` to produce a guider object, which is then passed to `SamplerCustomAdvanced`. The looper inlines this as `_build_scheduled_cfg_guider`.

**What to audit:**
- Compare against KJ's `ScheduledCFGGuidance` source. Verify the guider class the looper returns implements the same `predict_noise` / `__call__` interface that `SamplerCustomAdvanced` expects.
- Check that `start_percent` and `end_percent` are interpreted the same way (fraction of sampling, not absolute step number).
- Verify that when the guider is called with a sigma outside its active range, it correctly falls back to unconditional (or whatever the native behavior is).

### 3. `SamplerCustomAdvanced.execute(...)` invocation

Location: `nodes_wan_v2.py` lines 797-803 (high pass) and 812-818 (low pass).

The reference uses two native SamplerCustomAdvanced nodes in series, each taking its guider / sigma / latent / noise inputs as node connections. The looper calls `SamplerCustomAdvanced.execute()` directly with the same arguments.

**What to audit:**
- Native node execution goes through ComfyUI's scheduler (the `prompt_executor`, not the diffusion scheduler) which may do additional preprocessing — tensor device transfers, dtype casting, memory pinning — that direct `.execute()` calls bypass.
- Check whether `latent_image` needs to be wrapped differently when passed to native vs inline. The looper does `{"samples": high_result[0]["samples"]}` to pass from high pass to low pass — verify this matches what node-connected execution would pass.
- Verify `Noise_RandomNoise(seed)` produces the same noise tensor that a native `RandomNoise` node with the same seed would produce. They should be equivalent but worth confirming.

### 4. `prev_samples` carry-forward

Location: `nodes_wan_v2.py` line 821: `prev_latent_samples = {"samples": low_result[0]["samples"].clone()}`.

**What to audit:**
- Is the `.clone()` doing the right thing? It should be producing a detached copy that can't be affected by downstream tensor operations. Verify no gradient tracking, no shared storage.
- Compare against how the reference workflow's Set/Get of the `latent` output from `SamplerCustomAdvanced` propagates. The native Set/Get should be storing a reference or a serialized copy — check which, and whether the looper's `.clone()` matches that semantics.

## Recommended workflow for the audit

If audit is needed, CC should:

1. Start by reading `nodes_wan_v2.py` fully to get full mental context of the file.
2. Read the reference workflow JSON (`svi_revisited_local_4-15-26_dancer_test_5_.json` in the test files collection) to see exactly how the native path is wired.
3. Read the relevant KJNodes source files from the local install path. Specifically:
   - The `WanImageToVideoSVIPro` V3 node definition
   - The `ScheduledCFGGuidance` class
   - The `ImageBatchExtendWithOverlap` class (already verified this is correct; can skip)
4. Write a test that calls the native node path and the inline path with identical inputs and compares output tensors. This is the definitive way to confirm architectural equivalence.
5. Only make code changes after confirming the specific divergence. No speculative fixes.

## Known-good reference files

When doing the audit, these files on disk represent the verified-good state:
- **Reference workflow JSON:** `svi_revisited_local_4-15-26_dancer_test_5_.json` (chained-subgraph workflow, same seed 41169, produced clean mid-chain content)
- **Reference output (PNG sequence):** exported to `Wan22_SVI_Pro_dance_test_00002_5_.zip`, 461 frames at 640×960 (DaVinci-upscaled) or wherever the native MP4 was saved at 480×640

Claude's analysis in this Claude.ai session compared looper vs reference at frames 182, 220, 430, 460. The reference was clearly superior at all three positions.

## Separate cleanup work (not dependent on audit outcome)

Regardless of what the audit finds, the following code changes are worth doing:

### Fix 1: The overlap re-save bug

In `nodes_wan_v2.py` line 892-893, the re-save uses `prev_decoded[:-eff_overlap]` which contains raw pre-blend content. This destroys the blend frames that were saved at line 914 for the previous segment.

**Proposed fix** (to be discussed before implementing):
```python
# Re-save previous segment with its tail trimmed for the coming seam,
# preserving any leading blend frames in its existing disk save.
prev_saved = torch.load(segment_paths[-1], map_location="cpu", weights_only=True)
trimmed_prev = prev_saved[:-eff_overlap]
torch.save(trimmed_prev.cpu(), segment_paths[-1])
```

This loads what's actually on disk for segment N (which contains the blend from its own overlap with segment N-1 at positions 0..overlap-1) and trims only the tail, preserving the leading blend.

**Before implementing:** verify this doesn't break anything with `startup_trim > 0`. The interaction between overlap and startup_trim is what caused the 10-frame time-skip observed earlier; the fix needs to produce a coherent result, not just preserve the blend.

### Fix 2: Expose `motion_latent_count` as a widget

Currently hardcoded to 1 at line 661. Community consensus from user's notes is that values >1 don't help seam continuity but might help drift control (untested). User wants to experiment. Default should remain 1, min 0, with tooltip noting it is effectively 0 for segment 1 regardless of setting (because `prev_samples=None` short-circuits).

### Fix 3: Misleading log message

Line 738 prints `motion=1` for segment 1, but the effective behavior is motion=0 (prev_samples is None). Cosmetic — change log to read correctly.

### Fix 4: `startup_trim` → `trim_leading_frames` rename

The name `startup_trim` is not descriptive. `trim_leading_frames` is clearer and matches what the parameter actually does. Keep old name as backward-compat alias.

### Fix 5: Node naming finalization

Per project briefs: `Wan Looper SVI` (display) / `WanLooperNative` (class). Backward-compat aliases for old class names already planned.

## What to NOT do

- Do not change default values (`overlap=5`, `startup_trim=0`) yet. User may want these changed based on audit findings, but that's a decision for after diagnostics complete.
- Do not remove the `overlap > 0` code path even though it's currently buggy. It has a correct fix (Fix 1 above), and removing it would cut a feature users may want once fixed.
- Do not touch the KSA path. This audit is SVI-only. Per project state, KSA is still under repair and has its own open issues.
- Do not try to "fix" the content quality issue with hotfixes before the architectural audit confirms where the problem is. Claude cycled through several wrong diagnoses in the source session; the pattern to break is jumping from observation to confident fix without verification.

## User preferences (carried forward from prior sessions)

- Accuracy over speed. Say "uncertain" rather than guess. Correct errors when caught.
- Prefer minimal custom nodes. Research existing solutions before building new ones.
- Cloud portability matters (Runpod, Mimic PC). Any custom code should work there.
- Change one variable at a time when testing. Report results before proceeding.
- Prefer Claude Opus (not Sonnet) for CC sessions requiring precision — especially ones involving exact filenames, type-accurate JSON, and source signature matching.

## Immediate first action

Queue the `seed_mode=fixed` test as described above. Nothing else to do until that test's output is available and the user has compared it to the reference.
