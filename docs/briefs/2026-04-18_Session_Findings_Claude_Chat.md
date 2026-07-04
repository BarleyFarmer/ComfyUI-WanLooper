# Session Findings — Claude.ai chat, April 18, 2026
**Scope:** Multi-session debugging of `WanLooperNative` seam and content quality at 6-segment runs
**Repo:** `ComfyUI-WanLooper`
**Companion brief:** `2026-04-18_CC_Architecture_Audit_Brief.md`

---

## Summary

A long Claude.ai chat session on April 18, 2026 ran a series of 6-segment tests on the looper and compared against a reference chained-subgraph SVI Pro workflow. The session produced useful data on seam behavior and a strong-but-unverified hypothesis of architectural divergence between the looper and the reference.

Claude made several overconfident diagnoses during the session and had to walk them back. This report separates **what was measured** (reliable) from **what was hypothesized** (needs verification).

## Test configuration (constant across runs unless noted)

- 6 loop configs × 81 frames each
- 6 steps, split 3, cfg 1.0, sampler euler, scheduler simple
- seed 41169
- anchor_mode: fixed_initial, stitch_mode: workflow_style, anchor_frame_offset: -5, color_correction: false
- Dancer image, 6-prompt dance arc via `PromptSegmentSelector`
- Models: Wan 2.2 I2V A14B Q5_K_M GGUF (HN + LN)
- LoRAs @ 1.0: SVI Pro v2 HIGH/LOW, LightX2V HIGH (1030 rank 64), LightX2V LOW (old 480p rank 64)
- Resolution 480×640

## Tests run in this session

| Test | File | overlap | trim | blend mode | seed mode |
|------|------|---------|------|------------|-----------|
| A | `2026-04-18_WanLooper_testing_00002.mp4` | 5 | 5 | linear_blend | increment_per_segment |
| B | `2026-04-18_WanLooper_testing_00003.mp4` | 5 | 5 | ease_in_out | increment_per_segment |
| C | `2026-04-18_WanLooper_testing_00005.mp4` | 0 | 5 | linear_blend (no-op) | increment_per_segment |
| Ref | chained-subgraph workflow output | 5 | 0 | linear_blend | shared-noise (41169) |

A LightX2V LoRA A/B test was planned (test 00004) but never uploaded.

## What was measured (confident)

### Seam behavior

- Tests A and B (overlap=5, trim=5 with different blend modes) produced pixel-identical output for frames 0-323 at native resolution. Divergence begins gradually at frame 324, growing to MAE ~1.3 out of 255 by the end. Visually imperceptible difference confirmed by user via DaVinci comparison.
- Test C (overlap=0, trim=5) produced dramatically cleaner seams: max frame-to-frame diff 17.29 (not at a seam) vs test A's 17.39 at seam 2. Specifically, seam diffs in test C ranged 4-7; in tests A/B they ranged 15-17.
- All three tests produced similar large brightness drift across 6 segments (~+29 units seg 1 → seg 6).

### Re-save interaction bug

Traced through the code at `nodes_wan_v2.py` lines 882-922. Confirmed that with `overlap > 0`:

1. Segment N+1's iteration starts by re-saving segment N's disk content as `prev_decoded[:-overlap]`.
2. `prev_decoded` contains raw `decoded_for_stitch` content, NOT the previously-saved `segment_frames` which contained the blend.
3. The blend frames written at segment N's disk head get clobbered.
4. Only the final segment retains its blend frames (no subsequent segment to overwrite).
5. With `startup_trim > 0` also set, each seam becomes a hard cut between `seg_N[last_non-trimmed_frame]` and `seg_N+1[post-trim_frame]` — a 10-frame time-skip with no blend bridging it. This explains the 15-17 seam diffs in tests A/B.

### Reference workflow observations

The chained-subgraph workflow was run at 6 segments with the same seed, same LoRAs, same settings (with the difference that it uses shared RandomNoise across all 6 SamplerCustomAdvanced instances, while the looper uses increment_per_segment). Output was exported as a PNG sequence.

- Reference brightness trajectory: darkened by ~18 units seg 1 → seg 6 (opposite direction to the looper's ~+29).
- Reference output at mid-seg-3 (frames 182, 220 in final output) shows anatomically coherent bodies with legitimate motion blur.
- Reference output at late seg 6 (frames 430-460) retains clean hair, necklace chain detail, readable face structure.

## What was hypothesized (needs verification)

Update 2026-04-18 evening: Native-resolution reference frames were extracted and compared. Reference shows clean anatomy at frames 182, 220, and 430 where the looper shows degradation. This strengthens the architectural-divergence hypothesis but does not confirm it — seed trajectory is still a confound. The seed_mode=fixed test remains the next step.

### Looper diverges architecturally from reference

The looper's test C output at the same frame positions shows content-quality degradation not present in the reference:

- Frame 182: the dancer's forearm tattoo degrades into chaotic dark scribbles with greenish color bleeding.
- Frame 220: the extended arm smears into a shapeless mass with no anatomical articulation.
- Late-chain quality is expected to be worse than the reference based on the seg-3 trajectory, but was not systematically compared at native resolution.

**Claude's hypothesis:** something in the inlined SVIPro / ScheduledCFGGuidance / SamplerCustomAdvanced call path in the looper is not matching native node execution. This would produce clean seg 1 output (no prior context) and progressively degrading output as faulty `prev_samples` chain compounds.

**Why this is unverified:**

1. The looper uses `seed_mode=increment_per_segment` (41169, 41170, ..., 41174) while the reference uses a shared noise tensor. Different seed trajectories alone could produce different quality at any given frame. A seed-matched test is needed to separate seed variance from architectural divergence.

2. Some of Claude's reference-workflow frame analysis was done on DaVinci-upscaled 640×960 exports rather than native 480×640. Some of the "cleaner rendering" read may be smoothing from the upscale rather than true quality preservation.

3. Claude cycled through multiple incorrect diagnoses during the session (initially claiming `overlap_mode` was a no-op for all but the last seam, then claiming test 2 and test 3 were bit-identical for most frames at full resolution, then finding they were not). The pattern is jumping from observation to confident diagnosis without enough verification. Treat the architectural-divergence hypothesis as provisional until tested.

## What the diagnostic next step is

Run the looper at the same settings as test C, except change `seed_mode` from `increment_per_segment` to `fixed`. This makes all 6 segments use seed 41169, matching the reference's shared-noise behavior.

Compare the output at frames 182, 220, and late seg 6 (around 430-460) against both test C and the reference PNG sequence.

- If the seed_mode=fixed test matches the reference at these positions, the looper architecture is sound and the earlier degradation was seed-random. Overlap re-save bug (known) still needs fixing.
- If the seed_mode=fixed test still produces scribble textures and shapeless limbs, the inlined call path in the looper has a real divergence and needs a line-by-line audit against KJ's native nodes.

## Artifacts for reference

The following files in this repo's `docs/briefs/` folder and related test collections should be kept:

- `2026-04-18_CC_Architecture_Audit_Brief.md` (this report's companion)
- `2026-04-18_WanLooper_testing_00002.mp4` (overlap=5 trim=5 linear_blend)
- `2026-04-18_WanLooper_testing_00003.mp4` (overlap=5 trim=5 ease_in_out)
- `2026-04-18_WanLooper_testing_00005.mp4` (overlap=0 trim=5 — cleanest seams, content degradation at mid-chain)
- Reference workflow JSON: `svi_revisited_local_4-15-26_dancer_test_5_.json`
- Reference workflow PNG zip: `Wan22_SVI_Pro_dance_test_00002_5_.zip`

## User context

- User is no longer in "ship soon" mode. Project is hobby experimentation with no deadlines.
- User has equal priority on seam quality and drift — either failing independently makes output unusable.
- User has independently validated that dynamic anchors help with drift across longer chains, and has a separate (parked) refined-anchor workflow for anchor cleanup via image-edit models.
- User prefers one-variable-at-a-time testing with results reported before proceeding, and prefers accuracy over speed.

## Outstanding items across sessions (not this session's scope)

- LightX2V new (260412) vs old (1030) LoRA A/B comparison — never completed
- 6-segment drift mitigation testing (`color_correction=true`, dynamic anchors, LoRA strength tapering, frame-count tapering)
- Subgraph blueprint comparison — user raised this as an alternative to the looper. Worth building a blueprint version for direct comparison once looper architecture is verified, to determine whether the looper's policy-control advantages justify the custom node maintenance burden.
