# Native-Resolution Comparison — Reference vs WanLooper (overlap=0)
**Date:** April 18, 2026 (evening)
**Companion to:** `2026-04-18_Session_Findings_Claude_Chat.md`, `2026-04-18_CC_Architecture_Audit_Brief.md`

## Purpose

Replace the earlier DaVinci-upscaled 640×960 comparison with a native-resolution (480×640) apples-to-apples comparison between the reference chained-subgraph workflow output and WanLooper test C (overlap=0, trim=5, increment_per_segment, seed 41169).

All frames extracted at native 480×640. Reference from PNG sequence (`Wan22_SVI_Pro_dance_test_00002_5_.zip`). Looper from `2026-04-18_WanLooper_testing_00005.mp4`.

## Corrections to earlier session findings

### Brightness drift direction (looper)

The earlier session reported the looper drifts **+29 brightness units seg 1 → seg 6**. This is **incorrect**.

Measured at native resolution, content region only (letterbox excluded):

| frame | looper mean | looper drift | ref mean | ref drift |
|-------|-------------|--------------|----------|-----------|
| 0 | 122.15 | — | 106.47 | — |
| 182 | 109.15 | −12.99 | 99.42 | −7.05 |
| 220 | 112.16 | −9.99 | 100.47 | −6.00 |
| 430 | 95.59 | −26.56 | 88.42 | −18.05 |
| 460 | 98.21 | −23.94 | 90.66 | −15.81 |

**Both systems darken across the chain.** The looper darkens faster (~−24 vs ~−16 by frame 460) and starts from a higher baseline (~+16 units brighter at frame 0).

The earlier session's "+29 brightness" figure may have been measured including letterbox pixels, on DaVinci-upscaled exports, on a different test, or via a different methodology. Flagging but not investigating — the direction of drift is what matters and both systems drift the same direction.

### Content-quality characterization (looper)

The earlier session reported looper frame 182 shows "tattoo degrading into scribbles with greenish color bleeding" and frame 220 shows "extended arm smearing into shapeless mass."

Re-examining at native resolution: **the looper output at 182 and 220 is not degraded in that way.** Frame 182 shows a legitimate tattoo on the extended left forearm, anatomically coherent. Frame 220 shows arms extended in motion with mild motion blur but no shapeless-mass failure.

This does not mean the looper output matches the reference — it doesn't. But the failure mode is **character identity drift and scene drift**, not latent-space coherence collapse:

- Character identity shifts from the sharp-jawed blonde at frame 0 to a softer-featured person by frame 430.
- At frame 430, an arched window emerges in the background that wasn't in the start image.
- The necklace at frame 0 (two chains with ornate pendants) is a single thin chain by frame 430 and gone entirely by frame 460.
- Tattoo position and style shifts across the chain.

The reference workflow shows none of these drifts. Character and scene are preserved across all 460 frames.

## What's confirmed by this comparison

1. **Both systems render coherent images at every tested frame.** No latent collapse in either. Any earlier claim of latent-space coherence failure in the looper is not supported by the native-resolution evidence.
2. **The reference workflow preserves identity and scene better than the looper** across 6 segments. This is a real, visible difference that holds up at native resolution.
3. **Both systems drift in brightness** — same direction, looper darkens faster.

## What remains unverified

The character/scene drift difference could be caused by:

1. **Seed trajectory** (looper increments per segment; reference shares one noise tensor). Different noise seeds produce different drift trajectories.
2. **Architectural divergence** in the inlined SVIPro / ScheduledCFGGuidance / SamplerCustomAdvanced call path.
3. **Some combination** — for example, shared-noise being *more forgiving* of architectural differences by stabilizing the latent trajectory.

The `seed_mode=fixed` diagnostic test described in the audit brief isolates (1). Same priority as before.

## What changes for the CC audit brief

Nothing about the diagnostic plan changes. The `seed_mode=fixed` test is still the next move.

What changes is the **expected signal** from the test:

- If fixed seed makes the looper preserve identity and scene as well as the reference, the architecture is probably sound and seed trajectory was the dominant factor.
- If fixed seed still shows identity/scene drift vs the reference, architectural audit is warranted — but the audit should focus on **why identity-preserving conditioning behaves differently**, not on hunting for latent-collapse bugs. The failure mode is subtler than originally characterized.

## Artifacts

- `2026-04-18_ref_vs_looper_comparison.png` — side-by-side grid of reference (top row) vs looper (bottom row) at frames 0, 182, 220, 430, 460.
- Both source frame sets extracted and retained for any follow-up comparisons.
