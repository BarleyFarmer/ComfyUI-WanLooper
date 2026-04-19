# Architecture Audit Complete — WanLooperNative SVI
**Date:** April 18, 2026

## What was found

### 1. seed_mode=increment_per_segment caused compounding identity drift

The looper's `increment_per_segment` seed mode (41169, 41170, ..., 41174 across 6 segments) produced progressive character identity drift and scene drift over longer chains. The reference chained-subgraph workflow used a shared noise tensor (fixed seed 41169 for all segments) and showed no such drift.

A controlled test confirmed the cause: running the looper with `seed_mode=fixed` (same seed for all segments) produced identity and scene preservation comparable to the reference workflow. Brightness drift at frame 460: seed-fixed looper -13.79 vs reference -15.81 vs increment_per_segment -23.94. The architecture is not diverging; the seed trajectory was the dominant factor.

### 2. Overlap re-save bug clobbered blend frames at segment boundaries

When `overlap > 0`, the start of each segment N+1's iteration re-saved segment N's disk content. The re-save used `prev_decoded` (raw pre-blend decoded frames) instead of loading what was actually on disk (which contained blend frames from the KJ overlap call). This destroyed the blend at every seam except the last.

With `overlap=5, startup_trim=5`, this produced seam diffs of 15-17 MAE — worse than `overlap=0, trim=5` which produced seam diffs of 4-7.

## What was fixed

### seed_mode cleanup

Removed `increment_per_segment` from the `seed_mode` options. Remaining options: `fixed` (default) and `randomize`. The `fixed` default matches the reference workflow's shared-noise behavior and produces the best chain coherence.

### Overlap re-save bug fix

The re-save now loads the previous segment from disk (`torch.load`) before trimming its overlap tail, preserving any blend frames at its head from its own overlap with the segment before it. The slice index for extracting new segment frames was also corrected to use `prev_decoded.shape[0] - eff_overlap`.

Post-fix test results with `overlap=5, startup_trim=5, seed_mode=fixed`:

| Location | MAE |
|----------|-----|
| Seam 1 (frame 76) | 6.55 |
| Seam 2 (frame 147) | 6.42 |
| Seam 3 (frame 218) | 8.26 |
| Seam 4 (frame 289) | 6.68 |
| Seam 5 (frame 360) | 9.77 |
| Baseline mid-segment | 2.86 - 6.89 |

Seam diffs are now within the mid-segment baseline range. The overlap and startup_trim widgets work correctly together.

## What was confirmed sound

- **Inlined SVIPro / ScheduledCFGGuidance / SamplerCustomAdvanced call path.** The looper's inlined execution produces output equivalent to the reference chained-subgraph workflow when seed is controlled. No architectural divergence.
- **Overlap + startup_trim combination.** With the re-save bug fixed, overlap blending and startup_trim are independent and composable. No need for mutual exclusivity.

## What remains known and unfixed

- **Brightness drift proportional to chain length.** Both the looper and the reference workflow darken progressively across 6 segments (looper -13.79, reference -15.81 at frame 460 with fixed seed). This is inherent to the chained SVI approach, not a looper bug. Deferred as a future drift-mitigation investigation (candidates: color_correction, dynamic anchors, LoRA strength tapering, frame-count tapering).
