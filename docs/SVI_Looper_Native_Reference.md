# SVI Looper Native Reference

This document is the current reference for the SVI path in this repo. It is based on the live implementation in `nodes_wan_v2.py`, not on older design briefs.

The main shipping nodes are:

- `Wan Looper SVI` (`WanLooperNative`; legacy alias `SVILooperNative`)
- `Wan Loop Config SVI` (`LoopConfigWan`; legacy alias `LoopConfigSVI`)

Historical briefs in `docs/briefs/` are useful for design history and audit context, but several of them describe earlier interfaces or bugs that have since changed.

## What the node does

`Wan Looper SVI` runs a multi-segment WAN 2.2 image-to-video chain inside one node. Each segment:

- chooses an anchor image
- encodes that anchor through the VAE
- encodes the segment prompt through the text encoder
- calls KJNodes' real SVI conditioning path with `prev_samples`
- samples with the high-noise model first and the low-noise model second
- decodes the result
- extracts an anchor frame for later segments
- optionally trims and overlap-blends it into the running sequence

The implementation is deliberately policy-driven. The key behavior knobs are:

- `anchor_mode`
- `stitch_mode`
- `seed_mode`
- `startup_trim`
- `overlap`
- `keyframe_schedule`

## Required inputs

### `model_high` / `model_low` (`MODEL`)

Global high-noise and low-noise models. These are the default models for every segment unless a `Wan Loop Config SVI` node overrides them for a specific segment.

The looper expects fully prepared models. Any LoRA stack, shift node, or other model patching should happen upstream.

### `vae` (`VAE`)

Used twice:

- to encode the chosen anchor image into `anchor_samples`
- to decode each sampled segment back to frames

### `clip` (`CLIP`)

Used to encode the positive and negative prompts for each segment.

### `clip_vision` (`CLIP_VISION`)

Accepted for workflow compatibility but currently unused in the actual SVI conditioning path. Image conditioning comes from the VAE-encoded anchor image, not CLIP Vision.

### `start_image` (`IMAGE`)

The initial anchor image for the chain. The node resizes it internally to the requested `width` and `height`.

This image is always used for segment 1. After that, later segments may keep using it or switch to extracted anchors depending on `anchor_mode` and any per-segment `anchor_image` overrides.

### `width` / `height` (`INT`)

Output size for generated segments and prepared anchors.

Defaults:

- `width = 480`
- `height = 640`

Both must be multiples of 16.

### `steps` (`INT`)

Total sampler steps per segment across both model passes.

Default:

- `steps = 8`

### `split_step` (`INT`)

The split point between the high-noise and low-noise passes.

Default:

- `split_step = 3`

The node builds one sigma schedule, then splits it so:

- the high model runs the early portion
- the low model runs the later portion

### `cfg` (`FLOAT`)

CFG strength used in both scheduled guiders.

Default:

- `cfg = 1.0`

### `sampler_name` / `scheduler`

Sampler and scheduler choices used for every segment. The node does not impose special policy here beyond reusing the same choices across the run.

### `initial_seed` (`INT`)

Seed for segment 1.

Default:

- `initial_seed = 0`

How later segments use seeds depends on `seed_mode`.

## Segment policy inputs

### `seed_mode`

Options:

- `fixed`
- `randomize`

Default:

- `fixed`

Current behavior:

- `fixed`: every segment uses `initial_seed`
- `randomize`: segment 1 uses `initial_seed`, each later segment gets a fresh random 64-bit seed

Important history:

- older briefs mention `increment_per_segment`
- that mode was removed after audit because it caused compounding identity drift relative to the reference workflow

### `anchor_mode`

Options:

- `keyframe_schedule`
- `fixed_initial`
- `dynamic_every_segment`

Default:

- `fixed_initial`

This setting controls which anchor image is used for segments that do not have an explicit per-segment `anchor_image`.

#### `fixed_initial`

Every segment without an explicit override uses the resized `start_image`.

Effects:

- segment 1 uses `start_image`
- segment 2 also uses `start_image`
- segment 3 also uses `start_image`
- extracted anchors are still computed and returned, but they do not automatically become the next segment's anchor

This is the most conservative mode and is the current default.

#### `dynamic_every_segment`

Segment 1 uses `start_image`. Segment 2 onward uses the most recently extracted anchor frame from the previous segment.

Effects:

- segment 1 anchor source: `start_image`
- segment 2 anchor source: extracted anchor from segment 1
- segment 3 anchor source: extracted anchor from segment 2

Use this when you want a true rolling-anchor chain.

#### `keyframe_schedule`

The node maintains a scheduled anchor image. It starts as `start_image` and only updates when a segment id is listed in `keyframe_schedule`, or when a segment provides an explicit `anchor_image`.

Effects:

- all segments use the current scheduled anchor
- extracted anchors do not automatically replace it every segment
- only scheduled refresh points promote a newly extracted anchor into future use

This is the mode to use when you want anchor persistence across several segments, then a deliberate refresh.

### `keyframe_schedule`

Optional string input used only with `anchor_mode = keyframe_schedule`.

Examples:

- `3,6,9`
- `4-6`
- `3, 5, 8-10`

Meaning:

- if segment 3 is listed, the anchor extracted from segment 3 becomes the scheduled anchor for segment 4 and later
- if segment 6 is listed, the anchor extracted from segment 6 becomes the scheduled anchor for segment 7 and later

### `stitch_mode`

Options:

- `workflow_style`
- `trim_to_anchor`

Default:

- `workflow_style`

This setting decides how much of each decoded segment is eligible for stitching.

#### `workflow_style`

Keeps the full decoded segment before any `startup_trim` or overlap processing.

#### `trim_to_anchor`

Trims each segment to `decoded[:anchor_idx + 1]` before later stitch steps. This means frames after the extracted anchor are discarded.

This is the closest mode to "make the extracted anchor the true segment endpoint."

### `anchor_frame_offset`

Default:

- `-5`

Range:

- `-50` to `-1`

The extracted anchor frame index is computed from the decoded segment length plus this offset.

Examples:

- `-1`: use the last frame
- `-5`: use the fifth frame from the end

In practice, `-5` is a sensible default because it avoids the noisiest end-of-segment frames.

### `overlap`

Default:

- `5`

Number of frames used for overlap stitching between the previous segment and the current segment.

Behavior:

- `0`: no blend, hard join
- `> 0`: use KJNodes' overlap helper unless `overlap_mode = cut`

### `startup_trim`

Default:

- `0`

For segments after the first, remove this many leading frames from the current segment before overlap stitching.

Important ordering:

1. decode segment
2. optionally trim to anchor if `stitch_mode = trim_to_anchor`
3. apply `startup_trim` for segments after the first
4. perform overlap stitching if enabled

This means `startup_trim` affects what enters the overlap blend.

### `overlap_mode`

Options:

- `linear_blend`
- `ease_in_out`
- `filmic_crossfade`
- `cut`

Default:

- `linear_blend`

Special case:

- `cut` bypasses blend behavior even if `overlap > 0`

### `overlap_side`

Options:

- `source`
- `new_images`

Default:

- `source`

Passed directly into the KJ overlap helper to determine seam directionality.

### `color_correction`

Default:

- `False`

If enabled, the extracted anchor frame is normalized back toward the color statistics of the resized `start_image`.

This happens after anchor extraction and before the anchor is stored as:

- `last_extracted_anchor`
- `dynamic_anchor_image`
- `scheduled_anchor_image` when a scheduled refresh occurs

## Optional prompt and segment inputs

### `positive_prompt`

Optional global fallback positive prompt.

Per-segment prompt resolution is:

1. `loop_config["prompt"].strip()`
2. `positive_prompt`
3. empty string

### `negative_prompt`

Optional global negative prompt applied to every segment.

### `loop_1` through `loop_10`

Up to ten segment config inputs. Only connected configs run.

The node does not use a separate `num_loops` input. The active loop count is derived from however many `loop_n` inputs are connected.

## `Wan Loop Config SVI`

This companion node builds one per-segment config object.

Required inputs:

- `prompt`
- `frames`

Optional inputs:

- `model_high`
- `model_low`
- `anchor_image`

### `prompt`

Positive prompt for that segment.

### `frames`

Target frame count for that segment before trimming/stitching.

Default:

- `49`

### `model_high` / `model_low`

Optional per-segment model overrides. If connected, they replace the main node's global `model_high` and `model_low` for that segment only.

### `anchor_image`

Optional per-segment explicit anchor override.

This has the highest anchor precedence for that segment.

Actual behavior when present:

- the explicit image is resized to `width` and `height`
- that resized image becomes the anchor for the current segment
- it also replaces `scheduled_anchor_image`

That last point matters:

- in `keyframe_schedule`, an explicit `anchor_image` immediately resets the scheduled anchor baseline
- in `dynamic_every_segment`, it overrides the current segment and also updates the scheduled anchor state, though the next segment will still use the dynamic extracted anchor path unless another explicit override appears

## Actual anchor precedence

For each segment, the current code chooses anchors in this order:

1. explicit `anchor_image` from that segment's config
2. `dynamic_every_segment` rolling anchor from the previous segment
3. `keyframe_schedule` scheduled anchor
4. original resized `start_image`

That means `anchor_mode` does not override an explicit segment anchor. The explicit anchor always wins.

## Stitching and save behavior

After sampling and decode, the node:

1. extracts the anchor frame
2. optionally color-corrects it
3. chooses `decoded_for_stitch` based on `stitch_mode`
4. optionally removes leading frames with `startup_trim`
5. optionally overlap-blends with the previous segment
6. saves the current segment frames to a temp `.pt` file

Important current detail from the overlap fix:

- when overlap blending is used, the node reloads the previous segment from disk before trimming its tail
- this preserves any already-blended head frames from the segment before it

That bug fix is one of the reasons older docs can be misleading here.

## Outputs

### `full_video` (`IMAGE`)

The final concatenated frame batch assembled from the saved segment tensors.

### `last_extracted_anchor` (`IMAGE`)

The processed anchor extracted from the final segment. This is the current output name; older docs may call it `last_anchor`.

### `All Segment Prompts` (`STRING`)

A newline-joined prompt log in the format:

`Segment N (Xf): prompt text`

## Logging and temp files

The node writes segment `.pt` files into a temporary directory under ComfyUI's temp directory when available.

You should see a log line like:

`[WanLooperNative] Segment temp dir: /data/app/temp/wan_native_xxxxxxxx`

This is especially relevant on hosted environments where `/tmp` may be too small.

Other useful logs include:

- selected anchor mode, stitch mode, and seed mode
- chosen anchor source for each segment
- extracted anchor frame index
- overlap stitch parameters
- saved frame counts per segment

## Recommended current baseline

The code defaults and the audit-tested baseline are not identical.

Current code defaults:

- `seed_mode = fixed`
- `anchor_mode = fixed_initial`
- `stitch_mode = workflow_style`
- `anchor_frame_offset = -5`
- `overlap = 5`
- `startup_trim = 0`
- `overlap_mode = linear_blend`
- `overlap_side = source`
- `color_correction = false`

Audit-confirmed strong baseline for seam quality and chain stability:

- `seed_mode = fixed`
- `overlap = 5`
- `startup_trim = 5`
- `anchor_mode = fixed_initial`
- `stitch_mode = workflow_style`
- `anchor_frame_offset = -5`

Use the defaults as the UI baseline, not as proof that every default is the best tested setting.

## Docs to trust versus docs to treat as historical

Most reliable current sources:

- `nodes_wan_v2.py`
- `README.md`
- `docs/briefs/2026-04-18_audit_complete.md`

Useful but historical:

- older design briefs in `docs/briefs/`
- architecture plans in `docs/superpowers/`
- archived code in `docs/archive/`

If this doc and an older brief disagree, trust the code first.
