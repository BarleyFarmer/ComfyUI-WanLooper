# Agent Handoff

This repository ships a ComfyUI custom node pack for multi-segment Wan 2.2
image-to-video looping. Treat the files below as canonical:

- `nodes_wan_v2.py` is the shipping implementation.
- `__init__.py` exposes the ComfyUI node mappings.
- `README.md` is the public user-facing overview.
- `docs/SVI_Looper_Native_Reference.md` is the current behavioral reference.
- `workflows/ComfyUI-WanLooper_example_workflow.json` is the basic example.
- `workflows/ComfyUI-WanLooper_example_workflow_advanced.json` is the annotated
  larger example.

Historical context lives under `docs/briefs/`, `docs/archive/`, and
`docs/superpowers/`. Those files are useful for understanding project history,
but many describe older node names, abandoned architectures, or bugs that have
since been fixed. Do not treat them as current requirements unless a newer
canonical file explicitly points to them.

## Current Node Surface

The current public nodes are:

- `Wan Loop Config SVI`, registered as `LoopConfigWan`
- `Wan Looper SVI`, registered as `WanLooperNative`

Backward-compatible aliases are intentionally retained:

- `LoopConfigSVI`
- `SVILooperNative`

Do not remove those aliases without a migration plan for saved workflow JSONs.

## Runtime Dependencies

The node pack expects to run inside ComfyUI. It imports ComfyUI internals and
therefore cannot be fully imported or executed in a plain Python environment.

Required at runtime:

- ComfyUI with Wan 2.2 image-to-video support
- KJNodes, specifically `WanImageToVideoSVIPro`, `ScheduledCFGGuidance`, and
  `ImageBatchExtendWithOverlap`

The example workflows also use extra node packs listed in `README.md`.

## Behavioral Baseline

The audited default trajectory is:

- `seed_mode = fixed`
- `anchor_mode = fixed_initial`
- `stitch_mode = workflow_style`
- `overlap = 5`
- `anchor_frame_offset = -5`

The strongest audited seam baseline uses `startup_trim = 5` with the same
settings. `increment_per_segment` seed behavior was removed after testing showed
that it caused longer-chain identity drift.

Known unresolved characteristic:

- brightness drift can accumulate over long chains in both this node and the
  hand-wired reference workflow.

## Verification

Useful lightweight checks before changing behavior:

```powershell
python -m py_compile nodes_wan_v2.py __init__.py
python -m json.tool workflows/ComfyUI-WanLooper_example_workflow.json > $null
python -m json.tool workflows/ComfyUI-WanLooper_example_workflow_advanced.json > $null
```

Full runtime validation requires launching ComfyUI with the required Wan/KJNodes
stack and loading the included workflows.
