# ComfyUI-WanLooper

> This project is a derivative work based on [ComfyUI-LooperNode](https://github.com/masteroleary/ComfyUI-LooperNode) by masteroleary, also MIT licensed:
> Copyright (c) 2026 masteroleary

A custom node pack for extending Wan 2.2 image-to-video clips across multiple internally managed segments.

The loop runs inside a single custom node rather than across repeated ComfyUI subgraphs. This is deliberate: ComfyUI's graph engine has no loop primitive, and subgraphs are static node copies rather than function calls. Any chain that needs per-iteration state or per-segment policy (different prompts, anchor frames computed from the previous segment, seed strategies across the chain) has to put the iteration logic inside Python.

## Status

Working but unpolished, in use by the author. The core node (`Wan Looper SVI`) has been audited end-to-end for identity drift and seam quality across 3- and 6-segment chains. Output is comparable to a hand-wired chained-subgraph reference workflow built on the same models and LoRAs.

Still considered pre-release; widgets and defaults may change.

## Included nodes

Two nodes ship in this repo:

- **`Wan Loop Config SVI`** — per-segment configuration (prompt, frame count, optional per-segment model override).
- **`Wan Looper SVI`** — the loop driver itself. Consumes a chain of Loop Config inputs, handles anchor frame extraction, SVI Pro conditioning, sampling, VAE decode, overlap stitching, and final video assembly.

Both nodes retain backward-compatibility aliases (`LoopConfigSVI`, `SVILooperNative`) so existing workflow JSONs that reference the older names continue to load.

Key design points of the SVI path:

- per-segment prompt and frame control
- explicit `anchor_mode`, `stitch_mode`, and `seed_mode` policies
- `anchor_frame_offset` for controlling which frame of segment N becomes segment N+1's starting image
- `overlap` and `startup_trim` widgets for seam handling
- keyframe scheduling support via comma-separated segment schedules
- KJNodes-backed SVI conditioning and overlap — real class calls, not inline approximations

Defaults (as of the most recent audit commit):

- `seed_mode = fixed` — using a single shared seed across all segments. `increment_per_segment` was removed after it was found to cause compounding identity drift at longer chain lengths.
- `overlap = 5`, `startup_trim = 5`, `linear_blend`, `anchor_frame_offset = -5`.

## Dependencies

Runtime:

- **ComfyUI** with Wan 2.2 I2V A14B model support.
- **[KJNodes](https://github.com/kijai/ComfyUI-KJNodes)** — required. `Wan Looper SVI` dynamically loads the following KJ classes at import time and fails loudly if they are missing:
  - `WanImageToVideoSVIPro`
  - `ScheduledCFGGuidance`
  - `ImageBatchExtendWithOverlap`

Recommended model and LoRA stack (not bundled):

- Wan 2.2 I2V A14B Q5_K_M GGUF (HighNoise + LowNoise split)
- `umt5_xxl_fp8_e4m3fn_scaled.safetensors` text encoder
- `wan_2.1_vae.safetensors` VAE
- SVI Pro v2 HIGH/LOW LoRAs (temporal consistency)
- LightX2V HIGH/LOW LoRAs (distillation/speed)

## Getting started

1. Clone this repo into `ComfyUI/custom_nodes/`.
2. Make sure KJNodes is installed and importable.
3. Copy `examples/example_start_image.png` into your `ComfyUI/input/` folder so the example workflow can find it.
4. Restart ComfyUI.
5. Load `workflows/ComfyUI-WanLooper_example_workflow.json` as a starting point. The preview PNG alongside it shows the expected node layout.
6. Run it, or swap in your own start image and per-segment prompts.

The `examples/` directory also includes a sample output MP4 (`2026-04-19_WanLooper_testing_00001.mp4`) showing what the workflow produces.

## Known characteristics

- **Brightness drift across long chains.** Both this looper and hand-wired chained-subgraph reference workflows show brightness drift proportional to chain length (around 15–25 brightness units over 6 segments at 81 frames each, direction depending on content and seed). Not a looper-specific bug; likely a characteristic of the underlying sampling chain. Not currently addressed.
- **VAE encode/decode loss accumulates.** Per-segment anchor re-encoding loses some fine detail over long chains (eye/eyebrow definition softens). Irreducible without architectural changes.
- **8 GB VRAM is the tested floor.** Development hardware is an RTX 3060 Ti; launch flags `--novram --disable-smart-memory --disable-pinned-memory --use-sage-attention` are the baseline configuration.

## Further reading

- [`docs/briefs/2026-04-18_audit_complete.md`](docs/briefs/2026-04-18_audit_complete.md) — Summary of the most recent architecture audit, the two bugs found and fixed, and what was confirmed sound.

Older markdown briefs in `docs/briefs/` are historical working documents and may not reflect current architecture.

## License

MIT. See [LICENSE](LICENSE) for full terms, including attribution to the upstream `ComfyUI-LooperNode` project.
