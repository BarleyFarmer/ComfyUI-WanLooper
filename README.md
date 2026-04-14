# ComfyUI-NativeLooper

Multi-loop video generation with **sharp output** and character consistency.

## Features

- **Multi-loop generation** — up to 10 segments, chained for seamless long videos
- **Per-loop prompts** — different prompt for each segment
- **Per-loop frame counts** — customize duration per segment
- **Per-loop LoRA** — different LoRA per segment
- **Memory efficient** — segments saved to disk between loops, VRAM cleaned between iterations

## Installation

### Option 1: Clone into custom_nodes
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/ComfyUI-NativeLooper.git
```

### Option 2: Manual
Copy the `nodes.py` and `__init__.py` files into `ComfyUI/custom_nodes/ComfyUI-NativeLooper/`.

No additional dependencies required — uses only ComfyUI built-in modules.

## How It Works

1. For each loop:
   - Encodes prompt and CLIP Vision features for the current start image
   - Builds WanImageToVideo conditioning
   - Optionally applies a per-loop LoRA
   - Samples with KSampler, decodes with VAE
   - Saves segment to disk; uses last frame as start image for the next loop
2. Concatenates all segments into the final video.

## License

MIT
