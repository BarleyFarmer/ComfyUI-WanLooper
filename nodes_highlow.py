"""
I2V Looper High/Low - Multi-loop I2V video generation with dual High/Low noise models.

Each loop uses WanImageToVideo conditioning with the last frame from the
previous loop as the start image. Matches APP VIDEO workflow approach.
"""

import torch
import gc
import os
import tempfile
import shutil

import comfy.model_management
import comfy.utils
import comfy.latent_formats
import comfy.sd
import folder_paths
import nodes
import node_helpers


class I2VLoopHL:
    """Per-loop config with High/Low LoRA support."""
    @classmethod
    def INPUT_TYPES(cls):
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "frames": ("INT", {"default": 81, "min": 1, "max": 200}),
            },
            "optional": {
                "lora_high": (lora_list, {"default": "None"}),
                "lora_high_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05}),
                "lora_low": (lora_list, {"default": "None"}),
                "lora_low_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("I2V_LOOP_HL",)
    RETURN_NAMES = ("loop_config",)
    FUNCTION = "build"
    CATEGORY = "video/i2v_looper"

    def build(self, prompt, frames, lora_high="None", lora_high_strength=0.8,
              lora_low="None", lora_low_strength=0.8):
        return ({
            "prompt": prompt,
            "frames": frames,
            "lora_high": lora_high if lora_high != "None" else None,
            "lora_high_strength": lora_high_strength,
            "lora_low": lora_low if lora_low != "None" else None,
            "lora_low_strength": lora_low_strength,
        },)


class I2VLooperHL:
    """
    Multi-loop I2V with High/Low noise models.
    Last frame of each loop becomes start image for the next.
    Matches APP VIDEO dual-pass KSamplerAdvanced approach.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_high": ("MODEL", {"tooltip": "High-noise model"}),
                "model_low": ("MODEL", {"tooltip": "Low-noise model"}),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "clip_vision": ("CLIP_VISION", {"tooltip": "CLIP Vision model (e.g. clip_vision_h.safetensors)"}),
                "start_image": ("IMAGE", {"tooltip": "Start image for loop 1"}),
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "split_step": ("INT", {"default": 4, "min": 1, "max": 50, "tooltip": "Step where high switches to low model"}),
                "sampler_name": (["euler", "uni_pc", "dpm++_sde", "dpm++_2m", "ddim"], {"default": "euler"}),
                "scheduler": (["simple", "normal", "karras", "sgm_uniform"], {"default": "simple"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "positive_prompt": ("STRING", {"forceInput": True, "tooltip": "Base/fallback prompt"}),
                "negative_prompt": ("STRING", {"forceInput": True, "tooltip": "Global negative prompt"}),
                "loop_1": ("I2V_LOOP_HL",),
                "loop_2": ("I2V_LOOP_HL",),
                "loop_3": ("I2V_LOOP_HL",),
                "loop_4": ("I2V_LOOP_HL",),
                "loop_5": ("I2V_LOOP_HL",),
                "loop_6": ("I2V_LOOP_HL",),
                "loop_7": ("I2V_LOOP_HL",),
                "loop_8": ("I2V_LOOP_HL",),
                "loop_9": ("I2V_LOOP_HL",),
                "loop_10": ("I2V_LOOP_HL",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("full_video", "used_prompts",)
    FUNCTION = "generate"
    CATEGORY = "video/i2v_looper"

    def _load_lora(self, model, clip, lora_name, strength):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            patched_model, patched_clip = comfy.sd.load_lora_for_models(
                model, clip, lora, strength, strength
            )
            del lora
            return patched_model, patched_clip
        print(f"WARNING: LoRA not found: {lora_name}")
        return model, clip

    def generate(self, model_high, model_low, vae, clip, clip_vision, start_image,
                 width, height,
                 steps, cfg, split_step, sampler_name, scheduler, seed,
                 positive_prompt="", negative_prompt="low quality, blurry, glitch, distortion",
                 loop_1=None, loop_2=None, loop_3=None, loop_4=None, loop_5=None,
                 loop_6=None, loop_7=None, loop_8=None, loop_9=None, loop_10=None):

        from comfy_extras.nodes_wan import WanImageToVideo

        loop_configs = [loop_1, loop_2, loop_3, loop_4, loop_5,
                        loop_6, loop_7, loop_8, loop_9, loop_10]

        # Count connected loops
        extension_loops = sum(1 for c in loop_configs if c is not None)
        if extension_loops == 0:
            extension_loops = 1  # At least 1 loop

        used_prompts_log = []
        segment_dir = tempfile.mkdtemp(prefix="i2v_hl_segments_")
        print(f"\n{'='*60}")
        print("I2V LOOPER HIGH/LOW")
        print(f"{'='*60}")
        print(f"Loops: {extension_loops} (auto-detected from connected loop nodes)")
        print(f"Steps: {steps} (high 0-{split_step}, low {split_step}-end)")

        # Resize start image
        current_start_image = comfy.utils.common_upscale(
            start_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)

        segment_paths = []
        current_seed = seed

        for loop_idx in range(10):
            loop_id = loop_idx + 1

            comfy.model_management.soft_empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            config = loop_configs[loop_idx] if loop_idx < len(loop_configs) else None

            if config is None:
                print(f"\n{'─'*60}")
                print(f"LOOP {loop_id}/{extension_loops} — SKIPPED")
                print(f"{'─'*60}")
                used_prompts_log.append(f"Loop {loop_id}: SKIPPED")
                current_seed += 1
                continue

            loop_frame_count = config.get("frames", 81)
            if loop_frame_count <= 0:
                loop_frame_count = 81

            active_prompt = config.get("prompt", "").strip() or positive_prompt

            lora_high_name = config.get("lora_high")
            lora_high_str = config.get("lora_high_strength", 0.8)
            lora_low_name = config.get("lora_low")
            lora_low_str = config.get("lora_low_strength", 0.8)

            print(f"\n{'─'*60}")
            print(f"LOOP {loop_id}/{extension_loops} ({loop_frame_count} frames)")
            if lora_high_name:
                print(f"LoRA HIGH: {lora_high_name} @ {lora_high_str}")
            if lora_low_name:
                print(f"LoRA LOW: {lora_low_name} @ {lora_low_str}")
            print(f"{'─'*60}")
            print(f"Prompt: {active_prompt[:80]}...")
            used_prompts_log.append(f"Loop {loop_id} ({loop_frame_count}f): {active_prompt[:80]}")

            # Encode text
            tokens_pos = clip.tokenize(active_prompt)
            cond_pos, pooled_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True)
            c_pos = [[cond_pos, {"pooled_output": pooled_pos}]]

            tokens_neg = clip.tokenize(negative_prompt)
            cond_neg, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)
            c_neg = [[cond_neg, {"pooled_output": pooled_neg}]]

            # CLIP Vision encode — use CLIPVisionEncode node (same as APP VIDEO)
            print("CLIP Vision encoding...")
            cv_result = nodes.CLIPVisionEncode().encode(clip_vision, current_start_image, "none")
            cv_out = cv_result[0]  # CLIP_VISION_OUTPUT

            # WanImageToVideo conditioning (same as APP VIDEO)
            print("WanImageToVideo conditioning...")
            i2v_result = WanImageToVideo().execute(
                positive=c_pos, negative=c_neg, vae=vae,
                width=width, height=height, length=loop_frame_count, batch_size=1,
                start_image=current_start_image, clip_vision_output=cv_out,
            )
            c_sample_pos = i2v_result[0]
            c_sample_neg = i2v_result[1]
            latent_dict = i2v_result[2]

            # Apply per-loop LoRAs
            high_model = model_high
            low_model = model_low

            if lora_high_name:
                print(f"Loading HIGH LoRA...")
                high_model, _ = self._load_lora(model_high, clip, lora_high_name, lora_high_str)

            if lora_low_name:
                print(f"Loading LOW LoRA...")
                low_model, _ = self._load_lora(model_low, clip, lora_low_name, lora_low_str)

            # TWO-PASS SAMPLING (matching APP VIDEO exactly)
            ksampler = nodes.KSamplerAdvanced()

            # Pass 1: High-noise model (steps 0 to split_step)
            # APP VIDEO: add_noise=enable, return_with_leftover_noise=enable
            print(f"Sampling HIGH (steps 0-{split_step})...")
            high_out = ksampler.sample(
                model=high_model,
                add_noise="enable",
                noise_seed=current_seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=c_sample_pos,
                negative=c_sample_neg,
                latent_image=latent_dict,
                start_at_step=0,
                end_at_step=split_step,
                return_with_leftover_noise="enable",
            )

            comfy.model_management.soft_empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Pass 2: Low-noise model (split_step to end)
            # APP VIDEO: add_noise=disable, noise_seed=0, end_at_step=1000, return_with_leftover_noise=disable
            print(f"Sampling LOW (steps {split_step}-end)...")
            low_out = ksampler.sample(
                model=low_model,
                add_noise="disable",
                noise_seed=0,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=c_sample_pos,
                negative=c_sample_neg,
                latent_image={"samples": high_out[0]["samples"]},
                start_at_step=split_step,
                end_at_step=1000,
                return_with_leftover_noise="disable",
            )

            sampled = low_out[0]["samples"]

            # Decode
            print("VAE decoding...")
            decoded = vae.decode(sampled)
            while decoded.ndim > 4:
                decoded = decoded.squeeze(0)
            print(f"Decoded: {decoded.shape[0]} frames")

            if decoded.shape[1] != height or decoded.shape[2] != width:
                decoded = comfy.utils.common_upscale(
                    decoded.movedim(-1, 1), width, height, "bilinear", "center"
                ).movedim(1, -1)

            # Save segment
            seg_path = os.path.join(segment_dir, f"segment_{loop_id:03d}.pt")
            torch.save(decoded.cpu(), seg_path)
            segment_paths.append(seg_path)

            # Last frame becomes next loop's start image
            current_start_image = decoded[-1:].clone()
            print(f"Next start image from frame {decoded.shape[0]-1}")

            # Cleanup
            if lora_high_name and high_model is not model_high:
                del high_model
            if lora_low_name and low_model is not model_low:
                del low_model
            del decoded, sampled, c_pos, c_neg
            comfy.model_management.soft_empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            current_seed += 1

        # Combine all segments
        print(f"\n{'='*60}")
        print("Combining segments...")
        print(f"{'='*60}")

        full_video = None
        for seg_path in segment_paths:
            seg = torch.load(seg_path, map_location="cpu")
            full_video = seg if full_video is None else torch.cat([full_video, seg], dim=0)
            del seg
            gc.collect()
            print(f"  Total: {full_video.shape[0]} frames")

        if full_video is None:
            full_video = torch.zeros((1, height, width, 3))

        try:
            shutil.rmtree(segment_dir)
        except:
            pass

        print(f"\nFinal video: {full_video.shape[0]} frames")
        return (full_video, "\n".join(used_prompts_log))


NODE_CLASS_MAPPINGS = {
    "I2VLooperHL": I2VLooperHL,
    "I2VLoopHL": I2VLoopHL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "I2VLooperHL": "I2V Looper High/Low",
    "I2VLoopHL": "I2V Loop High/Low",
}
