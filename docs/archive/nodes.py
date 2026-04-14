"""
Native Looper - Multi-loop video generation using WanImageToVideo conditioning.

Each loop uses the last frame from the previous loop as the start image,
chaining I2V generation for seamless long videos.
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


class NativeLoop:
    """Per-loop config: prompt, frames, LoRA."""
    @classmethod
    def INPUT_TYPES(cls):
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "frames": ("INT", {"default": 0, "min": 0, "max": 200, "tooltip": "Frames for this loop (0 = use 81)"}),
                "lora_name": (lora_list, {"default": "None"}),
                "lora_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("NATIVE_LOOP",)
    RETURN_NAMES = ("loop_config",)
    FUNCTION = "build"
    CATEGORY = "video/native"

    def build(self, prompt, frames, lora_name, lora_strength):
        return ({
            "prompt": prompt,
            "frames": frames,
            "lora_name": lora_name if lora_name != "None" else None,
            "lora_strength": lora_strength,
        },)


class NativeLooper:
    """
    Multi-loop video generation using WanImageToVideo.
    Last frame of each loop becomes start image for the next.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "clip_vision": ("CLIP_VISION", {"tooltip": "CLIP Vision model (e.g. clip_vision_h.safetensors)"}),
                "start_image": ("IMAGE", {"tooltip": "Start image for loop 1"}),
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (["uni_pc", "euler", "dpm++_sde", "dpm++_2m", "ddim"], {"default": "uni_pc"}),
                "scheduler": (["normal", "simple", "karras", "sgm_uniform"], {"default": "normal"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "positive_prompt": ("STRING", {"forceInput": True, "tooltip": "Base/fallback prompt"}),
                "negative_prompt": ("STRING", {"forceInput": True, "tooltip": "Global negative prompt"}),
                "loop_1": ("NATIVE_LOOP",),
                "loop_2": ("NATIVE_LOOP",),
                "loop_3": ("NATIVE_LOOP",),
                "loop_4": ("NATIVE_LOOP",),
                "loop_5": ("NATIVE_LOOP",),
                "loop_6": ("NATIVE_LOOP",),
                "loop_7": ("NATIVE_LOOP",),
                "loop_8": ("NATIVE_LOOP",),
                "loop_9": ("NATIVE_LOOP",),
                "loop_10": ("NATIVE_LOOP",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("full_video", "used_prompts",)
    FUNCTION = "generate"
    CATEGORY = "video/native"

    def generate(self, model, vae, clip, clip_vision, start_image,
                 width, height,
                 steps, cfg, sampler_name, scheduler, seed,
                 positive_prompt="", negative_prompt="low quality, blurry, glitch, distortion",
                 loop_1=None, loop_2=None, loop_3=None, loop_4=None, loop_5=None,
                 loop_6=None, loop_7=None, loop_8=None, loop_9=None, loop_10=None):

        from comfy_extras.nodes_wan import WanImageToVideo

        loop_configs = [loop_1, loop_2, loop_3, loop_4, loop_5,
                        loop_6, loop_7, loop_8, loop_9, loop_10]

        used_prompts_log = []
        segment_dir = tempfile.mkdtemp(prefix="native_i2v_segments_")
        print(f"\n{'='*60}")
        print("NATIVE I2V LOOPER")
        print(f"{'='*60}")
        extension_loops = sum(1 for c in loop_configs if c is not None)
        if extension_loops == 0:
            extension_loops = 1
        print(f"Loops: {extension_loops} (auto-detected)")

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

            loop_frame_count = config.get("frames", 0)
            if loop_frame_count <= 0:
                loop_frame_count = 81

            active_prompt = config.get("prompt", "").strip() or positive_prompt
            loop_lora_name = config.get("lora_name")
            loop_lora_strength = config.get("lora_strength", 0.8)

            print(f"\n{'─'*60}")
            print(f"LOOP {loop_id}/{extension_loops} ({loop_frame_count} frames)")
            if loop_lora_name:
                print(f"LoRA: {loop_lora_name} @ {loop_lora_strength}")
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

            # CLIP Vision encode
            cv_result = nodes.CLIPVisionEncode().encode(clip_vision, current_start_image, "none")
            cv_out = cv_result[0]

            # WanImageToVideo conditioning
            i2v_result = WanImageToVideo().execute(
                positive=c_pos, negative=c_neg, vae=vae,
                width=width, height=height, length=loop_frame_count, batch_size=1,
                start_image=current_start_image, clip_vision_output=cv_out,
            )
            c_sample_pos = i2v_result[0]
            c_sample_neg = i2v_result[1]
            latent_dict = i2v_result[2]

            # Apply LoRA
            loop_model = model
            if loop_lora_name:
                lora_path = folder_paths.get_full_path("loras", loop_lora_name)
                if lora_path:
                    print(f"Loading LoRA: {loop_lora_name}...")
                    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                    loop_model, _ = comfy.sd.load_lora_for_models(model, clip, lora, loop_lora_strength, loop_lora_strength)
                    del lora

            # Sample
            print(f"Sampling ({steps} steps, {sampler_name}, cfg={cfg})...")
            ksampler = nodes.KSampler()
            result = ksampler.sample(
                model=loop_model, seed=current_seed, steps=steps, cfg=cfg,
                sampler_name=sampler_name, scheduler=scheduler,
                positive=c_sample_pos, negative=c_sample_neg,
                latent_image=latent_dict, denoise=1.0
            )
            sampled = result[0]["samples"]

            # Decode
            print("VAE decoding...")
            decoded = vae.decode(sampled)
            while decoded.ndim > 4:
                decoded = decoded.squeeze(0)

            if decoded.shape[1] != height or decoded.shape[2] != width:
                decoded = comfy.utils.common_upscale(
                    decoded.movedim(-1, 1), width, height, "bilinear", "center"
                ).movedim(1, -1)

            print(f"Decoded: {decoded.shape[0]} frames")

            seg_path = os.path.join(segment_dir, f"segment_{loop_id:03d}.pt")
            torch.save(decoded.cpu(), seg_path)
            segment_paths.append(seg_path)

            # Last frame becomes next loop's start image
            current_start_image = decoded[-1:].clone()

            if loop_lora_name and loop_model is not model:
                del loop_model
            del decoded, sampled, c_pos, c_neg
            comfy.model_management.soft_empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            current_seed += 1

        # Combine
        print(f"\n{'='*60}")
        print("Combining segments...")
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
    "I2VLooper": NativeLooper,
    "I2VLoop": NativeLoop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "I2VLooper": "I2V Looper",
    "I2VLoop": "I2V Loop",
}
