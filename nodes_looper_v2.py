"""
I2V Looper v2 — Multi-segment I2V video looper with dual high/low noise models.

Two nodes:
  LoopConfigV2   — per-segment config (prompt, frames, LoRAs)
  I2VLooperV2    — main looper using KSamplerAdvanced two-pass sampling

Conditioning: CLIPVisionEncode + WanImageToVideo (proven working approach).
Overlap: inline ease_in_out blend — zero external dependencies.
Drift correction: per-channel mean/std normalization on each anchor frame.
"""

import torch
import gc
import os
import tempfile
import shutil

import comfy.model_management
import comfy.utils
import comfy.sd
import comfy.samplers
import folder_paths
import nodes


# ---------------------------------------------------------------------------
# NODE 1: LoopConfigV2
# ---------------------------------------------------------------------------

class LoopConfigV2:
    """Per-segment configuration: prompt, frames, optional LoRAs."""

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "frames": ("INT", {"default": 49, "min": 1, "max": 200}),
            },
            "optional": {
                "lora_high": (lora_list, {"default": "None"}),
                "lora_high_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
                "lora_low": (lora_list, {"default": "None"}),
                "lora_low_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
            },
        }

    RETURN_TYPES = ("LOOP_CONFIG_V2",)
    RETURN_NAMES = ("loop_config",)
    FUNCTION = "build"
    CATEGORY = "video/looper_v2"

    def build(self, prompt, frames,
              lora_high="None", lora_high_strength=1.0,
              lora_low="None",  lora_low_strength=1.0):
        return ({
            "prompt":            prompt,
            "frames":            frames,
            "lora_high":         None if lora_high == "None" else lora_high,
            "lora_high_strength": lora_high_strength,
            "lora_low":          None if lora_low  == "None" else lora_low,
            "lora_low_strength":  lora_low_strength,
        },)


# ---------------------------------------------------------------------------
# NODE 2: I2VLooperV2
# ---------------------------------------------------------------------------

class I2VLooperV2:
    """
    Multi-segment I2V looper with dual high/low noise models.

    Each segment uses CLIPVisionEncode + WanImageToVideo conditioning and
    KSamplerAdvanced two-pass sampling (high model → low model).

    The anchor frame (at anchor_frame_offset from segment end) becomes the
    next segment's start image. Color drift is corrected per-segment by
    normalizing the anchor frame's statistics back to the original start image.
    Overlap is blended inline with ease_in_out — no external dependencies.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_high":   ("MODEL",),
                "model_low":    ("MODEL",),
                "vae":          ("VAE",),
                "clip":         ("CLIP",),
                "clip_vision":  ("CLIP_VISION",),
                "start_image":  ("IMAGE",),
                "width":  ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 640, "min": 16, "max": 4096, "step": 16}),
                "steps":  ("INT", {"default": 8,   "min": 1,  "max": 100}),
                "split_step": ("INT", {
                    "default": 3, "min": 1, "max": 50,
                    "tooltip": "Step where high-noise model hands off to low-noise model",
                }),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler":    (comfy.samplers.KSampler.SCHEDULERS,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "overlap": ("INT", {
                    "default": 5, "min": 0, "max": 20, "step": 1,
                    "tooltip": "Overlap frames for blending between segments (0 = hard cut)",
                }),
                "overlap_mode": (
                    ["linear_blend", "ease_in_out", "filmic_crossfade", "cut"],
                    {"default": "ease_in_out"},
                ),
                "overlap_side": (
                    ["source", "new_images"],
                    {"default": "source"},
                ),
                "anchor_frame_offset": ("INT", {
                    "default": -5, "min": -50, "max": -1, "step": 1,
                    "tooltip": "Frames from end to extract anchor. -5 = 5th from last",
                }),
            },
            "optional": {
                "positive_prompt": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Global fallback positive prompt",
                }),
                "negative_prompt": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Global negative prompt",
                }),
                "loop_1":  ("LOOP_CONFIG_V2",),
                "loop_2":  ("LOOP_CONFIG_V2",),
                "loop_3":  ("LOOP_CONFIG_V2",),
                "loop_4":  ("LOOP_CONFIG_V2",),
                "loop_5":  ("LOOP_CONFIG_V2",),
                "loop_6":  ("LOOP_CONFIG_V2",),
                "loop_7":  ("LOOP_CONFIG_V2",),
                "loop_8":  ("LOOP_CONFIG_V2",),
                "loop_9":  ("LOOP_CONFIG_V2",),
                "loop_10": ("LOOP_CONFIG_V2",),
            },
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES  = ("full_video", "last_anchor", "used_prompts")
    FUNCTION      = "generate"
    CATEGORY      = "video/looper_v2"

    def _load_lora(self, model, clip, lora_name, strength):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            patched_model, patched_clip = comfy.sd.load_lora_for_models(
                model, clip, lora, strength, strength
            )
            del lora
            return patched_model, patched_clip
        print(f"[I2VLooperV2] WARNING: LoRA not found: {lora_name}")
        return model, clip

    def generate(self,
                 model_high, model_low, vae, clip, clip_vision, start_image,
                 width, height, steps, split_step, cfg, sampler_name, scheduler,
                 seed, overlap, overlap_mode, overlap_side, anchor_frame_offset,
                 positive_prompt="", negative_prompt="",
                 loop_1=None, loop_2=None, loop_3=None, loop_4=None, loop_5=None,
                 loop_6=None, loop_7=None, loop_8=None, loop_9=None, loop_10=None):

        from comfy_extras.nodes_wan import WanImageToVideo

        loop_configs = [loop_1, loop_2, loop_3, loop_4, loop_5,
                        loop_6, loop_7, loop_8, loop_9, loop_10]
        active_configs = [(i + 1, cfg) for i, cfg in enumerate(loop_configs) if cfg is not None]

        if not active_configs:
            print("[I2VLooperV2] No loop configs connected — returning black frame")
            return (torch.zeros((1, height, width, 3)),
                    torch.zeros((1, height, width, 3)),
                    "No loops connected")

        total_loops = len(active_configs)
        print(f"\n[I2VLooperV2] {'='*50}")
        print(f"[I2VLooperV2] Starting — {total_loops} segment(s)")
        print(f"[I2VLooperV2] {'='*50}")

        # ── Resize start image ───────────────────────────────────────────────
        current_start_image = comfy.utils.common_upscale(
            start_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)

        # ── Reference color statistics for drift correction ──────────────────
        ref_mean = current_start_image.float().mean(dim=(0, 1, 2), keepdim=True)
        ref_std  = current_start_image.float().std(dim=(0, 1, 2),  keepdim=True).clamp(min=1e-5)

        # ── State ────────────────────────────────────────────────────────────
        segment_dir      = tempfile.mkdtemp(prefix="i2v_looper_v2_")
        segment_paths    = []
        used_prompts_log = []
        prev_decoded     = None
        current_seed     = seed
        last_anchor      = current_start_image

        ksampler = nodes.KSamplerAdvanced()

        for loop_num, (loop_id, config) in enumerate(active_configs):
            print(f"\n[I2VLooperV2] ── Loop {loop_num + 1}/{total_loops} ──────────────────────────")

            # ── a) Extract config ────────────────────────────────────────────
            loop_prompt   = config["prompt"].strip() or positive_prompt or ""
            loop_frames   = config["frames"] if config["frames"] > 0 else 49
            lora_high_name = config.get("lora_high")
            lora_high_str  = config.get("lora_high_strength", 1.0)
            lora_low_name  = config.get("lora_low")
            lora_low_str   = config.get("lora_low_strength", 1.0)

            print(f"[I2VLooperV2] Loop {loop_id}: prompt = \"{loop_prompt[:80]}\"")
            print(f"[I2VLooperV2] Loop {loop_id}: {loop_frames} frames, seed={current_seed}")
            used_prompts_log.append(f"Loop {loop_id} ({loop_frames}f): {loop_prompt[:80]}")

            # ── b) Text conditioning ─────────────────────────────────────────
            tokens_pos = clip.tokenize(loop_prompt)
            cond_pos, pooled_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True)
            c_pos = [[cond_pos, {"pooled_output": pooled_pos}]]

            tokens_neg = clip.tokenize(negative_prompt)
            cond_neg, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)
            c_neg = [[cond_neg, {"pooled_output": pooled_neg}]]

            # ── c) CLIPVision + WanImageToVideo ──────────────────────────────
            print(f"[I2VLooperV2] Loop {loop_id}: CLIP Vision encoding...")
            cv_result = nodes.CLIPVisionEncode().encode(clip_vision, current_start_image, "none")
            cv_out    = cv_result[0]

            print(f"[I2VLooperV2] Loop {loop_id}: WanImageToVideo conditioning...")
            i2v_result  = WanImageToVideo().execute(
                positive           = c_pos,
                negative           = c_neg,
                vae                = vae,
                width              = width,
                height             = height,
                length             = loop_frames,
                batch_size         = 1,
                start_image        = current_start_image,
                clip_vision_output = cv_out,
            )
            pos_cond    = i2v_result[0]
            neg_cond    = i2v_result[1]
            latent_dict = i2v_result[2]

            # ── d) Per-segment LoRAs ─────────────────────────────────────────
            high_model = model_high
            low_model  = model_low

            if lora_high_name:
                print(f"[I2VLooperV2] Loop {loop_id}: loading lora_high ({lora_high_name})")
                high_model, _ = self._load_lora(model_high, clip, lora_high_name, lora_high_str)
            if lora_low_name:
                print(f"[I2VLooperV2] Loop {loop_id}: loading lora_low ({lora_low_name})")
                low_model, _ = self._load_lora(model_low, clip, lora_low_name, lora_low_str)

            # ── e) Two-pass KSamplerAdvanced ─────────────────────────────────
            print(f"[I2VLooperV2] Loop {loop_id}: sampling HIGH (steps 0-{split_step})")
            high_out = ksampler.sample(
                model                      = high_model,
                add_noise                  = "enable",
                noise_seed                 = current_seed,
                steps                      = steps,
                cfg                        = cfg,
                sampler_name               = sampler_name,
                scheduler                  = scheduler,
                positive                   = pos_cond,
                negative                   = neg_cond,
                latent_image               = latent_dict,
                start_at_step              = 0,
                end_at_step                = split_step,
                return_with_leftover_noise = "enable",
            )

            comfy.model_management.soft_empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            print(f"[I2VLooperV2] Loop {loop_id}: sampling LOW (steps {split_step}-end)")
            low_out = ksampler.sample(
                model                      = low_model,
                add_noise                  = "disable",
                noise_seed                 = 0,
                steps                      = steps,
                cfg                        = cfg,
                sampler_name               = sampler_name,
                scheduler                  = scheduler,
                positive                   = pos_cond,
                negative                   = neg_cond,
                latent_image               = {"samples": high_out[0]["samples"]},
                start_at_step              = split_step,
                end_at_step                = 1000,
                return_with_leftover_noise = "disable",
            )
            sampled = low_out[0]["samples"]

            # ── f) VAE decode ────────────────────────────────────────────────
            decoded = vae.decode(sampled)
            while decoded.ndim > 4:
                decoded = decoded.squeeze(0)
            if decoded.shape[1] != height or decoded.shape[2] != width:
                decoded = comfy.utils.common_upscale(
                    decoded.movedim(-1, 1), width, height, "bilinear", "center"
                ).movedim(1, -1)
            print(f"[I2VLooperV2] Loop {loop_id}: VAE decode → {decoded.shape[0]} frames")

            # ── g) Anchor frame extraction ───────────────────────────────────
            actual_frames = decoded.shape[0]
            anchor_idx    = max(0, min(actual_frames - 1, actual_frames + anchor_frame_offset))
            raw_anchor    = decoded[anchor_idx:anchor_idx + 1].clone()
            print(f"[I2VLooperV2] Loop {loop_id}: anchor at frame {anchor_idx}")

            # ── h) Color drift correction ────────────────────────────────────
            anchor_f = raw_anchor.float()
            a_mean   = anchor_f.mean(dim=(0, 1, 2), keepdim=True)
            a_std    = anchor_f.std(dim=(0, 1, 2),  keepdim=True).clamp(min=1e-5)
            anchor_normalized = ((anchor_f - a_mean) / a_std) * ref_std + ref_mean
            anchor_normalized = anchor_normalized.clamp(0.0, 1.0).to(raw_anchor.dtype)
            print(f"[I2VLooperV2] Loop {loop_id}: color-normalized "
                  f"(mean {a_mean.mean().item():.3f}→{ref_mean.mean().item():.3f})")
            current_start_image = anchor_normalized
            last_anchor         = anchor_normalized
            # BUG1 CHECK: confirm current_start_image is updated for next segment
            print(f"[I2VLooperV2] Loop {loop_id}: current_start_image updated "
                  f"shape={current_start_image.shape}, "
                  f"mean={current_start_image.float().mean().item():.4f}")

            # ── i) Trim segment to anchor point ──────────────────────────────
            decoded_trimmed = decoded[:anchor_idx + 1]
            print(f"[I2VLooperV2] Loop {loop_id}: trimmed to {decoded_trimmed.shape[0]} frames")

            # ── j) Overlap stitching ─────────────────────────────────────────
            # BUG2 FIX: do NOT include prefix in segment_frames.
            # Re-save the previous segment's .pt trimmed to remove its last
            # eff_overlap frames, then save only [blended + suffix] here.
            # Result: prev(N-eff_overlap) + current(eff_overlap + suffix) = correct total.
            if loop_num > 0 and overlap > 0 and prev_decoded is not None and overlap_mode != "cut":
                eff_overlap = min(overlap, prev_decoded.shape[0], decoded_trimmed.shape[0])
                print(f"[I2VLooperV2] Loop {loop_id}: overlap stitching "
                      f"({eff_overlap} frames, {overlap_mode}, side={overlap_side})")

                # overlap_side controls which end fades in vs fades out
                if overlap_side == "source":
                    fade_out = prev_decoded[-eff_overlap:]    # source fades out
                    fade_in  = decoded_trimmed[:eff_overlap]  # new fades in
                else:  # "new_images"
                    fade_out = decoded_trimmed[:eff_overlap]  # new fades out
                    fade_in  = prev_decoded[-eff_overlap:]    # source fades in

                suffix = decoded_trimmed[eff_overlap:]

                dev  = fade_out.device
                fdtype = fade_out.float().dtype

                if overlap_mode == "linear_blend":
                    alpha = torch.linspace(0, 1, eff_overlap + 2,
                                           device=dev, dtype=fdtype)[1:-1]
                    alpha = alpha.view(-1, 1, 1, 1)
                    blended = ((1 - alpha) * fade_out.float() + alpha * fade_in.float()).to(fade_out.dtype)

                elif overlap_mode == "ease_in_out":
                    t     = torch.linspace(0, 1, eff_overlap + 2,
                                           device=dev, dtype=fdtype)[1:-1]
                    eased = 3 * t * t - 2 * t * t * t
                    eased = eased.view(-1, 1, 1, 1)
                    blended = ((1 - eased) * fade_out.float() + eased * fade_in.float()).to(fade_out.dtype)

                elif overlap_mode == "filmic_crossfade":
                    # Blend in linear light space (gamma 2.2), then re-apply gamma
                    gamma = 2.2
                    src_lin = fade_out.float().clamp(0, 1) ** gamma
                    dst_lin = fade_in.float().clamp(0, 1) ** gamma
                    t     = torch.linspace(0, 1, eff_overlap + 2,
                                           device=dev, dtype=fdtype)[1:-1]
                    eased = 3 * t * t - 2 * t * t * t
                    eased = eased.view(-1, 1, 1, 1)
                    blended_lin = (1 - eased) * src_lin + eased * dst_lin
                    blended = blended_lin.clamp(0, 1) ** (1.0 / gamma)
                    blended = blended.to(fade_out.dtype)

                else:
                    # Fallback — should not reach here given "cut" guard above
                    blended = fade_in.clone()

                # Re-save previous segment with overlap tail removed
                trimmed_prev = prev_decoded[:-eff_overlap]
                torch.save(trimmed_prev.cpu(), segment_paths[-1])
                print(f"[I2VLooperV2] Loop {loop_id}: re-saved prev segment → {trimmed_prev.shape[0]} frames")

                segment_frames = torch.cat([blended, suffix], dim=0)

            else:
                # "cut" mode or first segment — no blending
                if loop_num > 0 and overlap_mode == "cut":
                    print(f"[I2VLooperV2] Loop {loop_id}: cut (no blend)")
                segment_frames = decoded_trimmed

            # ── k) Save segment ──────────────────────────────────────────────
            seg_path = os.path.join(segment_dir, f"segment_{loop_id:03d}.pt")
            torch.save(segment_frames.cpu(), seg_path)
            segment_paths.append(seg_path)
            print(f"[I2VLooperV2] Loop {loop_id}: saved segment → {segment_frames.shape[0]} frames "
                  f"(decoded={decoded.shape[0]}, anchor_idx={anchor_idx}, "
                  f"trimmed={decoded_trimmed.shape[0]})")

            # ── l) State update ──────────────────────────────────────────────
            prev_decoded  = decoded_trimmed
            current_seed += 1

            # ── m) Cleanup ───────────────────────────────────────────────────
            if lora_high_name and high_model is not model_high:
                del high_model
            if lora_low_name and low_model is not model_low:
                del low_model
            del decoded, sampled
            comfy.model_management.soft_empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # ── Final assembly ───────────────────────────────────────────────────
        print(f"\n[I2VLooperV2] Assembling {len(segment_paths)} segment(s)...")
        full_video = None
        for seg_path in segment_paths:
            seg = torch.load(seg_path, map_location="cpu", weights_only=True)
            full_video = seg if full_video is None else torch.cat([full_video, seg], dim=0)
            del seg
            gc.collect()

        shutil.rmtree(segment_dir, ignore_errors=True)

        if full_video is None:
            full_video = torch.zeros((1, height, width, 3))

        print(f"[I2VLooperV2] Done. Total frames: {full_video.shape[0]}")
        return (full_video, last_anchor, "\n".join(used_prompts_log))


# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "LoopConfigV2": LoopConfigV2,
    "I2VLooperV2":  I2VLooperV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoopConfigV2": "Loop Config v2",
    "I2VLooperV2":  "I2V Looper v2",
}
