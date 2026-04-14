"""
Wan Looper Native — Multi-segment Wan video looper using SVIPro-style conditioning.

Two nodes:
  LoopConfigWan    — per-segment config (prompt, frames, optional model overrides)
  WanLooperNative  — main looper using SamplerCustomAdvanced + Guider_ScheduledCFG

Unlike Phase 1 (I2VLooperV2), this node passes prev_samples (previous segment's
sampled latent) into the Wan conditioning so the model has temporal context at
the segment boundary — eliminating the ghosting that Phase 1 suffers from.

Sampling: SamplerCustomAdvanced with SplitSigmas + Guider_ScheduledCFG guiders
(high model for early steps, low model for late steps — no KSamplerAdvanced).

All KJNodes logic is inlined:
  _wan_pro_condition()          — inlined from WanImageToVideoSVIPro.execute()
  Guider_ScheduledCFG           — from Guider_ScheduledCFG class
  _build_scheduled_cfg_guider() — from ScheduledCFGGuidance.get_guider()

Inlining avoids the V3 ComfyNode / class-wrapper issues that occur when calling
KJNodes node methods from outside ComfyUI's execution framework.
"""

import torch
import gc
import os
import tempfile
import shutil

import comfy.model_management
import comfy.utils
import comfy.samplers
import comfy.sd
import comfy.latent_formats
import folder_paths
import nodes as _comfy_nodes
import node_helpers

from comfy.samplers import sampling_function, CFGGuider

# ComfyUI core custom sampler components
from comfy_extras.nodes_custom_sampler import (
    Noise_RandomNoise,
    Noise_EmptyNoise,
    BasicScheduler,
    SplitSigmas,
    SamplerCustomAdvanced,
    KSamplerSelect,
)


# ---------------------------------------------------------------------------
# Inlined from comfyui-kjnodes/nodes/nodes.py: Guider_ScheduledCFG
# ---------------------------------------------------------------------------

class Guider_ScheduledCFG(CFGGuider):
    """CFG guider that applies a specified CFG value only during a step-percent
    window; steps outside the window use CFG 1.0 (faster, no unconditional pass).

    Inlined from KJNodes Guider_ScheduledCFG — no KJNodes module dependency.
    """

    def set_cfg(self, cfg, start_percent, end_percent):
        self.cfg           = cfg
        self.start_percent = start_percent
        self.end_percent   = end_percent

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        steps = model_options["transformer_options"]["sample_sigmas"]
        if isinstance(timestep, torch.Tensor):
            timestep_value = timestep.reshape(-1)[0].to(steps)
        else:
            timestep_value = torch.tensor(timestep, device=steps.device,
                                          dtype=steps.dtype)
        matched = torch.isclose(steps, timestep_value).nonzero()
        assert not (isinstance(self.cfg, list) and
                    len(self.cfg) != (len(steps) - 1)), \
            "cfg list length must match step count"
        if len(matched) > 0:
            current_step_index = matched.item()
        else:
            current_step_index = 0
            for i in range(len(steps) - 1):
                if (steps[i] - timestep_value) * (steps[i + 1] - timestep_value) <= 0:
                    current_step_index = i
                    break
        current_percent = current_step_index / (len(steps) - 1)

        if self.start_percent <= current_percent <= self.end_percent:
            cfg   = self.cfg[current_step_index] if isinstance(self.cfg, list) else self.cfg
            uncond = self.conds.get("negative", None)
        else:
            uncond = None
            cfg    = 1.0

        return sampling_function(
            self.inner_model, x, timestep, uncond,
            self.conds.get("positive", None), cfg,
            model_options=model_options, seed=seed,
        )


def _build_scheduled_cfg_guider(model, cfg, positive, negative,
                                 start_percent, end_percent):
    """Inline equivalent of ScheduledCFGGuidance.get_guider().

    ScheduledCFGGuidance in KJNodes is a V1-style node, but accessing it through
    the sys.modules scan or importlib triggers torch class wrapper errors when
    hasattr() probes the module attributes during iteration.

    Returns the guider object directly — equivalent to ScheduledCFGGuidance(model).
    """
    guider = Guider_ScheduledCFG(model)
    guider.set_conds(positive, negative)
    guider.set_cfg(cfg, start_percent, end_percent)
    return guider


def _wan_pro_condition(c_pos, c_neg, loop_frames, motion_latent_count,
                       anchor_samples, prev_latent_samples):
    """Inline equivalent of WanImageToVideoSVIPro.execute().

    WanImageToVideoSVIPro is a V3 ComfyNode (io.ComfyNode / define_schema).
    Calling its .execute() from outside ComfyUI's execution framework is unreliable
    because the framework routes V3 calls through EXECUTE_NORMALIZED → PREPARE_CLASS_CLONE
    → make_locked_method_func — infrastructure not present in direct Python calls.

    This function replicates the exact logic from WanImageToVideoSVIPro.execute()
    (comfyui-kjnodes/nodes/nodes.py) inline, with no dependency on the KJNodes module.

    Returns: (pos_cond, neg_cond, latent_dict)
    """
    anchor_latent = anchor_samples["samples"].clone()
    B, C, T, H, W = anchor_latent.shape

    total_latents = (loop_frames - 1) // 4 + 1
    empty_latent  = torch.zeros(
        [B, 16, total_latents, H, W],
        device=comfy.model_management.intermediate_device(),
    )

    device = anchor_latent.device
    dtype  = anchor_latent.dtype

    if prev_latent_samples is None or motion_latent_count == 0:
        padding_size      = total_latents - T
        image_cond_latent = anchor_latent
    else:
        motion_latent     = prev_latent_samples["samples"][:, :, -motion_latent_count:].clone()
        padding_size      = total_latents - T - motion_latent.shape[2]
        image_cond_latent = torch.cat([anchor_latent, motion_latent], dim=2)

    padding           = torch.zeros(1, C, padding_size, H, W, dtype=dtype, device=device)
    padding           = comfy.latent_formats.Wan21().process_out(padding)
    image_cond_latent = torch.cat([image_cond_latent, padding], dim=2)

    mask            = torch.ones((1, 1, total_latents, H, W), device=device, dtype=dtype)
    mask[:, :, :1]  = 0.0

    pos_cond    = node_helpers.conditioning_set_values(
        c_pos, {"concat_latent_image": image_cond_latent, "concat_mask": mask}
    )
    neg_cond    = node_helpers.conditioning_set_values(
        c_neg, {"concat_latent_image": image_cond_latent, "concat_mask": mask}
    )
    latent_dict = {"samples": empty_latent}

    return pos_cond, neg_cond, latent_dict


# ---------------------------------------------------------------------------
# NODE 1: LoopConfigWan
# ---------------------------------------------------------------------------

class LoopConfigWan:
    """Per-segment configuration: prompt, frames, optional model overrides.

    model_high / model_low are optional MODEL inputs. If connected, they override
    the global models for this segment. Wire any LoRA stack or shift node upstream
    of these inputs — the looper receives fully-patched models.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "frames": ("INT",    {"default": 49, "min": 1, "max": 200}),
            },
            "optional": {
                "model_high": ("MODEL",),
                "model_low":  ("MODEL",),
            },
        }

    RETURN_TYPES  = ("WAN_LOOP_CONFIG",)
    RETURN_NAMES  = ("loop_config",)
    FUNCTION      = "build"
    CATEGORY      = "video/wan_looper_v2"

    def build(self, prompt, frames, model_high=None, model_low=None):
        return ({
            "prompt":     prompt,
            "frames":     frames,
            "model_high": model_high,
            "model_low":  model_low,
        },)


# ---------------------------------------------------------------------------
# NODE 2: WanLooperNative
# ---------------------------------------------------------------------------

class WanLooperNative:
    """
    Multi-segment Wan looper with temporal continuity via SVIPro conditioning.

    Each segment's sampled latent is passed as prev_samples to the next segment's
    Wan conditioning step, giving the model temporal context that eliminates
    the ghosting artefacts present in Phase 1's standard WanImageToVideo approach.

    Wan conditioning is inlined from WanImageToVideoSVIPro.execute() (KJNodes)
    to avoid calling V3 ComfyNode.execute() outside the ComfyUI execution framework.

    Sampling pipeline: SamplerCustomAdvanced with ScheduledCFGGuidance guiders
    (high model guider covers steps 0..split, low model guider covers split..end)
    split via SplitSigmas on a BasicScheduler sigma schedule.

    clip_vision is accepted but not used in the Wan conditioning path — image
    conditioning is provided through anchor_samples (VAE-encoded latent).
    Kept for forward compatibility with Phase 3.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_high":   ("MODEL",),
                "model_low":    ("MODEL",),
                "vae":          ("VAE",),
                "clip":         ("CLIP",),
                "clip_vision":  ("CLIP_VISION",),  # reserved — not used in Wan path
                "start_image":  ("IMAGE",),
                "width":  ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 640, "min": 16, "max": 4096, "step": 16}),
                "steps":  ("INT", {"default": 8,   "min": 1,  "max": 100}),
                "split_step": ("INT", {
                    "default": 3, "min": 1, "max": 50,
                    "tooltip": "Sigma split point — high model runs steps 0 to split_step, "
                               "low model runs split_step to end",
                }),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler":    (comfy.samplers.KSampler.SCHEDULERS,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "overlap": ("INT", {
                    "default": 3, "min": 0, "max": 20, "step": 1,
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
                "color_correction": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Normalize anchor frame color statistics to match original start image",
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
                "loop_1":  ("WAN_LOOP_CONFIG",),
                "loop_2":  ("WAN_LOOP_CONFIG",),
                "loop_3":  ("WAN_LOOP_CONFIG",),
                "loop_4":  ("WAN_LOOP_CONFIG",),
                "loop_5":  ("WAN_LOOP_CONFIG",),
                "loop_6":  ("WAN_LOOP_CONFIG",),
                "loop_7":  ("WAN_LOOP_CONFIG",),
                "loop_8":  ("WAN_LOOP_CONFIG",),
                "loop_9":  ("WAN_LOOP_CONFIG",),
                "loop_10": ("WAN_LOOP_CONFIG",),
            },
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES  = ("full_video", "last_anchor", "used_prompts")
    FUNCTION      = "generate"
    CATEGORY      = "video/wan_looper_v2"

    def generate(self,
                 model_high, model_low, vae, clip, clip_vision, start_image,
                 width, height, steps, split_step, cfg, sampler_name, scheduler,
                 seed, overlap, overlap_mode, overlap_side, anchor_frame_offset,
                 color_correction,
                 positive_prompt="", negative_prompt="",
                 loop_1=None, loop_2=None, loop_3=None, loop_4=None, loop_5=None,
                 loop_6=None, loop_7=None, loop_8=None, loop_9=None, loop_10=None):

        loop_configs = [loop_1, loop_2, loop_3, loop_4, loop_5,
                        loop_6, loop_7, loop_8, loop_9, loop_10]
        active_configs = [(i + 1, lc) for i, lc in enumerate(loop_configs)
                         if lc is not None]

        if not active_configs:
            print("[WanLooperNative] No loop configs connected — returning black frame")
            return (torch.zeros((1, height, width, 3)),
                    torch.zeros((1, height, width, 3)),
                    "No loops connected")

        total_segments = len(active_configs)
        print(f"\n[WanLooperNative] {'='*52}")
        print(f"[WanLooperNative] Starting — {total_segments} segment(s)")
        print(f"[WanLooperNative] {'='*52}")

        # ── Resize start image ───────────────────────────────────────────────
        current_start_image = comfy.utils.common_upscale(
            start_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)

        # ── Reference color stats (computed once from resized start image) ───
        if color_correction:
            ref_mean = current_start_image.float().mean(dim=(0, 1, 2), keepdim=True)
            ref_std  = current_start_image.float().std(dim=(0, 1, 2),  keepdim=True).clamp(min=1e-5)
        else:
            ref_mean = ref_std = None

        # ── State ────────────────────────────────────────────────────────────
        segment_dir         = tempfile.mkdtemp(prefix="wan_native_")
        segment_paths       = []
        used_prompts_log    = []
        prev_decoded        = None
        motion_latent_count = 1      # hardcoded; ignored for segment 1 (prev_samples is None)
        prev_latent_samples = None   # carries sampled latent to next segment
        current_seed        = seed
        last_anchor         = current_start_image

        # Pre-build reusable sampler object (same sampler for both passes)
        sampler_obj = KSamplerSelect.execute(sampler_name=sampler_name)[0]

        for seg_num, (loop_id, config) in enumerate(active_configs):
            print(f"\n[WanLooperNative] ═══ Segment {seg_num + 1}/{total_segments} "
                  f"{'═' * max(0, 40 - len(str(seg_num+1)))}")

            # ── a) Extract config ────────────────────────────────────────────
            loop_prompt = config["prompt"].strip() or positive_prompt or ""
            loop_frames = config["frames"] if config["frames"] > 0 else 49
            high_model  = config.get("model_high") or model_high
            low_model   = config.get("model_low")  or model_low

            print(f"[WanLooperNative] Segment {loop_id} | "
                  f"Prompt: \"{loop_prompt}\"")
            print(f"[WanLooperNative] Segment {loop_id} | "
                  f"{loop_frames} frames, seed={current_seed}")
            used_prompts_log.append(
                f"Segment {loop_id} ({loop_frames}f): {loop_prompt}"
            )

            # ── b) Text conditioning ─────────────────────────────────────────
            tokens_pos = clip.tokenize(loop_prompt)
            cond_pos, pooled_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True)
            c_pos = [[cond_pos, {"pooled_output": pooled_pos}]]

            tokens_neg = clip.tokenize(negative_prompt)
            cond_neg, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)
            c_neg = [[cond_neg, {"pooled_output": pooled_neg}]]

            # ── c) VAE encode anchor image → anchor_samples ──────────────────
            # WanImageToVideoSVIPro uses anchor_samples (VAE latent) for image
            # conditioning, NOT CLIP Vision. clip_vision input is unused here.
            print(f"[WanLooperNative] Segment {loop_id} | VAE encode anchor...")
            anchor_latent = vae.encode(current_start_image[:, :, :, :3])
            # Ensure 5D video latent format [B, C, T, H, W]
            if anchor_latent.ndim == 4:
                anchor_latent = anchor_latent.unsqueeze(2)
            anchor_samples = {"samples": anchor_latent}

            # ── d) Wan Pro conditioning (inlined WanImageToVideoSVIPro logic) ──
            prev_str = "yes" if prev_latent_samples is not None else "None"
            print(f"[WanLooperNative] Segment {loop_id} | "
                  f"Wan conditioning (prev_samples={prev_str}, "
                  f"motion={motion_latent_count})")

            pos_cond, neg_cond, latent_dict = _wan_pro_condition(
                c_pos, c_neg, loop_frames, motion_latent_count,
                anchor_samples, prev_latent_samples,
            )

            # ── e) Build sigma schedule and split ────────────────────────────
            split_percent = split_step / steps
            print(f"[WanLooperNative] Segment {loop_id} | "
                  f"Sigmas: {steps} total → split at {split_step} "
                  f"(high={split_step}, low={steps - split_step})")

            # BasicScheduler.execute(model, scheduler, steps, denoise) → (sigmas,)
            sigmas_full = BasicScheduler.execute(
                model=high_model, scheduler=scheduler, steps=steps, denoise=1.0
            )[0]

            # SplitSigmas.execute(sigmas, step) → (sigmas[:step+1], sigmas[step:])
            split_result = SplitSigmas.execute(sigmas=sigmas_full, step=split_step)
            sigmas_high  = split_result[0]
            sigmas_low   = split_result[1]

            # ── f) Build guiders ─────────────────────────────────────────────
            guider_high = _build_scheduled_cfg_guider(
                model         = high_model,
                cfg           = cfg,
                positive      = pos_cond,
                negative      = neg_cond,
                start_percent = 0.0,
                end_percent   = split_percent,
            )

            guider_low = _build_scheduled_cfg_guider(
                model         = low_model,
                cfg           = cfg,
                positive      = pos_cond,
                negative      = neg_cond,
                start_percent = split_percent,
                end_percent   = 1.0,
            )

            # ── g) Build noise objects ───────────────────────────────────────
            # Instantiate directly — no need for ComfyUI node wrappers
            noise_obj    = Noise_RandomNoise(current_seed)
            no_noise_obj = Noise_EmptyNoise()

            # ── h) Two-pass SamplerCustomAdvanced ────────────────────────────
            # SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas,
            #                               latent_image) → (out_latent, denoised)
            print(f"[WanLooperNative] Segment {loop_id} | "
                  f"▶ HIGH NOISE PASS ({split_step} steps)")
            high_result = SamplerCustomAdvanced.execute(
                noise        = noise_obj,
                guider       = guider_high,
                sampler      = sampler_obj,
                sigmas       = sigmas_high,
                latent_image = latent_dict,
            )

            comfy.model_management.soft_empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            print(f"[WanLooperNative] Segment {loop_id} | "
                  f"▶ LOW NOISE PASS ({steps - split_step} steps)")
            low_result = SamplerCustomAdvanced.execute(
                noise        = no_noise_obj,
                guider       = guider_low,
                sampler      = sampler_obj,
                sigmas       = sigmas_low,
                latent_image = {"samples": high_result[0]["samples"]},
            )

            # ── i) Store latent for next segment's Wan conditioning ──────────
            prev_latent_samples = {"samples": low_result[0]["samples"].clone()}

            # ── j) VAE decode ────────────────────────────────────────────────
            sampled = low_result[0]["samples"]
            print(f"[WanLooperNative] Segment {loop_id} | VAE decode...")
            decoded = vae.decode(sampled)
            while decoded.ndim > 4:
                decoded = decoded.squeeze(0)
            if decoded.shape[1] != height or decoded.shape[2] != width:
                decoded = comfy.utils.common_upscale(
                    decoded.movedim(-1, 1), width, height, "bilinear", "center"
                ).movedim(1, -1)
            print(f"[WanLooperNative] Segment {loop_id} | "
                  f"VAE decode → {decoded.shape[0]} frames")

            # ── k) Anchor frame extraction ───────────────────────────────────
            actual_frames = decoded.shape[0]
            anchor_idx    = max(0, min(actual_frames - 1,
                                       actual_frames + anchor_frame_offset))
            raw_anchor    = decoded[anchor_idx:anchor_idx + 1].clone()
            print(f"[WanLooperNative] Segment {loop_id} | Anchor at frame {anchor_idx}")

            # ── l) Color drift correction ────────────────────────────────────
            if color_correction:
                anchor_f = raw_anchor.float()
                a_mean   = anchor_f.mean(dim=(0, 1, 2), keepdim=True)
                a_std    = anchor_f.std(dim=(0, 1, 2),  keepdim=True).clamp(min=1e-5)
                anchor_normalized = ((anchor_f - a_mean) / a_std) * ref_std + ref_mean
                anchor_normalized = anchor_normalized.clamp(0.0, 1.0).to(raw_anchor.dtype)
                current_start_image = anchor_normalized
                print(f"[WanLooperNative] Segment {loop_id} | "
                      f"Color corrected "
                      f"(mean {a_mean.mean().item():.3f}→{ref_mean.mean().item():.3f})")
            else:
                current_start_image = raw_anchor

            last_anchor = current_start_image

            # ── m) Trim to anchor point ──────────────────────────────────────
            decoded_trimmed = decoded[:anchor_idx + 1]
            print(f"[WanLooperNative] Segment {loop_id} | "
                  f"Trimmed to {decoded_trimmed.shape[0]} frames")

            # ── n) Overlap stitching ─────────────────────────────────────────
            # Re-save previous segment with tail removed, save only [blend+suffix]
            # to avoid double-counting the overlap region.
            if seg_num > 0 and overlap > 0 and prev_decoded is not None and overlap_mode != "cut":
                eff_overlap = min(overlap, prev_decoded.shape[0], decoded_trimmed.shape[0])
                print(f"[WanLooperNative] Segment {loop_id} | "
                      f"Overlap stitch ({eff_overlap}f, {overlap_mode}, "
                      f"side={overlap_side})")

                if overlap_side == "source":
                    fade_out = prev_decoded[-eff_overlap:]
                    fade_in  = decoded_trimmed[:eff_overlap]
                else:  # "new_images"
                    fade_out = decoded_trimmed[:eff_overlap]
                    fade_in  = prev_decoded[-eff_overlap:]

                suffix = decoded_trimmed[eff_overlap:]
                dev    = fade_out.device
                fdt    = fade_out.float().dtype

                if overlap_mode == "linear_blend":
                    alpha   = torch.linspace(0, 1, eff_overlap + 2,
                                             device=dev, dtype=fdt)[1:-1].view(-1, 1, 1, 1)
                    blended = ((1 - alpha) * fade_out.float() +
                               alpha        * fade_in.float()).to(fade_out.dtype)

                elif overlap_mode == "ease_in_out":
                    t       = torch.linspace(0, 1, eff_overlap + 2,
                                             device=dev, dtype=fdt)[1:-1]
                    eased   = (3 * t * t - 2 * t * t * t).view(-1, 1, 1, 1)
                    blended = ((1 - eased) * fade_out.float() +
                               eased        * fade_in.float()).to(fade_out.dtype)

                elif overlap_mode == "filmic_crossfade":
                    gamma    = 2.2
                    src_lin  = fade_out.float().clamp(0, 1) ** gamma
                    dst_lin  = fade_in.float().clamp(0, 1)  ** gamma
                    t        = torch.linspace(0, 1, eff_overlap + 2,
                                              device=dev, dtype=fdt)[1:-1]
                    eased    = (3 * t * t - 2 * t * t * t).view(-1, 1, 1, 1)
                    blended  = ((1 - eased) * src_lin + eased * dst_lin).clamp(0, 1)
                    blended  = (blended ** (1.0 / gamma)).to(fade_out.dtype)

                else:
                    blended = fade_in.clone()  # unreachable given overlap_mode != "cut" guard

                # Re-save previous segment with overlap tail removed
                trimmed_prev = prev_decoded[:-eff_overlap]
                torch.save(trimmed_prev.cpu(), segment_paths[-1])
                print(f"[WanLooperNative] Segment {loop_id} | "
                      f"Re-saved prev → {trimmed_prev.shape[0]} frames")

                segment_frames = torch.cat([blended, suffix], dim=0)

            else:
                if seg_num > 0 and overlap_mode == "cut":
                    print(f"[WanLooperNative] Segment {loop_id} | Cut (no blend)")
                segment_frames = decoded_trimmed

            # ── o) Save segment ──────────────────────────────────────────────
            seg_path = os.path.join(segment_dir, f"segment_{loop_id:03d}.pt")
            torch.save(segment_frames.cpu(), seg_path)
            segment_paths.append(seg_path)
            print(f"[WanLooperNative] Segment {loop_id} | "
                  f"Saved → {segment_frames.shape[0]} frames "
                  f"(decoded={decoded.shape[0]}, anchor_idx={anchor_idx}, "
                  f"trimmed={decoded_trimmed.shape[0]})")

            # ── p) State update ──────────────────────────────────────────────
            prev_decoded  = decoded_trimmed
            current_seed += 1

            # ── q) Cleanup ───────────────────────────────────────────────────
            del decoded, sampled
            comfy.model_management.soft_empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # ── Final assembly ───────────────────────────────────────────────────
        print(f"\n[WanLooperNative] Assembling {len(segment_paths)} segment(s)...")
        full_video = None
        for seg_path in segment_paths:
            seg        = torch.load(seg_path, map_location="cpu", weights_only=True)
            full_video = seg if full_video is None else torch.cat([full_video, seg], dim=0)
            del seg
            gc.collect()

        shutil.rmtree(segment_dir, ignore_errors=True)

        if full_video is None:
            full_video = torch.zeros((1, height, width, 3))

        print(f"[WanLooperNative] Done. Total frames: {full_video.shape[0]}")
        return (full_video, last_anchor, "\n".join(used_prompts_log))


# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    # New canonical names
    "LoopConfigWan":   LoopConfigWan,
    "WanLooperNative": WanLooperNative,

    # Backward-compat aliases for saved workflow JSONs
    "LoopConfigSVI":    LoopConfigWan,
    "SVILooperNative":  WanLooperNative,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoopConfigWan":   "Wan Loop Config",
    "WanLooperNative": "Wan Looper Native",
}
