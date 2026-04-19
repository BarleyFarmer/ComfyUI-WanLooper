"""
Wan Looper Native — Multi-segment Wan video looper using KJNodes SVI Pro.

Two nodes:
  LoopConfigWan    — per-segment config (prompt, frames, optional model overrides)
  WanLooperNative  — main looper using SamplerCustomAdvanced + KJNodes SVI helpers

Unlike Phase 1 (I2VLooperV2), this node passes prev_samples (previous segment's
sampled latent) into the Wan conditioning so the model has temporal context at
the segment boundary — eliminating the ghosting that Phase 1 suffers from.

Sampling: SamplerCustomAdvanced with SplitSigmas + ScheduledCFGGuidance guiders
(high model for early steps, low model for late steps — no KSamplerAdvanced).

This native path now depends on comfyui-kjnodes and uses the real:
  WanImageToVideoSVIPro
  ScheduledCFGGuidance
  ImageBatchExtendWithOverlap

That keeps the high-level loop orchestration in this repo while matching the
cleaner reference workflow's core SVI conditioning and overlap behavior.
"""

import torch
import gc
import os
import random
import sys
import tempfile
import shutil
import types
import importlib.util

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
# KJNodes dependency loader
# ---------------------------------------------------------------------------

_CUSTOM_NODES = os.path.join(
    os.path.dirname(os.path.abspath(folder_paths.__file__)),
    "custom_nodes",
)


def _find_kjnodes_root(custom_nodes_dir):
    candidates = []

    exact = os.path.join(custom_nodes_dir, "comfyui-kjnodes")
    if os.path.isdir(exact):
        candidates.append(exact)

    try:
        for entry in os.scandir(custom_nodes_dir):
            if not entry.is_dir():
                continue
            lower_name = entry.name.lower()
            if "kjnodes" in lower_name or ("kj" in lower_name and "node" in lower_name):
                candidates.append(entry.path)
    except FileNotFoundError:
        return None

    seen = set()
    for root in candidates:
        norm = os.path.normcase(os.path.normpath(root))
        if norm in seen:
            continue
        seen.add(norm)
        if (os.path.isfile(os.path.join(root, "nodes", "nodes.py")) and
                os.path.isfile(os.path.join(root, "nodes", "image_nodes.py"))):
            return root

    return None


_KJNODES_ROOT = _find_kjnodes_root(_CUSTOM_NODES)
_KJ_LOAD_ERROR = None


def _ensure_package(pkg_name, pkg_dir):
    if pkg_name not in sys.modules:
        mod = types.ModuleType(pkg_name)
        mod.__path__ = [pkg_dir]
        mod.__package__ = pkg_name
        sys.modules[pkg_name] = mod


def _load_package_class(package_root, rel_file, class_name, pkg_name=None):
    if pkg_name is None:
        pkg_name = os.path.basename(package_root).replace("-", "_")

    _ensure_package(pkg_name, package_root)

    rel_no_ext = rel_file[:-3] if rel_file.endswith(".py") else rel_file
    parts = rel_no_ext.replace("\\", "/").split("/")

    current_pkg = pkg_name
    current_dir = package_root
    target_mod = None

    for i, part in enumerate(parts):
        mod_name = f"{current_pkg}.{part}"
        subpath = os.path.join(current_dir, part)

        if i < len(parts) - 1:
            _ensure_package(mod_name, subpath)
            current_pkg = mod_name
            current_dir = subpath
        else:
            file_path = subpath + ".py"
            if mod_name not in sys.modules:
                spec = importlib.util.spec_from_file_location(mod_name, file_path)
                target_mod = importlib.util.module_from_spec(spec)
                target_mod.__package__ = current_pkg
                sys.modules[mod_name] = target_mod
                spec.loader.exec_module(target_mod)
            else:
                target_mod = sys.modules[mod_name]

    return getattr(target_mod, class_name)


def _resolve_loaded_class(class_name, required_attr):
    for mod_name, module in list(sys.modules.items()):
        if module is None:
            continue
        if "kjnodes" not in mod_name.lower():
            continue
        try:
            cls = getattr(module, class_name, None)
        except Exception:
            cls = None
        if cls is None:
            continue
        if not isinstance(cls, type):
            continue
        if not callable(getattr(cls, required_attr, None)):
            continue
        module_name = getattr(cls, "__module__", "")
        if "kjnodes" not in module_name.lower():
            continue
        if cls is not None:
            return cls
    return None


def _ensure_kj_dependencies_loaded():
    global WanImageToVideoSVIPro, _KJ_ScheduledCFGGuidance, _KJ_ImageBatchExtend, _KJ_LOAD_ERROR

    if (WanImageToVideoSVIPro is not None and
            _KJ_ScheduledCFGGuidance is not None and
            _KJ_ImageBatchExtend is not None):
        return True

    loaded_svi = _resolve_loaded_class("WanImageToVideoSVIPro", "execute")
    loaded_guidance = _resolve_loaded_class("ScheduledCFGGuidance", "get_guider")
    loaded_overlap = _resolve_loaded_class("ImageBatchExtendWithOverlap", "imagesfrombatch")

    try:
        if WanImageToVideoSVIPro is None and loaded_svi is not None:
            WanImageToVideoSVIPro = loaded_svi
        if _KJ_ScheduledCFGGuidance is None and loaded_guidance is not None:
            _KJ_ScheduledCFGGuidance = loaded_guidance()
        if _KJ_ImageBatchExtend is None and loaded_overlap is not None:
            _KJ_ImageBatchExtend = loaded_overlap()
    except Exception as e:
        _KJ_LOAD_ERROR = f"resolved-loaded-class fallback failed: {e}"

    return (
        WanImageToVideoSVIPro is not None and
        _KJ_ScheduledCFGGuidance is not None and
        _KJ_ImageBatchExtend is not None
    )


try:
    if _KJNODES_ROOT is None:
        raise FileNotFoundError(
            f"Could not locate a KJNodes folder under {_CUSTOM_NODES}"
        )

    WanImageToVideoSVIPro = _load_package_class(
        _KJNODES_ROOT, "nodes/nodes.py", "WanImageToVideoSVIPro",
        pkg_name="comfyui_kjnodes",
    )
    ScheduledCFGGuidance = _load_package_class(
        _KJNODES_ROOT, "nodes/nodes.py", "ScheduledCFGGuidance",
        pkg_name="comfyui_kjnodes",
    )
    ImageBatchExtendWithOverlap = _load_package_class(
        _KJNODES_ROOT, "nodes/image_nodes.py", "ImageBatchExtendWithOverlap",
        pkg_name="comfyui_kjnodes",
    )
    _KJ_ScheduledCFGGuidance = ScheduledCFGGuidance()
    _KJ_ImageBatchExtend = ImageBatchExtendWithOverlap()
    print(f"[WanLooperNative] Loaded KJNodes SVI Pro dependencies from {_KJNODES_ROOT}")
except Exception as e:
    print(f"[WanLooperNative] WARNING: Could not load KJNodes dependencies: {e}")
    _KJ_LOAD_ERROR = str(e)
    WanImageToVideoSVIPro = None
    _KJ_ScheduledCFGGuidance = None
    _KJ_ImageBatchExtend = None


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


def _parse_segment_schedule(schedule_text):
    """Parse a comma-separated segment schedule into a set of positive ints.

    Supports simple values like "3,6,9" and inclusive ranges like "4-6".
    Invalid tokens are ignored so experimentation in the UI stays forgiving.
    """
    if not schedule_text:
        return set()

    schedule = set()
    for raw_token in schedule_text.replace(";", ",").split(","):
        token = raw_token.strip()
        if not token:
            continue

        if "-" in token:
            try:
                start_str, end_str = token.split("-", 1)
                start = int(start_str.strip())
                end   = int(end_str.strip())
            except ValueError:
                continue

            if start <= 0 or end <= 0:
                continue
            if start > end:
                start, end = end, start
            schedule.update(range(start, end + 1))
            continue

        try:
            value = int(token)
        except ValueError:
            continue
        if value > 0:
            schedule.add(value)

    return schedule


# ---------------------------------------------------------------------------
# NODE 1: LoopConfigWan
# ---------------------------------------------------------------------------

class LoopConfigWan:
    """Per-segment configuration: prompt, frames, optional model overrides.

    model_high / model_low are optional MODEL inputs. If connected, they override
    the global models for this segment. Wire any LoRA stack or shift node upstream
    of these inputs — the looper receives fully-patched models. anchor_image is
    optional and lets the workflow inject a fresh keyframe anchor on specific
    segments without relying on automatic dynamic-anchor extraction.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Positive prompt text for this segment.",
                }),
                "frames": ("INT", {
                    "default": 49, "min": 1, "max": 200,
                    "tooltip": "Target frame count for this segment before overlap stitching.",
                }),
            },
            "optional": {
                "model_high": ("MODEL", {
                    "tooltip": "Optional fully prepared high-noise model override for this segment.",
                }),
                "model_low":  ("MODEL", {
                    "tooltip": "Optional fully prepared low-noise model override for this segment.",
                }),
                "anchor_image": ("IMAGE", {
                    "tooltip": "Optional keyframe anchor image override for this segment.",
                }),
            },
        }

    RETURN_TYPES  = ("WAN_LOOP_CONFIG",)
    RETURN_NAMES  = ("loop_config",)
    FUNCTION      = "build"
    CATEGORY      = "wan_looper/native"

    def build(self, prompt, frames, model_high=None, model_low=None, anchor_image=None):
        return ({
            "prompt":       prompt,
            "frames":       frames,
            "model_high":   model_high,
            "model_low":    model_low,
            "anchor_image": anchor_image,
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
                "model_high":   ("MODEL", {
                    "tooltip": "Global high-noise model used unless a segment overrides it.",
                }),
                "model_low":    ("MODEL", {
                    "tooltip": "Global low-noise model used unless a segment overrides it.",
                }),
                "vae":          ("VAE", {
                    "tooltip": "VAE used to encode the anchor image and decode each generated segment.",
                }),
                "clip":         ("CLIP", {
                    "tooltip": "Text encoder used to turn each segment prompt into conditioning.",
                }),
                "clip_vision":  ("CLIP_VISION", {
                    "tooltip": "Reserved for future use. Accepted for workflow compatibility but currently unused in the Wan path.",
                }),
                "start_image":  ("IMAGE", {
                    "tooltip": "Initial image anchor for the looper. This is resized to the requested width and height.",
                }),
                "width":  ("INT", {
                    "default": 480, "min": 16, "max": 4096, "step": 16,
                    "tooltip": "Output width used for segment generation and anchor preparation.",
                }),
                "height": ("INT", {
                    "default": 640, "min": 16, "max": 4096, "step": 16,
                    "tooltip": "Output height used for segment generation and anchor preparation.",
                }),
                "steps":  ("INT", {
                    "default": 8,   "min": 1,  "max": 100,
                    "tooltip": "Total sampler steps per segment across both high-noise and low-noise passes.",
                }),
                "split_step": ("INT", {
                    "default": 3, "min": 1, "max": 50,
                    "tooltip": "Sigma split point — high model runs steps 0 to split_step, "
                               "low model runs split_step to end",
                }),
                "cfg": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "CFG strength used by both scheduled guiders during sampling.",
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                    "tooltip": "Sampler algorithm used for both passes of every segment.",
                }),
                "scheduler":    (comfy.samplers.KSampler.SCHEDULERS, {
                    "tooltip": "Sigma scheduler used to build the full denoising schedule before splitting.",
                }),
                "initial_seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "Seed used for segment 1. fixed reuses it for every segment, randomize keeps segment 1 on this seed then picks a fresh random seed for each later segment.",
                }),
                "seed_mode": (
                    ["fixed", "randomize"],
                    {
                        "default": "fixed",
                        "tooltip": "How seeds change across segments: fixed reuses initial_seed for every segment, randomize keeps segment 1 on initial_seed then uses a fresh random seed for each later segment.",
                    },
                ),
                "overlap": ("INT", {
                    "default": 5, "min": 0, "max": 20, "step": 1,
                    "tooltip": "Overlap frames for blending between segments (0 = hard cut)",
                }),
                "startup_trim": ("INT", {
                    "default": 0, "min": 0, "max": 20, "step": 1,
                    "tooltip": "For segments after the first, drop this many decoded startup frames before stitching and overlap blending.",
                }),
                "overlap_mode": (
                    ["linear_blend", "ease_in_out", "filmic_crossfade", "cut"],
                    {
                        "default": "linear_blend",
                        "tooltip": "Blend style used when overlapping adjacent segments.",
                    },
                ),
                "overlap_side": (
                    ["source", "new_images"],
                    {
                        "default": "source",
                        "tooltip": "Which side of the seam is treated as the outgoing source during overlap blending.",
                    },
                ),
                "anchor_mode": (
                    ["keyframe_schedule", "fixed_initial", "dynamic_every_segment"],
                    {
                        "default": "fixed_initial",
                        "tooltip": "How anchor images are chosen across segments.",
                    },
                ),
                "stitch_mode": (
                    ["workflow_style", "trim_to_anchor"],
                    {
                        "default": "workflow_style",
                        "tooltip": "Whether to keep each full decoded segment for stitching or trim at the extracted anchor frame.",
                    },
                ),
                "anchor_frame_offset": ("INT", {
                    "default": -5, "min": -50, "max": -1, "step": 1,
                    "tooltip": "Frames from end to extract anchor. -5 = 5th from last",
                }),
                "color_correction": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Normalize anchor frame color statistics to match original start image",
                }),
            },
            "optional": {
                "positive_prompt": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Global fallback positive prompt used when a segment config prompt is blank.",
                }),
                "negative_prompt": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Global negative prompt used for every segment.",
                }),
                "keyframe_schedule": ("STRING", {
                    "default": "",
                    "tooltip": "Segments that promote their extracted anchor for future segments, e.g. 3,6,9 or 4-6",
                }),
                "loop_1":  ("WAN_LOOP_CONFIG", {"tooltip": "Configuration for segment 1."}),
                "loop_2":  ("WAN_LOOP_CONFIG", {"tooltip": "Configuration for segment 2."}),
                "loop_3":  ("WAN_LOOP_CONFIG", {"tooltip": "Configuration for segment 3."}),
                "loop_4":  ("WAN_LOOP_CONFIG", {"tooltip": "Configuration for segment 4."}),
                "loop_5":  ("WAN_LOOP_CONFIG", {"tooltip": "Configuration for segment 5."}),
                "loop_6":  ("WAN_LOOP_CONFIG", {"tooltip": "Configuration for segment 6."}),
                "loop_7":  ("WAN_LOOP_CONFIG", {"tooltip": "Configuration for segment 7."}),
                "loop_8":  ("WAN_LOOP_CONFIG", {"tooltip": "Configuration for segment 8."}),
                "loop_9":  ("WAN_LOOP_CONFIG", {"tooltip": "Configuration for segment 9."}),
                "loop_10": ("WAN_LOOP_CONFIG", {"tooltip": "Configuration for segment 10."}),
            },
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES  = ("full_video", "last_extracted_anchor", "All Segment Prompts")
    FUNCTION      = "generate"
    CATEGORY      = "wan_looper/native"

    def generate(self,
                 model_high, model_low, vae, clip, clip_vision, start_image,
                 width, height, steps, split_step, cfg, sampler_name, scheduler,
                 initial_seed, overlap, startup_trim, overlap_mode, overlap_side, anchor_mode,
                 stitch_mode, seed_mode, anchor_frame_offset,
                 color_correction,
                 positive_prompt="", negative_prompt="", keyframe_schedule="",
                 loop_1=None, loop_2=None, loop_3=None, loop_4=None, loop_5=None,
                 loop_6=None, loop_7=None, loop_8=None, loop_9=None, loop_10=None):
        if not _ensure_kj_dependencies_loaded():
            raise RuntimeError(
                "[WanLooperNative] comfyui-kjnodes with WanImageToVideoSVIPro, "
                "ScheduledCFGGuidance, and ImageBatchExtendWithOverlap is required "
                "for the native SVI path. "
                f"custom_nodes={_CUSTOM_NODES} | "
                f"detected_kjnodes_root={_KJNODES_ROOT} | "
                f"load_error={_KJ_LOAD_ERROR}"
            )

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
        keyframe_refresh_segments = _parse_segment_schedule(keyframe_schedule)
        print(f"\n[WanLooperNative] {'='*52}")
        print(f"[WanLooperNative] Starting — {total_segments} segment(s)")
        print(f"[WanLooperNative] Anchor mode={anchor_mode} | "
              f"Stitch mode={stitch_mode} | Seed mode={seed_mode}")
        if keyframe_refresh_segments:
            print(f"[WanLooperNative] Keyframe schedule → "
                  f"{sorted(keyframe_refresh_segments)}")
        print(f"[WanLooperNative] {'='*52}")

        # ── Resize start image ───────────────────────────────────────────────
        initial_anchor_image = comfy.utils.common_upscale(
            start_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)

        # ── Reference color stats (computed once from resized start image) ───
        if color_correction:
            ref_mean = initial_anchor_image.float().mean(dim=(0, 1, 2), keepdim=True)
            ref_std  = initial_anchor_image.float().std(dim=(0, 1, 2),  keepdim=True).clamp(min=1e-5)
        else:
            ref_mean = ref_std = None

        # ── State ────────────────────────────────────────────────────────────
        segment_dir         = tempfile.mkdtemp(prefix="wan_native_")
        segment_paths       = []
        all_segment_prompts_log = []
        prev_decoded        = None
        motion_latent_count = 1      # hardcoded; ignored for segment 1 (prev_samples is None)
        prev_latent_samples = None   # carries sampled latent to next segment
        scheduled_anchor_image = initial_anchor_image.clone()
        dynamic_anchor_image   = initial_anchor_image.clone()
        last_extracted_anchor = initial_anchor_image

        # Pre-build reusable sampler object (same sampler for both passes)
        sampler_obj = KSamplerSelect.execute(sampler_name=sampler_name)[0]

        for seg_num, (loop_id, config) in enumerate(active_configs):
            if seg_num == 0 or seed_mode == "fixed":
                segment_seed = initial_seed
            else:  # "randomize"
                segment_seed = random.getrandbits(64)

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
                  f"{loop_frames} frames, seed={segment_seed}")
            all_segment_prompts_log.append(
                f"Segment {loop_id} ({loop_frames}f): {loop_prompt}"
            )

            explicit_anchor_image = config.get("anchor_image")
            if explicit_anchor_image is not None:
                anchor_image_for_segment = comfy.utils.common_upscale(
                    explicit_anchor_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
                ).movedim(1, -1)
                scheduled_anchor_image = anchor_image_for_segment.clone()
                anchor_source = "explicit_segment_anchor"
            elif anchor_mode == "dynamic_every_segment" and seg_num > 0:
                anchor_image_for_segment = dynamic_anchor_image
                anchor_source = "dynamic_previous_anchor"
            elif anchor_mode == "keyframe_schedule":
                anchor_image_for_segment = scheduled_anchor_image
                anchor_source = "scheduled_keyframe_anchor"
            else:
                anchor_image_for_segment = initial_anchor_image
                anchor_source = "initial_anchor"

            print(f"[WanLooperNative] Segment {loop_id} | "
                  f"Anchor source={anchor_source}")

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
            anchor_latent = vae.encode(anchor_image_for_segment[:, :, :, :3])
            # Ensure 5D video latent format [B, C, T, H, W]
            if anchor_latent.ndim == 4:
                anchor_latent = anchor_latent.unsqueeze(2)
            anchor_samples = {"samples": anchor_latent}

            # ── d) Wan Pro conditioning via KJNodes WanImageToVideoSVIPro ───
            prev_str = "yes" if prev_latent_samples is not None else "None"
            print(f"[WanLooperNative] Segment {loop_id} | "
                  f"Wan conditioning (prev_samples={prev_str}, "
                  f"motion={motion_latent_count})")

            svi_result = WanImageToVideoSVIPro.execute(
                positive=c_pos,
                negative=c_neg,
                length=loop_frames,
                motion_latent_count=motion_latent_count,
                anchor_samples=anchor_samples,
                prev_samples=prev_latent_samples,
            )
            pos_cond = svi_result[0]
            neg_cond = svi_result[1]
            latent_dict = svi_result[2]

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
            guider_high = _KJ_ScheduledCFGGuidance.get_guider(
                model=high_model,
                cfg=cfg,
                positive=pos_cond,
                negative=neg_cond,
                start_percent=0.0,
                end_percent=split_percent,
            )[0]

            guider_low = _KJ_ScheduledCFGGuidance.get_guider(
                model=low_model,
                cfg=cfg,
                positive=pos_cond,
                negative=neg_cond,
                start_percent=split_percent,
                end_percent=1.0,
            )[0]

            # ── g) Build noise objects ───────────────────────────────────────
            # Instantiate directly — no need for ComfyUI node wrappers
            noise_obj    = Noise_RandomNoise(segment_seed)
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
                processed_anchor = anchor_normalized
                print(f"[WanLooperNative] Segment {loop_id} | "
                      f"Color corrected "
                      f"(mean {a_mean.mean().item():.3f}→{ref_mean.mean().item():.3f})")
            else:
                processed_anchor = raw_anchor

            last_extracted_anchor = processed_anchor
            dynamic_anchor_image = processed_anchor

            if anchor_mode == "keyframe_schedule" and loop_id in keyframe_refresh_segments:
                scheduled_anchor_image = processed_anchor.clone()
                print(f"[WanLooperNative] Segment {loop_id} | "
                      f"Promoted extracted anchor into keyframe schedule")

            # ── m) Choose frames used for stitching ──────────────────────────
            if stitch_mode == "trim_to_anchor":
                decoded_for_stitch = decoded[:anchor_idx + 1]
                print(f"[WanLooperNative] Segment {loop_id} | "
                      f"Trimmed to {decoded_for_stitch.shape[0]} frames")
            else:
                decoded_for_stitch = decoded
                print(f"[WanLooperNative] Segment {loop_id} | "
                      f"Workflow-style stitch keeps all {decoded_for_stitch.shape[0]} frames")

            if seg_num > 0 and startup_trim > 0 and decoded_for_stitch.shape[0] > 1:
                effective_trim = min(startup_trim, decoded_for_stitch.shape[0] - 1)
                decoded_for_stitch = decoded_for_stitch[effective_trim:]
                print(f"[WanLooperNative] Segment {loop_id} | "
                      f"Startup trim removed {effective_trim} frame(s) "
                      f"→ {decoded_for_stitch.shape[0]} stitch frames")

            # ── n) Overlap stitching ─────────────────────────────────────────
            # Use KJ's overlap helper directly so seam behavior matches the
            # proven workflow path as closely as possible.
            if seg_num > 0 and overlap > 0 and prev_decoded is not None and overlap_mode != "cut":
                eff_overlap = min(overlap, prev_decoded.shape[0], decoded_for_stitch.shape[0])
                print(f"[WanLooperNative] Segment {loop_id} | "
                      f"Overlap stitch ({eff_overlap}f, {overlap_mode}, "
                      f"side={overlap_side})")

                # Re-save previous segment with overlap tail removed,
                # preserving any blend frames at its head from its own
                # overlap with the segment before it.
                prev_on_disk = torch.load(segment_paths[-1], map_location="cpu",
                                          weights_only=True)
                trimmed_prev = prev_on_disk[:-eff_overlap]
                torch.save(trimmed_prev.cpu(), segment_paths[-1])
                del prev_on_disk
                print(f"[WanLooperNative] Segment {loop_id} | "
                      f"Re-saved prev → {trimmed_prev.shape[0]} frames")

                overlap_result = _KJ_ImageBatchExtend.imagesfrombatch(
                    source_images=prev_decoded,
                    overlap=eff_overlap,
                    overlap_side=overlap_side,
                    overlap_mode=overlap_mode,
                    new_images=decoded_for_stitch,
                )
                extended_images = overlap_result[2]
                n_prev_keep = prev_decoded.shape[0] - eff_overlap
                segment_frames = extended_images[n_prev_keep:]

            else:
                if seg_num > 0 and overlap_mode == "cut":
                    print(f"[WanLooperNative] Segment {loop_id} | Cut (no blend)")
                segment_frames = decoded_for_stitch

            # ── o) Save segment ──────────────────────────────────────────────
            seg_path = os.path.join(segment_dir, f"segment_{loop_id:03d}.pt")
            torch.save(segment_frames.cpu(), seg_path)
            segment_paths.append(seg_path)
            print(f"[WanLooperNative] Segment {loop_id} | "
                  f"Saved → {segment_frames.shape[0]} frames "
                  f"(decoded={decoded.shape[0]}, anchor_idx={anchor_idx}, "
                  f"stitch_input={decoded_for_stitch.shape[0]})")

            # ── p) State update ──────────────────────────────────────────────
            prev_decoded = decoded_for_stitch

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
        all_segment_prompts = "\n".join(all_segment_prompts_log)
        return (full_video, last_extracted_anchor, all_segment_prompts)


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
    "LoopConfigWan":   "Wan Loop Config SVI",
    "WanLooperNative": "Wan Looper SVI",
    "LoopConfigSVI":   "LoopConfigSVI SVI Alias",
    "SVILooperNative": "SVI Looper SVI Alias",
}
