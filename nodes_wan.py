"""
Wan Looper - Multi-loop Wan video generation with dual High/Low noise models,
WanImageMotion conditioning, and overlap stitching between segments.

Architecture: WanLoopPrep → [user-wired Clownshark samplers] → WanLoopFinish
Per-loop Wavespeed refine: WanLoopFinish.anchor_frame → WaveSpeedAIPredictor
                                                        → WanLoopPrep_{N+1}.refined_anchor
"""

import torch
import gc
import os
import sys
import types
import tempfile
import shutil
import importlib.util

import comfy.model_management
import comfy.utils
import comfy.sd
import folder_paths
import nodes as _comfy_nodes


# ---------------------------------------------------------------------------
# External class loader
# ---------------------------------------------------------------------------

# Absolute path to ComfyUI's custom_nodes directory.
_CUSTOM_NODES = r"E:\ComfyUI\ComfyUI\custom_nodes"


def _load_class(rel_path, class_name):
    """Load a class from a file that contains no relative imports.

    rel_path is relative to _CUSTOM_NODES.
    """
    full_path = os.path.join(_CUSTOM_NODES, rel_path)
    mod_name = class_name + "_mod_" + rel_path.replace(os.sep, "_").replace("/", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(mod_name, full_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)


def _ensure_package(pkg_name, pkg_dir):
    """Register pkg_dir as a Python package in sys.modules without executing __init__.py.

    This gives Python's import system a __path__ to follow when resolving relative
    imports inside modules that belong to this package.
    """
    if pkg_name not in sys.modules:
        mod = types.ModuleType(pkg_name)
        mod.__path__ = [pkg_dir]
        mod.__package__ = pkg_name
        sys.modules[pkg_name] = mod


def _load_package_class(package_root, rel_file, class_name, pkg_name=None):
    """Load a class from a module inside a package, handling relative imports.

    Registers the full package hierarchy in sys.modules (with __path__ set)
    so that Python's import system can resolve any relative imports inside
    the target module and its transitive dependencies.

    package_root: absolute path to the package's root directory
    rel_file:     path to the .py file relative to package_root (forward slashes)
    class_name:   name of the class to extract
    pkg_name:     Python identifier to use as the package name (default: basename
                  of package_root with hyphens replaced by underscores)
    """
    if pkg_name is None:
        pkg_name = os.path.basename(package_root).replace("-", "_")

    _ensure_package(pkg_name, package_root)

    # Strip .py suffix safely
    rel_no_ext = rel_file[:-3] if rel_file.endswith(".py") else rel_file
    parts = rel_no_ext.replace("\\", "/").split("/")

    current_pkg = pkg_name
    current_dir = package_root
    target_mod = None

    for i, part in enumerate(parts):
        mod_name = f"{current_pkg}.{part}"
        subpath = os.path.join(current_dir, part)

        if i < len(parts) - 1:
            # Intermediate directory: register as subpackage so relative imports
            # from the final module can traverse up through it.
            _ensure_package(mod_name, subpath)
            current_pkg = mod_name
            current_dir = subpath
        else:
            # Final entry: load the actual .py file.
            file_path = subpath + ".py"
            if mod_name not in sys.modules:
                spec = importlib.util.spec_from_file_location(mod_name, file_path)
                target_mod = importlib.util.module_from_spec(spec)
                target_mod.__package__ = current_pkg
                # Register before exec so circular imports resolve correctly.
                sys.modules[mod_name] = target_mod
                spec.loader.exec_module(target_mod)
            else:
                target_mod = sys.modules[mod_name]

    return getattr(target_mod, class_name)


# ---------------------------------------------------------------------------
# Absolute paths to each external package
# ---------------------------------------------------------------------------
_KJNODES_ROOT = os.path.join(_CUSTOM_NODES, "comfyui-kjnodes")


# ---------------------------------------------------------------------------
# Load external node classes at import time
# ---------------------------------------------------------------------------
# IAMCCS_WanImageMotion — called internally inside WanLoopPrep
try:
    IAMCCS_WanImageMotion = _load_class(
        os.path.join("IAMCCS-nodes", "iamccs_wan_svipro_motion.py"),
        "IAMCCS_WanImageMotion",
    )
    _IAMCCS_WanImageMotion = IAMCCS_WanImageMotion()
    print("[WanLooper] Loaded IAMCCS_WanImageMotion")
except Exception as e:
    print(f"[WanLooper] WARNING: Could not load IAMCCS_WanImageMotion: {e}")
    _IAMCCS_WanImageMotion = None

# GetImageRangeFromBatch — called internally inside WanLoopFinish
try:
    GetImageRangeFromBatch = _load_package_class(
        _KJNODES_ROOT, "nodes/image_nodes.py", "GetImageRangeFromBatch",
        pkg_name="comfyui_kjnodes",
    )
    _GetImageRange = GetImageRangeFromBatch()
    print("[WanLooper] Loaded GetImageRangeFromBatch")
except Exception as e:
    print(f"[WanLooper] WARNING: Could not load GetImageRangeFromBatch: {e}")
    _GetImageRange = None

# ImageBatchExtendWithOverlap — called internally inside WanLoopFinish
try:
    # image_nodes.py is already in sys.modules from the block above.
    ImageBatchExtendWithOverlap = _load_package_class(
        _KJNODES_ROOT, "nodes/image_nodes.py", "ImageBatchExtendWithOverlap",
        pkg_name="comfyui_kjnodes",
    )
    _ImageBatchExtend = ImageBatchExtendWithOverlap()
    print("[WanLooper] Loaded ImageBatchExtendWithOverlap")
except Exception as e:
    print(f"[WanLooper] WARNING: Could not load ImageBatchExtendWithOverlap: {e}")
    _ImageBatchExtend = None


# ---------------------------------------------------------------------------
# Clownshark sampler classes (located by scanning sys.modules at load time)
# ---------------------------------------------------------------------------
_RES4LYF_ROOT = os.path.join(_CUSTOM_NODES, "RES4LYF")


def _find_clownshark():
    """Scan sys.modules for the real Python module containing ClownsharKSampler_Beta."""
    import inspect as _inspect

    def _is_comfyui_node_class(obj):
        """True only for actual ComfyUI node classes — they always define INPUT_TYPES.
        Torch op wrappers and _OpNamespace objects do not."""
        return (_inspect.isclass(obj)
                and hasattr(obj, "INPUT_TYPES")
                and hasattr(obj, "FUNCTION"))

    # Try known module name patterns first (fastest path)
    candidates = [
        "RES4LYF.beta.samplers",
        "RES4LYF_beta_samplers",
        "beta.samplers",
    ]
    for key in candidates:
        mod = sys.modules.get(key)
        if (mod is not None
                and _is_comfyui_node_class(getattr(mod, "ClownsharKSampler_Beta", None))):
            print(f"[WanLooper] Found Clownshark in sys.modules['{key}']")
            return mod

    # Broad scan — only match real ComfyUI node classes
    for key, mod in list(sys.modules.items()):
        if (mod is not None
                and _is_comfyui_node_class(getattr(mod, "ClownsharKSampler_Beta", None))
                and _is_comfyui_node_class(getattr(mod, "ClownsharkChainsampler_Beta", None))):
            print(f"[WanLooper] Found Clownshark in sys.modules['{key}'] (broad scan)")
            return mod

    return None


def _load_clownshark():
    """Return (CKSampler_instance, CKChain_instance) or (None, None)."""
    mod = _find_clownshark()
    if mod is not None:
        return mod.ClownsharKSampler_Beta(), mod.ClownsharkChainsampler_Beta()

    # Last resort: use _load_package_class (handles relative imports via sys.modules
    # registration — safe because RES4LYF sub-packages are already loaded by ComfyUI).
    try:
        CKCls    = _load_package_class(_RES4LYF_ROOT, "beta/samplers.py",
                                        "ClownsharKSampler_Beta",    pkg_name="RES4LYF")
        CKChCls  = _load_package_class(_RES4LYF_ROOT, "beta/samplers.py",
                                        "ClownsharkChainsampler_Beta", pkg_name="RES4LYF")
        return CKCls(), CKChCls()
    except Exception as e:
        print(f"[WanLooper] WARNING: _load_package_class also failed: {e}")
        return None, None


try:
    _CKSampler, _CKChain = _load_clownshark()
    if _CKSampler is not None:
        print("[WanLooper] Loaded ClownsharKSampler_Beta and ClownsharkChainsampler_Beta")
    else:
        print("[WanLooper] WARNING: Clownshark samplers not found — "
              "WanLooperCK will raise at execution time if used")
except Exception as _e:
    print(f"[WanLooper] WARNING: Could not load Clownshark samplers: {_e}")
    _CKSampler = None
    _CKChain   = None


# ---------------------------------------------------------------------------
# NODE 1: WanLoopConfig
# ---------------------------------------------------------------------------

class WanLoopConfig:
    """Per-loop configuration for WanLooper."""

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = sorted(["None"] + folder_paths.get_filename_list("loras"))
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "frames": ("INT", {"default": 49, "min": 1, "max": 300}),
                "lora_high": (lora_list, {"default": "None"}),
                "lora_high_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05}),
                "lora_low": (lora_list, {"default": "None"}),
                "lora_low_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05}),
                "anchor_frame_offset": ("INT", {
                    "default": -5,
                    "min": -100,
                    "max": -1,
                    "tooltip": "Frames from end of segment to extract anchor. e.g. -5 = 5th from last frame",
                }),
            },
            "optional": {
                "anchor_override": ("IMAGE", {
                    "tooltip": "If connected, skips Wavespeed refine for this loop and uses this image as anchor_samples directly",
                }),
            },
        }

    RETURN_TYPES = ("WAN_LOOP",)
    RETURN_NAMES = ("loop_config",)
    FUNCTION = "build"
    CATEGORY = "video/wan_looper"

    def build(self, prompt, frames, lora_high, lora_high_strength, lora_low, lora_low_strength,
              anchor_frame_offset, anchor_override=None):
        return ({
            "prompt": prompt,
            "frames": frames,
            "lora_high": None if lora_high == "None" else lora_high,
            "lora_high_strength": lora_high_strength,
            "lora_low": None if lora_low == "None" else lora_low,
            "lora_low_strength": lora_low_strength,
            "anchor_frame_offset": anchor_frame_offset,
            "anchor_override": anchor_override,
        },)


# ---------------------------------------------------------------------------
# NODE 2: WanLoopPrep
# ---------------------------------------------------------------------------

class WanLoopPrep:
    """
    Per-loop preparation node.

    Handles LoRA application, anchor/prev-sample setup, prompt encoding,
    and WanImageMotion conditioning. Outputs conditioned latents and
    LoRA-patched models for the user-wired Clownshark sampler pair.

    Graph wiring:
      WanLoopConfig  ──→ WanLoopPrep
      WanLoopFinish_{N-1}.next_loop_state ──→ WanLoopPrep.prev_loop_state  (loop N > 0)
      WaveSpeedAIPredictor_{N-1}          ──→ WanLoopPrep.refined_anchor   (loop N > 0)
      WanLoopPrep.model_high_out ──→ ClownsharKSampler_Beta
      WanLoopPrep.model_low_out  ──→ ClownsharkChainsampler_Beta
      WanLoopPrep.positive_out / negative_out / latent_out ──→ ClownsharKSampler_Beta
      WanLoopPrep.loop_state ──→ WanLoopFinish
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_high":  ("MODEL",),
                "model_low":   ("MODEL",),
                "vae":         ("VAE",),
                "clip":        ("CLIP",),
                "start_image": ("IMAGE",),
                "positive":    ("CONDITIONING",),
                "negative":    ("CONDITIONING",),
                "width":  ("INT", {"default": 672, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 896, "min": 16, "max": 4096, "step": 16}),
                "loop_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9,
                    "tooltip": "0-based index of this loop segment",
                }),
                "loop_config": ("WAN_LOOP",),
            },
            "optional": {
                "prev_loop_state": ("WAN_LOOP_STATE",),
                "refined_anchor": ("IMAGE", {
                    "tooltip": "Wavespeed-refined anchor from WanLoopFinish of the previous loop. "
                               "Required for loop_index > 0 unless anchor_override is set in loop_config.",
                }),
            },
        }

    RETURN_TYPES  = ("CONDITIONING", "CONDITIONING", "LATENT", "MODEL", "MODEL", "WAN_LOOP_STATE")
    RETURN_NAMES  = ("positive_out", "negative_out", "latent_out",
                     "model_high_out", "model_low_out", "loop_state")
    FUNCTION      = "prepare"
    CATEGORY      = "video/wan_looper"

    def prepare(self, model_high, model_low, vae, clip, start_image, positive, negative,
                width, height, loop_index, loop_config,
                prev_loop_state=None, refined_anchor=None):

        if _IAMCCS_WanImageMotion is None:
            raise RuntimeError("[WanLoopPrep] IAMCCS_WanImageMotion not loaded")

        # ── Extract loop_config fields ───────────────────────────────────────
        prompt              = loop_config.get("prompt", "")
        frames              = loop_config.get("frames", 49)
        lora_high_name      = loop_config.get("lora_high")
        lora_high_strength  = loop_config.get("lora_high_strength", 0.8)
        lora_low_name       = loop_config.get("lora_low")
        lora_low_strength   = loop_config.get("lora_low_strength", 0.8)
        anchor_frame_offset = loop_config.get("anchor_frame_offset", -5)
        anchor_override     = loop_config.get("anchor_override")
        loop_id = loop_index + 1

        print(f"[WanLoopPrep] Loop {loop_id}: starting prepare "
              f"({frames} frames, loop_index={loop_index})")

        # ── LoRA application (model-only; clip strength = 0.0) ───────────────
        high_model = model_high
        low_model  = model_low

        if lora_high_name:
            lora_path = folder_paths.get_full_path("loras", lora_high_name)
            lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
            high_model, _ = comfy.sd.load_lora_for_models(
                high_model, clip, lora_data, lora_high_strength, 0.0)
            del lora_data
            print(f"[WanLoopPrep] Loop {loop_id}: applied LoRA {lora_high_name} "
                  f"to high model (strength={lora_high_strength})")

        if lora_low_name:
            lora_path = folder_paths.get_full_path("loras", lora_low_name)
            lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
            low_model, _ = comfy.sd.load_lora_for_models(
                low_model, clip, lora_data, lora_low_strength, 0.0)
            del lora_data
            print(f"[WanLoopPrep] Loop {loop_id}: applied LoRA {lora_low_name} "
                  f"to low model (strength={lora_low_strength})")

        # ── Anchor samples and loop state initialisation ─────────────────────
        if loop_index == 0:
            # Loop 0: resize + VAE-encode start_image as the initial anchor.
            img_bchw = start_image.permute(0, 3, 1, 2)
            img_bchw = comfy.utils.common_upscale(img_bchw, width, height, "lanczos", "disabled")
            anchor_image = img_bchw.permute(0, 2, 3, 1)
            t = vae.encode(anchor_image)
            anchor_samples   = {"samples": t}
            prev_samples     = None
            prev_decoded     = None
            segment_paths    = []
            segment_dir      = tempfile.mkdtemp(prefix="wan_looper_")
            used_prompts_log = []
            print(f"[WanLoopPrep] Loop {loop_id}: encoded start_image. "
                  f"segment_dir={segment_dir}")
        else:
            # Loop N > 0: anchor comes from refined_anchor (Wavespeed output)
            # or anchor_override; prev state carries segment/latent history.
            if prev_loop_state is None:
                raise RuntimeError(
                    f"[WanLoopPrep] Loop {loop_id}: prev_loop_state is required "
                    f"for loop_index > 0")
            prev_samples     = prev_loop_state["prev_samples"]
            prev_decoded     = prev_loop_state["prev_decoded"]
            segment_paths    = prev_loop_state["segment_paths"]
            segment_dir      = prev_loop_state["segment_dir"]
            used_prompts_log = list(prev_loop_state.get("used_prompts_log", []))

            if anchor_override is not None:
                # anchor_override from loop_config takes priority.
                ao_bchw = anchor_override.permute(0, 3, 1, 2)
                ao_bchw = comfy.utils.common_upscale(
                    ao_bchw, width, height, "lanczos", "disabled")
                anchor_image = ao_bchw.permute(0, 2, 3, 1)
                t = vae.encode(anchor_image)
                anchor_samples = {"samples": t}
                print(f"[WanLoopPrep] Loop {loop_id}: using anchor_override from loop_config")
            elif refined_anchor is not None:
                # Wavespeed-refined IMAGE from previous WanLoopFinish.
                t = vae.encode(refined_anchor)
                anchor_samples = {"samples": t}
                print(f"[WanLoopPrep] Loop {loop_id}: VAE encoded refined_anchor from Wavespeed")
            else:
                raise RuntimeError(
                    f"[WanLoopPrep] Loop {loop_id}: loop_index > 0 requires either "
                    f"refined_anchor input or anchor_override in loop_config")

        # ── Prompt encoding ──────────────────────────────────────────────────
        if prompt.strip():
            tokens = clip.tokenize(prompt)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            c_pos = [[cond, {"pooled_output": pooled}]]
            used_prompts_log.append(f"Loop {loop_id}: {prompt}")
            print(f"[WanLoopPrep] Loop {loop_id}: encoded loop prompt")
        else:
            c_pos = positive
            used_prompts_log.append(f"Loop {loop_id}: (using default positive)")
            print(f"[WanLoopPrep] Loop {loop_id}: using default positive conditioning")

        # ── WanImageMotion conditioning ──────────────────────────────────────
        motion_val = 1.15 if loop_index == 0 else 1.3
        use_prev   = loop_index > 0

        print(f"[WanLoopPrep] Loop {loop_id}: calling IAMCCS_WanImageMotion.apply() "
              f"(motion={motion_val}, use_prev={use_prev})")
        positive_out, negative_out, latent_out = _IAMCCS_WanImageMotion.apply(
            positive              = c_pos,
            negative              = negative,
            anchor_samples        = anchor_samples,
            prev_samples          = prev_samples,
            length                = frames,
            motion_latent_count   = 1,
            motion                = motion_val,
            motion_mode           = "motion_only (prev_samples)",
            add_reference_latents = False,
            latent_precision      = "fp32",
            vram_profile          = "normal",
            include_padding_in_motion = loop_index == 0,
            safety_preset         = "safe" if loop_index == 0 else "safer",
            lock_start_slots      = 1,
            diagnostic_log        = False,
            use_prev_samples      = use_prev,
            latent_refresh        = 0.0,
            delta_max             = 0.0,
        )
        print(f"[WanLoopPrep] Loop {loop_id}: WanImageMotion done")

        # ── Build loop_state ─────────────────────────────────────────────────
        loop_state = {
            "model_high_patched":  high_model,
            "model_low_patched":   low_model,
            "model_high_original": model_high,
            "model_low_original":  model_low,
            "lora_high_name":      lora_high_name,
            "lora_low_name":       lora_low_name,
            "loop_index":          loop_index,
            "frames":              frames,
            "anchor_frame_offset": anchor_frame_offset,
            "anchor_override":     anchor_override,
            "width":               width,
            "height":              height,
            "vae":                 vae,
            "anchor_samples":      anchor_samples,
            "prev_samples":        prev_samples,
            "prev_decoded":        prev_decoded,
            "segment_paths":       segment_paths,
            "segment_dir":         segment_dir,
            "used_prompts_log":    used_prompts_log,
        }

        print(f"[WanLoopPrep] Loop {loop_id}: prepare complete")
        return (positive_out, negative_out, latent_out, high_model, low_model, loop_state)


# ---------------------------------------------------------------------------
# NODE 3: WanLoopFinish
# ---------------------------------------------------------------------------

class WanLoopFinish:
    """
    Per-loop finish node.

    Handles VAE decode, overlap stitching, anchor frame extraction, segment
    disk save, LoRA cleanup, and (on the final loop) full video assembly.

    Graph wiring:
      ClownsharkChainsampler_Beta ──→ WanLoopFinish.sampled_latent
      WanLoopPrep.loop_state      ──→ WanLoopFinish.loop_state
      WanLoopFinish.anchor_frame  ──→ WaveSpeedAIPredictor
                                      → WanLoopPrep_{N+1}.refined_anchor
      WanLoopFinish.next_loop_state ──→ WanLoopPrep_{N+1}.prev_loop_state
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampled_latent": ("LATENT",),
                "loop_state":     ("WAN_LOOP_STATE",),
                "overlap": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 50,
                    "tooltip": "Frame overlap for ImageBatchExtendWithOverlap",
                }),
                "is_final_loop": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Set True on the last loop to trigger full video assembly",
                }),
            },
        }

    RETURN_TYPES  = ("LATENT", "IMAGE", "IMAGE", "WAN_LOOP_STATE", "IMAGE", "STRING")
    RETURN_NAMES  = ("prev_samples", "anchor_frame", "segment_frames",
                     "next_loop_state", "full_video_so_far", "used_prompts")
    FUNCTION      = "finish"
    CATEGORY      = "video/wan_looper"

    def finish(self, sampled_latent, loop_state, overlap, is_final_loop):

        if _GetImageRange is None:
            raise RuntimeError("[WanLoopFinish] GetImageRangeFromBatch not loaded")
        if _ImageBatchExtend is None:
            raise RuntimeError("[WanLoopFinish] ImageBatchExtendWithOverlap not loaded")

        # ── Unpack loop_state ────────────────────────────────────────────────
        vae                  = loop_state["vae"]
        width                = loop_state["width"]
        height               = loop_state["height"]
        anchor_frame_offset  = loop_state["anchor_frame_offset"]
        model_high_patched   = loop_state["model_high_patched"]
        model_low_patched    = loop_state["model_low_patched"]
        model_high_original  = loop_state["model_high_original"]
        model_low_original   = loop_state["model_low_original"]
        lora_high_name       = loop_state["lora_high_name"]
        lora_low_name        = loop_state["lora_low_name"]
        loop_index           = loop_state["loop_index"]
        prev_decoded         = loop_state["prev_decoded"]
        segment_paths        = loop_state["segment_paths"]
        segment_dir          = loop_state["segment_dir"]
        used_prompts_log     = loop_state.get("used_prompts_log", [])
        loop_id = loop_index + 1

        print(f"[WanLoopFinish] Loop {loop_id}: finishing")

        # ── 1. VAE decode ────────────────────────────────────────────────────
        print(f"[WanLoopFinish] Loop {loop_id}: VAE decoding")
        decoded = vae.decode(sampled_latent["samples"])
        while decoded.ndim > 4:
            decoded = decoded.squeeze(0)
        if decoded.shape[1] != height or decoded.shape[2] != width:
            d_bchw = decoded.permute(0, 3, 1, 2)
            d_bchw = comfy.utils.common_upscale(d_bchw, width, height, "lanczos", "disabled")
            decoded = d_bchw.permute(0, 2, 3, 1)

        # ── 2. Overlap stitching ─────────────────────────────────────────────
        if loop_index == 0 or overlap == 0 or prev_decoded is None:
            segment_frames = decoded
            print(f"[WanLoopFinish] Loop {loop_id}: no overlap "
                  f"(loop_index={loop_index}, overlap={overlap})")
        else:
            print(f"[WanLoopFinish] Loop {loop_id}: "
                  f"applying ImageBatchExtendWithOverlap (overlap={overlap})")
            overlap_result = _ImageBatchExtend.imagesfrombatch(
                source_images = prev_decoded,
                new_images    = decoded,
                overlap       = overlap,
                overlap_side  = "source",
                overlap_mode  = "ease_in_out",
            )
            segment_frames = overlap_result[0]

        # ── 3. Anchor frame extraction ────────────────────────────────────────
        actual_frames    = decoded.shape[0]
        anchor_frame_idx = max(0, min(actual_frames - 1, actual_frames + anchor_frame_offset))
        print(f"[WanLoopFinish] Loop {loop_id}: "
              f"extracting anchor at frame {anchor_frame_idx}")
        anchor_result = _GetImageRange.imagesfrombatch(
            images      = decoded,
            start_index = anchor_frame_idx,
            num_frames  = 1,
        )
        # [1,H,W,C] — user connects this to WaveSpeedAIPredictor in the graph.
        # WaveSpeedAIPredictor output connects to WanLoopPrep_{N+1}.refined_anchor.
        anchor_frame = anchor_result[0]

        # ── 4. Save segment to disk ──────────────────────────────────────────
        seg_path = os.path.join(segment_dir, f"segment_{loop_id:03d}.pt")
        torch.save(segment_frames.cpu(), seg_path)
        new_segment_paths = list(segment_paths) + [seg_path]
        print(f"[WanLoopFinish] Loop {loop_id}: saved segment to {seg_path}")

        # ── 5. Build next_loop_state ─────────────────────────────────────────
        # anchor_samples is left None — WanLoopPrep_{N+1} will build it from
        # the refined_anchor (WaveSpeedAIPredictor output) or anchor_override.
        next_loop_state = {
            "anchor_samples":      None,
            "prev_samples":        sampled_latent,
            "prev_decoded":        decoded,
            "segment_paths":       new_segment_paths,
            "segment_dir":         segment_dir,
            "used_prompts_log":    used_prompts_log,
            # Pass through original model refs for safety; WanLoopPrep reapplies LoRAs fresh.
            "model_high_patched":  model_high_original,
            "model_low_patched":   model_low_original,
            "model_high_original": model_high_original,
            "model_low_original":  model_low_original,
            "lora_high_name":      None,
            "lora_low_name":       None,
            "loop_index":          loop_index,
            "frames":              loop_state.get("frames", 49),
            "anchor_frame_offset": anchor_frame_offset,
            "anchor_override":     None,
            "width":               width,
            "height":              height,
            "vae":                 vae,
        }

        # ── 6. LoRA cleanup ──────────────────────────────────────────────────
        if lora_high_name and model_high_patched is not model_high_original:
            del model_high_patched
            print(f"[WanLoopFinish] Loop {loop_id}: cleaned up patched high model")
        if lora_low_name and model_low_patched is not model_low_original:
            del model_low_patched
            print(f"[WanLoopFinish] Loop {loop_id}: cleaned up patched low model")
        comfy.model_management.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # ── 7. Full video assembly (final loop only) ─────────────────────────
        used_prompts_str = "\n".join(used_prompts_log)

        if is_final_loop:
            print("[WanLoopFinish] Final loop — assembling full video from all segments")
            full_video = None
            for sp in new_segment_paths:
                seg = torch.load(sp, map_location="cpu", weights_only=True)
                full_video = seg if full_video is None else torch.cat([full_video, seg], dim=0)
                del seg
                gc.collect()
            if full_video is None:
                full_video = torch.zeros((1, height, width, 3))
            try:
                shutil.rmtree(segment_dir)
            except Exception:
                pass
            print(f"[WanLoopFinish] Done. Total frames: {full_video.shape[0]}")
        else:
            full_video = segment_frames  # preview: just this segment

        print(f"[WanLoopFinish] Loop {loop_id}: finish complete")
        return (sampled_latent, anchor_frame, segment_frames,
                next_loop_state, full_video, used_prompts_str)


# ---------------------------------------------------------------------------
# NODE 4: WanLooperCK  (monolithic — all loops handled internally)
# ---------------------------------------------------------------------------

class WanLooperCK:
    """
    Monolithic Wan looper using ClownsharKSampler_Beta (high model, first N steps)
    chained with ClownsharkChainsampler_Beta (low model, remaining steps).

    All segment management — IAMCCS conditioning, anchor handoff, VAE encode/decode,
    overlap stitching, disk save, and final assembly — happens inside a single Python
    loop.  No per-segment graph duplication needed.

    Per-loop prompts: supply one prompt per line in loop_prompts.  Line N is used for
    loop N (0-based).  Blank lines or missing lines fall back to the global positive
    conditioning.  Fewer lines than num_loops → last line repeats.

    Note: bongmath is always forced True on the chain sampler because
    ClownsharkChainsampler_Beta reads sigmas from state_info, which requires
    bongmath state to be populated by the first sampler.  The widget controls
    only the first sampler.
    """

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = sorted(["None"] + folder_paths.get_filename_list("loras"))
        return {
            "required": {
                "model_high":          ("MODEL",),
                "model_low":           ("MODEL",),
                "vae":                 ("VAE",),
                "clip":                ("CLIP",),
                "start_image":         ("IMAGE",),
                "positive":            ("CONDITIONING",),
                "negative":            ("CONDITIONING",),
                "width":  ("INT", {"default": 832,  "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480,  "min": 16, "max": 4096, "step": 16}),
                "frames": ("INT", {"default": 49,   "min": 1,  "max": 300,
                                   "tooltip": "Frames per segment"}),
                "num_loops": ("INT", {"default": 1, "min": 1,  "max": 10,
                                      "tooltip": "Number of segments to generate"}),
                "overlap": ("INT", {"default": 0, "min": 0, "max": 50,
                                    "tooltip": "Frame overlap between segments"}),
                "anchor_frame_offset": ("INT", {
                    "default": -5, "min": -100, "max": -1,
                    "tooltip": "Frames from end of segment for anchor extraction. "
                               "-5 = 5th-from-last frame",
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "tooltip": "Base seed — incremented by 1 per loop"}),
                # ── Clownshark sampler params ──────────────────────────────────
                "eta": ("FLOAT", {
                    "default": 0.5, "min": -100.0, "max": 100.0, "step": 0.01,
                    "round": False,
                    "tooltip": "Noise amount added/removed each step (both samplers)",
                }),
                "sampler_name": ("STRING", {
                    "default": "linear/euler",
                    "tooltip": "RES4LYF sampler name (e.g. 'linear/euler', 'res_2m'). "
                               "Used for both samplers.",
                }),
                "scheduler": ("STRING", {
                    "default": "beta57",
                    "tooltip": "Sigma schedule for the first (high) sampler only.",
                }),
                "steps": ("INT", {"default": 8,  "min": 1, "max": 200,
                                  "tooltip": "Total denoising steps"}),
                "steps_to_run": ("INT", {
                    "default": 4, "min": 1, "max": 200,
                    "tooltip": "Steps run by the high model (first sampler). "
                               "Remaining steps go to the low model (chain sampler).",
                }),
                "cfg": ("FLOAT", {
                    "default": 1.0, "min": -10000.0, "max": 10000.0,
                    "step": 0.01, "round": False,
                    "tooltip": "CFG scale (both samplers)",
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Denoise strength for the first sampler",
                }),
                "bongmath": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable BONGMATH state tracking on the first sampler. "
                               "Always forced True on the chain sampler.",
                }),
                "sampler_mode_high": (["standard", "unsample", "resample"], {
                    "default": "standard",
                    "tooltip": "sampler_mode for the first (high) sampler",
                }),
                "sampler_mode_low": (["resample", "standard", "unsample"], {
                    "default": "resample",
                    "tooltip": "sampler_mode for the chain (low) sampler. "
                               "'resample' follows a standard high pass (established working pattern).",
                }),
            },
            "optional": {
                "loop_prompts": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "One prompt per line, one per loop segment. "
                               "Blank lines use global positive. "
                               "Fewer lines than num_loops: last line repeats.",
                }),
                "lora_high": (lora_list, {"default": "None",
                                           "tooltip": "LoRA applied to model_high each loop"}),
                "lora_high_strength": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
                "lora_low":  (lora_list, {"default": "None",
                                           "tooltip": "LoRA applied to model_low each loop"}),
                "lora_low_strength": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
            },
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES  = ("full_video", "last_anchor", "used_prompts")
    FUNCTION      = "generate"
    CATEGORY      = "video/wan_looper"

    def generate(self,
                 model_high, model_low, vae, clip, start_image, positive, negative,
                 width, height, frames, num_loops, overlap, anchor_frame_offset, seed,
                 eta, sampler_name, scheduler, steps, steps_to_run, cfg, denoise, bongmath,
                 sampler_mode_high="standard", sampler_mode_low="standard",
                 loop_prompts="",
                 lora_high="None", lora_high_strength=0.8,
                 lora_low="None",  lora_low_strength=0.8):

        # Attempt lazy load in case RES4LYF finished loading after our module did
        global _CKSampler, _CKChain
        if _CKSampler is None or _CKChain is None:
            _CKSampler, _CKChain = _load_clownshark()
        if _CKSampler is None or _CKChain is None:
            raise RuntimeError(
                "[WanLooperCK] ClownsharK samplers not loaded — "
                "is RES4LYF installed? Check startup log for the load warning.")
        if _IAMCCS_WanImageMotion is None:
            raise RuntimeError("[WanLooperCK] IAMCCS_WanImageMotion not loaded")
        if _GetImageRange is None:
            raise RuntimeError("[WanLooperCK] GetImageRangeFromBatch not loaded")

        # ── Parse per-loop prompts ───────────────────────────────────────────
        raw_lines = [p.strip() for p in loop_prompts.strip().split("\n")] if loop_prompts.strip() else []

        def _get_prompt(loop_idx):
            if not raw_lines:
                return ""
            if loop_idx < len(raw_lines):
                return raw_lines[loop_idx]
            return raw_lines[-1]  # repeat last line

        lora_high_name = None if lora_high == "None" else lora_high
        lora_low_name  = None if lora_low  == "None" else lora_low

        # ── Per-loop state ───────────────────────────────────────────────────
        segment_dir    = tempfile.mkdtemp(prefix="wan_looper_ck_")
        segment_paths  = []
        used_prompts_log = []

        anchor_samples   = None   # LATENT dict
        prev_samples     = None   # LATENT dict (raw chain output)
        prev_decoded     = None   # IMAGE tensor (for overlap)
        last_anchor_frame = None  # IMAGE tensor [1,H,W,C]

        for loop_idx in range(num_loops):
            loop_id   = loop_idx + 1
            loop_seed = seed + loop_idx
            print(f"\n[WanLooperCK] ── Loop {loop_id}/{num_loops} ──────────────────────────")

            # ── 1. LoRA application ──────────────────────────────────────────
            high_model = model_high
            low_model  = model_low

            if lora_high_name:
                lora_path = folder_paths.get_full_path("loras", lora_high_name)
                lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
                high_model, _ = comfy.sd.load_lora_for_models(
                    high_model, clip, lora_data, lora_high_strength, 0.0)
                del lora_data
                print(f"[WanLooperCK] Loop {loop_id}: applied lora_high ({lora_high_name}, str={lora_high_strength})")

            if lora_low_name:
                lora_path = folder_paths.get_full_path("loras", lora_low_name)
                lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
                low_model, _ = comfy.sd.load_lora_for_models(
                    low_model, clip, lora_data, lora_low_strength, 0.0)
                del lora_data
                print(f"[WanLooperCK] Loop {loop_id}: applied lora_low ({lora_low_name}, str={lora_low_strength})")

            # ── 2. Anchor setup ──────────────────────────────────────────────
            if loop_idx == 0:
                img_bchw = start_image.permute(0, 3, 1, 2)
                img_bchw = comfy.utils.common_upscale(img_bchw, width, height, "lanczos", "disabled")
                anchor_image  = img_bchw.permute(0, 2, 3, 1)
                t             = vae.encode(anchor_image)
                anchor_samples = {"samples": t}
                print(f"[WanLooperCK] Loop {loop_id}: encoded start_image as anchor")
            # else: anchor_samples set at end of previous iteration

            # ── 3. Prompt encoding ───────────────────────────────────────────
            loop_prompt = _get_prompt(loop_idx)
            if loop_prompt:
                tokens    = clip.tokenize(loop_prompt)
                cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
                c_pos     = [[cond, {"pooled_output": pooled}]]
                used_prompts_log.append(f"Loop {loop_id}: {loop_prompt}")
                print(f"[WanLooperCK] Loop {loop_id}: encoded loop prompt")
            else:
                c_pos = positive
                used_prompts_log.append(f"Loop {loop_id}: (global positive)")
                print(f"[WanLooperCK] Loop {loop_id}: using global positive conditioning")

            # ── 4. IAMCCS WanImageMotion conditioning ────────────────────────
            motion_val = 1.15 if loop_idx == 0 else 1.3
            use_prev   = loop_idx > 0
            print(f"[WanLooperCK] Loop {loop_id}: WanImageMotion "
                  f"(motion={motion_val}, use_prev={use_prev})")
            pos_cond, neg_cond, latent_dict = _IAMCCS_WanImageMotion.apply(
                positive              = c_pos,
                negative              = negative,
                anchor_samples        = anchor_samples,
                prev_samples          = prev_samples,
                length                = frames,
                motion_latent_count   = 1,
                motion                = motion_val,
                motion_mode           = "motion_only (prev_samples)",
                add_reference_latents = False,
                latent_precision      = "fp32",
                vram_profile          = "normal",
                include_padding_in_motion = (loop_idx == 0),
                safety_preset         = "safe" if loop_idx == 0 else "safer",
                lock_start_slots      = 1,
                diagnostic_log        = False,
                use_prev_samples      = use_prev,
                latent_refresh        = 0.0,
                delta_max             = 0.0,
            )

            # ── 5. First sampler — high model ────────────────────────────────
            effective_steps_to_run = min(steps_to_run, steps - 1)  # guard: must leave at least 1 for chain
            chain_steps = steps - effective_steps_to_run
            print(f"[WanLooperCK] Loop {loop_id}: high sampler "
                  f"({effective_steps_to_run}/{steps} steps, seed={loop_seed})")
            out_high, _, _ = _CKSampler.main(
                model        = high_model,
                positive     = pos_cond,
                negative     = neg_cond,
                latent_image = latent_dict,
                eta          = eta,
                sampler_name = sampler_name,
                scheduler    = scheduler,
                steps        = steps,
                steps_to_run = effective_steps_to_run,
                cfg          = cfg,
                seed         = loop_seed,
                sampler_mode = sampler_mode_high,
                bongmath     = bongmath,
                denoise      = denoise,
            )

            # ── 6. Chain sampler — low model ─────────────────────────────────
            print(f"[WanLooperCK] Loop {loop_id}: chain sampler "
                  f"(~{chain_steps} steps remaining, mode={sampler_mode_low}, seed={loop_seed})")
            out_low, out_low_denoised, _ = _CKChain.main(
                model        = low_model,
                positive     = pos_cond,
                negative     = neg_cond,
                latent_image = out_high,      # carries state_info['sigmas'] from first sampler
                eta          = eta,
                sampler_name = sampler_name,
                steps_to_run = -1,            # run all remaining sigmas from state_info
                cfg          = cfg,
                seed         = loop_seed,
                sampler_mode = sampler_mode_low,
                bongmath     = True,          # always True: chain reads state_info['sigmas']
            )

            # ── 7. VAE decode — use denoised output (slot 1), not raw output ─
            # The denoised latent is the clean x0 prediction; raw output carries
            # leftover noise that produces corrupted frames when decoded directly.
            print(f"[WanLooperCK] Loop {loop_id}: VAE decode (denoised)")
            decode_latent = out_low_denoised if out_low_denoised is not None else out_low
            decoded = vae.decode(decode_latent["samples"])
            while decoded.ndim > 4:
                decoded = decoded.squeeze(0)
            if decoded.shape[1] != height or decoded.shape[2] != width:
                d_bchw  = decoded.permute(0, 3, 1, 2)
                d_bchw  = comfy.utils.common_upscale(d_bchw, width, height, "lanczos", "disabled")
                decoded = d_bchw.permute(0, 2, 3, 1)

            # ── 8. Overlap stitching ─────────────────────────────────────────
            if loop_idx == 0 or overlap == 0 or prev_decoded is None:
                segment_frames = decoded
                print(f"[WanLooperCK] Loop {loop_id}: no overlap")
            else:
                print(f"[WanLooperCK] Loop {loop_id}: overlap stitching ({overlap} frames)")
                overlap_result = _ImageBatchExtend.imagesfrombatch(
                    source_images = prev_decoded,
                    new_images    = decoded,
                    overlap       = overlap,
                    overlap_side  = "source",
                    overlap_mode  = "ease_in_out",
                )
                segment_frames = overlap_result[0]

            # ── 9. Save segment ──────────────────────────────────────────────
            seg_path = os.path.join(segment_dir, f"segment_{loop_id:03d}.pt")
            torch.save(segment_frames.cpu(), seg_path)
            segment_paths.append(seg_path)
            print(f"[WanLooperCK] Loop {loop_id}: saved segment → {seg_path} "
                  f"({segment_frames.shape[0]} frames)")

            # ── 10. Extract anchor for next loop ─────────────────────────────
            actual_frames     = decoded.shape[0]
            anchor_frame_idx  = max(0, min(actual_frames - 1, actual_frames + anchor_frame_offset))
            anchor_result     = _GetImageRange.imagesfrombatch(
                images      = decoded,
                start_index = anchor_frame_idx,
                num_frames  = 1,
            )
            last_anchor_frame = anchor_result[0]  # [1, H, W, C]
            print(f"[WanLooperCK] Loop {loop_id}: anchor extracted at frame {anchor_frame_idx}")

            # ── 11. Encode anchor + carry state for next loop ────────────────
            t              = vae.encode(last_anchor_frame)
            anchor_samples = {"samples": t}
            prev_samples   = out_low   # raw output latent → WanImageMotion next loop
            prev_decoded   = decoded   # for overlap stitching next loop

            # ── 12. LoRA cleanup ─────────────────────────────────────────────
            if lora_high_name and high_model is not model_high:
                del high_model
            if lora_low_name and low_model is not model_low:
                del low_model
            comfy.model_management.soft_empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            print(f"[WanLooperCK] Loop {loop_id}/{num_loops}: complete")

        # ── Assemble full video ──────────────────────────────────────────────
        print(f"\n[WanLooperCK] Assembling {len(segment_paths)} segment(s) into full video")
        full_video = None
        for sp in segment_paths:
            seg = torch.load(sp, map_location="cpu", weights_only=True)
            full_video = seg if full_video is None else torch.cat([full_video, seg], dim=0)
            del seg
            gc.collect()
        try:
            shutil.rmtree(segment_dir)
        except Exception:
            pass

        if full_video is None:
            full_video = torch.zeros((1, height, width, 3))
        if last_anchor_frame is None:
            last_anchor_frame = torch.zeros((1, height, width, 3))

        used_prompts_str = "\n".join(used_prompts_log)
        print(f"[WanLooperCK] Done. Total frames: {full_video.shape[0]}")
        return (full_video, last_anchor_frame, used_prompts_str)


# ---------------------------------------------------------------------------
# NODE 5: WanLooperKSA  (monolithic — KSamplerAdvanced two-pass high/low)
# ---------------------------------------------------------------------------

class WanLooperKSA:
    """
    Monolithic Wan looper using ComfyUI's built-in KSamplerAdvanced with a
    two-pass high/low noise model split.

    Uses the same proven conditioning approach as I2VLooperHL:
    CLIPVisionEncode + WanImageToVideo for each segment.
    The anchor frame (at anchor_frame_offset from end) becomes each loop's start image.

    Per-segment models: wire LoraLoaderModelOnly output to model_high_N / model_low_N.
    Falls back to global model_high / model_low if a segment input is not connected.

    Pass 1 (high model): add_noise=enable (WanImageToVideo latent is zeros — sampler adds initial noise), steps 0→split_step, return_with_leftover_noise=enable
    Pass 2 (low model):  add_noise=disable, steps split_step→1000, return_with_leftover_noise=disable
    """

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers as _cs
        sampler_list   = _cs.KSampler.SAMPLERS
        scheduler_list = _cs.KSampler.SCHEDULERS
        return {
            "required": {
                "model_high":   ("MODEL",),
                "model_low":    ("MODEL",),
                "vae":          ("VAE",),
                "clip":         ("CLIP",),
                "clip_vision":  ("CLIP_VISION",),
                "start_image":  ("IMAGE",),
                "positive":     ("CONDITIONING",),
                "negative":     ("CONDITIONING",),
                "width":  ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "frames": ("INT", {"default": 49,  "min": 1,  "max": 300,
                                   "tooltip": "Frames per segment"}),
                "num_loops": ("INT", {"default": 1, "min": 1, "max": 10,
                                      "tooltip": "Number of segments to generate"}),
                "overlap": ("INT", {"default": 0, "min": 0, "max": 50,
                                    "tooltip": "Frame overlap between segments"}),
                "anchor_frame_offset": ("INT", {
                    "default": -5, "min": -100, "max": -1,
                    "tooltip": "Frames from end of segment for anchor extraction. "
                               "-5 = 5th-from-last frame. This frame starts the next segment.",
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "tooltip": "Base seed — incremented by 1 per loop"}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 200,
                                  "tooltip": "Total denoising steps"}),
                "split_step": ("INT", {"default": 4, "min": 1, "max": 199,
                                       "tooltip": "Step at which to hand off from high to low model. "
                                                  "Must be < steps."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0,
                                  "step": 0.01, "round": False}),
                "sampler_name": (sampler_list,),
                "scheduler":    (scheduler_list,),
            },
            "optional": {
                "loop_prompts": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "One prompt per line, one per loop segment. "
                               "Blank lines use global positive. "
                               "Fewer lines than num_loops: last line repeats.",
                }),
                # Per-segment models — connect LoraLoaderModelOnly output here.
                # Falls back to global model_high / model_low if not connected.
                "model_high_1":  ("MODEL",), "model_low_1":  ("MODEL",),
                "model_high_2":  ("MODEL",), "model_low_2":  ("MODEL",),
                "model_high_3":  ("MODEL",), "model_low_3":  ("MODEL",),
                "model_high_4":  ("MODEL",), "model_low_4":  ("MODEL",),
                "model_high_5":  ("MODEL",), "model_low_5":  ("MODEL",),
                "model_high_6":  ("MODEL",), "model_low_6":  ("MODEL",),
                "model_high_7":  ("MODEL",), "model_low_7":  ("MODEL",),
                "model_high_8":  ("MODEL",), "model_low_8":  ("MODEL",),
                "model_high_9":  ("MODEL",), "model_low_9":  ("MODEL",),
                "model_high_10": ("MODEL",), "model_low_10": ("MODEL",),
            },
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES  = ("full_video", "last_anchor", "used_prompts")
    FUNCTION      = "generate"
    CATEGORY      = "video/wan_looper"

    def generate(self,
                 model_high, model_low, vae, clip, clip_vision, start_image, positive, negative,
                 width, height, frames, num_loops, overlap, anchor_frame_offset, seed,
                 steps, split_step, cfg, sampler_name, scheduler,
                 loop_prompts="",
                 model_high_1=None,  model_low_1=None,
                 model_high_2=None,  model_low_2=None,
                 model_high_3=None,  model_low_3=None,
                 model_high_4=None,  model_low_4=None,
                 model_high_5=None,  model_low_5=None,
                 model_high_6=None,  model_low_6=None,
                 model_high_7=None,  model_low_7=None,
                 model_high_8=None,  model_low_8=None,
                 model_high_9=None,  model_low_9=None,
                 model_high_10=None, model_low_10=None):

        from comfy_extras.nodes_wan import WanImageToVideo

        if _GetImageRange is None:
            raise RuntimeError("[WanLooperKSA] GetImageRangeFromBatch not loaded")

        # Guard: split_step must be < steps
        split_step = min(split_step, steps - 1)

        # Per-segment model lookup tables (1-based index)
        seg_high = [None, model_high_1, model_high_2, model_high_3, model_high_4,
                    model_high_5, model_high_6, model_high_7, model_high_8,
                    model_high_9, model_high_10]
        seg_low  = [None, model_low_1,  model_low_2,  model_low_3,  model_low_4,
                    model_low_5,  model_low_6,  model_low_7,  model_low_8,
                    model_low_9,  model_low_10]

        # ── Parse per-loop prompts ───────────────────────────────────────────
        # Segments are delimited by --- so each entry can span multiple lines.
        raw_lines = [s.strip() for s in loop_prompts.split("---") if s.strip()] if loop_prompts.strip() else []

        def _get_prompt(loop_idx):
            if not raw_lines:
                return ""
            if loop_idx < len(raw_lines):
                return raw_lines[loop_idx]
            return raw_lines[-1]

        # ── Resize start image once — keep as color reference for drift correction ──
        current_start_image = comfy.utils.common_upscale(
            start_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        # Reference statistics for per-channel color normalization across segments.
        # We match each anchor frame's mean+std back to the original start image to
        # prevent progressive VAE encode/decode color drift from compounding.
        ref_mean = current_start_image.float().mean(dim=(0, 1, 2), keepdim=True)  # [1,1,1,C]
        ref_std  = current_start_image.float().std(dim=(0, 1, 2), keepdim=True).clamp(min=1e-5)

        # ── Per-loop state ───────────────────────────────────────────────────
        segment_dir      = tempfile.mkdtemp(prefix="wan_looper_ksa_")
        segment_paths    = []
        used_prompts_log = []
        prev_decoded      = None
        last_anchor_frame = None

        ksampler = _comfy_nodes.KSamplerAdvanced()

        for loop_idx in range(num_loops):
            loop_id   = loop_idx + 1
            loop_seed = seed + loop_idx
            print(f"\n[WanLooperKSA] ── Loop {loop_id}/{num_loops} ──────────────────────────")

            # ── 1. Per-segment model selection ──────────────────────────────
            high_model = seg_high[loop_id] if seg_high[loop_id] is not None else model_high
            low_model  = seg_low[loop_id]  if seg_low[loop_id]  is not None else model_low
            print(f"[WanLooperKSA] Loop {loop_id}: "
                  f"{'per-segment' if seg_high[loop_id] is not None else 'global'} model_high, "
                  f"{'per-segment' if seg_low[loop_id]  is not None else 'global'} model_low")

            # ── 2. Prompt encoding ───────────────────────────────────────────
            loop_prompt = _get_prompt(loop_idx)
            if loop_prompt:
                tokens       = clip.tokenize(loop_prompt)
                cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
                c_pos        = [[cond, {"pooled_output": pooled}]]
                used_prompts_log.append(f"Loop {loop_id}: {loop_prompt}")
                print(f"[WanLooperKSA] Loop {loop_id}: encoded loop prompt")
            else:
                c_pos = positive
                used_prompts_log.append(f"Loop {loop_id}: (global positive)")
                print(f"[WanLooperKSA] Loop {loop_id}: using global positive conditioning")

            # ── 3. CLIPVision + WanImageToVideo conditioning ─────────────────
            # Identical approach to I2VLooperHL (proven working)
            print(f"[WanLooperKSA] Loop {loop_id}: CLIP Vision encoding...")
            cv_result = _comfy_nodes.CLIPVisionEncode().encode(
                clip_vision, current_start_image, "none"
            )
            cv_out = cv_result[0]

            print(f"[WanLooperKSA] Loop {loop_id}: WanImageToVideo conditioning...")
            i2v_result = WanImageToVideo().execute(
                positive          = c_pos,
                negative          = negative,
                vae               = vae,
                width             = width,
                height            = height,
                length            = frames,
                batch_size        = 1,
                start_image       = current_start_image,
                clip_vision_output = cv_out,
            )
            pos_cond   = i2v_result[0]
            neg_cond   = i2v_result[1]
            latent_dict = i2v_result[2]

            # ── 4. Pass 1 — high model (steps 0 → split_step) ────────────────
            print(f"[WanLooperKSA DEBUG] latent shape: {latent_dict['samples'].shape}")
            print(f"[WanLooperKSA DEBUG] latent min/max: {latent_dict['samples'].min():.4f} / {latent_dict['samples'].max():.4f}")
            print(f"[WanLooperKSA DEBUG] latent mean/std: {latent_dict['samples'].mean():.4f} / {latent_dict['samples'].std():.4f}")
            print(f"[WanLooperKSA DEBUG] pos_cond type: {type(pos_cond)}, len: {len(pos_cond)}")
            print(f"[WanLooperKSA DEBUG] pos_cond[0][0] shape: {pos_cond[0][0].shape}")
            print(f"[WanLooperKSA DEBUG] pos_cond[0][1] keys: {list(pos_cond[0][1].keys())}")
            print(f"[WanLooperKSA DEBUG] start_image shape: {current_start_image.shape}, min: {current_start_image.min():.3f}, max: {current_start_image.max():.3f}")
            print(f"[WanLooperKSA] Loop {loop_id}: sampling HIGH "
                  f"(steps 0-{split_step}, seed={loop_seed})")
            high_out = ksampler.sample(
                model                      = high_model,
                add_noise                  = "enable",
                noise_seed                 = loop_seed,
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

            # ── 5. Pass 2 — low model (steps split_step → end) ───────────────
            print(f"[WanLooperKSA] Loop {loop_id}: sampling LOW "
                  f"(steps {split_step}-end)")
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

            # ── 6. VAE decode ────────────────────────────────────────────────
            print(f"[WanLooperKSA] Loop {loop_id}: VAE decode")
            decoded = vae.decode(sampled)
            while decoded.ndim > 4:
                decoded = decoded.squeeze(0)
            if decoded.shape[1] != height or decoded.shape[2] != width:
                d_bchw  = decoded.permute(0, 3, 1, 2)
                d_bchw  = comfy.utils.common_upscale(d_bchw, width, height, "lanczos", "disabled")
                decoded = d_bchw.permute(0, 2, 3, 1)
            print(f"[WanLooperKSA] Loop {loop_id}: decoded {decoded.shape[0]} frames")

            # ── 7. Extract anchor frame FIRST (needed for trim + drift fix) ────
            actual_frames    = decoded.shape[0]
            anchor_frame_idx = max(0, min(actual_frames - 1, actual_frames + anchor_frame_offset))
            anchor_result    = _GetImageRange.imagesfrombatch(
                images      = decoded,
                start_index = anchor_frame_idx,
                num_frames  = 1,
            )
            raw_anchor = anchor_result[0]  # [1, H, W, C]
            print(f"[WanLooperKSA] Loop {loop_id}: anchor at frame {anchor_frame_idx}")

            # ── 8. Color drift correction ─────────────────────────────────────
            # Each VAE encode/decode cycle shifts per-channel mean+std slightly.
            # Normalize the anchor frame back to the original start image's statistics
            # so drift doesn't compound across segments.
            anchor_f = raw_anchor.float()
            a_mean   = anchor_f.mean(dim=(0, 1, 2), keepdim=True)
            a_std    = anchor_f.std(dim=(0, 1, 2), keepdim=True).clamp(min=1e-5)
            anchor_normalized = ((anchor_f - a_mean) / a_std) * ref_std + ref_mean
            anchor_normalized = anchor_normalized.clamp(0.0, 1.0).to(raw_anchor.dtype)
            last_anchor_frame   = anchor_normalized
            current_start_image = anchor_normalized   # next loop's start image
            print(f"[WanLooperKSA] Loop {loop_id}: anchor color-normalized "
                  f"(mean {a_mean.mean().item():.3f}→{ref_mean.mean().item():.3f})")

            # ── 9. Trim segment to anchor frame — prevents time-jump at transitions ─
            # Without trimming: segment 1 saves frames 0–48, segment 2 starts
            # from frame 44's content → visible 5-frame backward jump.
            # With trimming: segment 1 saves frames 0–44, segment 2 continues
            # cleanly from frame 44.
            decoded_trimmed = decoded[:anchor_frame_idx + 1]
            print(f"[WanLooperKSA] Loop {loop_id}: trimmed to {decoded_trimmed.shape[0]} frames "
                  f"(anchor_frame_offset={anchor_frame_offset})")

            # ── 10. Overlap stitching ────────────────────────────────────────
            if loop_idx == 0 or overlap == 0 or prev_decoded is None or _ImageBatchExtend is None:
                segment_frames = decoded_trimmed
                print(f"[WanLooperKSA] Loop {loop_id}: no overlap")
            else:
                print(f"[WanLooperKSA] Loop {loop_id}: overlap stitching ({overlap} frames)")
                overlap_result = _ImageBatchExtend.imagesfrombatch(
                    source_images = prev_decoded,
                    new_images    = decoded_trimmed,
                    overlap       = overlap,
                    overlap_side  = "source",
                    overlap_mode  = "ease_in_out",
                )
                segment_frames = overlap_result[0]

            # ── 11. Save segment ─────────────────────────────────────────────
            seg_path = os.path.join(segment_dir, f"segment_{loop_id:03d}.pt")
            torch.save(segment_frames.cpu(), seg_path)
            segment_paths.append(seg_path)
            prev_decoded = decoded_trimmed
            print(f"[WanLooperKSA] Loop {loop_id}: saved segment → {seg_path} "
                  f"({segment_frames.shape[0]} frames)")

            # ── 10. Cache cleanup ────────────────────────────────────────────
            comfy.model_management.soft_empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            print(f"[WanLooperKSA] Loop {loop_id}/{num_loops}: complete")

        # ── Assemble full video ──────────────────────────────────────────────
        print(f"\n[WanLooperKSA] Assembling {len(segment_paths)} segment(s) into full video")
        full_video = None
        for sp in segment_paths:
            seg = torch.load(sp, map_location="cpu", weights_only=True)
            full_video = seg if full_video is None else torch.cat([full_video, seg], dim=0)
            del seg
            gc.collect()
        try:
            shutil.rmtree(segment_dir)
        except Exception:
            pass

        if full_video is None:
            full_video = torch.zeros((1, height, width, 3))
        if last_anchor_frame is None:
            last_anchor_frame = torch.zeros((1, height, width, 3))

        used_prompts_str = "\n".join(used_prompts_log)
        print(f"[WanLooperKSA] Done. Total frames: {full_video.shape[0]}")
        return (full_video, last_anchor_frame, used_prompts_str)


# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    # New canonical names
    "WanLoopConfig": WanLoopConfig,
    "WanLoopPrep":   WanLoopPrep,
    "WanLoopFinish": WanLoopFinish,
    "WanLooperCK":   WanLooperCK,
    "WanLooperKSA":  WanLooperKSA,

    # Backward-compat aliases for saved workflow JSONs
    "SVILoopConfig": WanLoopConfig,
    "SVILoopPrep":   WanLoopPrep,
    "SVILoopFinish": WanLoopFinish,
    "SVILooperCK":   WanLooperCK,
    "SVILooperKSA":  WanLooperKSA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanLoopConfig": "Wan Loop Config",
    "WanLoopPrep":   "Wan Loop Prep",
    "WanLoopFinish": "Wan Loop Finish",
    "WanLooperCK":   "Wan Looper CK (Clownshark)",
    "WanLooperKSA":  "Wan Looper KSA (KSamplerAdvanced)",
}
