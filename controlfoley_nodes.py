"""
ComfyUI-ControlFoley: Video-to-Audio Foley Generation Node
Based on ControlFoley by Xiaomi Research (https://github.com/xiaomi-research/controlfoley)

Generates synchronized sound effects from video, text prompts, and/or reference audio.
Supports: Video-to-Audio, Text-to-Audio, Image-to-Audio, and audio style transfer.
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torchaudio
import numpy as np
from PIL import Image
from torchvision.transforms import v2

# Setup paths
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from lib.flow_matching import FlowMatching
from controlfoley.inference_utils import (
    ModelConfig, all_model_cfg, generate, load_video, make_video, load_image
)
from controlfoley.audio_model import AudioGenerationNetwork, create_audio_generation_model
from controlfoley.feature_extractor import FeaturesUtils
from controlfoley.temporal_config import TemporalConfiguration, DEFAULT_44K_CONFIG
from controlfoley.media_utils import MediaClipData

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger(__name__)

# ============================================================================
# Model and Feature Utility Cache (global singletons for efficiency)
# ============================================================================

_model_cache: dict = {}
_feature_utils_cache: dict = {}


def _get_default_model_path() -> Path:
    """Get the default models directory."""
    # Check COMFYUI_MODEL_DIR environment variable or use default
    env_path = os.environ.get("COMFYUI_MODEL_DIR", "")
    if env_path:
        base = Path(env_path)
    else:
        # Default ComfyUI models path relative to this node
        base = Path(current_dir).parent.parent / "models"
    return base / "controlfoley"


def get_model(variant: str = "large_44k", device: str = "cuda", dtype: torch.dtype = torch.float32) -> AudioGenerationNetwork:
    """Load or retrieve cached model."""
    cache_key = (variant, device)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    if variant not in all_model_cfg:
        raise ValueError(f"Unknown model variant: {variant}. Available: {list(all_model_cfg.keys())}")

    model_cfg: ModelConfig = all_model_cfg[variant]
    model_dir = _get_default_model_path()
    model_path = model_dir / "weights" / "controlfoley.pth"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. "
            f"Please download from https://github.com/xiaomi-research/controlfoley "
            f"and place in models/controlfoley/weights/"
        )

    net: AudioGenerationNetwork = create_audio_generation_model(model_cfg.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(model_path, map_location=device, weights_only=True))

    _model_cache[cache_key] = net
    log.info(f"Loaded ControlFoley model from {model_path}")
    return net


def get_feature_utils(mode: str = "44k", device: str = "cuda", dtype: torch.dtype = torch.float32) -> FeaturesUtils:
    """Load or retrieve cached feature utilities."""
    cache_key = (mode, device)
    if cache_key in _feature_utils_cache:
        return _feature_utils_cache[cache_key]

    model_dir = _get_default_model_path()
    ext_dir = model_dir / "ext_weights"

    # Check for required weight files
    required_files = {
        "v1-44.pth": ext_dir / "v1-44.pth",
        "synchformer_state_dict.pth": ext_dir / "synchformer_state_dict.pth",
        "cav_mae_st.pth": ext_dir / "cav_mae_st.pth",
        "music_speech_audioset_epoch_15_esc_89.98.pt": ext_dir / "music_speech_audioset_epoch_15_esc_89.98.pt",
    }
    for name, fpath in required_files.items():
        if not fpath.exists():
            raise FileNotFoundError(
                f"Missing required weight file: {name} at {fpath}. "
                f"Please download all model weights to models/controlfoley/ext_weights/"
            )

    feature_utils = FeaturesUtils(
        tod_vae_ckpt=str(ext_dir / "v1-44.pth"),
        synchformer_ckpt=str(ext_dir / "synchformer_state_dict.pth"),
        cav_mae_ckpt=str(ext_dir / "cav_mae_st.pth"),
        clap_ckpt=str(ext_dir / "music_speech_audioset_epoch_15_esc_89.98.pt"),
        mode=mode,
        enable_conditions=True,
        need_vae_encoder=False,
    )
    feature_utils = feature_utils.to(device, dtype).eval()

    _feature_utils_cache[cache_key] = feature_utils
    log.info("Loaded ControlFoley feature utils")
    return feature_utils


# ============================================================================
# Video loading helpers
# ============================================================================

_CLIP_SIZE = 384
_CLIP_FPS = 8.0
_VISUAL_SIZE = 224
_VISUAL_FPS = 4.0
_SYNC_SIZE = 224
_SYNC_FPS = 25.0


def _get_video_transforms():
    """Get standard video transform pipelines."""
    clip_transform = v2.Compose([
        v2.Resize((_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    visual_transform = v2.Compose([
        v2.Resize(_VISUAL_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(_VISUAL_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
    ])

    sync_transform = v2.Compose([
        v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(_SYNC_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return clip_transform, visual_transform, sync_transform


def load_video_from_frames(frames: torch.Tensor, duration_sec: float):
    """
    Process video frames tensor into ControlFoley format.

    Args:
        frames: Video frames tensor, shape (B, H, W, C) or (B, C, H, W)
                B = number of frames, C = 3 (RGB), H/W = frame dimensions
        duration_sec: Duration of the video in seconds

    Returns:
        MediaClipData with processed frames
    """
    clip_transform, visual_transform, sync_transform = _get_video_transforms()

    # Handle input format: ensure (B, H, W, C) -> (B, C, H, W)
    if frames.dim() == 4:
        if frames.shape[-1] == 3:  # (B, H, W, C)
            frames = frames.permute(0, 3, 1, 2)
        elif frames.shape[1] == 3:  # Already (B, C, H, W)
            pass
        else:
            raise ValueError(f"Unexpected frame shape: {frames.shape}, expected 3 channel images")

    total_frames = frames.shape[0]
    orig_fps = total_frames / duration_sec if duration_sec > 0 else 25.0

    # Extract frames at different FPS rates
    def sample_frames(target_fps, total_frames, orig_fps, duration_sec):
        target_count = int(target_fps * duration_sec)
        if target_count <= 0:
            target_count = 1
        indices = torch.linspace(0, total_frames - 1, target_count, dtype=torch.long)
        return indices

    clip_indices = sample_frames(_CLIP_FPS, total_frames, orig_fps, duration_sec)
    visual_indices = sample_frames(_VISUAL_FPS, total_frames, orig_fps, duration_sec)
    sync_indices = sample_frames(_SYNC_FPS, total_frames, orig_fps, duration_sec)

    clip_frames = frames[clip_indices]
    visual_frames = frames[visual_indices]
    sync_frames = frames[sync_indices]

    # Apply transforms
    clip_frames = clip_transform(clip_frames)
    visual_frames = visual_transform(visual_frames)
    sync_frames = sync_transform(sync_frames)

    # Convert frames to list for frame_sequence
    frame_list = [frames[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8) for i in range(total_frames)]

    from fractions import Fraction
    video_info = MediaClipData(
        total_duration=duration_sec,
        frame_rate=Fraction(int(orig_fps), 1) if orig_fps == int(orig_fps) else Fraction(int(orig_fps * 1000), 1000),
        clip_embeddings=clip_frames,
        visual_features=visual_frames,
        sync_embeddings=sync_frames,
        frame_sequence=frame_list,
    )
    return video_info


def load_video_from_path(video_path: str, duration_sec: float):
    """Load video from file path."""
    from controlfoley.inference_utils import load_video as _load_video
    return _load_video(Path(video_path), duration_sec)


def load_image_from_path(image_path: str):
    """Load image from file path."""
    from controlfoley.inference_utils import load_image as _load_image
    return _load_image(Path(image_path))


def process_audio_reference(audio: torch.Tensor, sampling_rate: int, device: str, dtype: torch.dtype):
    """
    Process reference audio for timbre conditioning.
    Expects 2-4 second audio clip.

    Returns:
        audio_frames: (1, 1, T) at 16kHz for CLAP
        timbre_frames: (1, 1, T) at 32kHz for MusicGen style
        timbre_duration: float
    """
    audio = audio.to(device, dtype)
    timbre_frames = audio

    # Resample to 16kHz for CLAP
    if sampling_rate != 16000:
        audio = torchaudio.functional.resample(audio, sampling_rate, 16000)
    audio = audio.mean(dim=0, keepdim=True)
    audio = audio.reshape(1, -1)
    audio = audio.unsqueeze(0)  # [1, 1, T]

    # Resample to 32kHz for MusicGen
    if sampling_rate != 32000:
        timbre_frames = torchaudio.functional.resample(timbre_frames, sampling_rate, 32000)

    # Handle duration: clamp to 2-4 seconds
    target_sr = 32000
    min_length = 2 * target_sr
    max_length = 4 * target_sr

    if timbre_frames.dim() == 2:
        num_samples = timbre_frames.shape[-1]
    elif timbre_frames.dim() == 3:
        num_samples = timbre_frames.shape[-1]
    else:
        raise ValueError(f"Unexpected audio tensor shape: {timbre_frames.shape}")

    if num_samples < min_length:
        padding_length = min_length - num_samples
        if timbre_frames.dim() == 2:
            timbre_frames = torch.nn.functional.pad(timbre_frames, (0, padding_length), mode='constant', value=0)
        else:
            timbre_frames = torch.nn.functional.pad(timbre_frames, (0, padding_length), mode='constant', value=0)
    elif num_samples > max_length:
        timbre_frames = timbre_frames[..., :max_length]

    num_samples = timbre_frames.shape[-1]
    timbre_duration = num_samples / target_sr

    timbre_frames = timbre_frames.mean(dim=0, keepdim=True)
    timbre_frames = timbre_frames.reshape(1, -1)
    timbre_frames = timbre_frames.unsqueeze(0)  # [1, 1, T]

    return audio, timbre_frames, timbre_duration


# ============================================================================
# ComfyUI Node: ControlFoleyGenerate
# ============================================================================

class ControlFoleyGenerate:
    """
    ControlFoley: Generate synchronized audio from video, text, and/or reference audio.

    This node produces high-quality foley sound effects that match the visual content
    of a video, follow text descriptions, and can adopt the style of reference audio.

    Inputs:
        - video: ComfyUI IMAGE batch tensor (B, H, W, C) - video frames
        - prompt: Text description of desired sound effects
        - negative_prompt: Text description of sounds to avoid
        - reference_audio: Audio waveform for timbre/style conditioning (2-4s recommended)
        - fps: Frame rate of the input video frames
        - cfg_strength: Classifier-free guidance strength (higher = more prompt adherence)
        - steps: Number of diffusion sampling steps
        - seed: Random seed for reproducibility
        - variant: Model variant (currently only "large_44k")

    Outputs:
        - audio: Generated audio waveform in ComfyUI AUDIO format (dict with waveform + sample_rate)
        - video_path: Path to the final video with generated audio muxed in (empty if no video input)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Describe the sound you want to generate..."
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Sounds to avoid..."
                }),
                "fps": ("FLOAT", {
                    "default": 25.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 1.0,
                    "display": "number",
                }),
                "duration": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 300.0,
                    "step": 0.5,
                    "display": "number",
                }),
                "cfg_strength": ("FLOAT", {
                    "default": 4.5,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "display": "number",
                }),
                "steps": ("INT", {
                    "default": 25,
                    "min": 5,
                    "max": 100,
                    "step": 1,
                    "display": "number",
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "step": 1,
                    "display": "number",
                }),
                "variant": (["large_44k"], {
                    "default": "large_44k",
                }),
            },
            "optional": {
                "video": ("IMAGE",),
                "reference_audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("AUDIO", "IMAGE",)
    RETURN_NAMES = ("audio", "video",)
    FUNCTION = "generate"
    CATEGORY = "audio/ControlFoley"
    DESCRIPTION = "Generate synchronized foley sound effects from video, text prompts, and reference audio"

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        fps: float,
        duration: float,
        cfg_strength: float,
        steps: int,
        seed: int,
        variant: str,
        video: Optional[torch.Tensor] = None,
        reference_audio: Optional[dict] = None,
    ):
        # --- Determine device ---
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        dtype = torch.float32

        # --- Load model and feature utils ---
        net = get_model(variant, device, dtype)
        model_cfg: ModelConfig = all_model_cfg[variant]
        seq_cfg: TemporalConfiguration = model_cfg.seq_cfg
        feature_utils = get_feature_utils(model_cfg.mode, device, dtype)

        # --- Random generator ---
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # --- Flow matching ---
        fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=steps)

        # --- Determine duration ---
        # Priority: explicit duration > video frame_count/fps > default 4.0
        clip_frames = visual_frames = sync_frames = None
        if video is not None:
            log.info(f"ControlFoley: input video shape={video.shape}, fps={fps}")
            frame_count = video.shape[0]
            video_duration = frame_count / fps if fps > 0 else 4.0
            log.info(f"ControlFoley: {frame_count} frames @ {fps}fps => {video_duration:.2f}s (video)")
            if duration <= 0:
                duration = video_duration
            else:
                log.info(f"ControlFoley: using explicit duration={duration:.2f}s, overriding video duration={video_duration:.2f}s")

            video_info = load_video_from_frames(video, duration)
            clip_frames = video_info.clip_embeddings
            visual_frames = video_info.visual_features
            sync_frames = video_info.sync_embeddings

            clip_frames = clip_frames.unsqueeze(0)  # Add batch dim
            sync_frames = sync_frames.unsqueeze(0)
            visual_frames = visual_frames.unsqueeze(0)
        elif duration <= 0:
            duration = 4.0  # fallback for text-only mode
            log.info(f"ControlFoley: text-to-audio mode, default duration={duration:.2f}s")
        else:
            log.info(f"ControlFoley: text-to-audio mode, duration={duration:.2f}s")

        # --- Update sequence lengths based on duration ---
        seq_cfg.total_time_seconds = duration
        net.update_seq_lengths(
            seq_cfg.latent_sequence_length,
            seq_cfg.clip_sequence_length,
            seq_cfg.visual_sequence_length,
            seq_cfg.sync_sequence_length,
        )

        # --- Process reference audio input ---
        audio_frames = timbre_frames = None
        timbre_duration = 0.0
        if reference_audio is not None:
            log.info("Processing reference audio for timbre conditioning")
            ref_waveform = reference_audio.get("waveform")
            ref_sample_rate = reference_audio.get("sample_rate", 44100)

            if ref_waveform is not None:
                # Handle ComfyUI AUDIO format: shape is typically [channels, samples] or [1, channels, samples]
                if ref_waveform.dim() == 3:
                    ref_waveform = ref_waveform.squeeze(0)  # Remove batch dim
                audio_frames, timbre_frames, timbre_duration = process_audio_reference(
                    ref_waveform, ref_sample_rate, device, dtype
                )
        else:
            log.info("No reference audio provided")

        # --- Generate ---
        log.info(f"Generating audio: prompt='{prompt}', negative_prompt='{negative_prompt}', duration={duration:.2f}s, cfg={cfg_strength}, steps={steps}")
        start_time = time.time()

        audios = generate(
            clip_frames,
            visual_frames,
            sync_frames,
            audio_frames,
            timbre_frames,
            timbre_duration,
            [prompt] if prompt.strip() else [""],
            negative_text=[negative_prompt] if negative_prompt.strip() else [""],
            feature_utils=feature_utils,
            net=net,
            fm=fm,
            rng=rng,
            cfg_strength=cfg_strength,
        )

        audio = audios.float().cpu()[0]
        elapsed = time.time() - start_time
        log.info(f"Audio generated in {elapsed:.2f}s, shape={audio.shape}")

        # Return in ComfyUI AUDIO format: {"waveform": tensor [channels, samples], "sample_rate": int}
        # ComfyUI AUDIO format expects waveform as (B, C, T) [batch, channels, samples]
        if audio.dim() == 1:
            audio_output_waveform = audio.unsqueeze(0).unsqueeze(0)  # (T,) -> (1, 1, T)
        elif audio.dim() == 2:
            audio_output_waveform = audio.unsqueeze(0)  # (C, T) -> (1, C, T)
        else:
            audio_output_waveform = audio  # already (B, C, T)
        audio_output = {
            "waveform": audio_output_waveform,
            "sample_rate": seq_cfg.audio_sample_rate,
        }

        # VIDEO output: pass through original frames (or empty if text-only mode)
        video_output = video if video is not None else torch.zeros(1, 64, 64, 3)

        return (audio_output, video_output,)


# ============================================================================
# ComfyUI Node: ControlFoleyVideoLoader
# ============================================================================

class ControlFoleyVideoLoader:
    """
    Load a video file for ControlFoley processing.

    This node extracts all frames from a video file and prepares them for the
    ControlFoleyGenerate node, handling all necessary preprocessing.
    No cropping is applied — the full video is always loaded.

    Inputs:
        - video_path: Path to the video file (mp4, avi, mov, etc.)

    Outputs:
        - frames: Processed video frames tensor (B, H, W, C)
        - fps: Frame rate of the video
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Path to video file...",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT",)
    RETURN_NAMES = ("frames", "fps",)
    FUNCTION = "load"
    CATEGORY = "audio/ControlFoley"
    DESCRIPTION = "Load and preprocess video for ControlFoley (full video, no cropping)"

    def load(self, video_path: str):
        video_path = Path(video_path.strip())
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Load full video (pass large duration to get everything)
        video_info = load_video_from_path(str(video_path), 86400.0)
        actual_duration = video_info.total_duration

        # Extract FPS from video_info
        vid_fps = float(video_info.frame_rate) if video_info.frame_rate else 25.0

        # Convert video_info to ComfyUI IMAGE format (B, H, W, C)
        if video_info.frame_sequence and len(video_info.frame_sequence) > 0:
            frame_list = video_info.frame_sequence
            frames_tensor = torch.from_numpy(np.stack(frame_list)).float() / 255.0
        else:
            # Fallback: use clip_embeddings (though these are already processed)
            frames_tensor = video_info.clip_embeddings  # (B, C, H, W) -> convert
            frames_tensor = frames_tensor.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)

        return (frames_tensor, vid_fps,)


# ============================================================================
# ComfyUI Node: ControlFoleyImageToAudio
# ============================================================================

class ControlFoleyImageToAudio:
    """
    Generate audio from a single image using ControlFoley.

    Treats the image as a static video frame and generates appropriate
    ambient sound or sound effects.

    Inputs:
        - image: ComfyUI IMAGE tensor (B, H, W, C) - single image or batch
        - prompt: Text description of desired sound
        - negative_prompt: Text description of sounds to avoid
        - duration: Desired output duration in seconds
        - cfg_strength: Classifier-free guidance strength
        - steps: Number of diffusion sampling steps
        - seed: Random seed

    Outputs:
        - audio: Generated audio waveform
        - sample_rate: 44100
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Describe the sound..."
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
                "duration": ("FLOAT", {
                    "default": 8.0,
                    "min": 1.0,
                    "max": 60.0,
                    "step": 0.5,
                }),
                "cfg_strength": ("FLOAT", {
                    "default": 4.5,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                }),
                "steps": ("INT", {
                    "default": 25,
                    "min": 5,
                    "max": 100,
                    "step": 1,
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT",)
    RETURN_NAMES = ("audio", "sample_rate",)
    FUNCTION = "generate_from_image"
    CATEGORY = "audio/ControlFoley"
    DESCRIPTION = "Generate audio from a single image using ControlFoley"

    def generate_from_image(
        self,
        image: torch.Tensor,
        prompt: str,
        negative_prompt: str,
        duration: float,
        cfg_strength: float,
        steps: int,
        seed: int,
    ):
        # --- Device setup ---
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        dtype = torch.float32

        # --- Models ---
        variant = "large_44k"
        net = get_model(variant, device, dtype)
        model_cfg = all_model_cfg[variant]
        seq_cfg = model_cfg.seq_cfg
        feature_utils = get_feature_utils(model_cfg.mode, device, dtype)

        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=steps)

        # Update sequence lengths
        seq_cfg.total_time_seconds = duration
        net.update_seq_lengths(
            seq_cfg.latent_sequence_length,
            seq_cfg.clip_sequence_length,
            seq_cfg.visual_sequence_length,
            seq_cfg.sync_sequence_length,
        )

        # Convert image to video-like format
        # ComfyUI IMAGE: (B, H, W, C) -> we need (B, C, H, W) then replicate for fps
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dim

        batch_size, h, w, c = image.shape
        # Use the first image
        img = image[0]  # (H, W, C)

        # Create frame replicas for different FPS rates
        clip_transform, visual_transform, sync_transform = _get_video_transforms()

        # For CLIP: replicate at 8fps
        clip_count = int(_CLIP_FPS * duration)
        clip_img = img.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        clip_frames = clip_img.repeat(clip_count, 1, 1, 1)  # (T, C, H, W)
        clip_frames = clip_transform(clip_frames)

        # For visual features
        visual_count = int(_VISUAL_FPS * duration)
        visual_frames = clip_img.repeat(visual_count, 1, 1, 1)
        visual_frames = visual_transform(visual_frames)

        # For sync
        sync_count = int(_SYNC_FPS * duration)
        sync_frames = clip_img.repeat(sync_count, 1, 1, 1)
        sync_frames = sync_transform(sync_frames)

        # Generate
        audios = generate(
            clip_frames.unsqueeze(0),
            visual_frames.unsqueeze(0),
            sync_frames.unsqueeze(0),
            None,  # No reference audio
            None,  # No timbre audio
            0.0,   # No timbre duration
            [prompt] if prompt.strip() else [""],
            negative_text=[negative_prompt] if negative_prompt.strip() else [""],
            feature_utils=feature_utils,
            net=net,
            fm=fm,
            rng=rng,
            cfg_strength=cfg_strength,
            image_input=True,
        )

        audio = audios.float().cpu()[0]
        # ComfyUI AUDIO expects (B, C, T) shape
        if audio.dim() == 1:
            audio_output_waveform = audio.unsqueeze(0).unsqueeze(0)  # (T,) -> (1, 1, T)
        elif audio.dim() == 2:
            audio_output_waveform = audio.unsqueeze(0)  # (C, T) -> (1, C, T)
        else:
            audio_output_waveform = audio  # already (B, C, T)
        audio_output = {
            "waveform": audio_output_waveform,
            "sample_rate": seq_cfg.audio_sample_rate,
        }
        return (audio_output, seq_cfg.audio_sample_rate,)


# ============================================================================
# ComfyUI Node: ControlFoleySaveAudio
# ============================================================================

class ControlFoleySaveAudio:
    """
    Save audio generated by ControlFoley to a file (FLAC format).
    This node is self-contained and does not depend on ComfyUI's built-in SaveAudio.

    Inputs:
        - audio: Audio from ControlFoleyGenerate (ComfyUI AUDIO format)
        - filename_prefix: Prefix for the output filename

    Outputs:
        - STRING: Full path to the saved audio file
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {
                    "multiline": False,
                    "default": "controlfoley_audio",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_audio"
    CATEGORY = "audio/ControlFoley"
    DESCRIPTION = "Save ControlFoley-generated audio to a FLAC file"
    OUTPUT_NODE = True

    def save_audio(self, audio: dict, filename_prefix: str):
        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate", 44100)

        if waveform is None:
            raise ValueError("No waveform found in audio input")

        # waveform shape: (B, C, T) - we save the first batch item
        if waveform.dim() == 3:
            wf = waveform[0]  # (C, T)
        elif waveform.dim() == 2:
            wf = waveform  # (C, T) or (T,)
        else:
            raise ValueError(f"Unexpected audio waveform shape: {waveform.shape}")

        # Clamp to valid range
        wf = wf.clamp(-1.0, 1.0)

        # Output directory: output/controlfoley
        output_dir = Path(current_dir) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique filename
        import time as time_module
        timestamp = int(time_module.time())
        safe_prefix = filename_prefix.replace(" ", "_").replace("/", "_").replace("\\", "_")
        if not safe_prefix:
            safe_prefix = "controlfoley_audio"
        file_path = str(output_dir / f"{safe_prefix}_{timestamp}.flac")

        # Move tensor to CPU if needed
        wf_cpu = wf.cpu()

        # Save with torchaudio
        torchaudio.save(file_path, wf_cpu, sample_rate)

        log.info(f"ControlFoleySaveAudio: saved to {file_path}")
        return (file_path,)


# ============================================================================
# Node Mappings for ComfyUI
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "ControlFoleyGenerate": ControlFoleyGenerate,
    "ControlFoleyVideoLoader": ControlFoleyVideoLoader,
    "ControlFoleyImageToAudio": ControlFoleyImageToAudio,
    "ControlFoleySaveAudio": ControlFoleySaveAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ControlFoleyGenerate": "ControlFoley Generate 🎬🔊",
    "ControlFoleyVideoLoader": "ControlFoley Load Video 📹",
    "ControlFoleyImageToAudio": "ControlFoley Image to Audio 🖼️➡️🔊",
    "ControlFoleySaveAudio": "ControlFoley Save Audio 💾🔊",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
