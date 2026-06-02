#V8
import torch
import torch.nn.functional as F
import math
import json
import os
import hashlib
import time
import numpy as np
from PIL import Image, ImageOps, ImageSequence
from comfy.utils import common_upscale
import comfy.model_management as mm

from typing import TypedDict
import torchaudio

class WanVideoVACEExtend:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "vae": ("WANVAE",),
            "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 29048, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
            "vace_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the steps to apply VACE"}),
            "vace_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the steps to apply VACE"}),
            "overlap": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "tooltip": "Number of frames to overlap from previous generation"}),
            },
            "optional": {
                "input_frames": ("IMAGE",),
                "ref_images": ("IMAGE",),
                "input_masks": ("MASK",),
                "prev_vace_embeds": ("WANVIDIMAGE_EMBEDS",),
                "previous_latents": ("LATENT",),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for reduced memory use"}),
            },
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("vace_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, vae, width, height, num_frames, strength, vace_start_percent, vace_end_percent, overlap,
                input_frames=None, ref_images=None, input_masks=None, prev_vace_embeds=None, previous_latents=None, tiled_vae=False):

        self.device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        self.vae = vae.to(self.device)
        self.vae_stride = (4, 8, 8)

        width = (width // 16) * 16
        height = (height // 16) * 16

        print(f"\n=== WanVideoVACEExtend Debug ===")
        print(f"Total frames requested: {num_frames}")
        print(f"Overlap frames: {overlap}")
        print(f"VAE temporal stride: {self.vae_stride[0]}")

        if previous_latents is not None and overlap > 0:
            print(f"Previous latents shape: {previous_latents['samples'].shape}")
            # Calculate how many latent frames the overlap corresponds to
            overlap_latent_frames = (overlap - 1) // self.vae_stride[0] + 1
            print(f"Overlap {overlap} video frames = {overlap_latent_frames} latent frames")

        target_shape = (16, (num_frames - 1) // self.vae_stride[0] + 1,
                        height // self.vae_stride[1],
                        width // self.vae_stride[2])
        print(f"Target shape: {target_shape}")

        # Keep original frame processing unchanged
        # vace context encode
        if input_frames is None:
            input_frames = torch.zeros((1, 3, num_frames, height, width), device=self.device, dtype=self.vae.dtype)
        else:
            input_frames = input_frames[:num_frames]
            input_frames = common_upscale(input_frames.clone().movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)
            input_frames = input_frames.to(self.vae.dtype).to(self.device).unsqueeze(0).permute(0, 4, 1, 2, 3) # B, C, T, H, W
            input_frames = input_frames * 2 - 1

        if input_masks is None:
            input_masks = torch.ones_like(input_frames, device=self.device)
        else:
            print("input_masks shape", input_masks.shape)
            input_masks = input_masks[:num_frames]
            input_masks = common_upscale(input_masks.clone().unsqueeze(1), width, height, "nearest-exact", "disabled").squeeze(1)
            input_masks = input_masks.to(self.vae.dtype).to(self.device)
            input_masks = input_masks.unsqueeze(-1).unsqueeze(0).permute(0, 4, 1, 2, 3).repeat(1, 3, 1, 1, 1) # B, C, T, H, W

        print(f"Input frames shape: {input_frames.shape}")
        print(f"Input masks shape: {input_masks.shape}")

        if ref_images is not None:
            # Create padded image
            if ref_images.shape[0] > 1:
                ref_images = torch.cat([ref_images[i] for i in range(ref_images.shape[0])], dim=1).unsqueeze(0)

            B, H, W, C = ref_images.shape
            current_aspect = W / H
            target_aspect = width / height
            if current_aspect > target_aspect:
                # Image is wider than target, pad height
                new_h = int(W / target_aspect)
                pad_h = (new_h - H) // 2
                padded = torch.ones(ref_images.shape[0], new_h, W, ref_images.shape[3], device=ref_images.device, dtype=ref_images.dtype)
                padded[:, pad_h:pad_h+H, :, :] = ref_images
                ref_images = padded
            elif current_aspect < target_aspect:
                # Image is taller than target, pad width
                new_w = int(H * target_aspect)
                pad_w = (new_w - W) // 2
                padded = torch.ones(ref_images.shape[0], H, new_w, ref_images.shape[3], device=ref_images.device, dtype=ref_images.dtype)
                padded[:, :, pad_w:pad_w+W, :] = ref_images
                ref_images = padded
            ref_images = common_upscale(ref_images.movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)

            ref_images = ref_images.to(self.vae.dtype).to(self.device).unsqueeze(0).permute(0, 4, 1, 2, 3).unsqueeze(0)
            ref_images = ref_images * 2 - 1

        # Encode NEW frames only
        z0 = self.vace_encode_frames(input_frames, ref_images, masks=input_masks, tiled_vae=tiled_vae)
        self.vae.model.clear_cache()
        m0 = self.vace_encode_masks(input_masks, ref_images)

        print(f"\nAfter encoding new frames:")
        print(f"z0 length: {len(z0)}")
        for i, z in enumerate(z0):
            print(f"  z0[{i}] shape: {z.shape}")
            # Let's analyze the structure of the encoded latent
            if input_masks is not None:
                half_channels = z.shape[0] // 2
                print(f"  - First {half_channels} channels (inactive): min={z[:half_channels].min():.3f}, max={z[:half_channels].max():.3f}, mean={z[:half_channels].mean():.3f}")
                print(f"  - Last {half_channels} channels (reactive): min={z[half_channels:].min():.3f}, max={z[half_channels:].max():.3f}, mean={z[half_channels:].mean():.3f}")
        print(f"m0 length: {len(m0)}")
        for i, m in enumerate(m0):
            print(f"  m0[{i}] shape: {m.shape}")

        # Now handle overlap if we have previous latents
        if previous_latents is not None and overlap > 0:
            # Extract overlap latents from previous generation
            latent_samples = previous_latents['samples']  # Shape: [B, C, T, H, W]

            # Calculate how many latent frames we need for the overlap
            overlap_latent_frames = (overlap - 1) // self.vae_stride[0] + 1
            print(f"\nExtracting {overlap_latent_frames} latent frames for {overlap} video frames")

            if overlap_latent_frames >= latent_samples.shape[2]:
                overlap_latents = latent_samples
            else:
                overlap_latents = latent_samples[:, :, -overlap_latent_frames:, :, :]

            print(f"Overlap latents shape: {overlap_latents.shape}")
            print(f"Overlap latents stats: min={overlap_latents.min():.3f}, max={overlap_latents.max():.3f}, mean={overlap_latents.mean():.3f}")

            # Process overlap latents to match VACE format
            # We need to convert from [B, C, T, H, W] to the format that VACE expects
            overlap_latents = overlap_latents.to(self.device).to(self.vae.dtype)

            # Create a modified z0 that includes overlap
            # The key insight: we need to manually construct the overlap portion in the same format as z0
            if len(z0) > 0:
                # Get the encoded latent for new frames
                new_latent = z0[0]  # Shape: [2*C, T_new, H, W] if masks were used
                print(f"New latent shape: {new_latent.shape}")

                # Process overlap latents
                # Remove batch dimension
                overlap_latents_squeezed = overlap_latents.squeeze(0)  # [C, T_overlap, H, W]
                print(f"Overlap latents squeezed shape: {overlap_latents_squeezed.shape}")

                # If masks were used, we need to create the inactive/reactive structure
                if input_masks is not None:
                    # The diffusion model outputs latents with C=16 channels
                    # VACE expects 2*C=32 channels (inactive + reactive)
                    # For overlap frames from previous generation:
                    # - Inactive channels: the actual latent values (what we want to preserve)
                    # - Reactive channels: zeros (these frames won't be modified)
                    overlap_inactive = overlap_latents_squeezed
                    overlap_reactive = torch.zeros_like(overlap_inactive)
                    overlap_combined = torch.cat([overlap_inactive, overlap_reactive], dim=0)  # [2*C, T_overlap, H, W]
                else:
                    overlap_combined = overlap_latents_squeezed

                print(f"Overlap combined shape: {overlap_combined.shape}")

                # Add ref image dimension if needed
                if ref_images is not None:
                    # The ref latent is prepended, so we add it to overlap too
                    # But for overlap, we use the existing latent time dimension
                    final_latent = torch.cat([new_latent[:, :1, :, :], overlap_combined, new_latent[:, 1:, :, :]], dim=1)
                else:
                    final_latent = torch.cat([overlap_combined, new_latent], dim=1)

                print(f"Final latent shape: {final_latent.shape}")
                z0[0] = final_latent

                # Now we need to create matching masks for the overlap frames
                # The mask calculation uses the same formula as vace_encode_masks
                overlap_mask_depth = int((overlap + 3) // self.vae_stride[0])
                print(f"Overlap mask depth calculation: ({overlap} + 3) // {self.vae_stride[0]} = {overlap_mask_depth}")

                overlap_mask_shape = (self.vae_stride[1] * self.vae_stride[2],
                                     overlap_mask_depth,
                                     height // self.vae_stride[1],
                                     width // self.vae_stride[2])
                overlap_mask = torch.ones(overlap_mask_shape, device=self.device, dtype=m0[0].dtype)
                print(f"Overlap mask shape: {overlap_mask.shape}")

                # Combine masks
                if ref_images is not None:
                    # Include ref mask padding
                    final_mask = torch.cat([m0[0][:, :1, :, :], overlap_mask, m0[0][:, 1:, :, :]], dim=1)
                else:
                    final_mask = torch.cat([overlap_mask, m0[0]], dim=1)

                print(f"Final mask shape: {final_mask.shape}")
                m0[0] = final_mask

        print(f"\nBefore vace_latent:")
        print(f"z0 shapes: {[z.shape for z in z0]}")
        print(f"m0 shapes: {[m.shape for m in m0]}")

        # Combine latents and masks
        z = self.vace_latent(z0, m0)

        self.vae.to(offload_device)

        vace_input = {
            "vace_context": z,
            "vace_scale": strength,
            "has_ref": ref_images is not None,
            "num_frames": num_frames,
            "target_shape": target_shape,
            "vace_start_percent": vace_start_percent,
            "vace_end_percent": vace_end_percent,
            "vace_seq_len": math.ceil((z[0].shape[2] * z[0].shape[3]) / 4 * z[0].shape[1]),
            "additional_vace_inputs": [],
        }

        if prev_vace_embeds is not None:
            vace_input["additional_vace_inputs"].append(prev_vace_embeds)

        return (vace_input,)

    def vace_encode_frames(self, frames, ref_images, masks=None, tiled_vae=False):
        if ref_images is None:
            ref_images = [None] * len(frames)
        else:
            assert len(frames) == len(ref_images)

        if masks is None:
            latents = self.vae.encode(frames, device=self.device, tiled=tiled_vae)
        else:
            inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks)]
            reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks)]
            inactive = self.vae.encode(inactive, device=self.device, tiled=tiled_vae)
            reactive = self.vae.encode(reactive, device=self.device, tiled=tiled_vae)
            latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)]
        self.vae.model.clear_cache()
        cat_latents = []
        for latent, refs in zip(latents, ref_images):
            if refs is not None:
                if masks is None:
                    ref_latent = self.vae.encode(refs, device=self.device, tiled=tiled_vae)
                else:
                    print("refs shape", refs.shape)
                    ref_latent = self.vae.encode(refs, device=self.device, tiled=tiled_vae)
                    ref_latent = [torch.cat((u, torch.zeros_like(u)), dim=0) for u in ref_latent]
                assert all([x.shape[1] == 1 for x in ref_latent])
                latent = torch.cat([*ref_latent, latent], dim=1)
            cat_latents.append(latent)
        return cat_latents

    def vace_encode_masks(self, masks, ref_images=None):
        if ref_images is None:
            ref_images = [None] * len(masks)
        else:
            assert len(masks) == len(ref_images)

        result_masks = []
        for mask, refs in zip(masks, ref_images):
            c, depth, height, width = mask.shape
            new_depth = int((depth + 3) // self.vae_stride[0])
            height = 2 * (int(height) // (self.vae_stride[1] * 2))
            width = 2 * (int(width) // (self.vae_stride[2] * 2))

            # reshape
            mask = mask[0, :, :, :]
            mask = mask.view(
                depth, height, self.vae_stride[1], width, self.vae_stride[1]
            )  # depth, height, 8, width, 8
            mask = mask.permute(2, 4, 0, 1, 3)  # 8, 8, depth, height, width
            mask = mask.reshape(
                self.vae_stride[1] * self.vae_stride[2], depth, height, width
            )  # 8*8, depth, height, width

            # interpolation
            mask = F.interpolate(mask.unsqueeze(0), size=(new_depth, height, width), mode='nearest-exact').squeeze(0)

            if refs is not None:
                length = len(refs)
                mask_pad = torch.zeros_like(mask[:, :length, :, :])
                mask = torch.cat((mask_pad, mask), dim=1)
            result_masks.append(mask)
        return result_masks

    def vace_latent(self, z, m):
        return [torch.cat([zz, mm], dim=0) for zz, mm in zip(z, m)]


import numpy as np
import torch
from colour.models import (
    log_encoding_ARRILogC3,
    log_decoding_ARRILogC3,
)

import gc
from contextlib import nullcontext


class LogCRec709Convert:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "direction": (["LogC3 to Rec709", "Rec709 to LogC3"], {"default": "LogC3 to Rec709"}),
                "exposure_stops": ("FLOAT", {"default": -3.0, "min": -10.0, "max": 10.0, "step": 0.25}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05}),
                "pivot": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.9, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "sjnodes"

    def _apply_contrast(self, x, contrast, pivot):
        """Apply contrast around pivot point. Reversible by using 1/contrast."""
        # Normalize around pivot, apply power, restore
        x_shifted = x / pivot
        x_contrasted = torch.sign(x_shifted) * torch.pow(torch.abs(x_shifted), contrast)
        return x_contrasted * pivot

    def convert(self, image, direction, exposure_stops, contrast, pivot):
        # ARRI LogC3 constants (EI 800)
        a, b, c, d, e, f = 5.555556, 0.052272, 0.247190, 0.385537, 5.367655, 0.092809
        lin_cut = 0.010591
        log_cut = e * lin_cut + f

        x = image
        exposure_mult = 2.0 ** exposure_stops

        if direction == "LogC3 to Rec709":
            # LogC3 -> Linear
            lin = torch.where(
                x > log_cut,
                (torch.pow(10.0, (x - d) / c) - b) / a,
                (x - f) / e
            )
            # Apply exposure
            lin = lin * exposure_mult
            # Linear -> Rec709 gamma
            out = torch.sign(lin) * torch.pow(torch.abs(lin), 1.0 / 2.4)
            # Apply contrast (after gamma, in display space)
            out = self._apply_contrast(out, contrast, pivot)
        else:
            # Reverse contrast first
            x = self._apply_contrast(x, 1.0 / contrast, pivot)
            # Rec709 -> Linear
            lin = torch.sign(x) * torch.pow(torch.abs(x), 2.4)
            # Reverse exposure
            lin = lin / exposure_mult
            # Linear -> LogC3
            out = torch.where(
                lin > lin_cut,
                c * torch.log10(a * lin + b) + d,
                e * lin + f
            )

        return (out,)

class PadAudioToLength:
    """
    Pads the input audio with silence at the end to reach a desired length (in milliseconds).
    The sample rate is preserved.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_length_ms": ("INT", {
                    "default": 1000,
                    "min": 1,
                    "max": 600000,
                    "step": 1,
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("padded_audio",)
    FUNCTION = "process"
    CATEGORY = "sjnodes/audio"

    @staticmethod
    def process(audio, target_length_ms):
        # Extract waveform and sample rate from input
        if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
            raise ValueError("Audio input must be a dictionary with 'waveform' and 'sample_rate' keys")

        waveform = audio["waveform"]  # shape: [1, C, N]
        sample_rate = audio["sample_rate"]

        # Compute current and target lengths in samples
        current_length = waveform.shape[-1]
        target_length = int((target_length_ms / 1000.0) * sample_rate)

        if current_length >= target_length:
            # Already long enough — return original
            return (audio,)

        # Compute number of silence samples to add
        silence_length = target_length - current_length

        # Create silence tensor with same shape
        silence = torch.zeros((waveform.shape[0], waveform.shape[1], silence_length), dtype=waveform.dtype, device=waveform.device)

        # Concatenate waveform + silence
        padded_waveform = torch.cat([waveform, silence], dim=-1)

        return ({
            "waveform": padded_waveform,
            "sample_rate": sample_rate,
        },)

class AudioDict(TypedDict):
    """Comfy's representation of AUDIO data."""

    sample_rate: int
    waveform: torch.Tensor


AudioData = AudioDict | list[AudioDict]


class AudioBase:
    """Base class for audio processing."""

    @classmethod
    def is_stereo(
        cls,
        audios: AudioData,
    ) -> bool:
        if isinstance(audios, list):
            return any(cls.is_stereo(audio) for audio in audios)
        else:
            return audios["waveform"].shape[1] == 2

    @staticmethod
    def resample(audio: AudioDict, common_sample_rate: int) -> AudioDict:
        if audio["sample_rate"] != common_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=audio["sample_rate"], new_freq=common_sample_rate
            )
            return {
                "sample_rate": common_sample_rate,
                "waveform": resampler(audio["waveform"]),
            }
        else:
            return audio

    @staticmethod
    def to_stereo(audio: AudioDict) -> AudioDict:
        if audio["waveform"].shape[1] == 1:
            return {
                "sample_rate": audio["sample_rate"],
                "waveform": torch.cat(
                    [audio["waveform"], audio["waveform"]], dim=1
                ),
            }
        else:
            return audio

    @classmethod
    def preprocess_audios(
        cls, audios: list[AudioDict]
    ) -> tuple[list[AudioDict], bool, int]:
        max_sample_rate = max([audio["sample_rate"] for audio in audios])

        resampled_audios = [
            cls.resample(audio, max_sample_rate) for audio in audios
        ]

        is_stereo = cls.is_stereo(audios)
        if is_stereo:
            audios = [cls.to_stereo(audio) for audio in resampled_audios]

        return (audios, is_stereo, max_sample_rate)


class AudioStackWithVolume(AudioBase):
    """Stack/Overlay audio inputs with volume control.

    - Converts batched audio into continuous audio.
    - Pads audios to the longest input.
    - Resamples audios to the highest sample rate in the inputs.
    - Converts them all to stereo if one of the inputs is stereo.
    - Adjusts the volume of each input using provided multipliers.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Dynamic inputs for audio and their respective volumes
                "audio_1": ("AUDIO",),
                "volume_1": (
                    ("FLOAT"),
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1},
                ),
                "audio_2": ("AUDIO",),
                "volume_2": (
                    ("FLOAT"),
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1},
                ),
                # Add more audio/volume pairs as needed
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("stacked_audio",)
    CATEGORY = "audio"
    FUNCTION = "stack_with_volume"

    def stack_with_volume(self, **kwargs: AudioDict | float) -> tuple[AudioDict]:
        # Separate audio inputs and volume multipliers
        audio_keys = [key for key in kwargs if key.startswith("audio_")]
        volume_keys = [key for key in kwargs if key.startswith("volume_")]

        audio_inputs = [kwargs[key] for key in audio_keys]
        volume_multipliers = [kwargs[key] for key in volume_keys]

        # Preprocess audio inputs
        audios, is_stereo, max_rate = self.preprocess_audios(audio_inputs)

        # Adjust volumes of each audio
        for i, audio in enumerate(audios):
            multiplier = volume_multipliers[i] if i < len(volume_multipliers) else 1.0
            audios[i]["waveform"] *= multiplier

        # Convert batched audio inputs into continuous audio
        for i, audio in enumerate(audios):
            waveform = audio["waveform"]

            if len(waveform.shape) == 4:  # Handle batched inputs
                batch_size, channels, _, length = waveform.shape
                print(f"Converting batch of size {batch_size} to continuous audio.")
                audio["waveform"] = waveform.view(batch_size * channels, length)  # Flatten batch into one tensor

        # Ensure all waveforms have the same number of dimensions
        for i, audio in enumerate(audios):
            if len(audio["waveform"].shape) == 2:  # Add batch dimension
                audio["waveform"] = audio["waveform"].unsqueeze(0)

        # Debugging: Print all audio shapes after preprocessing
        print("Processed audio shapes:")
        for idx, audio in enumerate(audios):
            print(f"Audio {idx + 1}: {audio['waveform'].shape}")

        # Find the maximum length across all audio inputs
        max_length = max([audio["waveform"].shape[-1] for audio in audios])

        # Pad audios to match the maximum length
        padded_audios: list[torch.Tensor] = []
        for audio in audios:
            channels = audio["waveform"].shape[1]  # Dynamically determine channels
            padding = torch.zeros(
                (
                    1,
                    channels,  # Match the number of channels
                    max_length - audio["waveform"].shape[-1],
                )
            )
            padded_audio = torch.cat([audio["waveform"], padding], dim=-1)
            padded_audios.append(padded_audio)

        # Stack and sum the audio waveforms
        stacked_waveform = torch.stack(padded_audios, dim=0).sum(dim=0)

        return (
            {
                "sample_rate": max_rate,
                "waveform": stacked_waveform,
            },
        )


class FixedLengthAudioSequencer(AudioBase):
    """Sequences a list of audio inputs into fixed-length segments with crossfading.

    - Accepts a list of audio inputs
    - Accepts either a single segment length or a list of lengths
    - Outputs a continuous audio with variable-length segments
    - Handles overlap between segments with crossfading
    - Pads shorter segments to match the specified lengths
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),  # Accepts a list of audio inputs
                "segment_length": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.1, "max": 999.0, "step": 0.1}
                ),  # Length in seconds
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("sequenced_audio",)
    CATEGORY = "audio"
    FUNCTION = "sequence_fixed"
    INPUT_IS_LIST = True  # Enable list input processing
    CROSSFADE_DURATION = 0.5  # Fixed 0.5 second crossfade

    def sequence_fixed(self, audio: list[AudioDict], segment_length: list[float]):
        # Handle segment lengths
        if len(segment_length) == 1:
            segment_lengths = [segment_length[0]] * len(audio)
        elif len(segment_length) != len(audio):
            raise ValueError(f"Number of segment lengths ({len(segment_length)}) must match number of audio inputs ({len(audio)}) or be 1")
        else:
            segment_lengths = segment_length

        # Ensure all inputs have the same sample rate
        sample_rates = [a["sample_rate"] for a in audio]
        if len(set(sample_rates)) > 1:
            raise ValueError("All audio inputs must have the same sample rate")

        sample_rate = sample_rates[0]
        crossfade_samples = int(self.CROSSFADE_DURATION * sample_rate)

        # Preprocess audio inputs for stereo consistency
        audios, is_stereo, _ = self.preprocess_audios(audio)
        channels = 2 if is_stereo else 1

        # Normalize waveform shapes to [1, channels, samples]
        for i, audio in enumerate(audios):
            waveform = audio["waveform"]

            # Handle different tensor shapes
            if len(waveform.shape) == 1:
                # [samples] -> [1, 1, samples]
                waveform = waveform.unsqueeze(0).unsqueeze(0)
            elif len(waveform.shape) == 2:
                # [channels, samples] -> [1, channels, samples]
                waveform = waveform.unsqueeze(0)
            elif len(waveform.shape) == 3:
                # Already [batch, channels, samples]
                pass
            else:
                raise ValueError(f"Unexpected waveform shape: {waveform.shape}")

            # Ensure correct number of channels
            if waveform.shape[1] == 1 and channels == 2:
                waveform = waveform.repeat(1, 2, 1)
            elif waveform.shape[1] == 2 and channels == 1:
                waveform = waveform.mean(dim=1, keepdim=True)

            audios[i]["waveform"] = waveform

        # Initialize sequence for building the output
        sequence: list[torch.Tensor] = []

        for audio_idx, (audio, seg_length) in enumerate(zip(audios, segment_lengths)):
            segment_samples = int(seg_length * sample_rate)
            current_segment = torch.zeros((1, channels, segment_samples), device=audio["waveform"].device)
            current_position = 0

            waveform = audio["waveform"]
            remaining_samples = waveform.shape[-1]
            position_in_wave = 0

            while remaining_samples > 0:
                samples_to_take = min(
                    remaining_samples,
                    segment_samples - current_position
                )

                # Extract the slice we need
                slice_to_add = waveform[0, :, position_in_wave:position_in_wave + samples_to_take]

                # Add samples to current segment
                current_segment[0, :, current_position:current_position + samples_to_take] = slice_to_add

                current_position += samples_to_take
                position_in_wave += samples_to_take
                remaining_samples -= samples_to_take

                if current_position >= segment_samples - crossfade_samples or remaining_samples == 0:
                    is_last_audio = (audio_idx == len(audios) - 1)
                    if remaining_samples > 0 or not is_last_audio:
                        next_seg_length = segment_samples
                        if not is_last_audio:
                            next_seg_length = int(segment_lengths[audio_idx + 1] * sample_rate)

                        next_segment = torch.zeros((1, channels, next_seg_length), device=waveform.device)

                        if remaining_samples > 0:
                            overlap_samples = min(remaining_samples, crossfade_samples)
                            fade_out = torch.linspace(1, 0, overlap_samples, device=waveform.device)
                            fade_in = torch.linspace(0, 1, overlap_samples, device=waveform.device)

                            # Apply crossfade with proper broadcasting
                            current_segment[0, :, -overlap_samples:] = current_segment[0, :, -overlap_samples:] * fade_out.unsqueeze(0)
                            overlap_audio = waveform[0, :, position_in_wave:position_in_wave + overlap_samples]
                            next_segment[0, :, :overlap_samples] = overlap_audio * fade_in.unsqueeze(0)

                            position_in_wave += overlap_samples
                            remaining_samples -= overlap_samples

                    sequence.append(current_segment)

                    current_segment = next_segment if remaining_samples > 0 else \
                        torch.zeros((1, channels, segment_samples), device=waveform.device)
                    current_position = overlap_samples if remaining_samples > 0 else 0

        if current_position > 0:
            sequence.append(current_segment)

        if sequence:
            final_waveform = torch.cat(sequence, dim=-1)
        else:
            final_waveform = torch.zeros((1, channels, int(segment_lengths[0] * sample_rate)))

        return ({
            "sample_rate": sample_rate,
            "waveform": final_waveform
        },)


class JsonArrayExtractor:
    """
    Extracts a field from a JSON array of objects.

    Modes:
      single  — value of `key` at position `index` (dot notation supported)
      list    — all values of `key`, newline-separated
      flatten — concatenates prompt sub-fields into a single Flux-ready string

    Automatically strips ```json fences from the input.
    """

    FLATTEN_FIELDS = ["scene", "subjects", "style", "lighting", "mood", "background", "composition"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_string": ("STRING", {"multiline": True, "tooltip": "JSON array of objects"}),
                "key":         ("STRING", {"default": "prompt",
                                           "tooltip": "Key to extract (dot notation supported, e.g. prompt.camera.lens)"}),
                "mode":        (["single", "list", "flatten"],),
                "index":       ("INT",    {"default": 0, "min": 0, "max": 9999,
                                           "tooltip": "Zero-based index (single / flatten modes)"}),
                "skip_keys":   ("STRING", {"default": "entry_state",
                                           "tooltip": "Comma-separated keys to exclude in list mode"}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("value",  "count")
    FUNCTION = "extract"
    CATEGORY = "sjnodes/JSON"

    @staticmethod
    def _strip_fences(s):
        s = s.strip()
        if s.startswith("```"):
            s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
        return s.strip()

    @staticmethod
    def _extract_nested(obj, key_path):
        for k in key_path.split("."):
            if isinstance(obj, dict):
                obj = obj[k]
            elif isinstance(obj, list) and k.isdigit():
                obj = obj[int(k)]
            else:
                return None
        return obj

    @staticmethod
    def _to_str(v):
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False)
        return str(v)

    @staticmethod
    def _flatten_prompt(prompt_obj):
        parts = []
        for field in JsonArrayExtractor.FLATTEN_FIELDS:
            if field == "subjects":
                for s in prompt_obj.get("subjects", []):
                    for subfield in ("description", "position", "pose", "framing_visible"):
                        v = s.get(subfield, "")
                        if v:
                            parts.append(str(v))
            else:
                v = prompt_obj.get(field, "")
                if v:
                    if isinstance(v, list):
                        parts.append(", ".join(str(i) for i in v))
                    else:
                        parts.append(str(v))
        cam = prompt_obj.get("camera", {})
        cam_parts = [x for x in [
            f"shot on {cam.get('body', '')}" if cam.get("body") else "",
            f"{cam.get('lens', '')} lens" if cam.get("lens") else "",
            cam.get("f-number", ""),
            cam.get("depth_of_field", ""),
        ] if x]
        if cam_parts:
            parts.append(", ".join(cam_parts))
        return ", ".join(p for p in parts if p)

    def extract(self, json_string, key, mode, index, skip_keys="entry_state"):
        json_string = self._strip_fences(json_string)
        data = json.loads(json_string)
        if not isinstance(data, list):
            raise ValueError("JsonArrayExtractor: input must be a JSON array")

        count = len(data)
        excluded = {k.strip() for k in skip_keys.split(",") if k.strip()}

        if mode == "flatten":
            if index >= count:
                raise IndexError(f"JsonArrayExtractor: index {index} out of range (array length {count})")
            prompt_obj = self._extract_nested(data[index], key)
            if not isinstance(prompt_obj, dict):
                raise ValueError(f"JsonArrayExtractor: flatten mode requires key '{key}' to be an object")
            value = self._flatten_prompt(prompt_obj)

        elif mode == "single":
            if index >= count:
                raise IndexError(f"JsonArrayExtractor: index {index} out of range (array length {count})")
            value = self._to_str(self._extract_nested(data[index], key))

        else:  # list
            rows = []
            for item in data:
                filtered = {k: v for k, v in item.items() if k not in excluded}
                rows.append(self._to_str(self._extract_nested(filtered, key)))
            value = "\n".join(rows)

        return (value, count)


class LoadImageFromPath:
    """
    Loads all images from a directory path — identical behaviour to VHS
    'Load Images (Path)' — but with a graceful fallback instead of an error
    when the directory is missing or empty.

    Paths can be absolute or relative to the ComfyUI install root, e.g.:
        ./output/endless_story/test/characters/simon
    """

    IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff', '.tif'}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {
                    "placeholder": "./output/path/to/images",
                    # vhs_path_extensions triggers the VHS path-picker UI if VHS is installed
                    "vhs_path_extensions": [],
                    "tooltip": "Absolute or ComfyUI-relative path to a directory of images.",
                }),
            },
            "optional": {
                "fallback_image": ("IMAGE", {
                    "tooltip": "Used when the directory is missing or contains no images.",
                }),
                "image_load_cap": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1,
                                           "tooltip": "Max images to load. 0 = no limit."}),
                "skip_first_images": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 999999, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "frame_count")
    FUNCTION = "load"
    CATEGORY = "sjnodes/Image"

    @staticmethod
    def _resolve(directory: str) -> str:
        """Strip quotes/whitespace and resolve relative paths vs. ComfyUI root."""
        import folder_paths as fp
        directory = directory.strip().strip('"')
        if not os.path.isabs(directory):
            directory = os.path.join(fp.base_path, directory)
        return os.path.normpath(directory)

    @classmethod
    def _get_files(cls, directory: str, skip: int, every_nth: int, cap: int):
        all_files = sorted(
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in cls.IMG_EXTENSIONS
        )
        files = all_files[skip::every_nth]
        if cap > 0:
            files = files[:cap]
        return files

    @staticmethod
    def _load_single(path: str, target_size, as_rgba: bool):
        """Load one image, resize to target_size if needed, return numpy float32 array."""
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGBA" if as_rgba else "RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        h, w = target_size[1], target_size[0]
        if arr.shape[0] != h or arr.shape[1] != w:
            t = torch.from_numpy(arr).movedim(-1, 0).unsqueeze(0)
            t = common_upscale(t, w, h, "lanczos", "center")
            arr = t.squeeze(0).movedim(0, -1).numpy()
        if as_rgba:
            arr[:, :, 3] = 1.0 - arr[:, :, 3]  # invert alpha → mask convention
        return arr

    @staticmethod
    def _blank():
        image = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        mask  = torch.zeros((1, 1, 1),    dtype=torch.float32)
        return image, mask, 0

    def load(self, directory: str, fallback_image=None,
             image_load_cap: int = 0, skip_first_images: int = 0, select_every_nth: int = 1):

        directory = self._resolve(directory)

        # --- attempt to load from directory ---
        if directory and os.path.isdir(directory):
            try:
                files = self._get_files(directory, skip_first_images, select_every_nth, image_load_cap)
                if files:
                    # determine dominant size and whether any image has alpha
                    sizes = {}
                    has_alpha = False
                    for p in files:
                        img = Image.open(p)
                        img = ImageOps.exif_transpose(img)
                        has_alpha = has_alpha or "A" in img.getbands()
                        sizes[img.size] = sizes.get(img.size, 0) + 1
                    target_size = max(sizes, key=sizes.get)

                    arrays = [self._load_single(p, target_size, has_alpha) for p in files]
                    channels = 4 if has_alpha else 3
                    images_np = np.fromiter(
                        iter(arrays),
                        dtype=np.dtype((np.float32, (target_size[1], target_size[0], channels)))
                    )
                    images = torch.from_numpy(images_np[:, :, :, :3])
                    if has_alpha:
                        masks = torch.from_numpy(images_np[:, :, :, 3])  # (N, H, W)
                    else:
                        masks = torch.zeros((len(files), 64, 64), dtype=torch.float32)
                    return (images, masks, len(files))
                else:
                    print(f"[LoadImageFromPath] No images found in '{directory}' — using fallback.")
            except Exception as e:
                print(f"[LoadImageFromPath] Error loading '{directory}': {e} — using fallback.")
        else:
            if directory:
                print(f"[LoadImageFromPath] Directory not found: '{directory}' — using fallback.")

        # --- fallback ---
        if fallback_image is not None:
            n, h, w = fallback_image.shape[0], fallback_image.shape[1], fallback_image.shape[2]
            masks = torch.zeros((n, h, w), dtype=torch.float32)
            return (fallback_image, masks, n)

        print("[LoadImageFromPath] No directory and no fallback — returning blank image.")
        return self._blank()

    @classmethod
    def IS_CHANGED(cls, directory: str, **kwargs):
        directory = cls._resolve(directory)
        if not os.path.isdir(directory):
            return float("nan")
        try:
            files = cls._get_files(
                directory,
                kwargs.get("skip_first_images", 0),
                kwargs.get("select_every_nth", 1),
                kwargs.get("image_load_cap", 0),
            )
            m = hashlib.sha256()
            for p in files:
                stat = os.stat(p)
                m.update(f"{p}{stat.st_size}{stat.st_mtime}".encode())
            return m.digest().hex()
        except Exception:
            return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, directory: str, **kwargs):
        return True  # always accept — missing directory falls back gracefully


class SJAlwaysChanged:
    """
    Emits a changing value every run. Useful as a cache-buster trigger.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("tick", "tick_string")
    FUNCTION = "emit"
    CATEGORY = "sjnodes/Utils"

    def emit(self):
        tick = time.time_ns()
        return (int(tick), str(tick))

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")


class SJPathListDirLive:
    """
    Lists directory contents and invalidates cache when directory contents change.
    Optional `refresh` can force rerun. Optional `signal` can be wired from an
    upstream node to enforce execution order (run after producer branch).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
            },
            "optional": {
                "files_only": ("BOOLEAN", {"default": False}),
                "dirs_only": ("BOOLEAN", {"default": False}),
                "refresh": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "signal": ("*",),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("entries", "count")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "list_directory"
    CATEGORY = "sjnodes/Path"

    @staticmethod
    def _resolve(path: str) -> str:
        if not path:
            return os.getcwd()
        return os.path.abspath(os.path.expandvars(os.path.expanduser(path.strip().strip('"'))))

    @classmethod
    def _filtered_entries(cls, path: str, files_only: bool = False, dirs_only: bool = False) -> list[str]:
        entries = sorted(os.listdir(path))
        if files_only:
            entries = [e for e in entries if os.path.isfile(os.path.join(path, e))]
        elif dirs_only:
            entries = [e for e in entries if os.path.isdir(os.path.join(path, e))]
        return entries

    def list_directory(self, path: str, files_only: bool = False, dirs_only: bool = False, refresh: int = 0, signal=None):
        resolved = self._resolve(path)

        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Directory does not exist: {resolved}")
        if not os.path.isdir(resolved):
            raise NotADirectoryError(f"Path is not a directory: {resolved}")

        entries = self._filtered_entries(resolved, files_only, dirs_only)
        return (entries, len(entries))

    @classmethod
    def IS_CHANGED(cls, path: str, files_only: bool = False, dirs_only: bool = False, refresh: int = 0, signal=None, **kwargs):
        resolved = cls._resolve(path)
        if not os.path.isdir(resolved):
            return float("nan")

        try:
            m = hashlib.sha256()
            m.update(str(refresh).encode())
            # Include a lightweight signal token so upstream trigger changes can
            # invalidate cache without heavy serialization of arbitrary payloads.
            if isinstance(signal, (str, int, float, bool, type(None))):
                sig_token = signal
            elif isinstance(signal, (list, tuple, set, dict)):
                sig_token = f"{type(signal).__name__}:{len(signal)}"
            elif hasattr(signal, "shape"):
                sig_token = f"{type(signal).__name__}:{tuple(signal.shape)}"
            else:
                sig_token = type(signal).__name__
            m.update(str(sig_token).encode())
            entries = cls._filtered_entries(resolved, files_only, dirs_only)
            for e in entries:
                full = os.path.join(resolved, e)
                st = os.stat(full)
                m.update(f"{e}|{st.st_size}|{st.st_mtime_ns}|{int(os.path.isdir(full))}".encode())
            return m.hexdigest()
        except Exception:
            return float("nan")


class SmartVideoTrim:
    """
    Trims static frames from the start of an assembled film and returns
    IMAGE + AUDIO ready to pipe directly into VHS Video Combine.

    Two guards prevent over-trimming:
      Motion guard   — stops at the first frame where movement exceeds
                       motion_threshold (mean abs pixel diff in [0,1]).
      Dialogue guard — stops at the first frame whose audio RMS exceeds
                       dialogue_threshold, so speech is never cut.

    After trimming:
      • Audio is clipped/padded to exactly match the video frame count.
      • A fade-in of the same duration as the stripped section is applied
        to smooth the audio transition.
      • A length-match report is printed to the console.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images":             ("IMAGE",),
                "audio":              ("AUDIO",),
                "fps":                ("FLOAT", {"default": 24.0,  "min": 1.0,  "max": 120.0, "step": 0.01,
                                                  "tooltip": "Frame rate of the clip"}),
                "max_strip_seconds":  ("FLOAT", {"default": 1.0,   "min": 0.0,  "max": 10.0,  "step": 0.05,
                                                  "tooltip": "Hard upper limit on how much to strip (seconds)"}),
                "motion_threshold":   ("FLOAT", {"default": 0.02,  "min": 0.0,  "max": 1.0,   "step": 0.001,
                                                  "tooltip": "Mean abs frame diff to call 'motion'. ~0.01 = very sensitive, ~0.05 = relaxed"}),
                "dialogue_threshold": ("FLOAT", {"default": 0.015, "min": 0.0,  "max": 1.0,   "step": 0.001,
                                                  "tooltip": "Audio RMS per frame to call 'dialogue'. ~0.01 = whisper, ~0.05 = normal speech"}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "AUDIO", "INT",              "FLOAT")
    RETURN_NAMES  = ("images", "audio", "stripped_frames", "stripped_seconds")
    FUNCTION      = "process"
    CATEGORY      = "sjnodes/Video"

    @staticmethod
    def _exact_audio_samples(n_frames: int, fps: float, sample_rate: int) -> int:
        """Exact number of audio samples for n_frames at fps/sample_rate."""
        return round(n_frames * sample_rate / fps)

    def process(self, images, audio, fps, max_strip_seconds, motion_threshold, dialogue_threshold):
        waveform    = audio["waveform"]   # (batch, channels, samples)
        sample_rate = audio["sample_rate"]
        n_frames    = images.shape[0]

        # ── Input length check ───────────────────────────────────────────────────
        expected_samples = self._exact_audio_samples(n_frames, fps, sample_rate)
        actual_samples   = waveform.shape[-1]
        drift_frames     = (actual_samples - expected_samples) / (sample_rate / fps)
        if abs(drift_frames) > 0.5:
            print(
                f"[SmartVideoTrim] ⚠️  Input length mismatch: "
                f"{n_frames} video frames → {expected_samples} expected audio samples, "
                f"got {actual_samples} ({drift_frames:+.2f} frames drift)"
            )
        else:
            print(
                f"[SmartVideoTrim] Input: {n_frames} frames, "
                f"{actual_samples} audio samples  ✓ lengths match"
            )

        samples_per_frame = sample_rate / fps
        max_strip_frames  = min(int(max_strip_seconds * fps), max(n_frames - 1, 0))

        # ── Motion guard ─────────────────────────────────────────────────────────
        first_motion = max_strip_frames
        for i in range(min(max_strip_frames, n_frames - 1)):
            diff = torch.mean(torch.abs(images[i + 1].float() - images[i].float())).item()
            if diff >= motion_threshold:
                first_motion = i
                break

        # ── Dialogue guard ───────────────────────────────────────────────────────
        first_dialogue = max_strip_frames
        for i in range(min(max_strip_frames, n_frames)):
            s0 = int(i       * samples_per_frame)
            s1 = int((i + 1) * samples_per_frame)
            if s1 > waveform.shape[-1]:
                break
            rms = torch.sqrt(torch.mean(waveform[:, :, s0:s1] ** 2)).item()
            if rms >= dialogue_threshold:
                first_dialogue = i
                break

        # ── Strip decision ───────────────────────────────────────────────────────
        strip_frames = max(0, min(first_motion, first_dialogue, max_strip_frames))
        stripped_seconds = strip_frames / fps

        print(
            f"[SmartVideoTrim] motion_at={first_motion}f  "
            f"dialogue_at={first_dialogue}f  "
            f"→ stripping {strip_frames}f ({stripped_seconds:.3f}s)"
        )

        if strip_frames == 0:
            # Still enforce exact length match on the way out
            out_samples = self._exact_audio_samples(n_frames, fps, sample_rate)
            out_waveform = self._fit_audio(waveform, out_samples)
            print(f"[SmartVideoTrim] Output: {n_frames} frames, {out_samples} audio samples — no trim needed")
            return (images, {"waveform": out_waveform, "sample_rate": sample_rate}, 0, 0.0)

        # ── Trim video ───────────────────────────────────────────────────────────
        trimmed_images   = images[strip_frames:]
        out_frames       = trimmed_images.shape[0]

        # ── Trim + fade-in + exact-length audio ─────────────────────────────────
        strip_samples = round(strip_frames * samples_per_frame)
        out_samples   = self._exact_audio_samples(out_frames, fps, sample_rate)

        trimmed_waveform = waveform[:, :, strip_samples:].clone()

        # Fade in over the stripped duration (= smooth crossfade into the clip)
        fade_len = min(strip_samples, trimmed_waveform.shape[-1])
        if fade_len > 0:
            fade = torch.linspace(0.0, 1.0, fade_len, device=trimmed_waveform.device)
            trimmed_waveform[:, :, :fade_len] = trimmed_waveform[:, :, :fade_len] * fade

        # Enforce exact sample count so VHS sees perfectly matched lengths
        trimmed_waveform = self._fit_audio(trimmed_waveform, out_samples)

        print(
            f"[SmartVideoTrim] Output: {out_frames} frames, {out_samples} audio samples  ✓"
        )

        return (
            trimmed_images,
            {"waveform": trimmed_waveform, "sample_rate": sample_rate},
            strip_frames,
            stripped_seconds,
        )

    @staticmethod
    def _fit_audio(waveform: torch.Tensor, target_samples: int) -> torch.Tensor:
        """Clip or zero-pad waveform to exactly target_samples."""
        current = waveform.shape[-1]
        if current == target_samples:
            return waveform
        if current > target_samples:
            return waveform[:, :, :target_samples]
        # pad with silence
        pad = torch.zeros(
            (*waveform.shape[:-1], target_samples - current),
            dtype=waveform.dtype, device=waveform.device
        )
        return torch.cat([waveform, pad], dim=-1)
