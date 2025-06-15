import os
import torch
from comfy.utils import load_torch_file, ProgressBar, common_upscale
import comfy.model_management as mm
import math

class WanVideoVACEExtend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("WANVAE",),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 480, "min": 64, "max": 29048, "step": 8}),
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 1}),
                "overlap": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                "vace_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "vace_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "previous_latents": ("LATENT",),
            },
            "optional": {
                "input_frames": ("IMAGE",),
                "ref_images": ("IMAGE",),
                "input_masks": ("MASK",),
                "tiled_vae": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("vace_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, vae, width, height, num_frames, overlap, strength,
                vace_start_percent, vace_end_percent, previous_latents,
                input_frames=None, ref_images=None, input_masks=None,
                tiled_vae=False):

        self.device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        self.vae = vae.to(self.device)
        self.vae_stride = (4, 8, 8)

        width = (width // 16) * 16
        height = (height // 16) * 16

        target_shape = (
            16,
            (num_frames - 1) // self.vae_stride[0] + 1,
            height // self.vae_stride[1],
            width // self.vae_stride[2],
        )

        # Split overlap from previous latents
        overlap_latents = [x[:, :, -overlap:, :, :] for x in previous_latents]
        black_masks = [torch.zeros_like(l[0:1]) for l in overlap_latents]  # inactive only
        encoded_masks = self.vace_encode_masks(black_masks, [None] * len(black_masks))
        z0_overlap = [torch.cat([l * (1 - m), l * m], dim=0) for l, m in zip(overlap_latents, encoded_masks)]

        if input_frames is not None:
            # Standard frame encode
            input_frames = input_frames[:num_frames]
            input_frames = common_upscale(input_frames.clone().movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)
            input_frames = input_frames.to(self.vae.dtype).to(self.device).unsqueeze(0).permute(0, 4, 1, 2, 3)
            input_frames = input_frames * 2 - 1

            if input_masks is None:
                input_masks = torch.ones_like(input_frames, device=self.device)
            else:
                input_masks = input_masks[:num_frames]
                input_masks = common_upscale(input_masks.clone().unsqueeze(1), width, height, "nearest-exact", "disabled").squeeze(1)
                input_masks = input_masks.to(self.vae.dtype).to(self.device)
                input_masks = input_masks.unsqueeze(-1).unsqueeze(0).permute(0, 4, 1, 2, 3).repeat(1, 3, 1, 1, 1)

            z0 = self.vace_encode_frames(input_frames, ref_images, masks=input_masks, tiled_vae=tiled_vae)
            m0 = self.vace_encode_masks(input_masks, ref_images)
            z0_frames = [torch.cat([u, v], dim=0) for u, v in zip(z0, m0)]
        else:
            z0_frames = []

        z = z0_overlap + z0_frames
        self.vae.to(offload_device)

        vace_input = {
            "vace_context": z,
            "vace_scale": strength,
            "has_ref": ref_images is not None,
            "num_frames": overlap + (num_frames if input_frames is not None else 0),
            "target_shape": target_shape,
            "vace_start_percent": vace_start_percent,
            "vace_end_percent": vace_end_percent,
            "vace_seq_len": math.ceil((z[0].shape[2] * z[0].shape[3]) / 4 * z[0].shape[1]),
            "additional_vace_inputs": [],
        }

        return (vace_input,)

    def vace_encode_frames(self, frames, ref_images, masks=None, tiled_vae=False):
        if ref_images is None:
            ref_images = [None] * len(frames)
        else:
            assert len(frames) == len(ref_images)

        if masks is None:
            latents = self.vae.encode(frames, device=self.device, tiled=tiled_vae)
        else:
            inactive = [i * (1 - m) for i, m in zip(frames, masks)]
            reactive = [i * m for i, m in zip(frames, masks)]
            inactive = self.vae.encode(inactive, device=self.device, tiled=tiled_vae)
            reactive = self.vae.encode(reactive, device=self.device, tiled=tiled_vae)
            latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)]

        self.vae.model.clear_cache()
        return latents

    def vace_encode_masks(self, masks, ref_images=None):
        if ref_images is None:
            ref_images = [None] * len(masks)

        result_masks = []
        for mask in masks:
            c, d, h, w = mask.shape
            d_new = (d + 3) // self.vae_stride[0]
            h = 2 * (h // (self.vae_stride[1] * 2))
            w = 2 * (w // (self.vae_stride[2] * 2))

            mask = mask[0].view(d, h, 8, w, 8).permute(2, 4, 0, 1, 3).reshape(64, d, h, w)
            mask = F.interpolate(mask.unsqueeze(0), size=(d_new, h, w), mode='nearest-exact').squeeze(0)

            result_masks.append(mask)
        return result_masks
