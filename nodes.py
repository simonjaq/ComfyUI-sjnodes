#V8
import torch
import torch.nn.functional as F
import math
from comfy.utils import common_upscale
import comfy.model_management as mm

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
    

class LogCRec709Convert:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "direction": (["LogC3 to Rec709", "Rec709 to LogC3"], {"default": "LogC3 to Rec709"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("converted_image",)
    FUNCTION = "convert"
    CATEGORY = "sjnodes/color"

    def convert(self, image, direction):
        # Ensure input is float32 and in range [0, 1]
        image = image.clamp(0, 1).float()
        device = image.device

        def logc_to_lin(v):
            a = 5.555556
            b = 0.047996
            c = 0.244161
            d = 0.386036
            return torch.where(v > d, torch.exp((v - c) / a) - b, (v - 0.092809) / 5.367655)

        def lin_to_logc(v):
            a = 5.555556
            b = 0.047996
            c = 0.244161
            d = 0.010591
            return torch.where(v > d, a * torch.log(v + b) + c, v * 5.367655 + 0.092809)

        def lin_to_rec709(v):
            gamma = 1 / 2.4
            return torch.clamp(v, 0, 1) ** gamma

        def rec709_to_lin(v):
            gamma = 2.4
            return torch.clamp(v, 0, 1) ** gamma

        if direction == "LogC3 to Rec709":
            linear = logc_to_lin(image)
            converted = lin_to_rec709(linear)
        else:  # Rec709 to LogC3
            linear = rec709_to_lin(image)
            converted = lin_to_logc(linear)

        return (converted.clamp(0, 1),)
