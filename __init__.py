from .inpaint_cropandstitch_gpu import InpaintCropImproved, InpaintStitchImproved

WEB_DIRECTORY = "js"

NODE_CLASS_MAPPINGS = {
    "InpaintCropImprovedGPU": InpaintCropImproved,
    "InpaintStitchImprovedGPU": InpaintStitchImproved,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InpaintCropImprovedGPU": "✂️ Inpaint Crop (GPU)",
    "InpaintStitchImprovedGPU": "✂️ Inpaint Stitch (GPU)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']