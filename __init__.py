from .inpaint_cropandstitch_gpu import InpaintCropImproved, InpaintStitchImproved, SaveStitcherToFile, LoadStitcherFromFile, SmoothTemporalMask, CrossFadeVideo
from .nodes import WanVideoVACEExtend

WEB_DIRECTORY = "js"

NODE_CLASS_MAPPINGS = {
    "InpaintCropImprovedGPU": InpaintCropImproved,
    "InpaintStitchImprovedGPU": InpaintStitchImproved,
    "SaveStitcherToFile": SaveStitcherToFile,
    "LoadStitcherFromFile": LoadStitcherFromFile,
    "SmoothTemporalMask" : SmoothTemporalMask,
    "CrossFadeVideo" : CrossFadeVideo,
    "WanVideoVACEExtend" : WanVideoVACEExtend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InpaintCropImprovedGPU": "✂️ Inpaint Crop (GPU)",
    "InpaintStitchImprovedGPU": "✂️ Inpaint Stitch (GPU)",
    "SaveStitcherToFile": "💾 Save Stitcher to File",
    "LoadStitcherFromFile": "📂 Load Stitcher from File",
    "SmoothTemporalMask" : "SmoothTemporalMask",
    "CrossFadeVideo" : "CrossFadeVideo",
    "WanVideoVACEExtend" : "WanVideoVACEExtend",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']