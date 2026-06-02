from .inpaint_cropandstitch_gpu import InpaintCropImproved, InpaintStitchImproved, SaveStitcherToFile, LoadStitcherFromFile, SmoothTemporalMask, CrossFadeVideo
from .nodes import (
    WanVideoVACEExtend,
    LogCRec709Convert,
    AudioStackWithVolume,
    FixedLengthAudioSequencer,
    PadAudioToLength,
    JsonArrayExtractor,
    LoadImageFromPath,
    SmartVideoTrim,
    SJAlwaysChanged,
    SJPathListDirLive,
)
from .distributed_batch import SJDistributedImageBatch

WEB_DIRECTORY = "js"

NODE_CLASS_MAPPINGS = {
    "InpaintCropImprovedGPU": InpaintCropImproved,
    "InpaintStitchImprovedGPU": InpaintStitchImproved,
    "SaveStitcherToFile": SaveStitcherToFile,
    "LoadStitcherFromFile": LoadStitcherFromFile,
    "SmoothTemporalMask" : SmoothTemporalMask,
    "CrossFadeVideo" : CrossFadeVideo,
    "WanVideoVACEExtend" : WanVideoVACEExtend,
    "LogCRec709Convert" : LogCRec709Convert,
    "AudioStackWithVolume": AudioStackWithVolume,
    "FixedLengthAudioSequencer": FixedLengthAudioSequencer,
    "PadAudioToLength": PadAudioToLength,
    "JsonArrayExtractor": JsonArrayExtractor,
    "LoadImageFromPath": LoadImageFromPath,
    "SmartVideoTrim": SmartVideoTrim,
    "SJAlwaysChanged": SJAlwaysChanged,
    "SJPathListDirLive": SJPathListDirLive,
    "SJDistributedImageBatch": SJDistributedImageBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InpaintCropImprovedGPU": "✂️ Inpaint Crop (GPU)",
    "InpaintStitchImprovedGPU": "✂️ Inpaint Stitch (GPU)",
    "SaveStitcherToFile": "💾 Save Stitcher to File",
    "LoadStitcherFromFile": "📂 Load Stitcher from File",
    "SmoothTemporalMask" : "SmoothTemporalMask",
    "CrossFadeVideo" : "CrossFadeVideo",
    "WanVideoVACEExtend" : "WanVideoVACEExtend",
    "LogCRec709Convert" : "LogCRec709Convert",
    "AudioStackWithVolume": "🎵 Audio Stack with Volume",
    "FixedLengthAudioSequencer": "📏 Fixed-Length Audio Sequencer",
    "PadAudioToLength": "🧘 Pad Audio to Length (ms)",
    "JsonArrayExtractor": "📋 JSON Array Extractor",
    "LoadImageFromPath": "🖼️ Load Images from Path (with fallback)",
    "SmartVideoTrim": "✂️ Smart Video Trim",
    "SJAlwaysChanged": "🔁 Always Changed Trigger",
    "SJPathListDirLive": "📁 List Directory (Live)",
    "SJDistributedImageBatch": "🛰️ SJ Distributed Image Batch",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
