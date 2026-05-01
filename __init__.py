"""
ComfyUI-ControlFoley: Video-to-Audio Foley Generation for ComfyUI

基于小米研究院 ControlFoley 项目 (https://github.com/xiaomi-research/controlfoley)
实现视频/图像/文本到音频的拟音音效生成。

Nodes:
    - ControlFoleyGenerate: 主生成节点，支持视频+文本+参考音频输入
    - ControlFoleyVideoLoader: 视频文件加载和预处理节点
    - ControlFoleyImageToAudio: 单张图片生成音频节点
"""

import os
import sys
from pathlib import Path

# Ensure the package directory is in sys.path for imports
PACKAGE_DIR = Path(__file__).parent.resolve()
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

# Import node mappings
from .controlfoley_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Package metadata
__version__ = "1.0.0"
__author__ = "ComfyUI Community"
__description__ = "ControlFoley integration for ComfyUI - Generate synchronized foley sound effects from video, text, and reference audio"