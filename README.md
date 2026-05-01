# ComfyUI-ControlFoley

ControlFoley integration for ComfyUI — generate **synchronized foley sound effects** from video, images, and text prompts.

Based on the [ControlFoley](https://github.com/xiaomi-research/controlfoley) project by Xiaomi Research.

## 功能概述

ControlFoley 是一个视频到音频的拟音生成模型，可以为无声视频生成时间同步的音效（如脚步声、关门声、键盘敲击等）。该 ComfyUI 节点完整复现了 ControlFoley 的所有能力：

- **视频到音效**: 输入无声视频，生成与视频内容时间同步的音效
- **图片到音效**: 输入单张图片 + 可选的文本描述，生成对应音效
- **文本到音效**: 仅通过文本描述生成音效
- **参考音色控制**: 通过参考音频控制生成音效的音色风格
- **多模态控制**: 同时使用视频、文本、音频进行联合控制

## 安装

### 前置条件

- ComfyUI 已安装
- Python >= 3.10
- CUDA GPU (推荐至少 8GB VRAM)

### 1. 安装依赖

```bash
cd ComfyUI/custom_nodes/comfyui-controlfoley
pip install -r requirements.txt
```

**注意**: 
- `audiocraft` 需要从源码安装:
  ```bash
  cd ComfyUI/custom_nodes/comfyui-controlfoley/lib/audiocraft
  pip install -e .
  ```
- Windows 用户需要跳过 `flash-attn`:
  ```bash
  pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
  ```

### 2. 下载模型权重

将模型权重放在 `ComfyUI/models/controlfoley/` 目录下，结构如下：

```
ComfyUI/models/controlfoley/
├── weights/
│   └── controlfoley.pth          # 主模型权重
├── tod_vae_44k.pth                # TOLD VAE 编码器/解码器
├── bigvgan_vocoder_44k.pth        # BigVGAN 声码器
├── synchformer.pth                # Synchformer 模型
├── cav_mae.pth                    # CAV-MAE 模型
├── laion_clap/                    # LAION-CLAP 模型文件夹
│   ├── config.yml
│   └── music_speech_audioset_epoch_15_esc_89.98.pt
└── audio_w2vbert/                 # Audio W2V-BERT 模型文件夹
    ├── config.json
    └── model.safetensors
```

### 3. 重启 ComfyUI

安装完成后重启 ComfyUI 服务。

## 节点说明

### 1. ControlFoley Generate 🎬🔊

主生成节点，接受视频帧 + 文本提示 + 参考音频，生成同步音效。

**输入**:
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `prompt` | STRING | 是 | 文本描述要生成的音效 |
| `negative_prompt` | STRING | 是 | 反向提示词 |
| `fps` | FLOAT | 是 | 输入视频的帧率 (默认 25.0) |
| `cfg_strength` | FLOAT | 是 | CFG 引导强度 (默认 4.5) |
| `steps` | INT | 是 | 扩散采样步数 (默认 25) |
| `seed` | INT | 是 | 随机种子 |
| `variant` | 下拉 | 是 | 模型变体 (目前仅 "large_44k") |
| `video` | IMAGE | 否 | 视频帧序列 (B, H, W, C) |
| `reference_audio` | AUDIO | 否 | 参考音频，控制音色风格 |

**输出**:
| 参数 | 类型 | 描述 |
|------|------|------|
| `audio` | AUDIO | 生成的音频波形 |
| `video` | IMAGE | 原始视频帧（可连到保存视频节点） |

### 2. ControlFoley Video Loader 📹

加载视频文件，提取所有帧并返回帧率和帧序列。

**输入**:
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `video_path` | STRING | 是 | 视频文件路径 |

**输出**:
| 参数 | 类型 | 描述 |
|------|------|------|
| `frames` | IMAGE | 视频帧序列 (B, H, W, C) |
| `fps` | FLOAT | 视频帧率 |

### 3. ControlFoley Image to Audio 🖼️➡️🔊

从单张图片生成音效。

**输入**:
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `image` | IMAGE | 是 | 输入图片 |
| `prompt` | STRING | 是 | 文本提示词 |
| `negative_prompt` | STRING | 是 | 反向提示词 |
| `duration` | FLOAT | 是 | 生成音频时长 (默认 8.0s) |
| `cfg_strength` | FLOAT | 是 | CFG 引导强度 (默认 4.5) |
| `steps` | INT | 是 | 扩散采样步数 (默认 25) |
| `seed` | INT | 是 | 随机种子 |

**输出**:
| 参数 | 类型 | 描述 |
|------|------|------|
| `audio` | AUDIO | 生成的音频波形 |
| `sample_rate` | INT | 采样率 (44100) |

### 4. ControlFoley Save Audio 💾🔊

保存 ControlFoley 生成的音频为 FLAC 文件（自包含，不依赖 ComfyUI 内置 SaveAudio）。

**输入**:
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `audio` | AUDIO | 是 | ControlFoley 生成的音频 |
| `filename_prefix` | STRING | 是 | 输出文件名前缀 |

**输出**:
| 参数 | 类型 | 描述 |
|------|------|------|
| `file_path` | STRING | 保存的 FLAC 文件完整路径 |

## 使用示例

### 基本用法：视频生成音效

1. 添加 **ControlFoley Video Loader** 节点，设置视频路径
2. 添加 **ControlFoley Generate** 节点
3. 将 Video Loader 的 `frames` + `fps` 分别连接到 Generate 的 `video` + `fps`
4. 可选：输入文本描述 (如 "dog barking, footsteps")
5. `audio` 连 **ControlFoley Save Audio** 保存音频
6. `video` 帧连你已有的保存视频节点

### 图片生成音效

1. 添加 **LoadImage** 节点加载图片
2. 添加 **ControlFoley Image to Audio** 节点
3. 连接图片并设置文本描述
4. 输出音频到 Save Audio 节点

### 带音色参考的生成

1. 按照基本用法设置视频或图片输入
2. 使用 **LoadAudio** 加载参考音频
3. 将参考音频连接到 Generate 节点的 `reference_audio` 输入
4. 生成结果将具有参考音频的音色特征

## 技术细节

ControlFoley 使用以下技术组件：

- **CLIP (DFN5B-CLIP-ViT-H-14-384)**: 视觉语义理解
- **CAV-MAE**: 音视频联合特征提取
- **Synchformer**: 时间同步特征提取
- **LAION-CLAP**: 音频内容理解
- **MusicGen Style Model**: 音色/风格特征提取
- **TOLD-AE + BigVGAN**: 音频编解码 (梅尔频谱 ↔ 波形)
- **Flow Matching**: 生成建模

## 致谢

- 原始项目: [ControlFoley](https://github.com/xiaomi-research/controlfoley) by Xiaomi Research
- 论文: ControlFoley: Controllable Foley Sound Effect Generation
- ComfyUI: [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## License

本项目基于 MIT 许可证发布，继承自 ControlFoley 项目的协议。

---

## 引用

```bibtex
@article{controlfoley2025,
  title={ControlFoley: Controllable Foley Sound Effect Generation},
  author={Xiaomi Research},
  journal={arXiv preprint},
  year={2025}
}