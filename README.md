# ComfyUI_RH_Univideo

<p align="center">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License">
  <img src="https://img.shields.io/badge/Python-3.11-blue" alt="Python">
  <img src="https://img.shields.io/badge/Platform-ComfyUI-orange" alt="Platform">
</p>

ComfyUI custom nodes for [UniVideo](https://github.com/KlingTeam/UniVideo) - A unified framework for video understanding, generation, and editing, developed by **Kling Team (Kuaishou Technology)** and **University of Waterloo**.

> **UniVideo** uses HunyuanVideo as the base video generation model and Qwen2.5-VL as the multimodal language model backbone.
>
> 📄 Paper: [arXiv:2510.08377](https://arxiv.org/abs/2510.08377) | 🌐 Project Page: [congwei1230.github.io/UniVideo](https://congwei1230.github.io/UniVideo/)

## ✨ Features

This ComfyUI node currently supports the following UniVideo tasks:

- **Video-to-Video Editing (v2v_edit)**: Apply free-form edits to existing videos based on text instructions
- **In-Context Video Editing (i+v2v_edit)**: Use reference images to guide video editing (e.g., face/identity swapping)
- **INT8 Quantization**: Optimized memory usage with INT8 quantization via optimum-quanto
- **Seamless ComfyUI Integration**: Node-based workflow for easy use

> **Note**: The original UniVideo framework supports additional tasks including `understanding`, `multiid`, `t2v`, `t2i`, `i2i_edit`, and `i2v`. These may be added in future updates.

## 📋 Nodes

| Node Name | Description |
|-----------|-------------|
| `RunningHub Univideo Loader` | Loads the UniVideo pipeline (Transformer, VAE, Scheduler, MLLM Encoder) |
| `RunningHub Univideo Editor` | Performs video editing with optional reference image for identity transfer |

## 🛠️ Installation

### Method 1: ComfyUI Manager (Recommended)

1. Install [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)
2. Search for `ComfyUI_RH_Univideo` in the manager
3. Click Install

### Method 2: Manual Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/HM-RunningHub/ComfyUI_RH_Univideo.git
cd ComfyUI_RH_Univideo
pip install -r requirements.txt
```

## 📦 Model Downloads

You need to download the following models:

### 1. HunyuanVideo Base Model

Download from 🤗 [hunyuanvideo-community/HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo)

Place in: `ComfyUI/models/HunyuanVideo/`

### 2. UniVideo Checkpoint

Download using the official script from [KlingTeam/UniVideo](https://github.com/KlingTeam/UniVideo):

```bash
python download_ckpt.py
```

This node uses **Variant 1** (`univideo_qwen2p5vl7b_hidden_hunyuanvideo`):
> Image, video, and text inputs are processed by the MLLM, and the final hidden states are fed into the MMDiT backbone.

| Model Variant | Description |
|---------------|-------------|
| `univideo_qwen2p5vl7b_hidden_hunyuanvideo` | Variant 1: MLLM last layer hidden → MMDiT |

Place in: `ComfyUI/models/UniVideo/univideo_qwen2p5vl7b_hidden_hunyuanvideo/model.ckpt`

### 3. Qwen2.5-VL-7B-Instruct

Download from 🤗 [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

Place in: `ComfyUI/models/Qwen/Qwen_Qwen2.5-VL-7B-Instruct/`

### Directory Structure

```
ComfyUI/
└── models/
    ├── HunyuanVideo/
    │   ├── vae/
    │   ├── transformer/
    │   └── scheduler/
    ├── UniVideo/
    │   └── univideo_qwen2p5vl7b_hidden_hunyuanvideo/
    │       └── model.ckpt
    └── Qwen/
        └── Qwen_Qwen2.5-VL-7B-Instruct/
```

## 🚀 Usage

### Basic Workflow

1. Add `RunningHub Univideo Loader` node to load the pipeline
2. Add `RunningHub Univideo Editor` node
3. Connect a video input to the `ref_video` input
4. (Optional) Connect a reference image to the `ref_image` input for identity swapping
5. Enter your editing prompt
6. Run the workflow

### Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `prompt` | Text instruction for editing | - | - |
| `width` | Output video width | 832 | 64-2048 |
| `height` | Output video height | 480 | 64-2048 |
| `num_frames` | Number of output frames | 81 | 1+ (step: 4) |
| `sample_steps` | Denoising steps | 20 | - |
| `fps` | Output frames per second | 24 | - |
| `seed` | Random seed | 42 | 0-2^64 |

### Task Types

- **v2v_edit** (Video-to-Video Editing): Edit video without reference image
  - Example: `"Change the man to look like he is sculpted from chocolate."`

- **i+v2v_edit** (In-Context Video Editing): Edit video with reference image for identity transfer
  - Example: `"Use the man's face in the reference image to replace the man's face in the video."`

## 💻 System Requirements

- **GPU**: NVIDIA GPU with at least 24GB VRAM (recommended)
- **CUDA**: 12.1 or higher (recommended)
- **Python**: 3.11 (tested)
- **PyTorch**: 2.4.1+ with CUDA support
- **OS**: Windows, Linux

## 📝 Dependencies

Core dependencies (aligned with official UniVideo):

- torch >= 2.4.1
- torchvision
- transformers >= 4.51.0
- diffusers >= 0.34.0
- optimum-quanto >= 0.2.0
- decord >= 0.6.0
- einops >= 0.7.0
- opencv-python >= 4.8.0
- PyYAML >= 6.0
- Pillow >= 9.0.0

## 🙏 Acknowledgements

- [UniVideo](https://github.com/KlingTeam/UniVideo) - The original UniVideo framework by Kling Team (Kuaishou Technology)
- [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) - Base video generation model by Tencent
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) - Multimodal language model by Alibaba
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - The powerful node-based UI

## 🌟 Citation

If you use this project, please cite the original UniVideo paper:

```bibtex
@article{wei2025univideo,
  title={UniVideo: Unified Understanding, Generation, and Editing for Videos},
  author={Wei, Cong and Liu, Quande and Ye, Zixuan and Wang, Qiulin and Wang, Xintao and Wan, Pengfei and Gai, Kun and Chen, Wenhu},
  journal={arXiv preprint arXiv:2510.08377},
  year={2025}
}
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This project is for research and educational purposes only. Users are responsible for ensuring their use complies with applicable laws and regulations. The authors are not responsible for any misuse of this software.
