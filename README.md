# ComfyUI_RH_Univideo

<p align="center">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License">
  <img src="https://img.shields.io/badge/Python-3.10+-blue" alt="Python">
  <img src="https://img.shields.io/badge/Platform-ComfyUI-orange" alt="Platform">
</p>

ComfyUI custom nodes for [UniVideo](https://github.com/KlingTeam/UniVideo) - A unified multimodal video generation and editing framework powered by HunyuanVideo and Qwen2.5-VL, developed by Kling Team.

## ✨ Features

- **Video-to-Video Editing (v2v_edit)**: Apply free-form edits to existing videos based on text instructions
- **In-Context Video Editing (i+v2v_edit)**: Use reference images to guide video editing (e.g., face swapping)
- **Integrated with ComfyUI**: Seamless workflow integration with ComfyUI's node-based interface
- **INT8 Quantization**: Optimized memory usage with INT8 quantization support

## 📋 Nodes

| Node Name | Description |
|-----------|-------------|
| `RunningHub Univideo Loader` | Loads the UniVideo pipeline including transformer, VAE, scheduler, and MLLM encoder |
| `RunningHub Univideo Editor` | Performs video editing tasks with optional reference image input |

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

Download from the [UniVideo official repository](https://github.com/KlingTeam/UniVideo)

| Model | Description |
|-------|-------------|
| `univideo_qwen2p5vl7b_hidden_hunyuanvideo` | Main UniVideo model checkpoint |

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
4. (Optional) Connect a reference image to the `ref_image` input for face swapping
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

- **v2v_edit**: Video-to-video editing without reference image
  - Example prompt: "Change the man to look like he is sculpted from chocolate."

- **i+v2v_edit**: In-context video editing with reference image
  - Example prompt: "Use the person's face in the reference image to replace the person's face in the video."

## 💻 System Requirements

- **GPU**: NVIDIA GPU with at least 24GB VRAM (recommended)
- **CUDA**: 11.8 or higher
- **Python**: 3.10+
- **OS**: Windows, Linux, macOS

## 📝 Dependencies

- torch >= 2.0.0
- torchvision >= 0.15.0
- transformers >= 4.40.0
- diffusers >= 0.27.0
- optimum-quanto >= 0.2.0
- decord >= 0.6.0
- einops >= 0.7.0
- opencv-python >= 4.8.0
- PyYAML >= 6.0
- Pillow >= 9.0.0

## 🙏 Acknowledgements

- [UniVideo](https://github.com/KlingTeam/UniVideo) - The original UniVideo framework by Kling Team
- [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) - Base video generation model
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) - Multimodal language model
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - The amazing node-based UI

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This project is for research and educational purposes only. Users are responsible for ensuring their use complies with applicable laws and regulations. The authors are not responsible for any misuse of this software.
