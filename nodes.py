import comfy.utils

import os
import io
import tempfile
import torch
import numpy as np
import yaml
import argparse
from PIL import Image

from diffusers.utils import export_to_video
from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import AutoencoderKLHunyuanVideo
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from .transformer_univideo_hunyuan_video import HunyuanVideoTransformer3DModel, TwoLayerMLP
from .mllm_encoder import MLLMInContext, MLLMInContextConfig
from .pipeline_univideo import UniVideoPipeline, UniVideoPipelineConfig

from .utils import pad_image_pil_to_square, load_model

import folder_paths
try:
    from comfy_api.input_impl.video_types import VideoFromFile
except ImportError:
    VideoFromFile = None
from pathlib import Path
from optimum.quanto import freeze, qint8, quantize 
import uuid

class RunningHub_Univideo_Loader:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # "type": (["V1"], {"default": "V1"}),
            }
        }

    RETURN_TYPES = ('RH_Univideo_Pipeline', )
    RETURN_NAMES = ('Univideo Pipeline', )
    FUNCTION = "load"
    CATEGORY = "RunningHub/Univideo"

    OUTPUT_NODE = True

    def load(self, **kwargs):

        current_file = Path(__file__).resolve()
        current_dir = current_file.parent
        config_path = current_dir / "configs" / "univideo_qwen2p5vl7b_hidden_hunyuanvideo.yaml"

        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)

        if "mllm_config" not in raw:
            raise KeyError("Missing required config section: mllm_config")
        if "pipeline_config" not in raw:
            raise KeyError("Missing required config section: pipeline_config")

        mllm_config = MLLMInContextConfig(**raw["mllm_config"])
        pipe_cfg    = UniVideoPipelineConfig(**raw["pipeline_config"])
        transformer_ckpt_path   = raw.get("transformer_ckpt_path")
        mllm_encoder_ckpt_path  = raw.get("mllm_encoder_ckpt", None)

        # Create MLLM encoder from config
        mllm_encoder = MLLMInContext(mllm_config)

        # Load mllm_encoder checkpoint if provided (queries version requires mllm_encoder ckpt)
        if mllm_encoder_ckpt_path is not None:
            print(f"[INIT] loading mllm_encoder ckpt from {mllm_encoder_ckpt_path}")
            mllm_encoder = load_model(mllm_encoder, mllm_encoder_ckpt_path)
        mllm_encoder.requires_grad_(False)
        mllm_encoder.eval()

        #kiki hardcode
        pipe_cfg.hunyuan_model_id = os.path.join(folder_paths.models_dir, 'HunyuanVideo')
        transformer_ckpt_path = os.path.join(folder_paths.models_dir, 'UniVideo', \
            'univideo_qwen2p5vl7b_hidden_hunyuanvideo', 'model.ckpt')

        # Load HunyuanVideo VAE
        vae = AutoencoderKLHunyuanVideo.from_pretrained(
            pipe_cfg.hunyuan_model_id,
            subfolder="vae", 
            low_cpu_mem_usage=False,  
            # kiki
            torch_dtype=torch.bfloat16,
            device_map=None 
        )
        vae.eval()
        vae.enable_tiling()
        vae.enable_slicing()
            
        # Load HunyuanVideo transformer and reinitialize connector.
        qwenvl_txt_dim = 3584
        # transformer = None
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            pipe_cfg.hunyuan_model_id,
            subfolder="transformer", 
            low_cpu_mem_usage=False,  # Avoid meta tensors
            device_map=None,  # Let us handle device placement manually
            # kiki
            torch_dtype=torch.bfloat16,
            text_embed_dim=qwenvl_txt_dim  # QwenVL 2.5-7B hidden size
        )
        transformer.qwen_project_in = TwoLayerMLP(qwenvl_txt_dim, qwenvl_txt_dim * 4, 4096) 
        with torch.no_grad():
            torch.nn.init.ones_(transformer.qwen_project_in.ln.weight)
            for layer in transformer.qwen_project_in.mlp:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight, gain=1.0)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)
        print(f"[INIT] Reinitialized qwen_project_in ({qwenvl_txt_dim} -> {qwenvl_txt_dim * 4} -> 4096)")
        # Load ckpt
        def rename_func(state_dict):
                new_state_dict = {}
                for k, v in state_dict.items():
                    # remove leading "transformer." if present
                    new_k = k.replace("transformer.", "", 1) if k.startswith("transformer.") else k
                    new_state_dict[new_k] = v
                return new_state_dict
        if isinstance(transformer_ckpt_path, str):
            print(f"[INIT] loading ckpt from {transformer_ckpt_path}")
            transformer = load_model(transformer, transformer_ckpt_path, rename_func=rename_func)
            
        # Load scheduler
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            pipe_cfg.hunyuan_model_id,
            subfolder="scheduler"
        )
        
        # Build Univideo pipeline
        pipeline = UniVideoPipeline(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            mllm_encoder=mllm_encoder,
            univideo_config=pipe_cfg
        )
        # ).to(device="cuda", dtype=torch.bfloat16)
        # ).to(device="cuda", dtype=torch.float32)
        quantize(pipeline.transformer, qint8)
        freeze(pipeline.transformer)
        
        return (pipeline, )

class RunningHub_Univideo_Editor:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("RH_Univideo_Pipeline", ),
                "ref_video": ("VIDEO", ),
                "prompt": ("STRING", {"default": "Use the person's face in the reference image to replace the person's face in the video.", \
                    "multiline": True}),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 8}),
                "num_frames": ("INT", {"default": 81, "min": 1, 'step': 4}),
                "sample_steps": ("INT", {"default": 20,}),
                "fps": ("INT", {"default": 24,}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "ref_image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ('VIDEO',)
    RETURN_NAMES = ('video',)
    FUNCTION = "sample"
    CATEGORY = "RunningHub/Univideo"

    def tensor_2_pil(self, img_tensor):
        i = 255. * img_tensor.squeeze().cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return img

    def get_video_path(self, video_input):
        """Extract video file path from VIDEO input, handling various input types.
        
        If the video is stored as BytesIO (e.g., loaded from URL), it will be saved
        to a temporary file and the path to that file will be returned.
        """
        if video_input is None:
            raise ValueError("Video input is None")
        
        # If it's already a string path, return it
        if isinstance(video_input, str):
            return video_input
        
        # If it's a Path object, convert to string
        if isinstance(video_input, Path):
            return str(video_input)
        
        # If it's a BytesIO object, save to temp file
        if isinstance(video_input, io.BytesIO):
            return self._save_bytesio_to_temp_file(video_input)
        
        # For VideoFromFile objects, use get_stream_source() method
        # This returns either a string path or a BytesIO object
        if hasattr(video_input, 'get_stream_source'):
            source = video_input.get_stream_source()
            print(f"[Univideo] get_stream_source() returned type: {type(source)}")
            
            # If it's a string path, return it directly
            if isinstance(source, str):
                print(f"[Univideo] Using video path: {source}")
                return source
            
            # If it's a Path object, convert to string
            if isinstance(source, Path):
                return str(source)
            
            # If it's a BytesIO object, save to temp file
            if isinstance(source, io.BytesIO):
                print(f"[Univideo] Video is in memory (BytesIO), saving to temp file...")
                return self._save_bytesio_to_temp_file(source)
        
        # Try to access the private __file attribute directly (name mangled)
        # VideoFromFile stores the file as self.__file which becomes _VideoFromFile__file
        private_attr = '_VideoFromFile__file'
        if hasattr(video_input, private_attr):
            file_obj = getattr(video_input, private_attr)
            if isinstance(file_obj, str):
                return file_obj
            if isinstance(file_obj, Path):
                return str(file_obj)
            if isinstance(file_obj, io.BytesIO):
                return self._save_bytesio_to_temp_file(file_obj)
        
        # Fallback: try other common attributes
        for attr_name in ['path', 'file_path', 'filename', 'source', '_path', '_file_path']:
            if hasattr(video_input, attr_name):
                path = getattr(video_input, attr_name)
                if isinstance(path, str) and path:
                    return path
                if isinstance(path, Path):
                    return str(path)
        
        raise ValueError(f"Cannot extract video path from input type: {type(video_input)}, value: {video_input}")
    
    def _save_bytesio_to_temp_file(self, bytesio_obj):
        """Save BytesIO video data to a temporary file and return the path."""
        # Create temp file with .mp4 extension (most common video format)
        temp_dir = folder_paths.get_temp_directory()
        temp_filename = f"univideo_input_{uuid.uuid4()}.mp4"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        # Ensure BytesIO is at the beginning
        bytesio_obj.seek(0)
        
        # Write to temp file
        with open(temp_path, 'wb') as f:
            f.write(bytesio_obj.read())
        
        print(f"[Univideo] Saved video to temp file: {temp_path}")
        return temp_path

    def sample(self, **kwargs):
        
        negative_prompt="Bright tones, overexposed, oversharpening, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, walking backwards, computer-generated environment, weak dynamics, distorted and erratic motions, unstable framing and a disorganized composition."

        pipeline = kwargs.get('pipeline')
        ref_video = kwargs.get('ref_video')
        cond_video_path = self.get_video_path(ref_video)
        print(f"[Univideo] Video path resolved: {cond_video_path} (type: {type(cond_video_path)})")
        width = kwargs.get('width')
        height = kwargs.get('height')
        num_frames = kwargs.get('num_frames')
        sample_steps = kwargs.get('sample_steps')
        seed = kwargs.get('seed') ^ (2 ** 32)
        ref_image = kwargs.get('ref_image', None)
        prompt = kwargs.get('prompt')
        fps = kwargs.get('fps')

        self.pbar = comfy.utils.ProgressBar(sample_steps)

        if ref_image is None:
            task = 'v2v_edit'
        else:
            task = 'i+v2v_edit'

        if task == "i+v2v_edit":
            print('[kiki univideo] -- i+v2v_edit --')
            ref_image_list = [self.tensor_2_pil(ref_image)]
            ref_images_pil_list = [[pad_image_pil_to_square(p.convert("RGB")) for p in ref_image_list]]
            # cond_video_path = "demo/in-context-v2v/id_swap/origin.mp4"
            pipeline_kwargs = dict(
                prompts=[prompt],
                negative_prompt=negative_prompt,
                ref_images=ref_images_pil_list,
                cond_video_path=cond_video_path,
                height=height,
                width=width,
                # num_frames=129,
                num_frames=num_frames,
                num_inference_steps=sample_steps,
                guidance_scale=7.0,
                image_guidance_scale=2.0,
                seed=seed,
                timestep_shift=7.0,
                task=task,
                update_fn=self.update,
            )

        # free form v2v editing
        elif task == "v2v_edit":
            print('[kiki univideo] -- v2v_edit --')
            # cond_video_path = "demo/v2v_edit/video.mp4"
            pipeline_kwargs = dict(
                prompts=[prompt],
                negative_prompt=negative_prompt,
                cond_video_path=cond_video_path,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=sample_steps,
                guidance_scale=7.0,
                image_guidance_scale=2.0,
                seed=seed,
                timestep_shift=7.0,
                task=task,
                update_fn=self.update,
            )
        output_filename = f"univideo_{uuid.uuid4()}.mp4"
        output_path = os.path.join(folder_paths.get_output_directory(), output_filename)

        output = pipeline(**pipeline_kwargs)

        # text output
        if hasattr(output, "text") and output.text is not None:
            for i, text in enumerate(output.text):
                print(f"Output {i}:\n{repr(text)}")
        # image / video output
        elif hasattr(output, "frames"):
            output = output.frames[0]   # (F, H, W, C)
            print(f"data.shape: {output.shape}, type: {type(output)}")
            print(f"min: {output.min()}, max: {output.max()}, dtype: {output.dtype}")

            if hasattr(output, "detach"):
                output = output.detach().cpu().float().numpy()
            F, H, W, C = output.shape
            assert C == 3, f"Expected RGB, got C={C}"
            # ---------- image ----------
            if F == 1:
                img = output[0]  # (H, W, C)
                # normalize if needed
                if img.min() < 0:
                    img = (img + 1.0) / 2.0
                img = (img * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(img).save(output_path)
            # ---------- video ----------
            else:
                export_to_video(output, output_path, fps=fps)
        else:
            raise ValueError(f"Unsupported pipeline output type: {type(output)}")
        return (self.create_video_object(output_path), )

    def create_video_object(self, video_path):
        """Create ComfyUI VIDEO object"""
        if VideoFromFile is not None:
            return VideoFromFile(video_path)
        else:
            # Fallback: return file path as string
            return video_path

    def update(self):
        self.pbar.update(1)

class RunningHub_Univideo_Editor_MultiID(RunningHub_Univideo_Editor):
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("RH_Univideo_Pipeline", ),
                "prompt": ("STRING", {"default": "", \
                    "multiline": True}),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 8}),
                "num_frames": ("INT", {"default": 81, "min": 1, 'step': 4}),
                "sample_steps": ("INT", {"default": 20,}),
                "fps": ("INT", {"default": 24,}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "ref_image1": ("IMAGE", ),
                "ref_image2": ("IMAGE", ),
                "ref_image3": ("IMAGE", ),
            }
        }

    def sample(self, **kwargs):
        
        negative_prompt="Bright tones, overexposed, oversharpening, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, walking backwards, computer-generated environment, weak dynamics, distorted and erratic motions, unstable framing and a disorganized composition."

        pipeline = kwargs.get('pipeline')
        width = kwargs.get('width')
        height = kwargs.get('height')
        num_frames = kwargs.get('num_frames')
        sample_steps = kwargs.get('sample_steps')
        seed = kwargs.get('seed') ^ (2 ** 32)
        ref_image1 = kwargs.get('ref_image1', None)
        ref_image2 = kwargs.get('ref_image2', None)
        ref_image3 = kwargs.get('ref_image3', None)
        ref_image_list = list(filter(lambda x: x is not None, [ref_image1, ref_image2, ref_image3]))
        print(len(ref_image_list))
        if len(ref_image_list) == 0:
            raise ValueError("At least one reference image is required")
        prompt = kwargs.get('prompt')
        fps = kwargs.get('fps')

        self.pbar = comfy.utils.ProgressBar(sample_steps)

        task = 'multiid'
        ref_images_pil_list = [[pad_image_pil_to_square(self.tensor_2_pil(p).convert("RGB")) for p in ref_image_list]]
        pipeline_kwargs = dict(
            prompts=[prompt],
            negative_prompt=negative_prompt,
            ref_images=ref_images_pil_list,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=sample_steps,
            guidance_scale=5.0,
            image_guidance_scale=3.0,
            seed=seed,
            timestep_shift=7.0,
            task=task,
            update_fn=self.update,
        )
        output_filename = f"univideo_{uuid.uuid4()}.mp4"
        output_path = os.path.join(folder_paths.get_output_directory(), output_filename)

        output = pipeline(**pipeline_kwargs)

        # text output
        if hasattr(output, "text") and output.text is not None:
            for i, text in enumerate(output.text):
                print(f"Output {i}:\n{repr(text)}")
        # image / video output
        elif hasattr(output, "frames"):
            output = output.frames[0]   # (F, H, W, C)
            print(f"data.shape: {output.shape}, type: {type(output)}")
            print(f"min: {output.min()}, max: {output.max()}, dtype: {output.dtype}")

            if hasattr(output, "detach"):
                output = output.detach().cpu().float().numpy()
            F, H, W, C = output.shape
            assert C == 3, f"Expected RGB, got C={C}"
            # ---------- image ----------
            if F == 1:
                img = output[0]  # (H, W, C)
                # normalize if needed
                if img.min() < 0:
                    img = (img + 1.0) / 2.0
                img = (img * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(img).save(output_path)
            # ---------- video ----------
            else:
                export_to_video(output, output_path, fps=fps)
        else:
            raise ValueError(f"Unsupported pipeline output type: {type(output)}")
        return (self.create_video_object(output_path), )

NODE_CLASS_MAPPINGS = {
    "RunningHub Univideo Loader": RunningHub_Univideo_Loader,
    "RunningHub Univideo Editor": RunningHub_Univideo_Editor,
    "RunningHub Univideo Generator(MultiID)": RunningHub_Univideo_Editor_MultiID,
}