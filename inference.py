import argparse
import itertools
import logging
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from torchvision import transforms

#from accelerate import Accelerator
##from accelerate.logging import get_logger
#from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel, LMSDiscreteScheduler, ControlNetModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, whoami

from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.models.clip.configuration_clip import CLIPTextConfig
from omegaconf import OmegaConf

from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel

from typing import Optional, Tuple, Union
import datasets
from datasets import load_dataset
from dreamtuner.models.subject_encoder import SubjectEncoder
from dreamtuner.models.unet import SDUNet2DConditionModel
from dreamtuner.models.attention import SETransformerBlock

from dreamtuner.models.utils import freeze_params, unfreeze_params
from dreamtuner.pipelines.pipeline_dreamtuner import DreamTunerPipeline

def main(args):
    height = width = args.resolution

    pipeline = DreamTunerPipeline.from_pretrained(args.model_name_or_path)
    pipeline.set_subject_encoder_beta(args.subject_encoder_beta)
    pipeline = pipeline.to(args.device)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=args.device).manual_seed(args.seed)

    images = []
    prompt = args.prompt
    nagetive_prompt = args.negative_prompt
    reference_image = Image.open(args.reference_image).convert("RGB")
    num_inference_steps = args.num_inference_steps
    guidance_scale = args.guidance_scale
    
    for _ in range(args.num_samples):
        image = pipeline(
            prompt, height, width, 
            negative_prompt=nagetive_prompt,
            reference_images=reference_image,  
            num_inference_steps=num_inference_steps, 
            generator=generator, 
            guidance_scale=guidance_scale
        ).images[0]

        images.append(image)

    # Save images
    for i, image in enumerate(images):
        image.save(os.path.join(args.output_dir, f"image_{i}.png"))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="./models/dreamtuner", help="Model name or path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--prompt", type=str, default="1girl", help="Prompt text")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt text")
    parser.add_argument("--reference_image", type=str, default="datasets/sample/00008_rgb.png", help="Reference image path")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Enable xformers memory efficient attention")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--subject_encoder_beta", type=float, default=1.0, help="subject encoder beta")
    
    args = parser.parse_args()
    
    main(args)