import argparse
import itertools
import logging
import math
import os
from pathlib import Path
from dotenv import load_dotenv
import random


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from torchvision import transforms

import transformers
import diffusers

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration
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

from dreamtuner.data.dataset import CustomDataset


load_dotenv()

if is_wandb_available():
    import wandb
    wandb.login()

logger = get_logger(__name__)


def log_validation(accelerator, vae, text_encoder, tokenizer, unet, controlnet, subject_encoder, scheduler, args):
    logger.info("Running validation... ")

    unet = accelerator.unwrap_model(unet)
    subject_encoder = accelerator.unwrap_model(subject_encoder)
    
    
    height = width = args.resolution

    pipeline = DreamTunerPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        subject_encoder=subject_encoder,
        scheduler=scheduler,
    )
    # pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    # pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()
    if args.seed is None:
        generator = None
    else:
        # generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
        generator = torch.Generator(device=args.device).manual_seed(args.seed)

    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        reference_images = args.validation_reference_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        reference_images = args.validation_reference_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_reference_image
        reference_images = args.reference_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )
        
    print("validation_images", validation_images, "reference_images", reference_images, "validation_prompts", validation_prompts)

    image_logs = []

    for validation_prompt, validation_image, reference_image in zip(validation_prompts, validation_images, reference_images):
        validation_image = Image.open(validation_image).convert("RGB")
        reference_image = Image.open(reference_image).convert("RGB")

        images = []

        for _ in range(args.num_validation_images):
            with torch.autocast("cuda"):
                image = pipeline(
                    validation_prompt, height, width, reference_images=reference_image,  num_inference_steps=50, generator=generator, guidance_scale=7.5 # controlnet_images=validation_image,
                ).images[0]

            images.append(image)

        image_logs.append(
            {"validation_image": validation_image, "reference_image": reference_image, "images": images, "validation_prompt": validation_prompt}
        )
        
    log = {
        "images" : [],
        "validation_images": [],
        "reference_images": [],
    }
        
    for index, il in  enumerate(image_logs):
        for i, image in enumerate(il["images"]):
            log["images"].append(wandb.Image(image, caption=il["validation_prompt"]))

        log["validation_images"].append(wandb.Image(il["validation_image"], caption="Validation image"))
        log["reference_images"].append(wandb.Image(il["reference_image"], caption="Reference image"))

    accelerator.log(log)

    del pipeline

    return image_logs

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--pretrained_subejct_encoder_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help=(
            "device"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5000,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--validation_reference_image",
        type=str,
        default=None,
        nargs="+",
        help="reference image for validation",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_dreamtuner",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "config"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "seed"
        ),
    )
    # enable mask for loss
    parser.add_argument(
        "--enable_mask",
        action="store_true",
        help="enable mask for loss",
    )
    
    args = parser.parse_args()
    # env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    # if env_local_rank != -1 and env_local_rank != args.local_rank:
    #     args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args

def main(
        args
    ):
    
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    
    
    accelerator.init_trackers(
        project_name="dreamtuner", 
        config=args,
        init_kwargs={"wandb": { 
            "group" : "group1",
            "notes" : "subject encoder training",
        }}
    )
    
    
    
    print(args.validation_image)
    print(args.validation_prompt)
    
    # is_main_process = True

    # accelerator = Accelerator(
    #     gradient_accumulation_steps=args.gradient_accumulation_steps,
    #     mixed_precision=args.mixed_precision,
    #     log_with="tensorboard",
    #     logging_dir=logging_dir,
    # )

    # # Handle the repository creation
    # if accelerator.is_main_process:
    #     if args.push_to_hub:
    #         if args.hub_model_id is None:
    #             repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
    #         else:
    #             repo_name = args.hub_model_id
    #         repo = Repository(args.output_dir, clone_from=repo_name)

    #         with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
    #             if "step_*" not in gitignore:
    #                 gitignore.write("step_*\n")
    #             if "epoch_*" not in gitignore:
    #                 gitignore.write("epoch_*\n")
    #     elif args.output_dir is not None:
    #         os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
        
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        # if args.push_to_hub:
        #     repo_id = create_repo(
        #         repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
        #     ).repo_id   
    
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    
    unet_additional_kwargs = {}
    if args.pretrained_subejct_encoder_model_name_or_path is not None:
        print("loading pretrained SDUNet2DConditionModel. ", args.pretrained_subejct_encoder_model_name_or_path)
        unet = SDUNet2DConditionModel.from_pretrained(args.pretrained_subejct_encoder_model_name_or_path, subfolder="unet")
    else:
        unet = SDUNet2DConditionModel.from_pretrained_vanilla(args.pretrained_model_name_or_path, subfolder="unet")
        
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    
    if args.pretrained_subejct_encoder_model_name_or_path is not None:
        print("loading pretrained SubjectEncoder. ", args.pretrained_subejct_encoder_model_name_or_path)
        subject_encoder = SubjectEncoder.from_pretrained(args.pretrained_subejct_encoder_model_name_or_path, subfolder="subject_encoder")
    else:
        subject_encoder = SubjectEncoder()    
        
   
    # initialize SDUNet2DConditionModel with pretrained IPAdapter weights
    if args.pretrained_subejct_encoder_model_name_or_path is None and args.pretrained_ip_adapter_model_name_or_path is not None:
        print("loading pretrained IPAdapter. ", args.pretrained_ip_adapter_model_name_or_path)
        ip_adapter_state_dict = torch.load(args.pretrained_ip_adapter_model_name_or_path, map_location="cpu")
        print(ip_adapter_state_dict["ip_adapter"].keys())
        print(ip_adapter_state_dict["image_proj"].keys())
        
        unet_sd = unet.state_dict()
        key_id = 1
        ip_attn_state_dict = ip_adapter_state_dict["ip_adapter"] # f"{key_id}.to_k_ip.weight" 
        for name in unet.attn_processors.keys():
            print("key_id", key_id)
            weight_k_ip  = ip_attn_state_dict[f"{key_id}.to_k_ip.weight"]
            weight_v_ip  = ip_attn_state_dict[f"{key_id}.to_v_ip.weight"]
            
            cross_attention_dim = None if name.endswith("attn1.processor") or name.endswith("se_attn.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
                
            if cross_attention_dim is not None:
                base_layer_name = name.split(".attn2.processor")[0]
                print(base_layer_name)
                attn1_layer_name = base_layer_name + ".attn1"
                attn2_layer_name = base_layer_name + ".attn2"
                norm2_layer_name = base_layer_name + ".norm2"
                           
                se_norm_layer_name = base_layer_name + ".se_norm"
                se_attn_layer_name = base_layer_name + ".se_attn"
                
                # show unet_sd keys that starts with base_layer_name
                # print("unet_sd keys", [k for k in unet_sd.keys() if k.startswith(base_layer_name)])
                
                unet_sd.update({
                    f"{se_norm_layer_name}.weight" : unet_sd[norm2_layer_name + ".weight"],
                    f"{se_norm_layer_name}.bias" : unet_sd[norm2_layer_name + ".bias"],
                    f"{se_attn_layer_name}.to_q.weight" : unet_sd[attn2_layer_name + ".to_q.weight"],
                    f"{se_attn_layer_name}.to_k.weight" : weight_k_ip,
                    f"{se_attn_layer_name}.to_v.weight" : weight_v_ip,
                })     

                key_id += 2
        
        unet.load_state_dict(unet_sd)
        # initialize subject_encoder with pretrained IPAdapter weights
        # subject_encoder_sd = subject_encoder.state_dict()
        # image_proj_state_dict = ip_adapter_state_dict["image_proj"] # f"{key_id}.to_k_ip.weight" 
        # print("subject_encoder_sd keys", subject_encoder_sd.keys())
        # subject_encoder_sd.update({
        #             f"mapper.proj.weight" : image_proj_state_dict["proj.weight"],
        #             f"mapper.proj.bias" : image_proj_state_dict["proj.bias"],
        #             f"mapper.out_norm.weight" : image_proj_state_dict["norm.weight"],
        #             f"mapper.out_norm.bias" : image_proj_state_dict["norm.bias"],
        #         })    
        
        # subject_encoder.load_state_dict(subject_encoder_sd)
        
        # exit()
  
    

    # Freeze vae and unet, encoder
    #freeze_params(vae.parameters())
    #freeze_params(unet.parameters())
    # freeze_params(tokenizer.parameters())
    #freeze_params(text_encoder.parameters())
    #freeze_params(subject_encoder.parameters())
    #freeze_params(controlnet.parameters())
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    subject_encoder.requires_grad_(False)
    controlnet.requires_grad_(False)
    
    
    for name, module in unet.named_modules():
        if(isinstance(module, SETransformerBlock)):
            # print("unfreezing", name)
            unfreeze_params(module.se_norm.parameters())
            unfreeze_params(module.se_attn.parameters())
    # Unfreeze the mapper
    unfreeze_params(subject_encoder.mapper.parameters())

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        
    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
        
    unet_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    subject_encoder_params = list(filter(lambda p: p.requires_grad, subject_encoder.parameters()))
    params_to_opt = itertools.chain(unet_params,  subject_encoder_params)
    optimizer = optimizer_class(
        params=params_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )    
    
    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")
    
     # Enable xformers
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        subject_encoder.enable_gradient_checkpointing()
    
    train_dataset = CustomDataset(args.train_data_dir, tokenizer, enable_mask=args.enable_mask)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # Prepare everything with our `accelerator`.
    unet, subject_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, subject_encoder, optimizer, train_dataloader, lr_scheduler
    )
    
    vae.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
        

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initialize automatically on the main process.
    # if accelerator.is_main_process:
    #     accelerator.init_trackers("elite", config=vars(args))

    # Train!
    # total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    global_step = 0
    first_epoch = 0

   # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        subject_encoder.train()
        
        # parameter log 
        # print("subject_encoder.mapper.proj.weight", accelerator.unwrap_model(subject_encoder).mapper.proj.weight.detach().cpu().numpy())
        # for name, module in accelerator.unwrap_model(unet).named_modules():
        #     if(isinstance(module, SETransformerBlock)):
        #         print("first module.se_attn.to_q.weight", module.se_attn.to_q.weight.detach().cpu().numpy())
        #         print("first module.se_attn.to_out.weight", module.se_attn.to_out[0].weight.detach().cpu().numpy())
        #         break
        
        for batch in train_dataloader:
            with accelerator.accumulate([unet, subject_encoder]):            
                with torch.no_grad():                
                    input_ids = batch["text_input_ids"].to(device=accelerator.device)
                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(input_ids)[0]
                    
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    
                # subject_encoder
                with torch.no_grad():
                    ref_image_embedding = subject_encoder.encode_image(batch["reference_pixel_values"])
                    
                
                # mask 
                if args.enable_mask:
                    mask_images = batch["mask"].to(dtype=weight_dtype).to(args.device)
                    mask_images = mask_images.repeat(1, latents.shape[1], 1, 1) # to latent channels
                    
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)         
                
                controlnet_images = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
                
                # drop image embeds
                ref_image_embeds_ = []               
                controlnet_images_ = []
                if args.enable_mask:
                     mask_images_ = []
                else:
                    mask_images = range(bsz)
                    
                assert len(ref_image_embedding) == len(mask_images) == len(controlnet_images) == len(batch["drop_image_embed"])     
                     
                for ref_image_embed, mask_image, controlnet_imag, drop_image_embed in zip(ref_image_embedding, mask_images, controlnet_images, batch["drop_image_embed"]):
                    if drop_image_embed == 1:
                        ref_image_embeds_.append(torch.zeros_like(ref_image_embed))
                        controlnet_images_.append(torch.zeros_like(controlnet_imag))
                        if args.enable_mask:
                            mask_images_.append(torch.ones_like(mask_image))
                    else:
                        ref_image_embeds_.append(ref_image_embed)                        
                        controlnet_images_.append(controlnet_imag)
                        if args.enable_mask:
                            mask_images_.append(mask_image)
                        
                ref_image_embedding = torch.stack(ref_image_embeds_, dim=0)
                controlnet_images = torch.stack(controlnet_images_, dim=0)
                if args.enable_mask: 
                    mask_images = torch.stack(mask_images_, dim=0)
                    
                subject_encoder_embedding = subject_encoder(ref_image_embedding)
                

                with torch.no_grad():
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=controlnet_images,
                        return_dict=False,
                    )

                # Predict the noise residual
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    subject_encoder_hidden_states=subject_encoder_embedding,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.enable_mask:                
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    resized_mask = F.interpolate(mask_images.float(), size=(loss.shape[-2], loss.shape[-1]), mode='bilinear', align_corners=False)
                    loss = (resized_mask * loss).mean() / resized_mask.mean()
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), subject_encoder.parameters())
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                # for removing_checkpoint in removing_checkpoints:
                                #     removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                #     shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.unwrap_model(unet).save_pretrained(os.path.join(save_path, "unet"))
                        accelerator.unwrap_model(subject_encoder).save_pretrained(os.path.join(save_path, "subject_encoder"))
                        logger.info(f"Saved state to {save_path}")

                    if global_step == 1 or (global_step % args.validation_steps == 0):
                        image_logs = log_validation(
                            accelerator,
                            vae,
                            text_encoder,
                            tokenizer,
                            unet,
                            controlnet,
                            subject_encoder,
                            noise_scheduler,
                            args,
                        )
                        
                        # save images in image_logs to output_dir
                        save_path = os.path.join(args.output_dir, f"validation-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        for i, log in enumerate(image_logs):
                            validation_image = log["validation_image"]
                            reference_image = log["reference_image"]
                            images = log["images"]
                            validation_prompt = log["validation_prompt"]
                            
                            validation_image.save(os.path.join(save_path, f"validation_image_{i}.png"))
                            reference_image.save(os.path.join(save_path, f"reference_image_{i}.png"))
                            
                            for j, image in enumerate(images):
                                image.save(os.path.join(save_path, f"image_{i}_{j}.png"))
                                
                        # parameter log 
                        # print("subject_encoder.mapper.proj.weight", subject_encoder.mapper.proj.weight.detach().cpu().numpy())
                        # for name, module in unet.named_modules():
                        #     if(isinstance(module, SETransformerBlock)):
                        #         print("first module.se_attn.to_q.weight", module.se_attn.to_q.weight.detach().cpu().numpy())
                        #         print("first module.se_attn.to_out.weight", module.se_attn.to_out[0].weight.detach().cpu().numpy())
                        #     break
                        
            logs = {'epoch': epoch, "loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            accelerator.log(logs, step=global_step)
            logger.info(logs)

            if global_step >= args.max_train_steps:
                break
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Training completed.")
        
        # save last checkpoint
        save_path = os.path.join(args.output_dir, f"checkpoint-last")
        accelerator.unwrap_model(unet).save_pretrained(os.path.join(save_path, "unet"))
        accelerator.unwrap_model(subject_encoder).save_pretrained(os.path.join(save_path, "subject_encoder"))
        logger.info(f"Saved state to {save_path}")
        
    accelerator.end_training()

if __name__ == "__main__":
    
    args = parse_args()
    # config = OmegaConf.load(args.config)
    #print(config)
    main(args) # **config)