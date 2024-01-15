# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py

import inspect
import random
from typing import Callable, List, Optional, Union, Dict
from dataclasses import dataclass
import PIL

import numpy as np
import torch
from tqdm import tqdm

from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer


from diffusers import StableDiffusionControlNetPipeline
from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.models import ControlNetModel
from diffusers.loaders import TextualInversionLoaderMixin
from diffusers.pipelines import DiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput 
from diffusers.utils.torch_utils import randn_tensor

from einops import rearrange

from dreamtuner.models.attention_processor import SelfSubjectAttnProcessor

from ..models.unet import SDUNet2DConditionModel
from ..models.subject_encoder import SubjectEncoder
import pdb
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPImageProcessor

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


@dataclass
class DreamTunerPipelineOutput(BaseOutput):
    images: Union[torch.Tensor, np.ndarray]


class DreamTunerPipelineSelfSubject(DiffusionPipeline, TextualInversionLoaderMixin):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: SDUNet2DConditionModel,
        subject_encoder: SubjectEncoder,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        unet_reference: SDUNet2DConditionModel = None,
        controlnet: Union[ControlNetModel, None] = None,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)
        
        if unet_reference is None:
            unet_reference = SDUNet2DConditionModel().from_config(unet.config)
            unet_reference.load_state_dict(unet.state_dict())
            unet_reference.to(dtype=unet.dtype, device=unet.device)
        
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            unet_reference=unet_reference,
            subject_encoder=subject_encoder,
            scheduler=scheduler,
            controlnet=controlnet,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=True
        )
        self.reference_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        self.clip_image_processor = CLIPImageProcessor()
        self.mask_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_grayscale=True, do_normalize=False
        )
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
                
        self.load_self_subject()
        self.unet.set_self_subject_mode(mode="reader")
        self.unet_reference.set_self_subject_mode(mode="writer")

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)


    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        
        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)
            
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt
                
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def decode_latents(self, latents):    
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = shape
                # shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def prepare_reference_latents(self, image, batch_size, num_images_per_prompt, dtype, device, generator=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            init_latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                init_latents = [
                    retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = retrieve_latents(self.vae.encode(image), generator=generator)

            init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        # noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # get latents
        #init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents

        return latents    

    def prepare_control_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        refernce_guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        # subject encoder
        use_reference_guidance = False,
        refrence_guidance_probability = 0.9,
        reference_images: torch.FloatTensor = None,
        refernece_beta : Union[float,None] = 0.5,      
        mask_images: torch.FloatTensor = None,  
        # support controlnet
        controlnet_images: torch.FloatTensor = None,
        #controlnet_image_index: list = [0],
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,

        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        # batch_size = 1 if isinstance(prompt, str) else len(prompt)
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        if negative_prompt is not None:
            negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size 
        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )
        latents_dtype = latents.dtype
        
        if controlnet_images is not None:
            controlnet_images = self.prepare_control_image(
                    image=controlnet_images,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=latents.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode
                )
            
        # latents of reference images for unet_reference
        reference_images_for_unet_ref = self.image_processor.preprocess(reference_images)       
        
        reference_latents = self.prepare_reference_latents(
            reference_images_for_unet_ref,
            batch_size,
            num_images_per_prompt,
            text_embeddings.dtype,
            device,
            generator,
        ) 
    
        
        # height, width = controlnet_images.shape[-2:]
        
        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        delta_t = 1    
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                
                if use_reference_guidance and do_classifier_free_guidance:
                    if random.random() <= refrence_guidance_probability:
                        reference_guidance_r = True # reference guidance
                    else:
                        reference_guidance_r = False # text guidance
                
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # reference_images_input = self.reference_image_processor.preprocess(reference_images, height=height, width=width).to(dtype=latents.dtype)
                clip_reference_image = self.clip_image_processor(images=reference_images, return_tensors="pt").pixel_values
                                               
                if mask_images is not None:
                    mask_images_input = self.mask_image_processor.preprocess(mask_images, height=height, width=width).to(dtype=latents.dtype)
                else:   
                    mask_images_input = torch.ones((1, 1, height, width)).to(dtype=latents.dtype)
                    
                if use_reference_guidance and do_classifier_free_guidance:
                    if reference_guidance_r:
                        uncond_text_embeddings, cond_text_embeddings = text_embeddings.chunk(2)
                        text_embeddings = torch.cat([cond_text_embeddings, cond_text_embeddings])
                    else:
                        pass # use cond and uncond text embeddings
                
                # latents of reference images for unet_reference
                noise = randn_tensor(reference_latents.shape, generator=generator, device=device, dtype=reference_latents.dtype)

                # get reference latents with timestep
                # use t - delta_t and t + delta_t for reference image timpstep
                if use_reference_guidance and do_classifier_free_guidance:
                    t_minus = max(t - delta_t, 0)
                    t_plus = min(t + delta_t, 1000)
                    
                    if reference_guidance_r:
                        noisy_reference_latents_input_minus = self.scheduler.add_noise(reference_latents, noise, t_minus)
                        reference_latents_zero_input = torch.zeros_like(reference_latents)
                        noisy_reference_latents_input = torch.cat([reference_latents_zero_input, noisy_reference_latents_input_minus])
                    else:      
                        noisy_reference_latents_input_minus = self.scheduler.add_noise(reference_latents, noise, t_minus)                  
                        noisy_reference_latents_input_plus = self.scheduler.add_noise(reference_latents, noise, t_plus)
                        noisy_reference_latents_input = torch.cat([noisy_reference_latents_input_plus, noisy_reference_latents_input_minus])
                
                else :
                    noisy_reference_latents_input = self.scheduler.add_noise(reference_latents, noise, t)
                    noisy_reference_latents_input = torch.cat([noisy_reference_latents_input] * 2) if do_classifier_free_guidance else noisy_reference_latents_input
                                
                # subject_encoder
                # reference_images_input = reference_images_input.to(latents.device)
                clip_reference_image = clip_reference_image.to(dtype=latents.dtype).to(latents.device)
                mask_images_input = mask_images_input.to(latents.device)
                
                reference_image_embeds = self.subject_encoder.encode_image(clip_reference_image)                
                subject_encoder_embedding = self.subject_encoder(reference_image_embeds)
                if do_classifier_free_guidance:
                    subject_encoder_embedding = torch.cat([subject_encoder_embedding] * 2)
                    mask_images_input = torch.cat([mask_images_input] * 2)
                                    
                down_block_additional_residuals = mid_block_additional_residual = None
                if (getattr(self, "controlnet", None) != None) and (controlnet_images != None):
                    
                    if guess_mode and do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                        control_model_input = latents
                        control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                        controlnet_prompt_embeds = text_embeddings.chunk(2)[1]
                    else:
                        control_model_input = latent_model_input
                        controlnet_prompt_embeds = text_embeddings
                        

                    # controlnet_cond_shape    = list(controlnet_images.shape)
                    # controlnet_cond_shape[2] = video_length
                    # controlnet_cond = torch.zeros(controlnet_cond_shape).to(latents.device)

                    # controlnet_conditioning_mask_shape    = list(controlnet_cond.shape)
                    # controlnet_conditioning_mask_shape[1] = 1
                    # controlnet_conditioning_mask          = torch.zeros(controlnet_conditioning_mask_shape).to(latents.device)

                    # assert controlnet_images.shape[2] >= len(controlnet_image_index)
                    # controlnet_cond[:,:,controlnet_image_index] = controlnet_images[:,:,:len(controlnet_image_index)]
                    # controlnet_conditioning_mask[:,:,controlnet_image_index] = 1
                    
                    # print("control_model_input.shape", control_model_input.shape)
                    # print("controlnet_prompt_embeds.shape", controlnet_prompt_embeds.shape)
                    # print("controlnet_images.shape", controlnet_images.shape)

                    down_block_additional_residuals, mid_block_additional_residual = self.controlnet(
                        control_model_input, 
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=controlnet_images,
                        # conditioning_mask=controlnet_conditioning_mask,
                        conditioning_scale=controlnet_conditioning_scale,
                        guess_mode=guess_mode,
                        return_dict=False,
                    )
                    
                    if guess_mode and self.do_classifier_free_guidance:
                        # Infered ControlNet only for the conditional batch.
                        # To apply the output of ControlNet to both the unconditional and conditional batches,
                        # add 0 to the unconditional batch to keep it unchanged.
                        down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                        mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])
                        
                # unet reference
                # predict the noise residual
                _ = self.unet_reference(
                    noisy_reference_latents_input,
                    t, 
                    encoder_hidden_states=text_embeddings,
                    subject_encoder_hidden_states=subject_encoder_embedding,
                    # down_block_additional_residuals = down_block_additional_residuals,
                    # mid_block_additional_residual   = mid_block_additional_residual,
                ).sample.to(dtype=latents_dtype)
                
                # hidden_states transfer from unet_reference to unet
                self.unet.copy_self_subject_hidden_state(self.unet_reference)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t, 
                    encoder_hidden_states=text_embeddings,
                    subject_encoder_hidden_states=subject_encoder_embedding,
                    self_subject_attention_mask=mask_images_input,
                    down_block_additional_residuals = down_block_additional_residuals,
                    mid_block_additional_residual   = mid_block_additional_residual,
                ).sample.to(dtype=latents_dtype)

                # perform guidance
                if use_reference_guidance and do_classifier_free_guidance:
                    if reference_guidance_r:
                        # use reference guidance scale 
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + refernce_guidance_scale * (noise_pred_text - noise_pred_uncond)
                    else:
                        # use text gudiance scale
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                elif do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # Post-processing
        # images = self.decode_latents(latents)
        images = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
        do_denormalize = [True] * images.shape[0]
        
        images = self.image_processor.postprocess(images, output_type=output_type, do_denormalize=do_denormalize)

        if not return_dict:
            return images

        return DreamTunerPipelineOutput(images=images)
    
    def load_self_subject(self, omega_ref = 2.5):
        processor = SelfSubjectAttnProcessor(omega_ref)
        self.unet.set_attn_processor(processor)
        self.unet_reference.set_attn_processor(processor)
        
    def set_subject_encoder_beta(self, beta):
        self.unet.set_subject_encoder_beta(beta)
        self.unet_reference.set_subject_encoder_beta(beta)
        

