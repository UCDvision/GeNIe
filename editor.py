import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
import random
import os 
import numpy as np 
import PIL
from PIL import Image
import pdb 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from imagenet_labels import ind2name
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
import torch.nn.functional as F
from torchvision import transforms as T



@torch.no_grad()
def __call__(
    self,
    prompt: Union[str, List[str]],
    percent_noise: int,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
):

    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor

    self.check_inputs(
        prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
    )

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0

    prompt_embeds = self._encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )

    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):

            if t - 1 > 1000 * percent_noise:
                continue

            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
            ).sample

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    if output_type == "latent":
        image = latents
        has_nsfw_concept = None
    elif output_type == "pil":
        image = self.decode_latents(latents)
        image = self.numpy_to_pil(image)
    else:
        image = self.decode_latents(latents)

    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        self.final_offload_hook.offload()

    if not return_dict:
        return (image, has_nsfw_concept)

    return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=False)



class ImageEditor():
    
    def __init__(self, ): 
        self.pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        self.pipe = self.pipe.to('cuda')
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        type(self.pipe).__call__ = __call__
        self.NUM_INFERENCE_STEPS = 50 if isinstance(self.pipe.scheduler, DPMSolverMultistepScheduler) else 100
        
        self.transform = T.Compose([T.PILToTensor(),
                    T.ConvertImageDtype(torch.float),
                    T.Resize(512),
#                     T.CenterCrop(512),
                    T.Normalize([0.5], [0.5])])
    
    def edit(self, img, prompt, percent_noise):
        latents = self.encode_latents(img)
        latents = self.diffusion_forward(percent_noise, latents)
        output  = self.diffusion_reverse(prompt, percent_noise, latents, 
                            num_inference_steps=self.NUM_INFERENCE_STEPS, output_type='pil').images[0]
        return output
        
    def encode_latents(self, img):

        with torch.no_grad():
            img = self.transform(img).half().to(self.pipe.device)
            img = torch.unsqueeze(img, 0)

        # Project image into the latent space
        latents = self.pipe.vae.encode(img).latent_dist.mode() 
        return 0.18215 * latents


    def diffusion_forward(self, percent_noise, latents):

        noise = torch.randn(latents.shape).to(self.pipe.device)
        timestep = torch.Tensor([int(self.pipe.scheduler.config.num_train_timesteps * percent_noise)]).to(self.pipe.device).long()
        if percent_noise > 0.999: 
            return noise 
        z = self.pipe.scheduler.add_noise(latents, noise, timestep).half()
        return z


    def diffusion_reverse(self, prompt, percent_noise, latents, num_inference_steps=100, output_type='pil', guidance_scale=7.5):

        with autocast('cuda'):
            return self.pipe(prompt=prompt, percent_noise=percent_noise, latents=latents,
                        num_inference_steps=num_inference_steps, output_type=output_type)  