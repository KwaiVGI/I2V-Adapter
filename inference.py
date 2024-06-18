import os
import cv2
import torch
import random
import imageio
import argparse
import numpy as np
from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection

from src.models.unet import UNet3DConditionModel
from src.utils.util import save_videos_grid, load_weights, imread_resize, color_match_frames
from src.models.ip_adapter import Resampler
from src.pipelines.pipeline_i2v_adapter import I2VIPAdapterPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default=None, help='Path of trained model weights')
    parser.add_argument('--pretrain_weight', type=str, default='./checkpoints/stable-diffusion-v1-4', help='Path for pretrained weight (SD v1.4)')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('--first_frame_path', type=str, default=None, help='The path for first frame image')
    parser.add_argument('-p', '--prompt', type=str, default=None, help='The video prompt. Default value: same to the filename of the first frame image')
    parser.add_argument('-hs', '--height', type=int, default=256, help='video height')
    parser.add_argument('-ws', '--width', type=int, default=256, help='video width')
    parser.add_argument('-l', '--length', type=int, default=16, help='video length')
    parser.add_argument('--cfg', type=float, default=8, help='classifier-free guidance scale')
    parser.add_argument('--infer_config', type=str, default='./configs/inference/inference_i2v.yaml', help='Path for inference config')
    parser.add_argument('--dreambooth_path', type=str, default='', help='Path for dreambooth model')
    parser.add_argument('--i2v_module_path', type=str, default='', help='Path for i2v module')
    parser.add_argument('--neg_prompt', type=str, default=None, help='The negative prompt')
    parser.add_argument('--motion_lora', action='store_true', help='if use motion lora model')
    parser.add_argument('--pretrained_image_encoder_path', type=str, default='', help='Path for pretrained image encoder')
    parser.add_argument('--pretrained_ipadapter_path', type=str, default='', help='Path for pretrained ipadapter encoder')
    args = parser.parse_args()

    pretrained_model_path = args.pretrain_weight
    pretrained_image_encoder_path = args.pretrained_image_encoder_path
    pretrained_ipadapter_path = args.pretrained_ipadapter_path
    inference_config = OmegaConf.load(args.infer_config)
    global_seed = inference_config.global_seed
    
    # load checkpoints during training
    unet = UNet3DConditionModel.from_pretrained_ip(pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)).to(torch.float16).to('cuda')
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae", torch_dtype=torch.float16).to('cuda')
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", torch_dtype=torch.float16)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder", torch_dtype=torch.float16).to('cuda')
    clip_image_processor = CLIPImageProcessor()
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_image_encoder_path, torch_dtype=torch.float16).to('cuda')
    image_proj_model = Resampler(
            dim=768,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=16,
            embedding_dim=image_encoder.config.hidden_size,
            output_dim=768,
            ff_mult=4
        )
    image_proj_model.load_state_dict(torch.load(pretrained_ipadapter_path, "cpu")["image_proj"], strict=True)
    image_proj_model.to(torch.float16).to('cuda')
    print("Load pretrained clip image encoder and ipadapter model successfully")
    
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    image_encoder.requires_grad_(False)
    image_proj_model.requires_grad_(False)

    # builds pipeline
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs))
    pipe = I2VIPAdapterPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, 
                                scheduler=noise_scheduler)
    if args.motion_lora:
        motion_module_lora_configs = inference_config.motion_module_lora_configs
    
    pipe = load_weights(pipe, 
                        dreambooth_model_path=args.dreambooth_path,
                        i2v_module_path=args.i2v_module_path,
                        motion_module_lora_configs=motion_module_lora_configs if args.motion_lora else [])
    pipe.to(torch.float16).to('cuda')
    pipe.enable_vae_slicing()
    
    # read the first frame
    img_path = args.first_frame_path
    if args.prompt is None:
        prompt = img_path.split('/')[-1][:-4].replace('_', ' ')
    else:
        prompt = args.prompt
    if args.neg_prompt is not None:
        neg_prompt = args.neg_prompt
    else:
        neg_prompt = None
    print('Prompt: ', prompt)
    print('Negative Prompt: ', neg_prompt)
    
    # Get ip-adapter image embeddings
    image = Image.open(img_path).convert("RGB")
    image = clip_image_processor(images=image, return_tensors="pt").pixel_values.to('cuda').to(torch.float16)
    with torch.no_grad():
        clip_image_embeds = image_encoder(image, output_hidden_states=True).hidden_states[-2]
        clip_image_embeds = image_proj_model(clip_image_embeds)
        un_cond_image_embeds = image_encoder(torch.zeros_like(image).to(image.device).to(torch.float16), output_hidden_states=True).hidden_states[-2]
        un_cond_image_embeds = image_proj_model(un_cond_image_embeds)
    
    print("Using seed {} for generation".format(global_seed))
    generator = torch.Generator(device="cuda").manual_seed(global_seed)
    # Get first frame latents as usual
    image = imread_resize(img_path, args.height, args.width)
    first_frame_latents = torch.Tensor(image.copy()).to('cuda').type(torch.float16).permute(2, 0, 1).repeat(1, 1, 1, 1)
    first_frame_latents = first_frame_latents / 127.5 - 1.0
    first_frame_latents = vae.encode(first_frame_latents).latent_dist.sample(generator) * 0.18215
    first_frame_latents = first_frame_latents.repeat(1, 1, 1, 1, 1).permute(1, 2, 0, 3, 4)
    
    # video generation
    video = pipe(prompt=prompt, generator=generator, latents=first_frame_latents, 
                 video_length=args.length, height=image.shape[0], width=image.shape[1], 
                 num_inference_steps=25, guidance_scale=args.cfg, 
                 noise_mode="iid", negative_prompt=neg_prompt, 
                 repeat_latents=True, gaussian_blur=True,
                 cond_image_embeds=clip_image_embeds,
                 un_cond_image_embeds=un_cond_image_embeds).videos

    # histogram matching post processing
    for f in range(1, video.shape[2]):
        former_frame = video[0, :, 0, :, :].permute(1, 2, 0).cpu().numpy()
        frame = video[0, :, f, :, :].permute(1, 2, 0).cpu().numpy()
        result = color_match_frames(former_frame, frame)
        result = torch.Tensor(result).type_as(video).to(video.device)
        video[0, :, f, :, :] = result.permute(2, 0, 1)

    save_path = args.output
    save_path = os.path.join(save_path, img_path.split('/')[-1][:-4] + '.gif')
    save_videos_grid(video, save_path)

if __name__ == '__main__':
    main()