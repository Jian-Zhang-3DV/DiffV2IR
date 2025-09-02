#!/usr/bin/env python
"""
Optimized inference script for DiffV2IR with memory reduction techniques
Supports FP16 (half precision) and other memory optimizations
"""
from __future__ import annotations

import math
import random
import sys
from argparse import ArgumentParser
import os
import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast
import gc
from contextlib import contextmanager
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from blip_models.blip import blip_decoder
from model_paths import MODEL_PATHS, check_model_exists

sys.path.append("./stable_diffusion")
from stable_diffusion.ldm.util import instantiate_from_config

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale, seg_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=4)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=4)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0],uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat1": [torch.cat([cond["c_concat1"][0], cond["c_concat1"][0], uncond["c_concat1"][0], uncond["c_concat1"][0]])],
            "c_concat2": [torch.cat([cond["c_concat2"][0], cond["c_concat2"][0],cond["c_concat2"][0], uncond["c_concat2"][0]])],
        }
        out_cond, out_img_cond, out_seg_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(4)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_seg_cond) + seg_cfg_scale * (out_seg_cond - out_uncond)

@contextmanager
def torch_gc():
    """Context manager for aggressive garbage collection"""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

def load_model_from_config_fp16(config, ckpt, vae_ckpt=None, use_fp16=True, verbose=False):
    """Load model with optional FP16 support"""
    print(f"Loading model from {ckpt}")
    if use_fp16:
        print("Using FP16 (half precision) for reduced memory usage")
    
    # Load checkpoint to CPU first to save GPU memory
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    
    if len(m) > 0 and verbose:
        print("missing keys:", m)
    if len(u) > 0 and verbose:
        print("unexpected keys:", u)
    
    # Convert to fp16 if requested
    if use_fp16:
        model = model.half()
    
    # Clear the loaded state dict from memory
    del pl_sd, sd
    gc.collect()
    
    return model

def load_demo_image(image_size, device, img_url):
    raw_image = Image.open(img_url).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0)
    return image

def main():
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--edit", default="turn the RGB image into the infrared one", type=str)
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    parser.add_argument("--cfg-seg", default=1.5, type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fp16", action="store_true", help="Use FP16 (half precision) to reduce memory usage")
    parser.add_argument("--cpu-offload", action="store_true", help="Offload models to CPU when not in use")
    parser.add_argument("--batch-size", default=1, type=int, help="Batch size for processing")
    parser.add_argument("--no-blip", action="store_true", help="Skip BLIP caption generation to save memory")
    args = parser.parse_args()

    # Memory optimization settings
    if args.fp16:
        print("FP16 mode enabled - memory usage will be reduced")
    if args.cpu_offload:
        print("CPU offloading enabled - models will be moved to CPU when not in use")
    if args.no_blip:
        print("BLIP caption generation disabled to save memory")

    # Load config and model
    config = OmegaConf.load(args.config)
    
    with torch_gc():
        model = load_model_from_config_fp16(config, args.ckpt, args.vae_ckpt, use_fp16=args.fp16)
        model.eval()
        
        if args.fp16:
            model = model.cuda()
        else:
            model = model.cuda()
    
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])
    
    # Load BLIP model only if needed
    blip_model = None
    if not args.no_blip:
        blip_model_path = MODEL_PATHS.get('blip')
        if not check_model_exists('blip'):
            print(f"BLIP model not found at {blip_model_path}")
            blip_model_path = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth"
        else:
            print(f"Using local BLIP model from {blip_model_path}")
        
        blip_model = blip_decoder(pretrained=blip_model_path, image_size=384, vit='base')
        blip_model.eval()
        
        if args.fp16:
            blip_model = blip_model.half()
        
        if args.cpu_offload:
            blip_model = blip_model.cpu()  # Keep on CPU until needed
    
    seed = random.randint(0, 100000) if args.seed is None else args.seed
    
    # Process images
    for root, dirs, files in os.walk(args.input):
        for file in files:
            print(f"\nProcessing: {file}")
            
            # Generate caption if BLIP is enabled
            if blip_model is not None:
                if args.cpu_offload:
                    blip_model = blip_model.cuda()
                
                image = load_demo_image(image_size=384, device='cuda', img_url=os.path.join(root, file))
                with torch.no_grad():
                    if args.fp16:
                        image = image.half()
                    caption = blip_model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
                args.edit = "turn the visible image of " + caption[0] + " into infrared"
                
                # Free memory
                del image
                if args.cpu_offload:
                    blip_model = blip_model.cpu()
                torch.cuda.empty_cache()
            
            # Load and prepare images
            input_image = Image.open(os.path.join(args.input, file)).convert("RGB")
            input_seg = Image.open(os.path.join(args.input + "_seg", file.split(".")[0] + ".png")).convert("RGB")
            
            width, height = input_image.size
            factor = args.resolution / max(width, height)
            factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
            width = int((width * factor) // 64) * 64
            height = int((height * factor) // 64) * 64
            input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
            input_seg = ImageOps.fit(input_seg, (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
            
            if args.edit == "":
                input_image.save(os.path.join(args.output, file))
                continue
            
            # Inference with memory management
            with torch.no_grad(), torch_gc():
                if args.fp16:
                    with autocast("cuda", dtype=torch.float16):
                        # Prepare conditions
                        cond = {}
                        cond["c_crossattn"] = [model.get_learned_conditioning([args.edit])]
                        
                        input_image_t = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
                        input_seg_t = 2 * torch.tensor(np.array(input_seg)).float() / 255 - 1
                        input_image_t = rearrange(input_image_t, "h w c -> 1 c h w").to(model.device)
                        input_seg_t = rearrange(input_seg_t, "h w c -> 1 c h w").to(model.device)
                        
                        if args.fp16:
                            input_image_t = input_image_t.half()
                            input_seg_t = input_seg_t.half()
                        
                        cond["c_concat1"] = [model.encode_first_stage(input_image_t).mode()]
                        cond["c_concat2"] = [model.encode_first_stage(input_seg_t).mode()]
                        
                        uncond = {}
                        uncond["c_crossattn"] = [null_token]
                        uncond["c_concat1"] = [torch.zeros_like(cond["c_concat1"][0])]
                        uncond["c_concat2"] = [torch.zeros_like(cond["c_concat2"][0])]
                        
                        sigmas = model_wrap.get_sigmas(args.steps)
                        
                        extra_args = {
                            "cond": cond,
                            "uncond": uncond,
                            "text_cfg_scale": args.cfg_text,
                            "image_cfg_scale": args.cfg_image,
                            "seg_cfg_scale": args.cfg_seg,
                        }
                        
                        torch.manual_seed(seed)
                        z = torch.randn_like(cond["c_concat1"][0]) * sigmas[0]
                        z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
                        x = model.decode_first_stage(z)
                        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
                        x = 255.0 * rearrange(x, "1 c h w -> h w c")
                        edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
                else:
                    # Standard FP32 inference
                    with autocast("cuda"):
                        cond = {}
                        cond["c_crossattn"] = [model.get_learned_conditioning([args.edit])]
                        
                        input_image_t = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
                        input_seg_t = 2 * torch.tensor(np.array(input_seg)).float() / 255 - 1
                        input_image_t = rearrange(input_image_t, "h w c -> 1 c h w").to(model.device)
                        input_seg_t = rearrange(input_seg_t, "h w c -> 1 c h w").to(model.device)
                        
                        cond["c_concat1"] = [model.encode_first_stage(input_image_t).mode()]
                        cond["c_concat2"] = [model.encode_first_stage(input_seg_t).mode()]
                        
                        uncond = {}
                        uncond["c_crossattn"] = [null_token]
                        uncond["c_concat1"] = [torch.zeros_like(cond["c_concat1"][0])]
                        uncond["c_concat2"] = [torch.zeros_like(cond["c_concat2"][0])]
                        
                        sigmas = model_wrap.get_sigmas(args.steps)
                        
                        extra_args = {
                            "cond": cond,
                            "uncond": uncond,
                            "text_cfg_scale": args.cfg_text,
                            "image_cfg_scale": args.cfg_image,
                            "seg_cfg_scale": args.cfg_seg,
                        }
                        
                        torch.manual_seed(seed)
                        z = torch.randn_like(cond["c_concat1"][0]) * sigmas[0]
                        z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
                        x = model.decode_first_stage(z)
                        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
                        x = 255.0 * rearrange(x, "1 c h w -> h w c")
                        edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
            
            edited_image.save(os.path.join(args.output, file))
            print(f"Saved: {os.path.join(args.output, file)}")
            
            # Clear memory after each image
            torch.cuda.empty_cache()
    
    print("\nInference completed!")

if __name__ == "__main__":
    main()