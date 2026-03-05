#!/usr/bin/env python
# coding: utf-8
import os
import sys
import json
import argparse
import numpy as np
import math
from einops import rearrange
import time
import random
import string
import h5py
from tqdm import tqdm
import webdataset as wds

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from accelerate import Accelerator

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('generative_models/')
import sgm
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder, FrozenOpenCLIPEmbedder2
from generative_models.sgm.models.diffusion import DiffusionEngine
from generative_models.sgm.util import append_dims
from omegaconf import OmegaConf

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# custom functions #
import utils
from models_our import *
import re

accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
device = accelerator.device
print("device:",device)

parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument("--model_name", type=str, default="testing", help="will load ckpt for model")
parser.add_argument("--data_path", type=str, default="/data/lsm/hub/datasets--pscotti--mindeyev2/snapshots/26421f100e4c6012a35ecadb272a0ec1d999202d", help="Path to where NSD data is stored / where to download it to")
parser.add_argument("--cache_dir", type=str, default="/data/lsm/hub/datasets--pscotti--mindeyev2/snapshots/26421f100e4c6012a35ecadb272a0ec1d999202d", help="Path to where misc. files downloaded from huggingface are stored. Defaults to current src directory.")
parser.add_argument("--subj", type=int, default=1, choices=[1,2,3,4,5,6,7,8], help="Validate on which subject?")
parser.add_argument("--blurry_recon", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--n_blocks", type=int, default=4)
parser.add_argument("--hidden_dim",type=int, default=2048)
parser.add_argument("--new_test", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)

# seed all random functions
utils.seed_everything(seed)

# make output directory
eval_dir = "/data/lsm/model_mindeyev2/evals"
os.makedirs(eval_dir, exist_ok=True)
os.makedirs(f"{eval_dir}/{model_name}", exist_ok=True)

voxels = {}
# Load hdf5 data for betas
f = h5py.File(f'{data_path}/betas_all_subj0{subj}_fp32_renorm.hdf5', 'r')
betas = f['betas'][:]
betas = torch.Tensor(betas).to("cpu")
num_voxels = betas[0].shape[-1]
voxels[f'subj0{subj}'] = betas
print(f"num_voxels for subj0{subj}: {num_voxels}")

if not new_test: # using old test set from before full dataset released (used in original MindEye paper)
    if subj==3:
        num_test=2113
    elif subj==4:
        num_test=1985
    elif subj==6:
        num_test=2113
    elif subj==8:
        num_test=1985
    else:
        num_test=2770
    test_url = f"{data_path}/wds/subj0{subj}/test/" + "0.tar"
else: # using larger test set from after full dataset released
    if subj==3:
        num_test=2371
    elif subj==4:
        num_test=2188
    elif subj==6:
        num_test=2371
    elif subj==8:
        num_test=2188
    else:
        num_test=3000
    test_url = f"{data_path}/wds/subj0{subj}/new_test/" + "0.tar"

print(test_url)
def my_split_by_node(urls): return urls
test_data = wds.WebDataset(test_url,resampled=False,nodesplitter=my_split_by_node)\
                    .decode("torch")\
                    .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                    .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False, drop_last=True, pin_memory=True)
print(f"Loaded test dl for subj{subj}!\n")

# Prep images but don't load them all to memory
f = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')
images = f['images']

# Prep test voxels and indices of test images
test_images_idx = []
test_voxels_idx = []
for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dl):
    test_voxels = voxels[f'subj0{subj}'][behav[:,0,5].cpu().long()]
    test_voxels_idx = np.append(test_images_idx, behav[:,0,5].cpu().numpy())
    test_images_idx = np.append(test_images_idx, behav[:,0,0].cpu().numpy())
test_images_idx = test_images_idx.astype(int)
test_voxels_idx = test_voxels_idx.astype(int)

assert (test_i+1) * num_test == len(test_voxels) == len(test_images_idx)
print(test_i, len(test_voxels), len(test_images_idx), len(np.unique(test_images_idx)))

clip_img_embedder = FrozenOpenCLIPImageEmbedder(
    arch="ViT-bigG-14",
    version="laion2b_s39b_b160k",
    output_tokens=True,
    only_tokens=True,
)
clip_img_embedder.to(device)
clip_seq_dim = 256
clip_emb_dim = 1664

if blurry_recon:
    from diffusers import AutoencoderKL
    autoenc = AutoencoderKL(
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256,
    )
    ckpt = torch.load(f'{cache_dir}/sd_image_var_autoenc.pth')
    autoenc.load_state_dict(ckpt)
    autoenc.eval()
    autoenc.requires_grad_(False)
    autoenc.to(device)
    utils.count_params(autoenc)

class MindEyeModule(nn.Module):
    def __init__(self):
        super(MindEyeModule, self).__init__()
    def forward(self, x):
        return x

model = MindEyeModule()

class RidgeRegression(torch.nn.Module):
    # make sure to add weight_decay when initializing optimizer to enable regularization
    def __init__(self, input_sizes, out_features): 
        super(RidgeRegression, self).__init__()
        self.out_features = out_features
        self.linears = torch.nn.ModuleList([
                torch.nn.Linear(input_size, out_features) for input_size in input_sizes
            ])
    def forward(self, x, subj_idx):
        out = self.linears[subj_idx](x[:,0]).unsqueeze(1)
        return out

model.ridge = RidgeRegression([num_voxels], out_features=hidden_dim)

from diffusers.models.vae import Decoder
from models_our import BrainNetwork
model.backbone = BrainNetwork(h=hidden_dim, in_dim=hidden_dim, seq_len=1, 
                          clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim) 
utils.count_params(model.ridge)
utils.count_params(model.backbone)
utils.count_params(model)

# setup diffusion prior network
out_dim = clip_emb_dim
depth = 6
dim_head = 52
heads = clip_emb_dim//52 # heads * dim_head = clip_emb_dim
timesteps = 100

prior_network = PriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        num_tokens = clip_seq_dim,
        learned_query_mode="pos_emb"
    )

model.diffusion_prior = BrainDiffusionPrior(
    net=prior_network,
    image_embed_dim=out_dim,
    condition_on_text_encodings=False,
    timesteps=timesteps,
    cond_drop_prob=0.2,
    image_embed_scale=None,
)
model.to(device)

utils.count_params(model.diffusion_prior)
utils.count_params(model)

# Load pretrained model ckpt
tag='last'
outdir = os.path.abspath(f'/data/lsm/model_mindeyev2/train_logs/{model_name}')
print(f"\n---loading {outdir}/{tag}.pth ckpt---\n")
def detect_and_enable_adapters(backbone: nn.Module, state_dict):
    """
    Inspect state_dict to decide whether to enable LoRA / Skip-LoRA and with what ranks.
    We infer ranks from tensor shapes and default alpha=rank if not recorded.
    """
    # Detect LoRA
    has_lora = any(k.startswith('backbone.') and (k.endswith('.lora_A') or '.lora_A' in k) for k in state_dict.keys())
    if has_lora:
        # try to infer a representative rank from backbone_linear if present
        rep_key = None
        for k, v in state_dict.items():
            if k.endswith('backbone_linear.lora_A'):
                rep_key = k; break
        if rep_key is None:
            for k, v in state_dict.items():
                if k.endswith('.lora_A'):
                    rep_key = k; break
        rank = state_dict[rep_key].shape[0] if rep_key is not None else 8
        backbone.enable_lora(rank=rank, alpha=rank)

    # Detect Skip-LoRA
    has_skip_blocks = any(k.startswith('backbone._skip_adapters_block1') for k in state_dict.keys())
    has_skip_final = any(k.startswith('backbone._skip_adapter_backbone_linear') for k in state_dict.keys())
    has_skip_align = any(k.startswith('backbone._skip_adapter_align') for k in state_dict.keys())
    if has_skip_blocks or has_skip_final:
        # infer rank from any A of skip adapters
        rep_key = None
        for k, v in state_dict.items():
            if '_skip_adapters_block1' in k and k.endswith('.A'):
                rep_key = k; break
        if rep_key is None:
            for k, v in state_dict.items():
                if '_skip_adapter_backbone_linear' in k and k.endswith('.A'):
                    rep_key = k; break
        rank = state_dict[rep_key].shape[0] if rep_key is not None else 8
        # v_in_dim is subject voxel dim
        v_in_dim = num_voxels
        backbone.enable_skip_lora(
            v_in_dim=v_in_dim,
            activation='gelu',
            rank=rank,
            alpha=rank,
            include_final=has_skip_final,
            include_align=has_skip_align,
        )

def detect_and_construct_cs_proj(state_dict: dict, model: nn.Module, device: torch.device):
    """If checkpoint contains keys for `cs_proj.net.*`, construct a matching nn.Sequential
    in `model.cs_proj` so that loading its weights succeeds. We infer layer types from
    parameter shapes: 2D weight -> Linear(out,in), 1D weight -> LayerNorm(normalized_shape).
    Missing indices are filled with GELU activations by default.
    """
    # find cs_proj keys
    proj_keys = [k for k in state_dict.keys() if k.startswith('cs_proj.net.')]
    if len(proj_keys) == 0:
        return False
    # collect indices that have params
    idx_set = set()
    param_map = {}
    for k in proj_keys:
        m = re.match(r'^cs_proj\.net\.(\d+)\.(.+)$', k)
        if not m:
            continue
        idx = int(m.group(1))
        name = m.group(2)
        idx_set.add(idx)
        param_map.setdefault(idx, {})[name] = state_dict[k]
    max_idx = max(idx_set)
    modules = [None] * (max_idx + 1)
    # build modules in order
    for i in range(max_idx + 1):
        if i in param_map:
            pm = param_map[i]
            if 'weight' in pm:
                w = pm['weight']
                # Linear if 2D weight, LayerNorm if 1D
                if isinstance(w, torch.Tensor) and w.dim() == 2:
                    out_dim, in_dim = w.shape[0], w.shape[1]
                    modules[i] = nn.Linear(in_dim, out_dim)
                elif isinstance(w, torch.Tensor) and w.dim() == 1:
                    modules[i] = nn.LayerNorm(w.shape[0])
                else:
                    # fallback: attempt Linear with inferred dims if possible
                    try:
                        out_dim = pm['bias'].shape[0]
                        in_dim = out_dim
                        modules[i] = nn.Linear(in_dim, out_dim)
                    except Exception:
                        modules[i] = nn.Identity()
            else:
                modules[i] = nn.Identity()
        else:
            # likely an activation slot
            modules[i] = nn.GELU()
    seq = nn.Sequential(*modules)
    # attach to model
    model.cs_proj = seq.to(device)
    return True

try:
    checkpoint = torch.load(outdir+f'/{tag}.pth', map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    # enable adapters as needed before loading
    detect_and_enable_adapters(model.backbone, state_dict)
    # if checkpoint contains cs_proj, dynamically construct it so weights can be loaded
    try:
        created = detect_and_construct_cs_proj(state_dict, model, device)
        if created:
            print("[INFO] Constructed cs_proj in model from checkpoint keys.")
    except Exception as _e_csproj:
        print(f"[WARN] Failed to construct cs_proj from checkpoint: {_e_csproj}")
    # ensure any newly created adapter params are moved to the active device
    model.to(device)
    # load with strict=False to be robust to minor key mismatches (we already constructed cs_proj when possible)
    model.load_state_dict(state_dict, strict=False)
    del checkpoint
except Exception as e: # probably ckpt is saved using deepspeed format
    print("Standard ckpt load failed, trying deepspeed format:", e)
    import deepspeed
    state_dict = deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir=outdir, tag=tag)
    detect_and_enable_adapters(model.backbone, state_dict)
    model.load_state_dict(state_dict, strict=False)
    del state_dict
print("ckpt loaded!")

# setup text caption networks
from transformers import AutoProcessor, AutoModelForCausalLM
from modeling_git import GitForCausalLMClipEmb
processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
clip_text_model = GitForCausalLMClipEmb.from_pretrained("microsoft/git-large-coco")
clip_text_model.to(device) # if you get OOM running this script, you can switch this to cpu and lower minibatch_size to 4
clip_text_model.eval().requires_grad_(False)
clip_text_seq_dim = 257
clip_text_emb_dim = 1024

class CLIPConverter(torch.nn.Module):
    def __init__(self):
        super(CLIPConverter, self).__init__()
        self.linear1 = nn.Linear(clip_seq_dim, clip_text_seq_dim)
        self.linear2 = nn.Linear(clip_emb_dim, clip_text_emb_dim)
    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.linear1(x)
        x = self.linear2(x.permute(0,2,1))
        return x

clip_convert = CLIPConverter()
state_dict = torch.load(f"{cache_dir}/bigG_to_L_epoch8.pth", map_location='cpu')['model_state_dict']
clip_convert.load_state_dict(state_dict, strict=True)
clip_convert.to(device) # if you get OOM running this script, you can switch this to cpu and lower minibatch_size to 4
del state_dict

# prep unCLIP
config = OmegaConf.load("generative_models/configs/unclip6.yaml")
config = OmegaConf.to_container(config, resolve=True)
unclip_params = config["model"]["params"]
network_config = unclip_params["network_config"]
denoiser_config = unclip_params["denoiser_config"]
first_stage_config = unclip_params["first_stage_config"]
conditioner_config = unclip_params["conditioner_config"]
sampler_config = unclip_params["sampler_config"]
scale_factor = unclip_params["scale_factor"]
disable_first_stage_autocast = unclip_params["disable_first_stage_autocast"]
offset_noise_level = unclip_params["loss_fn_config"]["params"]["offset_noise_level"]

first_stage_config['target'] = 'sgm.models.autoencoder.AutoencoderKL'
sampler_config['params']['num_steps'] = 38

diffusion_engine = DiffusionEngine(network_config=network_config,
                       denoiser_config=denoiser_config,
                       first_stage_config=first_stage_config,
                       conditioner_config=conditioner_config,
                       sampler_config=sampler_config,
                       scale_factor=scale_factor,
                       disable_first_stage_autocast=disable_first_stage_autocast)
# set to inference
diffusion_engine.eval().requires_grad_(False)
diffusion_engine.to(device)

ckpt_path = f'{cache_dir}/unclip6_epoch0_step110000.ckpt'
ckpt = torch.load(ckpt_path, map_location='cpu')
diffusion_engine.load_state_dict(ckpt['state_dict'])

batch={"jpg": torch.randn(1,3,1,1).to(device), # jpg doesnt get used, it's just a placeholder
      "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
      "crop_coords_top_left": torch.zeros(1, 2).to(device)}
out = diffusion_engine.conditioner(batch)
vector_suffix = out["vector"].to(device)
print("vector_suffix", vector_suffix.shape)

# get all reconstructions
model.to(device)
model.eval().requires_grad_(False)

# all_images = None
all_blurryrecons = None
all_recons = None
all_predcaptions = []
all_clipvoxels = None

minibatch_size = 1
num_samples_per_image = 1
assert num_samples_per_image == 1

# if utils.is_interactive(): plotting=True
plotting=False

with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
    for batch in tqdm(range(0,len(np.unique(test_images_idx)),minibatch_size)):
        uniq_imgs = np.unique(test_images_idx)[batch:batch+minibatch_size]
        voxel = None
        for uniq_img in uniq_imgs:
            locs = np.where(test_images_idx==uniq_img)[0]
            if len(locs)==1:
                locs = locs.repeat(3)
            elif len(locs)==2:
                locs = locs.repeat(2)[:3]
            assert len(locs)==3
            if voxel is None:
                voxel = test_voxels[None,locs] # 1, num_image_repetitions, num_voxels
            else:
                voxel = torch.vstack((voxel, test_voxels[None,locs]))
        voxel = voxel.to(device)

        for rep in range(3):
            voxel_ridge = model.ridge(voxel[:,[rep]],0) # 0th index of subj_list
            # if Skip-LoRA enabled, pass raw voxels as skip_input (single-subject inference)
            skip_in = voxel[:,rep]
            try:
                backbone0, clip_voxels0, blurry_image_enc0 = model.backbone(voxel_ridge, skip_input=skip_in, return_skip_stats=False)
            except TypeError:
                # fallback for older backbone signature
                backbone0, clip_voxels0, blurry_image_enc0 = model.backbone(voxel_ridge)
            if rep==0:
                clip_voxels = clip_voxels0
                backbone = backbone0
                blurry_image_enc = blurry_image_enc0[0]
            else:
                clip_voxels += clip_voxels0
                backbone += backbone0
                blurry_image_enc += blurry_image_enc0[0]
        clip_voxels /= 3
        backbone /= 3
        blurry_image_enc /= 3

        # Save retrieval submodule outputs
        if all_clipvoxels is None:
            all_clipvoxels = clip_voxels.cpu()
        else:
            all_clipvoxels = torch.vstack((all_clipvoxels, clip_voxels.cpu()))

        # Feed voxels through OpenCLIP-bigG diffusion prior
        prior_out = model.diffusion_prior.p_sample_loop(backbone.shape, 
                        text_cond = dict(text_embed = backbone), 
                        cond_scale = 1., timesteps = 20)

        pred_caption_emb = clip_convert(prior_out)
        generated_ids = clip_text_model.generate(pixel_values=pred_caption_emb, max_length=20)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
        all_predcaptions = np.hstack((all_predcaptions, generated_caption))
        print(generated_caption)

        # Feed diffusion prior outputs through unCLIP
        for i in range(len(voxel)):
            samples = utils.unclip_recon(prior_out[[i]],
                             diffusion_engine,
                             vector_suffix,
                             num_samples=num_samples_per_image)
            if all_recons is None:
                all_recons = samples.cpu()
            else:
                all_recons = torch.vstack((all_recons, samples.cpu()))
            if plotting:
                for s in range(num_samples_per_image):
                    plt.figure(figsize=(2,2))
                    plt.imshow(transforms.ToPILImage()(samples[s]))
                    plt.axis('off')
                    plt.show()

        if blurry_recon:
            blurred_image = (autoenc.decode(blurry_image_enc/0.18215).sample/ 2 + 0.5).clamp(0,1)

            for i in range(len(voxel)):
                im = torch.Tensor(blurred_image[i])
                if all_blurryrecons is None:
                    all_blurryrecons = im[None].cpu()
                else:
                    all_blurryrecons = torch.vstack((all_blurryrecons, im[None].cpu()))
                if plotting:
                    plt.figure(figsize=(2,2))
                    plt.imshow(transforms.ToPILImage()(im))
                    plt.axis('off')
                    plt.show()

        if plotting: 
            print(model_name)
            err # dont actually want to run the whole thing with plotting=True

# resize outputs before saving
imsize = 256
all_recons = transforms.Resize((imsize,imsize))(all_recons).float()
if blurry_recon: 
    all_blurryrecons = transforms.Resize((imsize,imsize))(all_blurryrecons).float()

# saving
print(all_recons.shape)
# # You can find the all_images file on huggingface: https://huggingface.co/datasets/pscotti/mindeyev2/tree/main/evals
# torch.save(all_images,"evals/all_images.pt") 
if blurry_recon:
    torch.save(all_blurryrecons,f"{eval_dir}/{model_name}/{model_name}_all_blurryrecons.pt")
torch.save(all_recons,f"{eval_dir}/{model_name}/{model_name}_all_recons.pt")
torch.save(all_predcaptions,f"{eval_dir}/{model_name}/{model_name}_all_predcaptions.pt")
torch.save(all_clipvoxels,f"{eval_dir}/{model_name}/{model_name}_all_clipvoxels.pt")
print(f"saved {model_name} outputs!")

if not utils.is_interactive():
    sys.exit(0)

