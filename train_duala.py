import os
import sys
import json
import h5py
import torch
import wandb
import random
import argparse
from collections import defaultdict
import numpy as np
import torch.nn as nn
import webdataset as wds
import matplotlib.pyplot as plt
from tqdm import tqdm
from accelerate import Accelerator

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('generative_models/')
import sgm
import utils
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder # bigG embedder
from modules import (
    MindEyeModule, RidgeRegression,
    freeze_module_params, count_trainable_params, trainable_params,
    print_trainable_params_breakdown,
)
from ssp_sdp import (
    load_reference_mats, get_ssp_category_names,
    _ssp_build_imageid_to_label_from_json, SSPCategoryAssigner,
    get_labels_for_batch, FeatureStatsAugmenter,
    ssp_triplet_loss_cosine_xbank, abs_pearson_corr,
)

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

data_type = torch.float16 # 根据 accelerate 的 mixed_precision 调整（fp16/fp32/bf16）

accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
global_batch_size = 32

print("PID of this process =",os.getpid())
device = accelerator.device
print("device:",device)
world_size = accelerator.state.num_processes
distributed = not accelerator.state.distributed_type == 'NO'
num_devices = torch.cuda.device_count()
if num_devices==0 or not distributed: num_devices = 1
print(accelerator.state)

local_rank=0
print("distributed =",distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =", world_size, "data_type =", data_type)
print = accelerator.print # only print if local_rank=0

parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument("--model_name", type=str, default="testing", help="name of model, used for ckpt saving and wandb logging (if enabled)")
parser.add_argument("--data_path", type=str, help="Path to where NSD data is stored / where to download it to")
parser.add_argument("--subj", type=int, default=1, choices=[1,2,3,4,5,6,7,8], help="Validate on which subject?")
parser.add_argument("--num_sessions", type=int, default=1, help="Number of training sessions to include")
parser.add_argument("--num_epochs", type=int, default=150, help="number of epochs of training")
parser.add_argument("--multisubject_ckpt", type=str, default=None, help="Path to pre-trained multisubject model to finetune a single subject from. multisubject must be False.")
parser.add_argument("--multi_subject", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--new_test", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--seed", type=int, default=42)
# decoding model arguments
parser.add_argument("--n_blocks", type=int, default=4)
parser.add_argument("--hidden_dim", type=int, default=4096)
parser.add_argument("--ckpt_saving", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--ckpt_interval", type=int, default=5, help="save backup ckpt and reconstruct every x epochs")
# training arguments
parser.add_argument("--batch_size", type=int, default=10, help="Batch size can be increased by 10x if only training retreival submodule and not diffusion prior")
parser.add_argument("--max_lr", type=float, default=3e-4)
parser.add_argument("--lr_scheduler_type",type=str, default='cycle', choices=['cycle','linear'])
parser.add_argument("--mixup_pct", type=float, default=.33, help="proportion of way through training when to switch from BiMixCo to SoftCLIP")
parser.add_argument("--use_image_aug", action=argparse.BooleanOptionalAction, default=False, help="whether to use image augmentation")
# loss arguments
parser.add_argument("--clip_scale", type=float, default=1., help="multiply contrastive loss by this number")
parser.add_argument("--blurry_recon", action=argparse.BooleanOptionalAction, default=True, help="whether to output blurry reconstructions")
parser.add_argument("--blur_scale", type=float, default=.5, help="multiply loss from blurry recons by this number")
parser.add_argument("--use_prior", action=argparse.BooleanOptionalAction, default=True, help="whether to train diffusion prior (True) or just rely on retrieval part of the pipeline (False)")
parser.add_argument("--prior_scale", type=float, default=30, help="multiply diffusion prior loss by this")
# finetune mode and adapters
parser.add_argument("--finetune_mode", type=str, default="lora+skip", choices=["full","lora","skip-lora","lora+skip"], help="Finetune strategy: full params, LoRA, Skip-LoRA, or both")
parser.add_argument("--lora_rank", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=8)
parser.add_argument("--skip_loss_weight", type=float, default=1.5, help="Weight for Skip-LoRA Pearson correlation loss")
# SSP module
# semantic alignment (intra-subject)
parser.add_argument("--ssp_mb_weight", type=float, default=1.0, help="Weight for memory-bank SSP loss; 0 to disable")
parser.add_argument("--ssp_mb_size", type=int, default=8192, help="Memory bank size (number of past samples)")
parser.add_argument("--ssp_mb_warmup", type=int, default=512, help="Warmup entries before enabling memory bank loss")
parser.add_argument("--ssp_margin", type=float, default=0.2, help="Margin for SSP triplet loss with cosine similarity")
parser.add_argument("--ssp_use_clip_categories", action=argparse.BooleanOptionalAction, default=True, help="If true (and no JSON), use CLIP ViT-L/14 to pseudo-label categories each batch")
# relational consistency (cross-subject)
parser.add_argument("--semantic_align_weight", type=float, default=0.1, help="Weight for cross-subject semantic similarity alignment loss; 0 to disable")
parser.add_argument("--semantic_align_ref_dir", type=str, default="/class_corr_clip", help="Directory containing subjXX_cosine.npy and subjXX_categories.txt for reference subjects (e.g., 02..08)")
parser.add_argument("--semantic_align_momentum", type=float, default=0.9, help="EMA momentum for per-image memory features (0 disables EMA, uses last feature)")
# SDP module
parser.add_argument("--feat_stats_dir", type=str, default="/feat_dis", help="Directory containing global_stats(.npz) and subjXX_sigma_wrt_global(.npz) saved by offline_feature_stats.py")
parser.add_argument("--feat_sigma_subjs", type=str, default="2,3,4,5,6,7,8", help="Subject IDs to use for subject-specific sigma (comma-separated).")
# logging arguments
parser.add_argument("--wandb_log", action=argparse.BooleanOptionalAction, default=False, help="whether to log to wandb")
parser.add_argument("--wandb_project",type=str, default="mindeye", help="wandb project name")
args = parser.parse_args()

# 将 args 解包到全局变量（便于下方直接访问变量名）
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)

# seed all random functions
utils.seed_everything(seed)

outdir = os.path.abspath(f'./train_logs/{model_name}')
if not os.path.exists(outdir) and ckpt_saving:
    os.makedirs(outdir, exist_ok=True)
feat_stats_dir = os.path.join(feat_stats_dir, f'subj0{args.subj}')

if use_image_aug or blurry_recon:
    import kornia
    from kornia.augmentation.container import AugmentationSequential
if use_image_aug:
    img_augment = AugmentationSequential(
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.3),
        same_on_batch=False,
        data_keys=["input"],
    )

if multi_subject:
    subj_list = np.arange(1,9)
    subj_list = subj_list[subj_list != subj]
else:
    subj_list = [subj]

print("subj_list", subj_list, "num_sessions", num_sessions)

# # Prep data, models, and dataloaders
def my_split_by_node(urls): return urls
num_voxels_list = []

if multi_subject:
    nsessions_allsubj=np.array([40, 40, 32, 30, 40, 32, 40, 30])
    num_samples_per_epoch = (750*40) // num_devices 
else:
    num_samples_per_epoch = (750*num_sessions) // num_devices 

print("dividing batch size by subj_list, which will then be concatenated across subj during training...") 
batch_size = batch_size // len(subj_list)

num_iterations_per_epoch = num_samples_per_epoch // (batch_size*len(subj_list))

print("batch_size =", batch_size, "num_iterations_per_epoch =",num_iterations_per_epoch, "num_samples_per_epoch =",num_samples_per_epoch)

train_data = {}
train_dl = {}
num_voxels = {}
voxels = {}
for s in subj_list:
    print(f"Training with {num_sessions} sessions")
    if multi_subject:
        train_url = f"{data_path}/wds/subj0{s}/train/" + "{0.." + f"{nsessions_allsubj[s-1]-1}" + "}.tar"
    else:
        train_url = f"{data_path}/wds/subj0{s}/train/" + "{0.." + f"{num_sessions-1}" + "}.tar"
    print(train_url)

    train_data[f'subj0{s}'] = wds.WebDataset(train_url,resampled=True,nodesplitter=my_split_by_node)\
                        .shuffle(750, initial=1500, rng=random.Random(42))\
                        .decode("torch")\
                        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                        .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
    train_dl[f'subj0{s}'] = torch.utils.data.DataLoader(train_data[f'subj0{s}'], batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True, num_workers=4, persistent_workers=True, prefetch_factor=2)

    f = h5py.File(f'{data_path}/betas_all_subj0{s}_fp32_renorm.hdf5', 'r')
    betas = f['betas'][:]
    betas = torch.Tensor(betas).to("cpu").to(data_type)
    num_voxels_list.append(betas[0].shape[-1])
    num_voxels[f'subj0{s}'] = betas[0].shape[-1]
    voxels[f'subj0{s}'] = betas
    print(f"num_voxels for subj0{s}: {num_voxels[f'subj0{s}']}")

print("Loaded all subj train dls and betas!\n")

# Validate only on one subject
if multi_subject: 
    subj = subj_list[0] # cant validate on the actual held out person so picking first in subj_list
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
test_data = wds.WebDataset(test_url,resampled=False,nodesplitter=my_split_by_node)\
                    .shuffle(750, initial=1500, rng=random.Random(42))\
                    .decode("torch")\
                    .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                    .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False, drop_last=True, pin_memory=True, num_workers=2)
print(f"Loaded test dl for subj{subj}!\n")

f = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')
images = f['images'][:]
images = torch.from_numpy(images).to(data_type)
print("Loaded all 73k possible NSD images to cpu!", images.shape)

json_file_main = f'/category_image_idx_subj{subj}.json'

# ## Load models
# ### CLIP image embeddings  model
clip_img_embedder = FrozenOpenCLIPImageEmbedder(
    arch="ViT-bigG-14",
    version="laion2b_s39b_b160k",
    output_tokens=True,
    only_tokens=True,
)
clip_img_embedder.to(device)

clip_seq_dim = 256
clip_emb_dim = 1664

# ### SD VAE
if blurry_recon:
    from diffusers import AutoencoderKL    
    autoenc = AutoencoderKL(
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256,
    )
    ckpt = torch.load(f'{data_path}/sd_image_var_autoenc.pth')
    autoenc.load_state_dict(ckpt)

    autoenc.eval()
    autoenc.requires_grad_(False)
    autoenc.to(device)
    utils.count_params(autoenc)

    from autoencoder.convnext import ConvnextXL
    cnx = ConvnextXL(f'{data_path}/convnext_xlarge_alpha0.75_fullckpt.pth')
    cnx.requires_grad_(False)
    cnx.eval()
    cnx.to(device)

    mean = torch.tensor([0.485, 0.456, 0.406]).to(device).reshape(1,3,1,1)
    std = torch.tensor([0.228, 0.224, 0.225]).to(device).reshape(1,3,1,1)

    blur_augs = AugmentationSequential(
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        kornia.augmentation.RandomGrayscale(p=0.1),
        kornia.augmentation.RandomSolarize(p=0.1),
        kornia.augmentation.RandomResizedCrop((224,224), scale=(.9,.9), ratio=(1,1), p=1.0),
        data_keys=["input"],
    )

# ### MindEye modules
model = MindEyeModule()

model.ridge = RidgeRegression(num_voxels_list, out_features=hidden_dim)
utils.count_params(model.ridge)
utils.count_params(model)

from models_our import BrainNetwork
model.backbone = BrainNetwork(h=hidden_dim, in_dim=hidden_dim, seq_len=1, n_blocks=n_blocks,
                          clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim, 
                          blurry_recon=blurry_recon, clip_scale=clip_scale)
utils.count_params(model.backbone)
utils.count_params(model)

if use_prior:
    from models_our import *

    out_dim = clip_emb_dim
    depth = 6
    dim_head = 52
    heads = clip_emb_dim//52 # heads * dim_head = clip_emb_dim
    timesteps = 100

    prior_network = PriorNetwork(dim=out_dim, depth=depth, dim_head=dim_head, heads=heads, causal=False, num_tokens = clip_seq_dim, learned_query_mode="pos_emb")
    model.diffusion_prior = BrainDiffusionPrior(net=prior_network, image_embed_dim=out_dim, condition_on_text_encodings=False, timesteps=timesteps, cond_drop_prob=0.2, image_embed_scale=None)
    utils.count_params(model.diffusion_prior)
    utils.count_params(model)

# === Enable LoRA / Skip-LoRA finetuning if requested ===
if finetune_mode in ("lora", "lora+skip", "skip-lora"):
    # Replace Linear layers with LoRA and/or attach Skip-LoRA
    if finetune_mode in ("lora", "lora+skip"):
        model.backbone.enable_lora(rank=lora_rank, alpha=lora_alpha)
    if finetune_mode in ("skip-lora", "lora+skip"):
        # raw voxel dim for current subject
        v_in_dim = num_voxels[f'subj0{subj}']
        model.backbone.enable_skip_lora(
            v_in_dim=v_in_dim,
            activation="gelu",
            rank=lora_rank,
            alpha=lora_alpha,
            include_final=False,
            include_align=False,
        )
    # freeze all base params of backbone; train only LoRA/Skip-LoRA
    freeze_module_params(model.backbone)
    # re-enable LoRA params
    for n, p in model.backbone.named_parameters():
        if getattr(p, 'is_lora', False) or ('lora_' in n):
            p.requires_grad = True
    # keep RidgeRegression trainable (subject adapter)
    for p in model.ridge.parameters():
        p.requires_grad = True
    if use_prior:
        freeze_module_params(model.diffusion_prior)
    print_trainable_params_breakdown(model)

# ### Setup optimizer / lr / ckpt saving
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']  # 不参与权重衰减的参数名片段
opt_grouped_parameters = [
    {'params': trainable_params(model.ridge), 'weight_decay': 1e-2},
    {'params': [p for n, p in model.backbone.named_parameters() if p.requires_grad and (not any(nd in n for nd in no_decay))], 'weight_decay': 1e-2},
    {'params': [p for n, p in model.backbone.named_parameters() if p.requires_grad and (any(nd in n for nd in no_decay))], 'weight_decay': 0.0},
]
if use_prior:
    opt_grouped_parameters.extend([
        {'params': [p for n, p in model.diffusion_prior.named_parameters() if p.requires_grad and (not any(nd in n for nd in no_decay))], 'weight_decay': 1e-2},
        {'params': [p for n, p in model.diffusion_prior.named_parameters() if p.requires_grad and (any(nd in n for nd in no_decay))], 'weight_decay': 0.0}
    ])
optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=max_lr)

if lr_scheduler_type == 'linear':
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=int(np.floor(num_epochs*num_iterations_per_epoch)), last_epoch=-1)
elif lr_scheduler_type == 'cycle':
    total_steps=int(np.floor(num_epochs*num_iterations_per_epoch))
    print("total_steps", total_steps)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, final_div_factor=1000, last_epoch=-1, pct_start=2/num_epochs)

def save_ckpt(tag):
    ckpt_path = outdir+f'/{tag}.pth'
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save({
            'epoch': epoch,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'train_losses': losses,
            'test_losses': test_losses,
            'lrs': lrs,
            }, ckpt_path)
    print(f"\n---saved {outdir}/{tag} ckpt!---\n")

def load_ckpt(tag,load_lr=True,load_optimizer=True,load_epoch=True,strict=True,outdir=outdir,multisubj_loading=False): 
    print(f"\n---loading {outdir}/{tag}.pth ckpt---\n")
    checkpoint = torch.load(outdir+'/last.pth', map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    if multisubj_loading: # remove incompatible ridge layer that will otherwise error
        state_dict.pop('ridge.linears.0.weight',None)
    model.load_state_dict(state_dict, strict=strict)
    if load_epoch:
        globals()["epoch"] = checkpoint['epoch']
        print("Epoch",epoch)
    if load_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if load_lr:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    del checkpoint

print("\nDone with model preparations!")
num_params = utils.count_params(model)

# Global memory bank for semantic alignment (image_id -> {feat, label})
semantic_mem = {}

# # Weights and Biases
if local_rank==0 and wandb_log: # 仅主进程进行 wandb 日志记录
    print(f"wandb {wandb_project} run {model_name}")
    # need to configure wandb beforehand in terminal with "wandb init"!
    wandb_config = {
      "model_name": model_name,
      "global_batch_size": global_batch_size,
      "batch_size": batch_size,
      "num_epochs": num_epochs,
      "num_sessions": num_sessions,
      "num_params": num_params,
      "clip_scale": clip_scale,
      "prior_scale": prior_scale,
      "blur_scale": blur_scale,
      "ssp_margin": ssp_margin,
      "ssp_use_clip_categories": ssp_use_clip_categories,
      "ssp_mb_weight": ssp_mb_weight,
      "ssp_mb_size": ssp_mb_size,
      "ssp_mb_warmup": ssp_mb_warmup,
      "use_image_aug": use_image_aug,
      "max_lr": max_lr,
      "mixup_pct": mixup_pct,
      "num_samples_per_epoch": num_samples_per_epoch,
      "num_test": num_test,
      "ckpt_interval": ckpt_interval,
      "ckpt_saving": ckpt_saving,
      "seed": seed,
      "distributed": distributed,
      "num_devices": num_devices,
      "world_size": world_size,
      "train_url": train_url,
      "test_url": test_url,
    }
    print("wandb_config:\n",wandb_config)
    print("wandb_id:",model_name)
    wandb.init(
        id=model_name,
        project=wandb_project,
        name=model_name,
        config=wandb_config,
        resume="allow",
    )
else:
    wandb_log = False

# # Main
epoch = 0
losses, test_losses, lrs = [], [], []
torch.cuda.empty_cache()

# load multisubject stage1 ckpt if set
if multisubject_ckpt is not None:
    load_ckpt("last", outdir=multisubject_ckpt, load_lr=False, load_optimizer=False, load_epoch=False, strict=False, multisubj_loading=True)

# Prepare semantic alignment resources (optional)
semantic_active = semantic_align_weight > 0.0
_, pair2val_mean = (None, None)
if semantic_active:
    _, pair2val_mean = load_reference_mats(semantic_align_ref_dir, subj_ids=(2,3,4,5,6,7,8))

train_dls = [train_dl[f'subj0{s}'] for s in subj_list]
model, optimizer, *train_dls, lr_scheduler = accelerator.prepare(model, optimizer, *train_dls, lr_scheduler)

print(f"{model_name} starting with epoch {epoch} / {num_epochs}")
progress_bar = tqdm(range(epoch,num_epochs), ncols=1200, disable=(local_rank!=0))
test_image, test_voxel = None, None
mse = nn.MSELoss()
l1 = nn.L1Loss()
soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, num_epochs - int(mixup_pct * num_epochs))

# Build category names and id2lab mapping (used by feat_aug and semantic alignment)
ssp_category_names = get_ssp_category_names()
id2lab = _ssp_build_imageid_to_label_from_json(json_file_main, ssp_category_names)

# Build feature-space augmenter (optional, uses offline global stats)
sigma_subj_list = [int(s.strip()) for s in str(feat_sigma_subjs).split(',') if s.strip()]
feat_aug = FeatureStatsAugmenter(
    stats_dir=feat_stats_dir,
    clip_seq_dim=clip_seq_dim,
    device=device,
    ssp_category_names=ssp_category_names,
    sigma_subjs=sigma_subj_list,
)

for epoch in progress_bar:
    model.train()

    fwd_percent_correct = 0.
    bwd_percent_correct = 0.
    test_fwd_percent_correct = 0.
    test_bwd_percent_correct = 0.

    recon_cossim = 0.
    recon_mse = 0.

    loss_clip_total = 0.
    loss_blurry_total = 0.
    loss_blurry_cont_total = 0.
    test_loss_clip_total = 0.

    loss_prior_total = 0.
    test_loss_prior_total = 0.
    loss_ssp_mb_total = 0.
    sem_classes_total = 0.
    sem_pairs_total = 0.
    loss_sem_total = 0.

    blurry_pixcorr = 0.
    test_blurry_pixcorr = 0. # needs >.456 to beat low-level subj01 results in mindeye v1

    # pre-load all batches for this epoch (it's MUCH faster to pre-load in bulk than to separate loading per batch)
    voxel_iters = {} # empty dict because diff subjects have differing # of voxels
    image_iters = torch.zeros(num_iterations_per_epoch, batch_size*len(subj_list), 3, 224, 224).float()
    perm_iters, betas_iters, select_iters = {}, {}, {}
    image_id_iters = torch.zeros(num_iterations_per_epoch, batch_size*len(subj_list), dtype=torch.long)
    for s, train_dl in enumerate(train_dls):
        with torch.cuda.amp.autocast(dtype=data_type):
            iter = -1
            for behav0, past_behav0, future_behav0, old_behav0 in train_dl: 
                # Load images to cpu from hdf5 (requires sorted indexing)
                image_idx = behav0[:,0,0].cpu().long().numpy()				# [52247,  2019,  8926, 50102, 16548, 70129, 15555,   204, 40422, 11018]
                image0, image_sorted_idx = np.unique(image_idx, return_index=True)                
                if len(image0) != len(image_idx): # hdf5 cant handle duplicate indexing
                    continue
                iter += 1
                image0 = torch.tensor(images[image0], dtype=data_type)
                image_iters[iter,s*batch_size:s*batch_size+batch_size] = image0
                image_id_iters[iter, s*batch_size:s*batch_size+batch_size] = torch.from_numpy(image_idx)

                # Load voxels for current batch, matching above indexing
                voxel_idx = behav0[:,0,5].cpu().long().numpy()
                voxel_sorted_idx = voxel_idx[image_sorted_idx]
                voxel0 = voxels[f'subj0{subj_list[s]}'][voxel_sorted_idx]
                voxel0 = torch.Tensor(voxel0).unsqueeze(1)

                if epoch < int(mixup_pct * num_epochs):
                    voxel0, perm, betas, select = utils.mixco(voxel0)
                    perm_iters[f"subj0{subj_list[s]}_iter{iter}"] = perm
                    betas_iters[f"subj0{subj_list[s]}_iter{iter}"] = betas
                    select_iters[f"subj0{subj_list[s]}_iter{iter}"] = select

                voxel_iters[f"subj0{subj_list[s]}_iter{iter}"] = voxel0

                if iter >= num_iterations_per_epoch-1:
                    break

    # 准备 SSP 所需的组件（避免在循环内重复初始化）
    ssp_need_labels = (ssp_mb_weight > 0.0)
    if ssp_need_labels and ('ssp_assigner' not in globals()):
        globals()['ssp_assigner'] = SSPCategoryAssigner(
            use_clip=bool(ssp_use_clip_categories),
            clip_variant="ViT-L/14",
            device=device,
            category_names=ssp_category_names,
        )
    if ('ssp_bank_feats' not in globals()):
        globals()['ssp_bank_feats'] = None
        globals()['ssp_bank_labels'] = None
        globals()['ssp_bank_ptr'] = 0
        globals()['ssp_bank_count'] = 0
    for train_i in range(num_iterations_per_epoch):
        with torch.cuda.amp.autocast(dtype=data_type):
            optimizer.zero_grad()
            loss=0.

            voxel_list = [voxel_iters[f"subj0{s}_iter{train_i}"].detach().to(device, non_blocking=True) for s in subj_list]
            image = image_iters[train_i].detach()
            image = image.to(device, non_blocking=True)

            if use_image_aug: 
                image = img_augment(image)

            clip_target = clip_img_embedder(image)  # 冻结的 CLIP 生成目标 token
            assert not torch.any(torch.isnan(clip_target))

            if epoch < int(mixup_pct * num_epochs):
                perm_list = [perm_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
                perm = torch.cat(perm_list, dim=0)
                betas_list = [betas_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
                betas = torch.cat(betas_list, dim=0)
                select_list = [select_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
                select = torch.cat(select_list, dim=0)

            voxel_ridge_list = [model.ridge(voxel_list[si],si) for si,s in enumerate(subj_list)]
            voxel_ridge = torch.cat(voxel_ridge_list, dim=0)

            if finetune_mode in ("skip-lora", "lora+skip"):
                skip_in = torch.cat(voxel_list, dim=0) if len(subj_list) == 1 else None
                backbone, clip_voxels, blurry_image_enc_, skip_stats = model.backbone(voxel_ridge, skip_input=skip_in, return_skip_stats=True)  # 主干网络输出：
            else:
                backbone, clip_voxels, blurry_image_enc_ = model.backbone(voxel_ridge)  # 主干网络输出：

            # Optional: feature-space augmentation using offline stats (subjects 2..8)
            if feat_aug is not None:
                img_ids_cur = image_id_iters[train_i].to(device)
                labels_aug = get_labels_for_batch(img_ids_cur, id2lab, device)
                clip_voxels = feat_aug(clip_voxels, labels_aug)

            if clip_scale>0:
                clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

            if use_prior:
                loss_prior, prior_out = model.diffusion_prior(text_embed=backbone, image_embed=clip_target)
                loss_prior_total += loss_prior.item()
                loss_prior *= prior_scale
                loss += loss_prior

                recon_cossim += nn.functional.cosine_similarity(prior_out, clip_target).mean().item()
                recon_mse += mse(prior_out, clip_target).item()

            if clip_scale>0:
                if epoch < int(mixup_pct * num_epochs):                
                    loss_clip = utils.mixco_nce(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=.006,
                        perm=perm, betas=betas, select=select)
                else:
                    epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
                    loss_clip = utils.soft_clip_loss(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=epoch_temp)

                loss_clip_total += loss_clip.item()
                loss_clip *= clip_scale
                loss += loss_clip


            # Semantic structure alignment loss over full fine-tune set via memory bank (optional)
            if semantic_active:
                # Normalize current batch features
                feats_now = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                ids_now = image_id_iters[train_i].long().cpu().numpy()
                labels_now = get_labels_for_batch(image_id_iters[train_i].to(device), id2lab, device)
                labels_now = [str(ssp_category_names[i]) for i in labels_now]

                # Update per-image memory with EMA
                mem = semantic_mem
                m = float(semantic_align_momentum)
                for idx_in_batch, img_id in enumerate(ids_now):
                    key = int(img_id)
                    lab = labels_now[idx_in_batch]
                    v = feats_now[idx_in_batch].detach()
                    if key in mem:
                        old = mem[key]['feat']
                        mem[key]['feat'] = (m * old + (1.0 - m) * v).detach()
                        mem[key]['label'] = lab  # update in case
                    else:
                        mem[key] = {'feat': v, 'label': lab}

                # Build label -> list of features from memory
                label_groups = defaultdict(list)
                for rec in mem.values():
                    label_groups[rec['label']].append(rec['feat'])
                # Also include current batch features (kept with gradients)
                for idx_in_batch, lab in enumerate(labels_now):
                    label_groups[lab].append(feats_now[idx_in_batch])

                # Build prototypes for labels meeting global min count
                labs = []
                protos = []
                for lab, vecs in label_groups.items():                    
                    if len(vecs) < 3:
                        continue
                    X = torch.stack(vecs, dim=0)
                    p = nn.functional.normalize(X.mean(dim=0, keepdim=False), dim=-1)
                    labs.append(lab)
                    protos.append(p)
                if len(protos) >= 2:
                    P = torch.stack(protos, dim=0)
                    S_cur = (P @ P.t()).clamp(-1, 1)

                    # Build target S according to strategy
                    n = len(labs)
                    S_tgt = torch.full((n, n), float('nan'), device=S_cur.device)
                    mask = torch.zeros((n, n), dtype=torch.bool, device=S_cur.device)
                    for i in range(n):
                        S_tgt[i, i] = 1.0

                    for i in range(n):
                        for j in range(i+1, n):
                            a, b = labs[i], labs[j]
                            key = (a, b) if a < b else (b, a)
                            if (pair2val_mean is not None) and (key in pair2val_mean):
                                val = pair2val_mean[key]
                                S_tgt[i, j] = val
                                S_tgt[j, i] = val
                                mask[i, j] = True
                                mask[j, i] = True

                    # logging helpers: count classes used and valid pair count (unique off-diagonals)
                    sem_classes_total += n
                    sem_pairs_total += int((mask.sum().item()) // 2)

                    if mask.any():
                        diff = S_cur[mask] - S_tgt[mask]
                        loss_sem = (diff * diff).mean()
                        loss = loss + semantic_align_weight * loss_sem
                        # accumulate semantic loss for logging
                        loss_sem_total += float(loss_sem.item())

            if (ssp_mb_weight > 0.0):
                emb = voxel_ridge[:,0,:].float()
                img_ids = image_id_iters[train_i].to(device)
                labels = globals()['ssp_assigner'].labels_for_batch(image, img_ids)
                # 初始化 bank（首次）
                if globals()['ssp_bank_feats'] is None:
                    D = emb.shape[-1]
                    globals()['ssp_bank_feats'] = torch.zeros((ssp_mb_size, D), dtype=torch.float32, device=device)
                    globals()['ssp_bank_labels'] = torch.zeros((ssp_mb_size,), dtype=torch.long, device=device)
                    globals()['ssp_bank_ptr'] = 0
                    globals()['ssp_bank_count'] = 0
                # 计算 bank 损失（warmup 后）
                if int(globals()['ssp_bank_count']) >= int(ssp_mb_warmup):
                    Q = int(globals()['ssp_bank_count'])
                    l_mb = ssp_triplet_loss_cosine_xbank(
                        emb, labels,
                        globals()['ssp_bank_feats'][:Q], globals()['ssp_bank_labels'][:Q],
                        margin=ssp_margin,
                    )
                    loss_ssp_mb_total += l_mb.item()
                    loss = loss + ssp_mb_weight * l_mb
                # 更新 bank（detach，fp32，L2 归一化）
                with torch.no_grad():
                    e = nn.functional.normalize(emb.to(torch.float32), dim=-1).detach()
                    bsz = e.size(0)
                    ptr = int(globals()['ssp_bank_ptr'])
                    end = ptr + bsz
                    if end <= ssp_mb_size:
                        globals()['ssp_bank_feats'][ptr:end].copy_(e)
                        globals()['ssp_bank_labels'][ptr:end].copy_(labels.detach())
                    else:
                        first = ssp_mb_size - ptr
                        globals()['ssp_bank_feats'][ptr:].copy_(e[:first])
                        globals()['ssp_bank_labels'][ptr:].copy_(labels[:first].detach())
                        rem = bsz - first
                        if rem > 0:
                            globals()['ssp_bank_feats'][:rem].copy_(e[first:])
                            globals()['ssp_bank_labels'][:rem].copy_(labels[first:].detach())
                    globals()['ssp_bank_ptr'] = (ptr + bsz) % ssp_mb_size
                    globals()['ssp_bank_count'] = min(ssp_mb_size, int(globals()['ssp_bank_count']) + bsz)

            if blurry_recon:     
                image_enc_pred, transformer_feats = blurry_image_enc_

                image_enc = autoenc.encode(2*image-1).latent_dist.mode() * 0.18215
                loss_blurry = l1(image_enc_pred, image_enc)
                loss_blurry_total += loss_blurry.item()

                if epoch < int(mixup_pct * num_epochs):
                    image_enc_shuf = image_enc[perm]
                    betas_shape = [-1] + [1]*(len(image_enc.shape)-1)
                    image_enc[select] = image_enc[select] * betas[select].reshape(*betas_shape) + \
                        image_enc_shuf[select] * (1 - betas[select]).reshape(*betas_shape)

                image_norm = (image - mean)/std
                image_aug = (blur_augs(image) - mean)/std
                _, cnx_embeds = cnx(image_norm)
                _, cnx_aug_embeds = cnx(image_aug)

                cont_loss = utils.soft_cont_loss(
                    nn.functional.normalize(transformer_feats.reshape(-1, transformer_feats.shape[-1]), dim=-1),
                    nn.functional.normalize(cnx_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                    nn.functional.normalize(cnx_aug_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                    temp=0.2)
                loss_blurry_cont_total += cont_loss.item()

                loss += (loss_blurry + 0.1*cont_loss) * blur_scale #/.18215

            # Skip-LoRA nonlinear correlation regularizer
            if finetune_mode in ("skip-lora", "lora+skip"):
                if isinstance(skip_stats, list) and len(skip_stats) > 0:
                    corr_vals = []
                    # If final mapping skip adapter is present, exclude it from Lskip; otherwise include all block pairs
                    for lin, sk in skip_stats:
                        # ensure shapes (B, D)
                        lin_flat = lin.view(lin.size(0), -1)
                        sk_flat = sk.view(sk.size(0), -1)
                        corr = abs_pearson_corr(lin_flat, sk_flat)
                        corr_vals.append(corr.mean())
                    if len(corr_vals):
                        l_skip = torch.stack(corr_vals).mean()
                        loss = loss + skip_loss_weight * l_skip

            if clip_scale>0:
                # forward and backward top 1 accuracy        
                labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device) 
                fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
                bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()

            if blurry_recon:
                with torch.no_grad():
                    # only doing pixcorr eval on a subset of the samples per batch because its costly & slow to compute autoenc.decode()
                    random_samps = np.random.choice(np.arange(len(image)), size=len(image)//5, replace=False)
                    blurry_recon_images = (autoenc.decode(image_enc_pred[random_samps]/0.18215).sample/ 2 + 0.5).clamp(0,1)
                    pixcorr = utils.pixcorr(image[random_samps], blurry_recon_images)
                    blurry_pixcorr += pixcorr.item()

            utils.check_loss(loss)
            accelerator.backward(loss)
            optimizer.step()

            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])

            if lr_scheduler_type is not None:
                lr_scheduler.step()

    model.eval()
    if local_rank==0:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=data_type): 
            for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dl):  
                # all test samples should be loaded per batch such that test_i should never exceed 0
                assert len(behav) == num_test

                ## Average same-image repeats ##
                if test_image is None:
                    voxel = voxels[f'subj0{subj}'][behav[:,0,5].cpu().long()].unsqueeze(1)

                    image = behav[:,0,0].cpu().long()

                    unique_image = torch.unique(image)
                    for im in unique_image:
                        locs = torch.where(im == image)[0]
                        if len(locs)==1:
                            locs = locs.repeat(3)
                        elif len(locs)==2:
                            locs = locs.repeat(2)[:3]
                        assert len(locs)==3
                        if test_image is None:
                            test_image = torch.Tensor(images[im][None])
                            test_voxel = voxel[locs][None]
                        else:
                            test_image = torch.vstack((test_image, torch.Tensor(images[im][None])))
                            test_voxel = torch.vstack((test_voxel, voxel[locs][None]))

                loss = torch.tensor(0.0, device=device)

                test_indices = torch.arange(len(test_voxel))[:300]
                voxel = test_voxel[test_indices].to(device)
                image = test_image[test_indices].to(device)
                assert len(image) == 300

                clip_target = clip_img_embedder(image.float())

                for rep in range(3):
                    voxel_ridge = model.ridge(voxel[:,rep],0) # 0th index of subj_list
                    if finetune_mode in ("skip-lora", "lora+skip"):
                        backbone0, clip_voxels0, blurry_image_enc_, _ = model.backbone(voxel_ridge, skip_input=voxel[:,rep], return_skip_stats=True)
                    else:
                        backbone0, clip_voxels0, blurry_image_enc_ = model.backbone(voxel_ridge)
                    if rep==0:
                        clip_voxels = clip_voxels0
                        backbone = backbone0
                    else:
                        clip_voxels += clip_voxels0
                        backbone += backbone0
                clip_voxels /= 3
                backbone /= 3

                if clip_scale>0:
                    clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                    clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

                # for some evals, only doing a subset of the samples per batch because of computational cost
                random_samps = np.random.choice(np.arange(len(image)), size=len(image)//5, replace=False)

                if use_prior:
                    loss_prior, contaminated_prior_out = model.diffusion_prior(text_embed=backbone[random_samps], image_embed=clip_target[random_samps])
                    test_loss_prior_total += loss_prior.item()
                    loss_prior *= prior_scale
                    loss += loss_prior

                if clip_scale>0:
                    loss_clip = utils.soft_clip_loss(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=.006)

                    test_loss_clip_total += loss_clip.item()
                    loss_clip = loss_clip * clip_scale
                    loss += loss_clip

                if blurry_recon:
                    image_enc_pred, _ = blurry_image_enc_
                    blurry_recon_images = (autoenc.decode(image_enc_pred[random_samps]/0.18215).sample / 2 + 0.5).clamp(0,1)
                    pixcorr = utils.pixcorr(image[random_samps], blurry_recon_images)
                    test_blurry_pixcorr += pixcorr.item()

                if clip_scale>0:
                    # forward and backward top 1 accuracy        
                    labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device) 
                    test_fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
                    test_bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()

                utils.check_loss(loss)                
                test_losses.append(loss.item())

            assert (test_i+1) == 1
            logs = {"train/loss": np.mean(losses[-(train_i+1):]),
                "test/loss": np.mean(test_losses[-(test_i+1):]),
                "train/lr": lrs[-1],
                "train/num_steps": len(losses),
                "test/num_steps": len(test_losses),
                "train/fwd_pct_correct": fwd_percent_correct / (train_i + 1),
                "train/bwd_pct_correct": bwd_percent_correct / (train_i + 1),
                "test/test_fwd_pct_correct": test_fwd_percent_correct / (test_i + 1),
                "test/test_bwd_pct_correct": test_bwd_percent_correct / (test_i + 1),
                "train/loss_clip_total": loss_clip_total / (train_i + 1),
                "train/loss_blurry_total": loss_blurry_total / (train_i + 1),
                "train/loss_blurry_cont_total": loss_blurry_cont_total / (train_i + 1),
                "train/loss_skiplora_total": l_skip / (train_i + 1),
                "train/loss_ssp_mb_total": (loss_ssp_mb_total / (train_i + 1)) if (ssp_mb_weight>0.0) else 0.0,
                "test/loss_clip_total": test_loss_clip_total / (test_i + 1),
                "train/blurry_pixcorr": blurry_pixcorr / (train_i + 1),
                "test/blurry_pixcorr": test_blurry_pixcorr / (test_i + 1),
                "train/recon_cossim": recon_cossim / (train_i + 1),
                "train/recon_mse": recon_mse / (train_i + 1),
                "train/loss_prior": loss_prior_total / (train_i + 1),
                "test/loss_prior": test_loss_prior_total / (test_i + 1),
                "train/loss_semantic": (loss_sem_total / (train_i + 1)) if semantic_active else 0.0,
                "train/sem_pairs_avg": (sem_pairs_total / (train_i + 1)) if semantic_active else 0.0,
                "train/sem_classes_avg": (sem_classes_total / (train_i + 1)) if semantic_active else 0.0,
                }

            # if finished training, save jpg recons if they exist
            if (epoch == num_epochs-1) or (epoch % ckpt_interval == 0):
                if blurry_recon:    
                    image_enc = autoenc.encode(2*image[:4]-1).latent_dist.mode() * 0.18215
                    # transform blurry recon latents to images and plot it
                    fig, axes = plt.subplots(1, 8, figsize=(10, 4))
                    jj=-1
                    for j in [0,1,2,3]:
                        jj+=1
                        axes[jj].imshow(utils.torch_to_Image((autoenc.decode(image_enc[[j]]/0.18215).sample / 2 + 0.5).clamp(0,1)))
                        axes[jj].axis('off')
                        jj+=1
                        axes[jj].imshow(utils.torch_to_Image((autoenc.decode(image_enc_pred[[j]]/0.18215).sample / 2 + 0.5).clamp(0,1)))
                        axes[jj].axis('off')

                    if wandb_log:
                        logs[f"test/blur_recons"] = wandb.Image(fig, caption=f"epoch{epoch:03d}")
                        plt.close()
                    else:
                        plt.show()

            progress_bar.set_postfix(**logs)

            if wandb_log: wandb.log(logs)

    # Save model checkpoint and reconstruct
    if (ckpt_saving) and (epoch % ckpt_interval == 0):
        save_ckpt(f'last')

    # wait for other GPUs to catch up if needed
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()

print("\n===Finished!===\n")
if ckpt_saving:
    save_ckpt(f'last')
