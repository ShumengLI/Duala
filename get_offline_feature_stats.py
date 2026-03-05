import os, re, json, argparse
from typing import Dict, List, Tuple
from collections import defaultdict
import pdb
import numpy as np
import torch
import torch.nn as nn
import webdataset as wds
import h5py

from models import BrainNetwork
import utils

def _split_by_node(urls):
    return urls

class MindEyeModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.ridge = None
        self.backbone = None
    def forward(self, x):
        return x

class RidgeRegression(nn.Module):
    def __init__(self, input_sizes: List[int], out_features: int):
        super().__init__()
        self.out_features = out_features
        self.linears = nn.ModuleList([nn.Linear(inp, out_features) for inp in input_sizes])
    def forward(self, x: torch.Tensor, subj_idx: int):
        return self.linears[subj_idx](x[:, 0]).unsqueeze(1)


def infer_ridge_shapes_from_ckpt(state_dict: Dict[str, torch.Tensor]) -> Tuple[List[int], int, int]:
    input_sizes = []
    out_features = None
    for k, v in state_dict.items():
        if k.startswith('ridge.linears.') and k.endswith('.weight'):
            parts = k.split('.')
            try:
                idx = int(parts[2])
            except Exception:
                continue
            in_features = v.shape[1]
            out_features = v.shape[0]
            input_sizes.append((idx, in_features))
    input_sizes = [inp for _, inp in sorted(input_sizes, key=lambda x: x[0])]
    if out_features is None or len(input_sizes) == 0:
        raise RuntimeError('Could not infer ridge shapes from checkpoint state_dict.')
    return input_sizes, out_features, len(input_sizes)

def _save_txt(path: str, lines: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for x in lines:
            f.write(str(x) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Path to training log dir containing last.pth')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--category_json_dir', type=str, default='./')
    parser.add_argument('--subj', type=str, default='1')
    parser.add_argument('--subjects', type=str, default='2,3,4,5,6,7,8')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16','bfloat16','float32'])
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--min_images_per_class', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default='./feat_dis')
    parser.add_argument('--token_granularity', type=str, default='tokenwise', choices=['per_token', 'tokenwise', 'mean'],
                        help='per_token: 将所有 token 视为独立观测并跨 T 聚合；tokenwise: 保留每个 token 位置的统计（输出 [T,D]）；mean: 先对 T 取均值再统计（每图等权）')
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}[args.dtype]
    subjects = [int(s.strip()) for s in args.subjects.split(',') if s.strip()]

    ckpt_path = os.path.join(args.ckpt_dir, 'last.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']

    input_sizes, hidden_dim, num_linears = infer_ridge_shapes_from_ckpt(state_dict)
    print(f'Inferred ridge: linears={num_linears}, hidden_dim={hidden_dim}, inputs={input_sizes}')      # Inferred ridge: linears=7, hidden_dim=4096, inputs=[14278, 15226, 13153, 13039, 17907, 12682, 14386]

    clip_seq_dim = 256
    clip_emb_dim = 1664

    # build model
    model = MindEyeModule()
    model.ridge = RidgeRegression(input_sizes=input_sizes, out_features=hidden_dim)
    model.backbone = BrainNetwork(h=hidden_dim, in_dim=hidden_dim, seq_len=1, n_blocks=4,
                                  clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim,
                                  blurry_recon=True, clip_scale=1)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f'Loaded checkpoint: missing={len(missing)}, unexpected={len(unexpected)}')
    model.to(device).eval()

    out_dir = os.path.join(args.output_dir, f'subj0{args.subj}')
    os.makedirs(out_dir, exist_ok=True)

    voxels: Dict[str, torch.Tensor] = {}
    linear_index_for_subj: Dict[int, int] = {}
    for s in subjects:
        with h5py.File(f"{args.data_path}/betas_all_subj0{s}_fp32_renorm.hdf5", "r") as f:
            v = torch.tensor(f['betas'][:]).to('cpu').to(dtype)
        voxels[f'subj0{s}'] = v
        nvox = v.shape[-1]
        match_idx = None
        for i, in_feats in enumerate(input_sizes):
            if in_feats == nvox:
                match_idx = i
                break
        if match_idx is not None:
            linear_index_for_subj[s] = match_idx
        else:
            print(f'[warn] no matching ridge linear for subj{s} with {nvox} voxels; skip subject')
    if len(linear_index_for_subj) == 0:
        raise RuntimeError('No usable subjects found for the checkpoint ridge shapes')

    per_subject: Dict[int, Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]] = {}

    for s in subjects:
        if s not in linear_index_for_subj:
            continue
        print(f'Processing subject {s} ...')
        img2cats: Dict[int, List[str]] = defaultdict(list)
        json_path = os.path.join(args.category_json_dir, f'category_image_idx_subj{s}.json')
        if os.path.isfile(json_path):
            raw = json.load(open(json_path, 'r'))
            for cat, lst in raw.items():
                seen_ids = set()
                for item in lst:
                    iid = None
                    if isinstance(item, int):
                        iid = int(item)
                    elif isinstance(item, str):
                        head = item.strip().split('_', 1)[0]
                        m = re.match(r'^(\d+)$', head) or re.match(r'^(\d+)', item)
                        if m:
                            iid = int(m.group(1))
                    if iid is not None and iid not in seen_ids:
                        img2cats[iid].append(cat)
                        seen_ids.add(iid)
            print(f'  categories in JSON: {len(raw)}; images covered: {len(img2cats)}')
        else:
            print(f'  [warn] JSON not found for subj{s}; will fallback to单一类别')

        nsessions_allsubj=np.array([40, 40, 32, 30, 40, 32, 40, 30])
        train_url = f"{args.data_path}/wds/subj0{s}/train/{{0..{nsessions_allsubj[s-1]-1}}}.tar"
        ds = (
            wds.WebDataset(train_url, resampled=False, nodesplitter=_split_by_node)
            .decode('torch')
            .rename(behav='behav.npy', past_behav='past_behav.npy', future_behav='future_behav.npy', olds_behav='olds_behav.npy')
            .to_tuple('behav', 'past_behav', 'future_behav', 'olds_behav')
        )
        dl = torch.utils.data.DataLoader(
            ds, batch_size=args.batch, shuffle=False, drop_last=False,
            pin_memory=True, num_workers=4, persistent_workers=True, prefetch_factor=2,
        )
        print(train_url)

        sum_x: Dict[str, torch.Tensor] = {}
        sum_x2: Dict[str, torch.Tensor] = {}
        counts: Dict[str, object] = defaultdict(int)
        vox_bank = voxels[f'subj0{s}']
        lin_idx = linear_index_for_subj[s]

        for behav, _, _, _ in dl:
            img_ids = behav[:, 0, 0].cpu().long()
            vox_ids = behav[:, 0, 5].cpu().long()
            all_vox = vox_bank[vox_ids]
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype if device.type == 'cuda' else torch.float32):
                voxel_ridge = model.ridge(all_vox.unsqueeze(1).to(device), lin_idx)
                _, clip_vox, _ = model.backbone(voxel_ridge)
                pdb.set_trace()
                if isinstance(clip_vox, tuple):
                    clip_vox = clip_vox[0]
                feats = clip_vox.to(torch.float64)  # [B, T, D]     # float16改成float64，解决后面的inf
                feats = feats.detach().to('cpu')                # [256, 256, 1664]

            for i in range(feats.shape[0]):
                iid = int(img_ids[i])
                cats = img2cats.get(iid, None)
                if cats is None or len(cats) == 0:
                    continue
                x_tok = feats[i]  # [T, D]
                T_len = int(x_tok.shape[0])

                if args.token_granularity == 'mean':
                    x = x_tok.mean(dim=0)            # [D]
                    x2 = x * x                       # [D]
                    for c in cats:
                        if c not in sum_x:
                            sum_x[c] = x.clone()
                            sum_x2[c] = x2.clone()
                            counts[c] = 1
                        else:
                            sum_x[c] += x
                            sum_x2[c] += x2
                            counts[c] = int(counts[c]) + 1
                elif args.token_granularity == 'tokenwise':
                    x = x_tok                         # [T, D]
                    x2 = x * x                        # [T, D]
                    for c in cats:
                        if c not in sum_x:
                            sum_x[c] = x.clone()
                            sum_x2[c] = x2.clone()
                            counts[c] = torch.ones((T_len,), dtype=torch.long)
                        else:
                            sum_x[c] += x
                            sum_x2[c] += x2
                            counts[c] = counts[c] + 1  # 按 token 位置分别 +1（广播标量）
                else:
                    x_sum = x_tok.sum(dim=0)                 # [D]
                    x2_sum = (x_tok * x_tok).sum(dim=0)      # [D]
                    for c in cats:
                        if c not in sum_x:
                            sum_x[c] = x_sum.clone()
                            sum_x2[c] = x2_sum.clone()
                            counts[c] = T_len
                        else:
                            sum_x[c] += x_sum
                            sum_x2[c] += x2_sum
                            counts[c] = int(counts[c]) + T_len

        cats_sorted = sorted(counts.keys())
        print(len(cats_sorted))
        MU_list, VAR_list, CNT_list = [], [], []
        cats_kept = []
        if args.token_granularity == 'tokenwise':
            for c in cats_sorted:
                n_vec = counts[c]
                min_cnt = int(torch.min(n_vec).item()) if isinstance(n_vec, torch.Tensor) else int(n_vec)
                if min_cnt < max(1, args.min_images_per_class):
                    pass
                sx = sum_x[c].to(torch.float64)       # [T,D]
                sx2 = sum_x2[c].to(torch.float64)     # [T,D]
                n = n_vec.to(torch.float64).view(-1, 1)  # [T,1]
                mu = (sx / n).to(torch.float64)                   # [T,D]
                var_diag = (sx2 / n - mu * mu).clamp(min=0.0)     # [T,D]
                MU_list.append(mu.unsqueeze(0))
                VAR_list.append(var_diag.unsqueeze(0))
                CNT_list.append(n_vec.view(1, -1))
                cats_kept.append(c)
                if torch.inf in var_diag:
                    print(f'  [warn] inf detected in var_diag for class {c} in subj{s}')

            if len(MU_list) == 0:
                print(f'  [warn] no classes kept for subj{s}; skipping')
                continue
            MU = torch.cat(MU_list, dim=0).to(torch.float32).cpu().numpy()      # [C,T,D] (78, 256, 1664)
            VAR = torch.cat(VAR_list, dim=0).to(torch.float32).cpu().numpy()    # [C,T,D] (78, 256, 1664)
            CNT = torch.cat(CNT_list, dim=0).to(torch.long).cpu().numpy()       # [C,T] (78, 256)

            np.savez(
                os.path.join(out_dir, f"subj{s:02d}_mu_tokenwise.npz"),
                categories=np.array(cats_kept, dtype=object),
                mu_token=MU.astype(np.float32),
                counts_token=CNT,
                token_granularity=args.token_granularity,
                note="per-subject tokenwise mu saved; per-subject sigma stored as subjXX_sigma_wrt_global_tokenwise.npz"
            )
            per_subject[s] = (cats_kept, MU, VAR, CNT)

    # pooled global stats across subjects
    if len(per_subject) > 0:
        cat_set = set()
        for _, (cats, _, _, _) in per_subject.items():
            cat_set.update(cats)
        cats_all = sorted(cat_set)
        idx = {c: i for i, c in enumerate(cats_all)}

        example = next(iter(per_subject.values()))
        _, MU_ex, _, CNT_ex = example
        if args.token_granularity == 'tokenwise':
            T_dim, D_dim = int(MU_ex.shape[1]), int(MU_ex.shape[2])
            mu_g = np.zeros((len(cats_all), T_dim, D_dim), dtype=np.float64)
            cnt_g = np.zeros((len(cats_all), T_dim), dtype=np.int64)

            for _, (cats, MU, _, CNT) in per_subject.items():
                for j, c in enumerate(cats):
                    i = idx[c]
                    n_vec = CNT[j].astype(np.int64)  # [T]
                    valid = n_vec > 0
                    mu_g[i][valid] += MU[j][valid] * n_vec[valid][..., None]
                    cnt_g[i][valid] += n_vec[valid]
            for i in range(len(cats_all)):
                valid = cnt_g[i] > 0
                if np.any(valid):
                    mu_g[i][valid] /= cnt_g[i][valid][..., None]

            _save_txt(os.path.join(out_dir, 'global_categories.txt'), cats_all)
            np.savez(
                os.path.join(out_dir, 'global_stats_tokenwise.npz'),
                categories=np.array(cats_all, dtype=object),
                mu_global_token=mu_g.astype(np.float32),
                counts_total_token=cnt_g,
                token_granularity=args.token_granularity,
                note="only shared global mu saved; per-subject sigma saved separately"
            )
            print(f"Global saved (tokenwise mu only): classes={len(cats_all)}, T={T_dim}, D={D_dim}")

            mu_g_t = torch.from_numpy(mu_g).to(torch.float64)           # [Cg, T, D]
            for s, (cats, MU, VAR, CNT) in per_subject.items():
                MU_t = torch.from_numpy(MU).to(torch.float64)            # [C, T, D]
                VAR_t = torch.from_numpy(VAR).to(torch.float64)          # [C, T, D]
                C_local = len(cats)
                Ex = MU_t                                               # E[x]
                Ex2 = VAR_t + MU_t * MU_t                               # E[x^2]
                mu_g_this = []
                for c in cats:
                    i = idx[c]
                    mu_g_this.append(mu_g_t[i:i+1])
                mu_g_this = torch.cat(mu_g_this, dim=0)                  # [C, T, D]
                print(mu_g_this.min(), mu_g_this.max())

                var_wrt_global = (Ex2 - 2.0 * mu_g_this * Ex + mu_g_this * mu_g_this)
                var_wrt_global = var_wrt_global.clamp_min(0.0).to(torch.float32).cpu().numpy()
                print(var_wrt_global.min(), var_wrt_global.max())

                _save_txt(os.path.join(out_dir, f"subj{s:02d}_categories.txt"), cats)
                np.savez(
                    os.path.join(out_dir, f"subj{s:02d}_sigma_wrt_global_tokenwise.npz"),
                    categories=np.array(cats, dtype=object),
                    var_diag_wrt_global_token=var_wrt_global,
                    counts_token=CNT,
                    note="sigma per subject computed relative to shared global mu_token"
                )
                print(f"  saved subj{s:02d} sigma_wrt_global (tokenwise): classes={C_local}, T={T_dim}, D={D_dim}")


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512')
    utils.seed_everything(42)
    main()

