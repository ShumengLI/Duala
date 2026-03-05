"""
ssp_sdp.py
SSP (Semantic Structure Preservation) and SDP (Subject Distribution Prior) module helpers.
Contains: category labeling, feature augmentation, triplet loss, and semantic alignment utilities.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from typing import List, Optional


# ---------------------------------------------------------------------------
# Semantic alignment (cross-subject relational consistency)
# ---------------------------------------------------------------------------

def load_reference_mats(ref_dir: str, subj_ids=(2, 3, 4, 5, 6, 7, 8)):
    """Load per-subject category lists and cosine matrices.
    Returns: refs = [{sid, labels, S, label_to_idx}], and also a mean pair dict.
    """
    refs = []
    pair_vals = {}
    labels_all = set()
    ok = 0
    nsessions_allsubj = np.array([40, 40, 32, 30, 40, 32, 40, 30])
    for sid in subj_ids:
        cat_path = os.path.join(ref_dir, f"subj{sid:02d}_session{nsessions_allsubj[sid-1]}_categories.txt")
        mat_path = os.path.join(ref_dir, f"subj{sid:02d}_session{nsessions_allsubj[sid-1]}_cosine.npy")
        if (not os.path.isfile(cat_path)) or (not os.path.isfile(mat_path)):
            continue
        try:
            with open(cat_path, "r") as f:
                cats = [line.strip() for line in f if line.strip()]
            S = np.load(mat_path)
            if S.shape[0] != len(cats):
                print(f"[semantic] size mismatch for subj{sid:02d}: S={S.shape}, cats={len(cats)}; skip")
                continue
            label_to_idx = {c: i for i, c in enumerate(cats)}
            refs.append({"sid": sid, "labels": cats, "S": S, "label_to_idx": label_to_idx})
            ok += 1
        except Exception as e:
            print(f"[semantic] failed to read subj{sid:02d}: {e}")
            continue
        labels_all.update(cats)
        for i in range(len(cats)):
            for j in range(i + 1, len(cats)):
                a, b = cats[i], cats[j]
                key = (a, b) if a < b else (b, a)
                pair_vals.setdefault(key, []).append(float(S[i, j]))
    if ok == 0:
        print("[semantic] no reference similarity found; disable semantic align")
        return None, None
    pair2val_mean = {k: float(np.mean(v)) for k, v in pair_vals.items()}
    print(f"[semantic] loaded {ok} reference subjects; pair entries={len(pair2val_mean)}, label_space={len(labels_all)}")
    return refs, pair2val_mean


# ---------------------------------------------------------------------------
# SSP category names and label helpers
# ---------------------------------------------------------------------------

def get_ssp_category_names() -> List[str]:
    return [
        "person", "bicycle", "car", "motorcycle", "airplane", "train", "truck", "boat", "traffic",
        "fire hydrant", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball", "skateboard", "surfboard",
        "tennis racket", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "laptop", "keyboard", "cell phone", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "computer",
        "people", "A variety of fruits", "girl", "boy", "men", "woman", "computer", "surf"
    ]


def _ssp_build_imageid_to_label_from_json(json_path: str, category_names: list):
    """
    Load a json mapping {category_name: ["<image_id>_<localidx>", ...], ...}
    and return a dict {int(image_id): int(category_idx)}. If an image appears in multiple
    categories, the first encountered mapping is used.
    """
    if (json_path is None) or (not os.path.isfile(json_path)):
        return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        name_to_idx = {n: i for i, n in enumerate(category_names)}
        id2lab = {}
        for cat, arr in data.items():
            if cat not in name_to_idx:
                continue
            lab = name_to_idx[cat]
            for s in arr:
                try:
                    imid = int(str(s).split('_')[0])
                except Exception:
                    continue
                if imid not in id2lab:
                    id2lab[imid] = lab
        return id2lab
    except Exception as e:
        print(f"[SSP] Failed loading category json: {e}")
        return None


def get_labels_for_batch(image_ids, id2lab: dict, device) -> torch.Tensor:
    """
    image_ids: (B,) tensor or array
    id2lab: dict[int, int]
    Returns LongTensor of shape (B,); unknown entries are -1.
    """
    labels = torch.full((len(image_ids),), -1, dtype=torch.long, device=device)
    for i, imid in enumerate(image_ids.tolist()):
        if imid in id2lab:
            labels[i] = id2lab[imid]
    return labels


class SSPCategoryAssigner:
    """Assign category indices per image batch.
    Priority: CLIP pseudo-labels (if enabled) -> all zeros.
    """

    def __init__(self, use_clip: bool, clip_variant: str, device: torch.device,
                 category_names: List[str]):
        self.device = device
        self.category_names = category_names
        self.use_clip = use_clip
        self._clip_model = None
        self._text_feats = None
        if self.use_clip:
            try:
                import clip as _clip
                self._clip = _clip
                self._clip_model, _ = _clip.load(clip_variant, device=device)
                self._clip_model.eval()
                with torch.no_grad():
                    text_tokens = _clip.tokenize(category_names).to(device)
                    self._text_feats = self._clip_model.encode_text(text_tokens)
                    self._text_feats = self._text_feats / self._text_feats.norm(dim=-1, keepdim=True)
                # image norm used by CLIP preprocess; our images are already 224 and in [0,1]
                self._img_norm = transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)
                )
            except Exception as e:
                print(f"[SSP] CLIP not available for pseudo-labels: {e}")
                self.use_clip = False

    @torch.no_grad()
    def labels_for_batch(self, images: torch.Tensor, image_ids: torch.Tensor) -> torch.Tensor:
        """
        images: (B,3,224,224) in [0,1]
        image_ids: (B,) long
        returns labels LongTensor shape (B,)
        """
        B = images.shape[0]
        labels = torch.full((B,), -1, dtype=torch.long, device=images.device)
        # If CLIP available, predict labels
        if self.use_clip:
            need_mask = labels < 0
            if need_mask.any():
                imgs = images[need_mask]
                imgs = self._img_norm(imgs)
                feats = self._clip_model.encode_image(imgs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                sims = feats @ self._text_feats.T
                pred = sims.argmax(dim=-1)
                labels[need_mask] = pred
        # Fallback: set remaining to 0
        labels[labels < 0] = 0
        return labels


# ---------------------------------------------------------------------------
# SDP: Feature-space augmentation using offline Gaussian stats
# ---------------------------------------------------------------------------

class FeatureStatsAugmenter:
    def __init__(self, stats_dir: str, clip_seq_dim: int, device: torch.device,
                 sigma_scale: float = 0.5, p: float = 0.5,
                 ssp_category_names: Optional[List[str]] = None,
                 sigma_subjs: Optional[List[int]] = None):
        self.device = device
        self.scale = float(sigma_scale)
        self.p = float(p)
        self.clip_T = int(clip_seq_dim)

        self.available = False
        tokenwise_path = os.path.join(stats_dir, 'global_stats_tokenwise.npz')
        self.tokenwise = False
        cats_stats = None
        if os.path.isfile(tokenwise_path):
            data = np.load(tokenwise_path, allow_pickle=True)
            mu = torch.from_numpy(data['mu_global_token']).to(torch.float32)   # [C, T, D]
            cats_stats = list(map(str, data['categories']))
            self.tokenwise = True
            mu_cls = mu.mean(dim=1, keepdim=True)

        self.stats_cat_to_idx = {str(n): i for i, n in enumerate(cats_stats)}
        self.mu_tok = mu.to(self.device)       # [C, T, D]
        self.mu_cls = mu_cls.to(self.device)   # [C, 1, D]

        sigma_list = []  # list of [C, T, D]
        if sigma_subjs is None:
            sigma_subjs = []
        for sid_ in sigma_subjs:
            sid_int = int(sid_)
            p_tok = os.path.join(stats_dir, f"subj{sid_int:02d}_sigma_wrt_global_tokenwise.npz")
            if os.path.isfile(p_tok):
                sd = np.load(p_tok, allow_pickle=True)
                var_s = torch.from_numpy(sd['var_diag_wrt_global_token']).to(torch.float32)  # [C, T, D]
            sigma_list.append(var_s)

        sigma_stack = torch.stack(sigma_list, dim=0)          # [S, C, T, D]
        self.var_sigma_stack = sigma_stack.to(self.device)    # [S, C, T, D]
        self.var_sigma_mean = self.var_sigma_stack.mean(dim=0)                    # [C, T, D]
        self.std_sigma_mean = self.var_sigma_mean.clamp_min(0.0).sqrt()           # [C, T, D]

        self.label_to_stats_idx = None
        if ssp_category_names is not None:
            mapping = {}
            for lid, name in enumerate(ssp_category_names):
                mapping[lid] = self.stats_cat_to_idx.get(str(name), -1)
            self.label_to_stats_idx = mapping

        self.available = True

    @torch.no_grad()
    def __call__(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        feats: [B, T, D] (brain clip tokens or image clip tokens)
        labels: [B] (category id aligned to SSP names; -1 = unknown)
        return: augmented feats (same shape)
        """
        if (not self.available) or (feats is None) or (feats.numel() == 0):
            return feats
        B, T, D = feats.shape
        assert T == self.clip_T, "Token length mismatch after normalization"
        if self.p >= 1.0:
            do_mask = torch.ones((B,), dtype=torch.bool, device=feats.device)
        elif self.p <= 0.0:
            return feats
        else:
            do_mask = torch.bernoulli(torch.full((B,), self.p, device=feats.device)).bool()

        out = feats
        with torch.cuda.amp.autocast(enabled=False):
            x = feats.to(torch.float32)
            y = x.clone()
            for i in range(B):
                if not bool(do_mask[i]):
                    continue
                lid = int(labels[i].item())
                if lid < 0:
                    continue
                sid = None
                if self.label_to_stats_idx is not None:
                    sid = self.label_to_stats_idx.get(lid, -1)
                if (sid is None) or (sid < 0) or (sid >= self.mu_tok.shape[0]):
                    sid = lid if (0 <= lid < self.mu_tok.shape[0]) else -1
                if sid < 0:
                    continue

                if self.var_sigma_stack.shape[0] > 1:
                    donor = int(torch.randint(0, self.var_sigma_stack.shape[0], (1,), device=x.device).item())
                    std_t = self.var_sigma_stack[donor, sid].clamp_min(0.0).sqrt()  # [T, D]
                else:
                    std_t = self.std_sigma_mean[sid]  # [T, D]

                mu_tok = self.mu_tok[sid]                        # [T, D]
                xc = x[i] - mu_tok
                std_x = xc.std(dim=0, keepdim=True) + 1e-6      # [1, D]
                std_target = std_t.mean(dim=0, keepdim=True)     # [1, D]
                scale_vec = (1.0 - self.scale) + self.scale * (std_target / std_x)
                y[i] = mu_tok + xc * scale_vec

            out = y.to(feats.dtype)
        return out


# ---------------------------------------------------------------------------
# SSP triplet loss
# ---------------------------------------------------------------------------

def abs_pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute absolute Pearson correlation per-sample between two tensors of shape (B, D).
    Returns shape (B,).
    """
    x = x - x.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1)
    x_norm = torch.sqrt((x * x).sum(dim=-1) + eps)
    y_norm = torch.sqrt((y * y).sum(dim=-1) + eps)
    corr = xy / (x_norm * y_norm + eps)
    return corr.abs()


def ssp_triplet_loss_cosine_xbank(
    emb: torch.Tensor,
    labels: torch.Tensor,
    bank_feats: torch.Tensor,
    bank_labels: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """
    Triplet-style loss vs memory bank entries.
    emb: (B,D), bank_feats: (Q,D), labels: (B,), bank_labels: (Q,)
    """
    if (bank_feats is None) or (bank_labels is None) or (bank_feats.numel() == 0):
        return emb.new_tensor(0.)
    with torch.cuda.amp.autocast(enabled=False):
        emb32 = nn.functional.normalize(emb.to(torch.float32), dim=-1)
        bank32 = nn.functional.normalize(bank_feats.to(torch.float32), dim=-1)
        S = emb32 @ bank32.T  # (B, Q)
        B, Q = S.shape
        y = labels.to(dtype=torch.long)
        same = y.view(-1, 1).expand(B, Q).eq(bank_labels.view(1, -1).expand(B, Q))
        pos_mask = same
        neg_mask = ~same
        pos_counts = pos_mask.sum(dim=1)
        has_pos = pos_counts > 0
        pos_counts_safe = pos_counts.clamp(min=1)
        s_pos_mean = (S * pos_mask.float()).sum(dim=1) / pos_counts_safe
        fill_val = torch.tensor(-1e4, dtype=S.dtype, device=S.device)
        s_neg = S.masked_fill(~neg_mask, fill_val).max(dim=1).values
        per_anchor = nn.functional.relu(margin + s_neg - s_pos_mean)
        if has_pos.any():
            return per_anchor[has_pos].mean().to(emb.dtype)
        return (per_anchor.mean() * 0.0).to(emb.dtype)
