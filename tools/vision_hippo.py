#!/usr/bin/env python3
"""Class-incremental vision: SEQ, SEQ* and DeltaHippo (utils/hippo_vis.py) on ViT-B/16 (ImageNet-21k) and ResNet-18/50.

  python tools/vision_hippo.py --model vit --dataset cifar100 --method ours --out experiments/vision/vit_c100_ours

Released configurations (DeltaHippo = --method ours):
  ViT-B/16:  --model vit --method ours --fresh_head --oldrow --oldrow_span --fresh_aug --wdfold --pdet --cmpowm
             --lastcls --tabln --sink --lr_bb 3e-5
  ResNet-50: --model resnet50 --method ours --fresh_head --oldrow --oldrow_span --no_owm --fresh_aug --wdfold --pdet
             --cmpowm --cmpcons   (backbone lr: the default 1e-4)

Protocol (identical for every method of a model family):
  * datasets: CIFAR-100 (10 tasks x 10 classes; images upsampled 32 -> 224 bilinear) and ImageNet-R (200 classes,
    10 tasks x 20; per-class fixed-seed 80/20 split, Resize 256 + CenterCrop 224, cached by tools/vision_data_prep.py).
    Class order: RandomState(1993).permutation (the PyCIL convention). No train augmentation: wake and sleep see
    the same inputs. CIL: one head per task, logits = concatenation of the heads seen so far, CE over all seen
    classes, argmax over all seen classes at test, no task id.
  * backbone in eval mode throughout: ViT has no dropout (config 0.0); ResNet BatchNorm uses its pretrained running
    statistics (never updated; gains and biases are trained). Full fine-tuning of every parameter.
  * recipe: AdamW (betas 0.9/0.999, eps 1e-8, weight decay 5e-4), batch 64, 3 epochs per task, fp32 weights with bf16
    autocast. Default backbone lr: ViT 2e-5, ResNet 1e-4 (override with --lr_bb); heads lr 1e-3. Relative step on
    every linear / conv map of the backbone: lr x min(1, rms(W) / mean rms).
  * SEQ: plain full fine-tuning. SEQ*: full fine-tuning during task 0's first epoch, then the backbone is frozen; a
    cosine classifier (scale 16, no bias); earlier heads frozen.
  * ours: SEQ + the hippocampus (task 0 identical to SEQ; sleep after every task; held learning from task 1).
"""
import argparse
import json
import logging
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.hippo_vis import HippoVis  # noqa: E402
from tools.vision_diag import EvalDiag, acquisition, downstream_gain  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
log = logging.getLogger("hippo_vis")


# ---------------------------------------------------------------------- data
class Data:
    def __init__(self, name, model, dev):
        self.dev = dev
        if name == "cifar100":
            d = os.path.join(ROOT, "data_vision", "cifar-100-python")
            ld = lambda f: pickle.load(open(os.path.join(d, f), "rb"), encoding="latin1")
            tr, te = ld("train"), ld("test")
            self.xtr = torch.tensor(tr["data"]).view(-1, 3, 32, 32).to(dev)
            self.xte = torch.tensor(te["data"]).view(-1, 3, 32, 32).to(dev)
            self.ytr, self.yte = np.array(tr["fine_labels"]), np.array(te["fine_labels"])
            self.K, self.up = 100, True
        else:
            # ImageNet-R / CUB-200-2011 / ImageNet-A: the same uint8 224x224 cache format
            # ({imr,cub,ina}_{train,test}_{x,y}.npy)
            c = os.path.join(ROOT, "data_vision", "cache")
            pf = {"imr": "imr", "cub": "cub", "ina": "ina"}[name]
            self.xtr = torch.from_numpy(np.load(os.path.join(c, pf + "_train_x.npy"))).permute(0, 3, 1, 2).pin_memory()
            self.xte = torch.from_numpy(np.load(os.path.join(c, pf + "_test_x.npy"))).permute(0, 3, 1, 2).pin_memory()
            self.ytr = np.load(os.path.join(c, pf + "_train_y.npy"))
            self.yte = np.load(os.path.join(c, pf + "_test_y.npy"))
            self.K, self.up = 200, False
        if model == "vit":
            self.mean, self.std = [0.5] * 3, [0.5] * 3
        else:
            self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.mean = torch.tensor(self.mean, device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor(self.std, device=dev).view(1, 3, 1, 1)

    def split(self, T, seed=1993):
        order = np.random.RandomState(seed).permutation(self.K)
        pos = np.empty(self.K, dtype=np.int64)
        pos[order] = np.arange(self.K)
        self.ytr_m, self.yte_m = pos[self.ytr], pos[self.yte]           # labels = position in the class order
        per = self.K // T
        self.per = per
        self.tr_idx = [np.where((self.ytr_m >= t * per) & (self.ytr_m < (t + 1) * per))[0] for t in range(T)]
        self.te_idx = [np.where((self.yte_m >= t * per) & (self.yte_m < (t + 1) * per))[0] for t in range(T)]

    def batch(self, x_all, y_all, idx, flip_gen=None):
        it = torch.as_tensor(idx)
        x = x_all[it.to(x_all.device)] if x_all.is_cuda else x_all[it]
        x = x.to(self.dev, non_blocking=True).float().div_(255.0)
        if self.up:
            x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        if flip_gen is not None:
            f = torch.rand(x.shape[0], generator=flip_gen) < 0.5
            if f.any():
                fi = f.nonzero(as_tuple=True)[0].to(self.dev)
                x[fi] = x[fi].flip(3)
        x = (x - self.mean) / self.std
        y = torch.as_tensor(y_all[idx], device=self.dev)
        return x, y

    def train_batches(self, t, bs, gen):
        idx = self.tr_idx[t][torch.randperm(len(self.tr_idx[t]), generator=gen).numpy()]
        for i in range(0, len(idx), bs):
            yield self.batch(self.xtr, self.ytr_m, idx[i:i + bs])           # no augmentation (see header)

    def plain_batches(self, t, bs, test=False):
        src = self.te_idx[t] if test else self.tr_idx[t]
        X, Y = (self.xte, self.yte_m) if test else (self.xtr, self.ytr_m)
        for i in range(0, len(src), bs):
            yield self.batch(X, Y, src[i:i + bs])


# ---------------------------------------------------------------------- model
class CosHead(nn.Module):
    def __init__(self, d, n, s=16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n, d) * 0.01)
        self.bias = None
        self.s = s
        self.out_features = n

    def forward(self, h):
        return self.s * F.normalize(h.float(), dim=-1) @ F.normalize(self.weight.float(), dim=-1).t()


class Net(nn.Module):
    def __init__(self, model, T, per, cosine=False):
        super().__init__()
        self.kind = model
        if model == "vit":
            from transformers import ViTModel
            self.bb = ViTModel.from_pretrained(os.path.join(ROOT, "hf_models", "vit-base-in21k"),
                                               add_pooling_layer=False, attn_implementation="sdpa",
                                               dtype=torch.float32)
            d = self.bb.config.hidden_size
            assert self.bb.config.hidden_dropout_prob == 0 and self.bb.config.attention_probs_dropout_prob == 0
        else:
            import torchvision.models as tvm
            w = {"resnet18": tvm.ResNet18_Weights.IMAGENET1K_V1, "resnet50": tvm.ResNet50_Weights.IMAGENET1K_V2}[model]
            self.bb = getattr(tvm, model)(weights=w)
            d = self.bb.fc.in_features
            self.bb.fc = nn.Identity()
        self.d = d
        self.heads = nn.ModuleList([CosHead(d, per) if cosine else nn.Linear(d, per) for _ in range(T)])
        self.n = 1

    def train(self, mode=True):
        super().train(mode)
        self.bb.eval()                       # no dropout (ViT) / pretrained BN statistics (ResNet), always
        return self

    def features(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.kind == "vit":
                return self.bb(pixel_values=x).last_hidden_state[:, 0]
            return self.bb(x)

    def logits(self, h):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return torch.cat([self.heads[i](h) for i in range(self.n)], -1).float()


def build_opt(net, hx, lr_bb, lr_head, wd, wref=0.0):
    areas = [a for a in hx.areas.values()]
    mods = [m for a in areas for m in a.mods]
    rms = lambda w: float(w.detach().float().pow(2).mean().sqrt())
    mean_ = sum(rms(m.weight) for m in mods) / len(mods)
    sc = {}
    nw_ = 0
    for m in mods:
        s = min(1.0, rms(m.weight) / mean_)
        if wref > 0:
            # width rule: a matrix whose input is wider than the reference steps by ref / fan_in
            fi_ = m.weight[0].numel()
            if fi_ > wref:
                s *= wref / fi_; nw_ += 1
        sc[id(m.weight)] = s
        if m.bias is not None:
            sc[id(m.bias)] = s
    head_ids = {id(p) for p in net.heads.parameters()}
    groups, rest, heads, ss = [], [], [], []
    for p in net.parameters():
        if id(p) in sc:
            groups.append({"params": [p], "lr": lr_bb * sc[id(p)]}); ss.append(sc[id(p)])
        elif id(p) in head_ids:
            heads.append(p)
        else:
            rest.append(p)
    groups = [{"params": rest, "lr": lr_bb}] + groups + [{"params": heads, "lr": lr_head}]
    log.info("relative step on %d maps/biases: scale mean %.3f min %.3f; %d other backbone params; heads lr %.1e; "
             "width ref %d: %d maps scaled" % (len(ss), sum(ss) / len(ss), min(ss), len(rest), lr_head, int(wref), nw_))
    return torch.optim.AdamW(groups, lr=lr_bb, betas=(0.9, 0.999), eps=1e-8, weight_decay=wd)


# ---------------------------------------------------------------------- evaluation + drift diagnostics
def stage_modules(net):
    if net.kind == "vit":
        return [net.bb.embeddings] + list(net.bb.encoder.layer)
    return [net.bb.conv1, net.bb.layer1, net.bb.layer2, net.bb.layer3, net.bb.layer4]


@torch.no_grad()
def evaluate(net, data, t, diag):
    net.eval()
    net.n = t + 1
    row = []
    # Diagnostic (evaluation only): task 0's per-stage decision-position state ([CLS] row / global pool) for the
    # per-layer drift; does not affect training.
    cap, hs = {}, []
    for i, m in enumerate(stage_modules(net)):
        def f(mod, inp, out, _i=i):
            o = out[0] if isinstance(out, tuple) else out
            v = o[:, 0] if o.dim() == 3 else o.mean((2, 3))
            cap.setdefault(_i, []).append(v.float())
        hs.append(m.register_forward_hook(f))
    try:
        for x, y in data.plain_batches(0, 256, test=True):
            net.features(x)
    finally:
        for h in hs:
            h.remove()
    if "L0" not in diag:
        diag["L0"] = [torch.cat(cap[i]) for i in sorted(cap)]
    else:
        msg = []
        for i in sorted(cap):
            a0, a1 = diag["L0"][i], torch.cat(cap[i])
            msg.append("%.3f" % float((a1 - a0).norm(dim=1).mean() / a0.norm(dim=1).mean()))
        log.info("[EVAL] after T%d task 0 per-stage per-image drift |d|/|h| since learnt: %s" % (t, " ".join(msg)))
    W = torch.cat([net.heads[i].weight.detach().float() for i in range(t + 1)])
    for e in range(t + 1):
        hit, n, H, Y, win = 0, 0, [], [], []
        for x, y in data.plain_batches(e, 256, test=True):
            h = net.features(x).float()
            z = net.logits(h)
            p = z.argmax(1)
            hit += int((p == y).sum()); n += y.numel()
            H.append(h); Y.append(y)
            win.append(p // data.per)
        row.append(100.0 * hit / n)
        H, Y, win = torch.cat(H), torch.cat(Y), torch.cat(win)
        ks = sorted(set(Y.tolist()))
        Hm = torch.stack([H[Y == k].mean(0) for k in ks])
        ws = torch.bincount(win, minlength=t + 1).float() / win.numel()
        if e not in diag:
            diag[e] = Hm
            msg = ""
        else:
            H0 = diag[e]
            dd = Hm - H0
            cm = float(dd.mean(0).pow(2).sum() / dd.pow(2).sum(1).mean().clamp(min=1e-30))
            msg = " | drift since learnt |d|/|h| %.3f common-mode share %.2f" % (
                float(dd.norm(dim=1).mean() / H0.norm(dim=1).mean()), cm)
        log.info("[EVAL] after T%d task %d acc %.2f | predicted-task share %s%s" % (
            t, e, row[-1], "[" + " ".join("%.2f" % v for v in ws.tolist()) + "]", msg))
    return row


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["vit", "resnet18", "resnet50"])
    ap.add_argument("--dataset", required=True, choices=["cifar100", "imr", "cub", "ina"])
    ap.add_argument("--method", required=True, choices=["seq", "seqstar", "ours"])
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--max_tasks", type=int, default=0, help="smoke test: stop after this many tasks")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1993)
    ap.add_argument("--fresh_head", action="store_true",
                    help="ours: the present task's head (no memory yet) is held only off the earlier class-mean patterns")
    ap.add_argument("--cmp_held", action="store_true",
                    help="ours: the CA1 comparator's mismatch is taken with the whole held span W (not only the "
                         "class-mean patterns) in every area, as conv areas always do")
    ap.add_argument("--cmp_conv_off", action="store_true",
                    help="ours: no CA1 comparator in conv areas (every position is an answer row there)")
    ap.add_argument("--dw", action="store_true",
                    help="ours: decision-weighted moments (rows weighted by |dL/dy|^2 of the present task's own CE)")
    ap.add_argument("--fresh_span", action="store_true",
                    help="ours (with --fresh_head): the fresh head held off the readout's whole held span W (U and the "
                         "class-mean patterns) at full scale, not only off the class means")
    ap.add_argument("--oldrow", action="store_true",
                    help="ours: old heads held per row off their own class record [mean; 1] (not the readout span)")
    ap.add_argument("--oldrow_span", action="store_true",
                    help="ours (with --oldrow): old row k held off its class's region (GD basis of its [h; 1] moment)")
    ap.add_argument("--no_owm", action="store_true",
                    help="ours: backbone operator = residual plasticity factor c (I - W W^T) instead of the "
                         "minimum-interference step (used for ResNet-50)")
    ap.add_argument("--owmg", action="store_true",
                    help="ours: gains take the per-coordinate OWM share instead of the novelty share with owned "
                         "coordinates held (ablation; off by default)")
    ap.add_argument("--fresh_common", action="store_true",
                    help="ours (with --fresh_aug): the fresh head's hold keeps its pattern across the earlier classes, "
                         "not the level common to all classes (old records and present class means) "
                         "(ablation; off by default)")
    ap.add_argument("--fresh_aug", action="store_true",
                    help="ours: the fresh head's [w; b] held jointly off the earlier classes' [mean; 1]")
    ap.add_argument("--wdfold", action="store_true",
                    help="ours: weight decay folded into the update before the hold")
    ap.add_argument("--pdet", action="store_true",
                    help="ours: state conflict only if closer to an old record than to its own present class")
    ap.add_argument("--cmpowm", action="store_true",
                    help="ours: comparator give-back restored only to the minimum-interference share")
    ap.add_argument("--owmx", action="store_true",
                    help="ours: exact minimum-interference operator (I + mu C / f)^-1 on the full earlier moment per area "
                         "(ablation; off by default)")
    ap.add_argument("--ro_common", action="store_true",
                    help="ours: old rows' common step through the augmented exact readout operator "
                         "(ablation; off by default)")
    ap.add_argument("--fresh_exact", action="store_true",
                    help="ours: fresh head [w; b] through P T_aug P (ablation; off by default)")
    ap.add_argument("--cmpcons", action="store_true",
                    help="ours: comparator give-back sized by the main rule (c / exact moment)")
    ap.add_argument("--lastcls", action="store_true",
                    help="ours (ViT): last block o / fc1 / fc2 and final LN count only the [CLS] row")
    ap.add_argument("--tabln", action="store_true",
                    help="ours (ViT): cls / position rows held in the units of the stream they feed")
    ap.add_argument("--biasjoint", action="store_true",
                    help="ours (with --owmx): [W, b] held jointly by the augmented input moment "
                         "(ablation; off by default)")
    ap.add_argument("--sink", action="store_true",
                    help="ours (ViT): attention-input areas count the decision row by 1 + sum_i a_i0^2")
    ap.add_argument("--trunk_c1", action="store_true",
                    help="ours: trunk areas held by their span alone (tail share c = 1): the tail is fully plastic "
                         "(ablation; off by default)")
    ap.add_argument("--pdir", action="store_true",
                    help="ours: per-direction tail novelty share in trunk areas in place of the scalar tail share c "
                         "(ablation; off by default)")
    ap.add_argument("--bn_area", action="store_true",
                    help="ours: BatchNorm gain/bias held by the tail share c of the conv area it scales "
                         "(ablation; off by default)")
    ap.add_argument("--no_diag", action="store_true",
                    help="skip the evaluation-only diagnostics (tools/vision_diag.py: margin split, block drift, "
                         "acquisition)")
    ap.add_argument("--width_ref", type=float, default=0,
                    help="width rule: maps with fan_in > ref step by ref / fan_in (0 = off)")
    ap.add_argument("--lr_bb", type=float, default=None,
                    help="backbone lr override (default: ViT 2e-5 / ResNet 1e-4)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reserve_gb", type=float, default=12.0,
                    help="reserve this much GPU memory up front in the caching allocator (0 = off)")
    args = ap.parse_args()
    if args.reserve_gb > 0:
        _r = torch.empty(int(args.reserve_gb * 2 ** 30), dtype=torch.uint8, device="cuda")
        del _r

    os.makedirs(args.out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    log.info("args %s" % vars(args))
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))   # limits host BLAS/LAPACK threads
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"
    # no TF32: the hold's projections and the tail energies are differences of nearly equal fp32 quantities, and
    # TF32's 10-bit mantissa would leak into every held update and inflate the tail share c
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"):
        # cuDNN can crash inside conv forward under an MPS active-thread-percentage cap; use the native
        # convolution kernels instead
        torch.backends.cudnn.enabled = False
        log.info("MPS thread cap %s%%: cuDNN disabled (native conv kernels)" % os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"])

    fam = "vit" if args.model == "vit" else "resnet"
    lr_bb = {"vit": 2e-5, "resnet": 1e-4}[fam] if args.lr_bb is None else float(args.lr_bb)
    lr_head, wd = 1e-3, 5e-4
    data = Data(args.dataset, args.model, dev)
    data.split(args.tasks)
    net = Net(args.model, args.tasks, data.per, cosine=args.method == "seqstar").to(dev)
    net.train()

    def probe():
        with torch.no_grad():
            net.features(torch.zeros(2, 3, 224, 224, device=dev))
    extra = {}
    if fam == "vit":
        emb = net.bb.embeddings
        extra = {emb.patch_embeddings.projection: [emb.cls_token, emb.position_embeddings]}
    hx = HippoVis(net.bb, net.heads, probe, extra_consts=extra, n_classes=data.K)
    hx.cmp_held_all = args.cmp_held
    hx.cmp_conv_off = args.cmp_conv_off
    hx.dw = args.dw
    hx.pdir = args.pdir
    hx.fresh_span = args.fresh_span
    hx.oldrow = args.oldrow
    hx.oldrow_span = args.oldrow_span
    hx.trunk_c1 = args.trunk_c1
    hx.owm = not args.no_owm
    hx.pnames = {id(p): n for n, p in net.named_parameters()}
    hx.owm_coord = args.owmg
    hx.fresh_aug = args.fresh_aug
    hx.fresh_common = args.fresh_common
    hx.pdet, hx.cmpowm, hx.sink = args.pdet, args.cmpowm, args.sink
    hx.owmx = args.owmx
    hx.ro_common, hx.fresh_exact, hx.cmpcons = args.ro_common, args.fresh_exact, args.cmpcons
    hx.lastcls, hx.tabln = args.lastcls, args.tabln
    hx.biasjoint = args.biasjoint
    if fam == "vit":
        hx.tabconst = {id(net.bb.embeddings.cls_token), id(net.bb.embeddings.position_embeddings)}
    if fam == "vit":
        hx.nheads = int(net.bb.config.num_attention_heads)
    hx.wdfold = args.wdfold
    if args.bn_area:
        hx.bn_to_area(net.bb, probe)
    opt = build_opt(net, hx, lr_bb, lr_head, wd, float(args.width_ref or 0))
    ours = args.method == "ours"
    gen = torch.Generator().manual_seed(args.seed)
    T_run = args.max_tasks or args.tasks
    mat, diag = [], {}
    edg = None if args.no_diag else EvalDiag(net, data)
    # Optional profiling (env VIS_PROF=1): per-component GPU time per step with CUDA events, steps 10.. of every task
    # (no host sync in the hot path; one sync per task when the events are read)
    VPROF = None
    if os.environ.get("VIS_PROF"):
        from utils.hippo_enc import _Prof
        VPROF = _Prof()
        VPROF.on = True
        if ours:
            hx._prof = VPROF
            for nm_ in ("readout", "pre_backward", "apply", "step", "_pdet_scores", "_sink_w", "_oldrows",
                        "_rowhold_all", "_build"):
                if hasattr(hx, nm_):
                    setattr(hx, nm_, VPROF.wrap("call:" + nm_, getattr(hx, nm_)))
    t_start = time.time()
    for t in range(T_run):
        net.n = t + 1
        if ours:
            hx.begin_task(t)
            hx.fresh = {id(p) for p in net.heads[t].parameters()} if args.fresh_head else set()
        if args.method == "seqstar":
            for i in range(t):
                net.heads[i].weight.requires_grad_(False)
        steps_per_ep = (len(data.tr_idx[t]) + args.bs - 1) // args.bs
        written, nstep = 0, 0
        t0 = time.time()
        for ep in range(args.epochs):
            if args.method == "seqstar" and not (t == 0 and ep == 0):
                for p in net.bb.parameters():
                    p.requires_grad_(False)
            ce_s, ce_n = 0.0, 0
            for bi, (x, y) in enumerate(data.train_batches(t, args.bs, gen)):
                if nstep == 10:
                    tw0_ = time.perf_counter()
                if VPROF is not None:
                    VPROF.live = nstep >= 10
                    if VPROF.live:
                        ev_ = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                        ev_[0].record()
                if ours:
                    hx.cls_cnt.index_add_(0, y, torch.ones_like(y, dtype=torch.float32))
                    hx._logstep = nstep % 10 == 0
                    hx.ep_prog = max(ep + (bi + 1) / steps_per_ep, 1.0)
                    hx.mode = "wake"
                if args.method == "seqstar" and not (t == 0 and ep == 0):
                    with torch.no_grad():
                        h = net.features(x).float()
                else:
                    h = net.features(x).float()
                z = net.logits(h)
                loss = F.cross_entropy(z, y)
                if ours:
                    hx.readout(h, z, y)
                if VPROF is not None and VPROF.live:
                    ev_[1].record(); VPROF.mark("L:forward+readout", ev_[0], ev_[1])
                opt.zero_grad(set_to_none=True)
                if ours:
                    hx.pre_backward()
                loss.backward()
                if VPROF is not None and VPROF.live:
                    ev_[2].record(); VPROF.mark("L:pre_backward+backward", ev_[1], ev_[2])
                if ours and hx.active:
                    hx.apply()
                    take = hx.prng.random() < hx.plast
                    written += int(take)
                    hx.step(opt, take)
                    hx.plast = 1.0
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                if VPROF is not None and VPROF.live:
                    ev_[3].record(); VPROF.mark("L:apply+step+zero", ev_[2], ev_[3]); VPROF.mark("L:TOTAL", ev_[0], ev_[3])
                if ours:
                    hx.mode = None
                nstep += 1
                ce_s += float(loss); ce_n += 1
                if ours and hx.active and nstep % 100 == 0:
                    hx.log_wake("T%d step %d: " % (t, nstep))
            log.info("T%d ep %d: ce %.4f (%.0fs)" % (t, ep + 1, ce_s / max(ce_n, 1), time.time() - t0))
        if nstep > 10:
            log.info("[SPEED] T%d: %.4f s/step over steps 10..%d (wall, incl. the per-step loss read)"
                     % (t, (time.perf_counter() - tw0_) / (nstep - 10), nstep))
        if VPROF is not None:
            VPROF.live = False
            VPROF.dump("T%d" % t, max(nstep - 10, 1))
        if ours and hx.active:
            hx.log_wake("T%d end: " % t)
            log.info("T%d: steps written %.3f" % (t, written / max(nstep, 1)))
        if edg is not None:
            try:
                acquisition(net, data, t, opt, hx if ours else None)
                if ours and os.environ.get("VIS_GAIN") and t in (0, 1, 5, 9):
                    downstream_gain(net, data, t, hx)
            except Exception as ex_:                       # diagnostics must never stop a run
                log.info("[DIAG-A] skipped: %s" % ex_)
                for p_ in net.parameters():
                    p_.grad = None
        if ours:
            ts = time.time()
            net.eval()
            rg = [p.requires_grad for p in net.parameters()]
            for p in net.parameters():
                p.requires_grad_(False)
            hx.sleep(t, lambda: data.plain_batches(t, 64 if args.dw else 128), lambda x: net.features(x),
                     lambda h: net.logits(h))
            for p, r in zip(net.parameters(), rg):
                p.requires_grad_(r)
            log.info("T%d sleep %.0fs, peak mem %.1f GB" % (t, time.time() - ts, torch.cuda.max_memory_allocated() / 1e9))
        row = evaluate(net, data, t, diag)
        if edg is not None:
            try:
                edg.after_eval(t)
            except Exception as ex_:                       # diagnostics must never stop a run
                log.info("[DIAG-M] skipped: %s" % ex_)
        net.train()
        mat.append(row)
        log.info("ROW T%d: %s | AA %.2f | %.0fs total" % (t, " ".join("%.1f" % v for v in row), np.mean(row),
                                                          time.time() - t_start))
        json.dump({"args": vars(args), "matrix": mat, "AA": float(np.mean(mat[-1])),
                   "AIA": float(np.mean([np.mean(r) for r in mat])),
                   "diag_acc": [mat[i][i] for i in range(len(mat))],
                   "recipe": {"lr_backbone": lr_bb, "lr_head": lr_head, "wd": wd, "bs": args.bs, "epochs": args.epochs}},
                  open(os.path.join(args.out, "result.json"), "w"), indent=1)
    log.info("FINAL AA %.2f peak mem %.1f GB" % (np.mean(mat[-1]), torch.cuda.max_memory_allocated() / 1e9))


if __name__ == "__main__":
    main()
