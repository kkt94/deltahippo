"""HippoVis: the DeltaHippo hippocampus for vision encoders (ViT, ResNet).

Same rules as utils/hippo_lite.py / utils/hippo_enc.py (not imported here); what is vision-specific is only how the
model is read:

  AREAS (probe forward). An area is the set of linear maps reading the SAME input vectors:
    * nn.Linear modules (heads excluded) that receive the same input tensor object (ViT: query/key/value; attention
      output; fc1; fc2). Their rows are token positions (B, L, d).
    * nn.Conv2d modules: a conv is a linear map on the unfolded C*k*k input; convs that read the same tensor with the
      same (kernel, stride, padding, dilation) read the same unfolded vectors and form one area (ViT patch embedding:
      kernel = stride = 16, i.e. a linear map on flattened patches; ResNet: every conv, the downsample conv joining
      the block's conv1 only when the unfold is identical). Rows are spatial positions; moments are taken on a fixed
      subset of at most POS positions per image (cost), patterns on the mean over all positions.
    * READOUT: the per-task Linear heads, reading the decision feature h (ViT: final-LN [CLS]; ResNet: global pool).
  ANSWER ROWS (where the loss enters). ViT token areas: the [CLS] row of each image (trunk moment = all positions +
    (n_all / n_ans) x the [CLS] rows, the same answer-row reweighting as for decoders). Conv areas (ResNet, ViT patch
    embedding): the decision is a function of every position, so every position is an answer row (plain moment).
    Readout: the decision feature.
  GAINS: LayerNorm (ViT) and BatchNorm2d (ResNet) gains are per-coordinate areas on the normalised input
    (LN: centred / std over features; BN: (x - running mean) / running std per channel, eval-mode statistics); their
    biases take the gain's per-coordinate share D.
  BIASES: linear / conv biases are the weight on a constant input, outside the held span: gradient and Adam step
    scaled by the area's tail share c. ViT's [CLS] token and position embeddings are constant inputs added at the
    patch embedding's output: they are biases of the patch-embedding area.
  RECORDS: one decision position per image, so a memory is a class (offset 0 only). The joint-share target (needs
    o_t > 0) has no effect here and is omitted.

Sleep (after every task, eval mode, no graph, the present task's training images only, no augmentation):
    per area the episode moment -> Gavish-Donoho keys -> gated-delta write (erase by beta (1 - rho_i), rho_i from a
    second read flagging rows in state conflict with earlier records), tail; gains: running energy, GD owned
    coordinates; class records (count, sum h) and every area's class-mean answer-row input (patterns).
Wake (task >= 1): T_a = c_a (I - W_a W_a^T), W_a = [U_a, patterns outside U_a], c_a = tail novelty share (rebuilt at
    steps 1, 2, 4, ...); the weight gradient is the held one, (dW - dW_A) T + dW_A (I - Q Q^T) (= dy^T (x T) with the
    comparator's armed rows entering as their mismatch with the pattern basis Q), formed in weight space (exactly
    equal); Adam's update u is held again, u T + (u M) Q~^T (M = (I - T) Q~, Q~ the armed rows' mismatch basis);
    the step is written whole with probability plast (state-conflict write gate).
"""
import logging
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("hippo_vis")

import os as _os
# Diagnostic ablations (not part of the method; unset by default): HV_DIAG is a comma list of
# gainhold, convhold, rohold, consthold, cmpoff, each freezing / disabling the named component.
_DIAG = {k: True for k in _os.environ.get("HV_DIAG", "").split(",") if k}

# HV_CMP_CONV: "held" (default) takes the conv areas' comparator mismatch off the whole held span W, any other
# value off the pattern basis Q only.
CMP_CONV_HELD = _os.environ.get("HV_CMP_CONV", "held") == "held"

POS = 64                      # conv areas: spatial positions per image in moments / energies (fixed subset)


def gd_rank(ev, n):
    """Gavish-Donoho hard threshold on sqrt(ev) (ev ascending, as from eigh) of a moment from n rows."""
    sv = ev.clamp(min=0).sqrt().flip(0)
    D = float(ev.shape[0])
    bt = min(float(n), D) / max(float(n), D)
    om = 0.56 * bt ** 3 - 0.95 * bt ** 2 + 1.82 * bt + 1.43
    tau = om * float(sv[:int(min(n, D))].median())
    return int((sv > tau).sum()), tau


def orth(M, tol=1e-4):
    """Orthonormal basis of the columns of M (QR, near-zero diagonal entries dropped)."""
    if M.shape[1] == 0:
        return M
    Q, R = torch.linalg.qr(M)
    dg = R.diagonal().abs()
    return Q[:, dg > tol * max(float(dg.max()), 1e-30)].contiguous()


def _kind(name):
    if name == "readout":
        return "ro"
    for k_, v_ in (("patch", "patch"), ("query", "qkv"), ("attention.output", "o"), ("intermediate", "fc1"),
                   ("output.dense", "fc2"), ("layer1", "L1"), ("layer2", "L2"), ("layer3", "L3"), ("layer4", "L4")):
        if k_ in name:
            return v_
    return "stem"


class Area:
    def __init__(self, name, mods, kind):
        self.name, self.mods, self.kind = name, mods, kind        # kind: "tok" | "conv" | "ro"
        self.tag = _kind(name)
        m0 = mods[0]
        self.d = m0.weight[0].numel()
        if kind == "conv":
            self.cv = dict(kernel_size=m0.kernel_size, dilation=m0.dilation, padding=m0.padding, stride=m0.stride)
        self.U = None            # d x r held subspace
        self.lam = None
        self.tr = None
        self.n = 0.0
        self.tail = None
        self.W = None            # hold basis of the task: [U, patterns outside U]
        self.c = 1.0
        self.Q = None            # comparator pattern basis
        self.floor, self.nlast = None, None   # the last write's GD noise floor (energy units) and episode count
        self.av, self.b, self.sb = None, None, None   # minimum-interference operator: T = b I + W diag(av - b) W^T
        self.consts = [m.bias for m in mods if getattr(m, "bias", None) is not None]


class HippoVis:
    def __init__(self, model, heads, probe, extra_consts=None, seed=12345, n_classes=1000):
        """model: the backbone (features); heads: nn.ModuleList of per-task Linear heads (the readout);
        probe(): one short forward of the backbone; extra_consts: {module: [params]} constant-input parameters added
        at that module's output (ViT [CLS] / position embeddings at the patch embedding)."""
        self.heads = heads
        self.areas, self.gains = self._detect(model, probe)
        for a in self.areas.values():
            # an area's biases are those of ALL its maps: Area() is created with the first map only and the others
            # (ViT key / value) are appended by _detect
            a.consts = [m.bias for m in a.mods if getattr(m, "bias", None) is not None]
        for mod, ps in (extra_consts or {}).items():
            for a in self.areas.values():
                if mod in a.mods:
                    a.consts += list(ps)
        self.ro = Area("readout", list(heads), "ro")
        self.ro.consts = [h.bias for h in heads if h.bias is not None]
        self.all_areas = list(self.areas.values()) + [self.ro]
        self.pmap = {}
        for a in self.all_areas:
            for m in a.mods:
                self.pmap[id(m.weight)] = ("w", a)
            for p in a.consts:
                self.pmap[id(p)] = ("b", a)
        for g, mod in self.gains.items():
            self.pmap[id(mod.weight)] = ("g", g)
            if getattr(mod, "bias", None) is not None:
                self.pmap[id(mod.bias)] = ("g", g)
        self.mode = None         # None | "wake" | "rec" | "g2s"
        self.handles = []
        for a in self.areas.values():
            self.handles.append(a.mods[0].register_forward_pre_hook(self._area_hook(a)))
            for m in a.mods:
                self.handles.append(m.register_forward_hook(self._out_hook(a, m)))
                self.handles.append(m.register_forward_hook(self._dw_hook(a, m)))
        for g, mod in self.gains.items():
            self.handles.append(mod.register_forward_pre_hook(self._gain_hook(g, mod)))
        self.active = False      # wake from task 1 on
        self.dw = False            # option: DECISION-WEIGHTED moments (rows weighted by |dL/dy|^2 of the area's output)
        self.bn_skip = set()       # BN gains held as constants of the conv area they scale (option bn_area)
        self._wacc, self._bpend = {}, False
        self._adv, self._cmpdone, self._cmpcache = False, False, {}
        self.cmp_conv_off = False  # option: no CA1 comparator in conv areas (no answer rows distinct from context)
        self.cmp_held_all = False  # option: the comparator's mismatch is taken with the whole held span in every area
        self.fresh = set()       # ids of parameters with no memory yet held only by the patterns (option)
        self.pdir = False          # option: PER-DIRECTION tail novelty share (see _build) in place of the scalar c
        self.fresh_span = False    # option: the fresh head held off the readout's whole held span W (c = 1), not Q
        self.oldrow = False        # option: OLD HEADS held per row off their own class record [mean; 1] only
        self.oldrow_span = False   # option (with oldrow): row k held off its class's REGION (GD basis of [h; 1] moment)
        self.trunk_c1 = False      # option: trunk areas held by the span alone (tail share c = 1), the GPM-like rule
        self.owm = True            # MINIMUM-INTERFERENCE STEP in every area (see begin_task); False: the tail share c
        self.owm_coord = False     # option: per-coordinate gains take share f / (f + mu e_i)
        self.fresh_aug = False     # option: fresh head: [w; b] held jointly off [class mean; 1] of the earlier classes
        self.fresh_common = False  # option: the fresh head's hold without the component common to all classes
        self.faug = None
        self.wdfold = False        # option: weight decay folded into the update before the hold
        self.pdet = False          # option: conflict only if closer to an old record than to the own present class
        self.pcm = None            # present classes' running decision-feature sums [count (K), sum (K, d)]
        self.cmpowm = False        # option: comparator give-back restored only to s_q = f / (f + mu q^T C q)
        self.cmpS = []
        self.sink = False          # option: attention-input areas: the decision row weighted by 1 + sum_i a_i0^2
        self.owmx = False          # EXACT minimum-interference operator (I + mu C / f)^-1 on the full earlier moment
        self.ro_common = False     # option: old rows' common step through the augmented exact readout operator
        self.fresh_exact = False   # option: fresh head: [w; b] through P T_aug P (class [mean; 1] still held)
        self.Caug, self.caug_n = None, 0.0     # running per-sample mean of [h; 1][h; 1]^T over every earlier image
        self.Taug, self.Tfresh = None, None
        self.cmpcons = False       # option: give-back follows the main rule: c under the tail share, C_full under owmx
        self.lastcls = False       # option: ViT last block (o / fc1 / fc2, final LN): only the [CLS] row reaches the decision
        self.tabln = False         # option: cls / position rows held in the units of the stream they feed
        self.tabconst = set()      # ids of the cls / position parameters (set by the trainer)
        self.tab_share = None
        self.cmpc = {}             # per-task cache of the give-back sizing's U / lam on the device
        self._lastblk = None
        self.gvar = {}
        self.biasjoint = False     # option (with owmx): [W, b] held jointly by the augmented moment E[[x; 1][x; 1]^T]
        self.rsum = {}
        self.sinkw = {}
        self.nheads = 12
        self.gfm = {}              # gain -> (per-coordinate noise floor, earlier-sample weight mu)
        self.arec, self.hoff = None, {}
        self.cbas = {}             # class -> (d + 1, r) basis of the class's augmented decision-feature moment
        self.tmom = {}             # wake (pdir): area -> [present tail moment, rows]
        self.steps, self.next_build = 0, 1
        self.now, self.gnow = {}, {}
        self.gold = {}           # gain -> [sum energy, n]
        self.gown, self.D = {}, {}
        self.K = int(n_classes)
        self.recs = None         # [count (K), sum h (K, d)] per class
        self.pat = {}            # area name -> [count (K), sum x (K, d)] (class-mean answer-row input)
        self.sc = None           # records' neighbourhoods
        self.plast = 1.0
        self.armed = None        # wake: sample indices armed for the comparator
        self.xin = {}            # wake: area -> (input tensor) of the present forward
        self.dWA = {}            # id(module) -> armed rows' dW
        self.cmpA = {}           # area name -> (Q~, M)
        self.cls_cnt = None
        self._logstep = False
        self.ep_prog = 1.0
        self._pos = {}
        self._zstat()
        self.prng = random.Random(seed)
        log.info("[HIPPO-VIS] %d trunk areas (%s), readout %d heads, %d gain areas" % (
            len(self.areas), ", ".join("%s:%d" % (k, sum(1 for a in self.areas.values() if a.tag == k))
                                       for k in sorted({a.tag for a in self.areas.values()})),
            len(heads), len(self.gains)))

    # ------------------------------------------------------------------ probe: areas
    @torch.no_grad()
    def _detect(self, model, probe):
        seen, keep, hs = {}, [], []
        for name, mod in model.named_modules():
            if isinstance(mod, (nn.Linear, nn.Conv2d, nn.LayerNorm, nn.BatchNorm2d)):
                def pre(m_, inp, _n=name):
                    x = inp[0]
                    if torch.is_tensor(x) and _n not in seen:
                        keep.append(x)
                        seen[_n] = id(x)
                hs.append(mod.register_forward_pre_hook(pre))
        try:
            probe()
        finally:
            for h in hs:
                h.remove()
        areas, by_key, gains = {}, {}, {}
        for name, mod in model.named_modules():
            if name not in seen:
                continue
            if isinstance(mod, nn.Linear):
                key = ("lin", seen[name])
            elif isinstance(mod, nn.Conv2d):
                assert mod.groups == 1, "grouped convs are not linear maps on one unfold"
                key = ("conv", seen[name], mod.kernel_size, mod.stride, mod.padding, mod.dilation)
            else:
                gains[name] = mod
                continue
            a = by_key.get(key)
            if a is None:
                by_key[key] = name
                areas[name] = Area(name, [mod], "conv" if key[0] == "conv" else "tok")
            else:
                areas[a].mods.append(mod)
        del keep
        return areas, gains

    # ------------------------------------------------------------------ rows
    def _pos_idx(self, L, dev):
        k = (L, str(dev))
        if k not in self._pos:
            g = torch.Generator().manual_seed(1000 + L)
            idx = torch.randperm(L, generator=g)[:min(L, POS)].sort().values
            self._pos[k] = idx.to(dev)
        return self._pos[k]

    def _unfold(self, a, x):
        return F.unfold(x, **a.cv)                                   # (B, d, L)

    def _rows(self, a, x):
        """(all rows (n, d), answer rows (B, d) or None, sample of each all-row)"""
        if a.kind == "tok":
            B, L, d = x.shape
            if self.lastcls and self._is_last(a):
                return x[:, 0], None, None                   # (lastcls) only the [CLS] row's output reaches the decision
            return x.reshape(-1, d), x[:, 0], None
        U = self._unfold(a, x)
        idx = self._pos_idx(U.shape[2], U.device)
        X = U.index_select(2, idx).transpose(1, 2)                   # (B, S, d)
        sid = torch.arange(x.shape[0], device=x.device).repeat_interleave(X.shape[1])
        return X.reshape(-1, a.d), None, sid

    def _is_last(self, a):
        """last transformer block, areas other than the attention input (q / k / v read every position)"""
        import re as _re
        if self._lastblk is None:
            ix = [int(m.group(1)) for m in (_re.search(r"layer\.(\d+)\.", n) for n in self.areas) if m]
            self._lastblk = max(ix) if ix else -1
        m = _re.search(r"layer\.(\d+)\.", a.name)
        return m is not None and int(m.group(1)) == self._lastblk and a.tag != "qkv"

    def _ans_mean(self, a, x):
        """per image, the area's answer-row input (tok: [CLS] row; conv: mean over all positions)"""
        if a.kind == "tok":
            return x[:, 0].float()
        return self._unfold(a, x.float()).mean(2)

    # ------------------------------------------------------------------ hooks
    def _area_hook(self, a):
        def hook(mod, inp):
            # (the forward runs under bf16 autocast: every statistic here is taken with autocast off)
            with torch.autocast("cuda", enabled=False):
                return hook_(mod, inp)

        def hook_(mod, inp):
            if self.mode is None:
                return None
            x = inp[0]
            if self.mode == "wake":
                if not self.active or not torch.is_grad_enabled():
                    return None
                self.xin[a.name] = x
                if self._energy_step() and a.U is not None and not self.dw:
                    with torch.no_grad():
                        Xa, Xc, _ = self._rows(a, x.detach())
                        e = self._energy(a, Xa, Xc)
                        w0 = self._sink_w(a, x.detach())
                        if w0 is not None:
                            X0 = Xc.double()
                            tot0 = (X0.pow(2).sum(1) * (w0.double() - 1.0)).sum()
                            tl0 = ((X0 - (X0 @ a.U) @ a.U.t()).pow(2).sum(1) * (w0.double() - 1.0)).sum()
                            e = [e[0] + tot0, e[1] + tl0, e[2]]
                        o = self.now.get(a.name)
                        self.now[a.name] = e if o is None else [o[0] + e[0], o[1] + e[1], o[2] + e[2]]
                        if self.pdir and a.W is not None:
                            # the present input's part outside the held span W, its moment (all rows alike)
                            Xf = Xa.float()
                            Xt = Xf - (Xf @ a.W) @ a.W.t()
                            tm = self.tmom.get(a.name)
                            C = Xt.t() @ Xt
                            self.tmom[a.name] = [C, float(Xa.shape[0])] if tm is None else [tm[0] + C, tm[1] + float(Xa.shape[0])]
                return None
            with torch.no_grad():
                xd = x.detach()
                Xa, Xc, sid = self._rows(a, xd)
                if self.mode == "rec" and self.dw:
                    self._pat_add(a.name, self._ans_mean(a, xd), self._ylab)
                elif self.mode == "g2s" and self.dw:
                    pass
                elif self.mode == "rec":
                    # (full-precision operands: rounding the input to bf16 adds a noise floor that is itself tail
                    # energy and would make every later present tail look old)
                    Xb = Xa.double()                     # (float64: the tail of some areas is a tiny share of the energy)
                    C = Xb.t() @ Xb
                    n = float(Xa.shape[0])
                    if Xc is not None:
                        Cb = Xc.double()
                        C.add_(Cb.t() @ Cb, alpha=n / Xc.shape[0])
                        w0 = self._sink_w(a, xd)
                        if w0 is not None:
                            X0 = Cb * (w0.double() - 1.0).clamp(min=0.0).sqrt().unsqueeze(1)
                            C.add_(X0.t() @ X0)
                    if self.biasjoint:
                        # (biasjoint) the same weighted rows' sum and weight: the moment's coupling with the constant input
                        sv = Xb.sum(0); ws = float(Xa.shape[0])
                        if Xc is not None:
                            sv = sv + Cb.sum(0) * (n / Xc.shape[0]); ws += n
                            if w0 is not None:
                                sv = sv + (Cb * (w0.double() - 1.0).clamp(min=0.0).unsqueeze(1)).sum(0)
                                ws += float((w0 - 1.0).clamp(min=0.0).sum())
                        er = self.rsum.get(a.name)
                        self.rsum[a.name] = [sv, ws] if er is None else [er[0] + sv, er[1] + ws]
                    e = self.rec.get(a.name)
                    if e is None:
                        self.rec[a.name] = [C, n]
                    else:
                        e[0] += C; e[1] += n
                    self._pat_add(a.name, self._ans_mean(a, xd), self._ylab)
                elif self.mode == "g2s":
                    kc = self.kc.get(a.name)
                    if kc is not None:
                        R = Xc if Xc is not None else Xa
                        self.g2P[a.name] = ((R.float() @ kc["K"]).pow(2), sid)
            return None
        return hook

    def _out_hook(self, a, m):
        def hook(mod, inp, out):
            if self.mode != "wake" or not self.active or not torch.is_grad_enabled() or a.Q is None \
                    or not out.requires_grad or a.kind == "conv":
                return None
            x = inp[0]

            def gh(g, _x=x):
                A = self.armed
                if A is None or A.numel() == 0:
                    return None
                with torch.no_grad():
                    if a.kind == "tok":
                        dW = g[A, 0].float().t() @ _x[A, 0].float()
                    else:
                        dW = torch.nn.grad.conv2d_weight(_x[A].float(), m.weight.shape, g[A].float(),
                                                         stride=m.stride, padding=m.padding, dilation=m.dilation)
                        dW = dW.reshape(dW.shape[0], -1)
                    self.dWA[id(m)] = dW
                return None
            out.register_hook(gh)
            return None
        return hook

    @staticmethod
    def _dz2(z, y):
        """|dL/dz|^2 per row for CE: |softmax(z) - onehot(y)|^2 (the readout's decision weight)"""
        q = torch.softmax(z.float(), dim=1)
        q[torch.arange(q.shape[0], device=q.device), y] -= 1.0
        return q.pow(2).sum(1)

    def _energy_w(self, a, Xa, w):
        """decision-weighted per-position energies [sum w|x|^2, sum w|x_tail|^2, sum w]"""
        Xf, wd = Xa.double(), w.double()
        tot = (Xf.pow(2).sum(1) * wd).sum()
        tl = ((Xf - (Xf @ a.U) @ a.U.t()).pow(2).sum(1) * wd).sum()
        return [tot, tl, wd.sum()]

    def _dw_hook(self, a, m):
        """DECISION-WEIGHTED MOMENTS: the area's rows weighted by |delta_t|^2, delta_t = the present task's CE gradient
        at the area's output row (all maps of the area: |delta_t|^2 summed over them). Used by the sleep moment, the
        second (rho) read and the wake tail energies alike, in place of the raw energy weighting."""
        nm = len(a.mods)

        def hook(mod, inp, out):
            if not self.dw or not torch.is_tensor(out) or not out.requires_grad:
                return None
            if self.mode == "wake":
                if not (self.active and a.U is not None and self._energy_step() and torch.is_grad_enabled()):
                    return None
            elif self.mode not in ("rec", "g2s"):
                return None
            x = inp[0].detach()

            def gh(g, _x=x):
                with torch.no_grad(), torch.autocast("cuda", enabled=False):
                    gf = g.float()
                    if a.kind == "tok":
                        w = gf.pow(2).sum(-1).reshape(-1)
                    else:
                        w = gf.pow(2).sum(1).reshape(g.shape[0], -1)
                        w = w.index_select(1, self._pos_idx(w.shape[1], w.device)).reshape(-1)
                    acc = self._wacc.get(a.name)
                    if acc is None:
                        acc = self._wacc[a.name] = [torch.zeros_like(w), 0]
                    acc[0] += w
                    acc[1] += 1
                    if acc[1] == nm:
                        del self._wacc[a.name]
                        self._dw_rows(a, _x, acc[0])
                return None
            out.register_hook(gh)
            return None
        return hook

    def _dw_rows(self, a, x, w):
        Xa, _, sid = self._rows(a, x)
        if self.mode == "rec":
            Xd = Xa.double()
            C = (Xd * w.double().unsqueeze(1)).t() @ Xd
            e = self.rec.get(a.name)
            if e is None:
                self.rec[a.name] = [C, float(Xa.shape[0]), float(w.sum())]
            else:
                e[0] += C; e[1] += float(Xa.shape[0]); e[2] += float(w.sum())
        elif self.mode == "g2s":
            kc = self.kc.get(a.name)
            if kc is not None:
                if sid is None:
                    sid = torch.arange(x.shape[0], device=x.device).repeat_interleave(x.shape[1])
                self.g2P[a.name] = ((Xa.float() @ kc["K"]).pow(2) * w.unsqueeze(1), sid)
        elif self.mode == "wake":
            e = self._energy_w(a, Xa, w)
            o = self.now.get(a.name)
            self.now[a.name] = e if o is None else [o[0] + e[0], o[1] + e[1], o[2] + e[2]]

    def bn_to_area(self, model, probe):
        """BN gains / biases held as constants of the conv area whose output they scale (that area's tail share c,
        gradient and step), instead of the per-coordinate rule."""
        outs, ins, hs, keep = {}, {}, [], []

        def fo(m_, i_, o_, _n):
            keep.append(o_)                      # (alive until the mapping is done: no id reuse)
            outs.setdefault(id(o_), _n)
            return None

        def fi(m_, i_, _n):
            ins.setdefault(_n, id(i_[0]))
            return None
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Conv2d):
                hs.append(mod.register_forward_hook(lambda m_, i_, o_, _n=name: fo(m_, i_, o_, _n)))
            if isinstance(mod, nn.BatchNorm2d):
                hs.append(mod.register_forward_pre_hook(lambda m_, i_, _n=name: fi(m_, i_, _n)))
        try:
            probe()
        finally:
            for h in hs:
                h.remove()
        by_mod = {}
        for a in self.areas.values():
            for m in a.mods:
                by_mod[m] = a
        named = dict(model.named_modules())
        n = 0
        for g, mod in self.gains.items():
            if not isinstance(mod, nn.BatchNorm2d) or g not in ins:
                continue
            src = outs.get(ins[g])
            if src is None:
                continue
            a = by_mod.get(named[src])
            if a is None:
                continue
            for p in (mod.weight, mod.bias):
                a.consts.append(p)
                self.pmap[id(p)] = ("b", a)
            self.bn_skip.add(g)
            n += 1
        log.info("[HIPPO-VIS] bn_area: %d BatchNorm layers held by the share c of the conv area they scale" % n)

    def _gain_hook(self, g, mod):
        bn = isinstance(mod, nn.BatchNorm2d)

        def hook(m_, inp):
            with torch.autocast("cuda", enabled=False):
                return hook_(m_, inp)

        def hook_(m_, inp):
            if self.mode not in ("wake", "rec"):
                return None
            if self.mode == "wake" and (not self.active or not self._energy_step() or not torch.is_grad_enabled()):
                return None
            with torch.no_grad():
                x = inp[0].detach().float()
                if bn:
                    xh = (x - mod.running_mean.view(1, -1, 1, 1)) * torch.rsqrt(mod.running_var.view(1, -1, 1, 1) + mod.eps)
                    S = xh.pow(2).sum((0, 2, 3))
                    n = float(x.shape[0] * x.shape[2] * x.shape[3])
                else:
                    if self.lastcls and g == "layernorm" and x.dim() == 3:
                        x = x[:, 0:1]                       # (lastcls) the final norm: only the [CLS] row is read
                    if self.mode == "rec" and self.tabln and g.endswith("layer.0.layernorm_before"):
                        e = self.gvar.get(g)
                        v_ = float(x.var(-1, unbiased=False).mean())
                        self.gvar[g] = [v_, 1.0] if e is None else [e[0] + v_, e[1] + 1.0]
                    xh = (x - x.mean(-1, keepdim=True)) * torch.rsqrt(x.var(-1, unbiased=False, keepdim=True) + mod.eps)
                    S = xh.reshape(-1, x.shape[-1]).pow(2).sum(0)
                    n = float(xh.numel() // x.shape[-1])
                st = self.gnow if self.mode == "wake" else self.grec
                e = st.get(g)
                st[g] = [S, n] if e is None else [e[0] + S, e[1] + n]
            return None
        return hook

    def _energy(self, a, Xa, Xc):
        """per-position energies [weighted total, weighted outside U (tail), weight]"""
        Uf = a.U                                                  # float64: the tail can be a tiny share of the energy
        n = float(Xa.shape[0])
        Xf = Xa.double()
        tot = Xf.pow(2).sum()
        tl = (Xf - (Xf @ Uf) @ Uf.t()).pow(2).sum()          # the tail energy itself (no cancellation)
        if Xc is not None:
            Cf = Xc.double()
            w = n / Cf.shape[0]
            tot = tot + w * Cf.pow(2).sum()
            tl = tl + w * (Cf - (Cf @ Uf) @ Uf.t()).pow(2).sum()
        return [tot, tl, n]

    def _energy_step(self):
        s = self.steps + 1
        return (s & (s - 1)) == 0

    def _pat_add(self, name, X, y):
        P = self.pat.get(name)
        if P is None:
            P = self.pat[name] = [torch.zeros(self.K, device=X.device), torch.zeros(self.K, X.shape[1], device=X.device)]
        P[0].index_add_(0, y, torch.ones_like(y, dtype=torch.float32))
        P[1].index_add_(0, y, X.float())

    def _zstat(self):
        self.stat = {"pass": {}, "conf": [], "armed": [], "plast": [], "step": {}}

    # ------------------------------------------------------------------ operator
    @staticmethod
    def _T(a, X):
        """X T_a = c (X - (X W) W^T) for rows X (n, d); with a per-direction share (pdir): X V D V^T, V the present
        tail's directions outside W, D their novelty shares"""
        if getattr(a, "dense", None) is not None:
            return X @ a.dense
        if getattr(a, "av", None) is not None:
            Y = ((X @ a.W) * (a.av - a.b)) @ a.W.t()
            return Y.add_(X, alpha=a.b)
        pd = getattr(a, "pd", None)
        if pd is not None:
            V, Dv = pd
            Y = ((X @ V) * Dv) @ V.t()
            return Y - (Y @ a.W) @ a.W.t()
        Y = X - (X @ a.W) @ a.W.t()
        return Y.mul_(a.c) if a.c != 1.0 else Y

    # ------------------------------------------------------------------ task start
    @torch.no_grad()
    def begin_task(self, t):
        self._tcur = t
        self.steps, self.next_build = 0, 1
        self._adv, self._cmpdone, self._cmpcache = False, False, {}   # cache of stacked comparator bases / operators
        self.now, self.gnow, self.D = {}, {}, {}
        self.tmom = {}
        self.pcm = None
        for a in self.all_areas:
            a.pd = None
        self.cls_cnt = torch.zeros(self.K, device=self.ro.mods[0].weight.device)
        self.active = t > 0 and self.ro.U is not None
        if not self.active:
            return
        # record neighbourhoods
        ks = (self.recs[0] > 0).nonzero(as_tuple=True)[0]
        H = self.recs[1][ks] / self.recs[0][ks].unsqueeze(1)
        cbar = H.mean(0, keepdim=True)
        M = F.normalize(H - cbar, dim=1)
        S = M @ M.t()
        S.fill_diagonal_(-2.0)
        self.sc = {"keys": ks, "c": cbar, "M": M, "r": S.max(1).values.clamp(min=-1.0),
                   "N": self.recs[0][ks].clone()}
        if self.fresh_aug or self.fresh_exact:
            Pr = self.pat.get("readout")
            Pm = (Pr[1][ks] / Pr[0][ks].unsqueeze(1)) if Pr is not None else H
            Aa = torch.cat([Pm, torch.ones(Pm.shape[0], 1, device=Pm.device)], 1).t()     # (d + 1, K)
            self._faugA = Aa
            self.faug = self._fresh_basis(Aa)
        self.tab_share = None
        if self.tabln and self.tabconst:
            # (tabln) a cls / position row adds dv at one of the L positions of every image; the first attention-input
            # area reads it through layer 0's pre-norm as (gamma / sigma) dv: share f_A / (f_A + mu (1 / L) gamma^2 / sigma^2)
            a0 = next((a for a in self.areas.values() if a.tag == "qkv" and ".layer.0." in a.name), None)
            g0 = next((g for g in self.gains if g.endswith("layer.0.layernorm_before")), None)
            if a0 is not None and g0 is not None and a0.floor is not None and g0 in self.gvar:
                sig2 = self.gvar[g0][0] / max(self.gvar[g0][1], 1.0)
                k_ = float(self.gains[g0].weight.detach().float().pow(2).mean()) / max(sig2, 1e-30)
                L_ = float(getattr(self, "seq_len", 197))
                mu0 = a0.n / max(a0.nlast, 1.0)
                self.tab_share = float(a0.floor / (a0.floor + mu0 * k_ / L_))
                log.info("[HIPPO-VIS] task %d cls / position rows (stream units): f_A %.3g mu %.2f gamma^2/sigma^2 %.3g "
                         "L %d -> share %.4f" % (t, a0.floor, mu0, k_, int(L_), self.tab_share))
        self.Taug = self.Tfresh = None
        if (self.ro_common or self.fresh_exact) and self.Caug is not None and self.ro.floor is not None:
            # AUGMENTED EXACT READOUT OPERATOR, built once per task: (I + mu C_aug / f)^-1, C_aug = E[[h; 1][h; 1]^T]
            # over every earlier image, f / mu the readout area's
            f_ = max(self.ro.floor, 1e-300)
            mu_ = self.ro.n / max(self.ro.nlast, 1.0)
            ev_, V_ = torch.linalg.eigh(self.Caug)
            sh_ = f_ / (f_ + mu_ * ev_.clamp(min=0.0))
            Ta_ = (V_ * sh_.unsqueeze(0)) @ V_.t()
            self.Taug = (0.5 * (Ta_ + Ta_.t())).float().contiguous()
            if self.fresh_exact and self.faug is not None:
                Qa_ = self.faug.double()
                Tf_ = Ta_ - Qa_ @ (Qa_.t() @ Ta_)
                Tf_ = Tf_ - (Tf_ @ Qa_) @ Qa_.t()
                self.Tfresh = (0.5 * (Tf_ + Tf_.t())).float().contiguous()
            log.info("[HIPPO-VIS] task %d augmented readout operator: eigen-share mean %.3f min %.4f max %.3f (f %.3g mu %.2f)"
                     % (t, float(sh_.mean()), float(sh_.min()), float(sh_.max()), f_, mu_))
        if self.oldrow:
            # PER-ROW HOLD OF THE OLD HEADS: row k (class k) is held off its own class record, augmented with the
            # constant input of its bias, [mean_k; 1] (unit). On a new image CE only lowers an old row; lowering row k
            # can cost only class k's own inputs (on class m != k it widens m's margin), so row k protects only the
            # place class k sits
            A = torch.zeros(self.K, H.shape[1] + 1, device=H.device)
            A[ks, :-1] = H
            A[ks, -1] = 1.0
            self.arec = F.normalize(A, dim=1)
            if self.oldrow_span and self.cbas:
                rmax = max(B.shape[1] for B in self.cbas.values())
                Bp = torch.zeros(self.K, H.shape[1] + 1, rmax, device=H.device)
                for k, B in self.cbas.items():
                    Bp[k, :, :B.shape[1]] = B.to(H.device)
                self.rbas = Bp
            o = 0
            self.hoff = {}
            for hd in self.heads:
                self.hoff[id(hd.weight)] = (hd, o)
                o += hd.out_features
        # pattern bases and hold bases
        for a in self.all_areas:
            P = self.pat.get(a.name)
            if P is not None:
                Pm = (P[1][ks] / P[0][ks].unsqueeze(1)).t()                       # d x R
                a.Q = orth(Pm)
                # PATTERN HOLD: the patterns' part outside U, kept only where it is real (>= 1e-3 of a unit pattern
                # direction, absolute) and re-orthogonalised against U in float64 -- a residual that is numerical
                # noise, normalised by QR, is not orthogonal to U and makes W W^T a non-projector (T then writes
                # inside U)
                Ud = a.U
                Qd = a.Q.double()
                R_ = Qd - Ud @ (Ud.t() @ Qd)
                Qr, Rr = torch.linalg.qr(R_)
                Rq = Qr[:, Rr.diagonal().abs() > 1e-3]
                Rq = Rq - Ud @ (Ud.t() @ Rq)
                Rq = torch.linalg.qr(Rq).Q if Rq.shape[1] else Rq
                a.W = torch.cat([Ud, Rq], 1).float().contiguous()
            else:
                a.Q, a.W = None, a.U.float()
            a.c = 1.0
            a.av = a.b = a.sb = None
            if self.owm and a.floor is not None:
                # MINIMUM-INTERFERENCE STEP (closed form, same rule as for decoders): the step minimises <G, dW> +
                # (1/2eta)(|dW|^2 + mu/f tr(dW C dW^T)), C the earlier inputs' moment (the memory), mu = n_old / n_last,
                # f the area's GD noise floor: dW = -eta G (I + mu C / f)^-1. With C = U diag(lam) U^T + tau (I - U U^T),
                # tau = tail / (d - r): each held direction keeps f / (f + mu lam_i), the tail f / (f + mu tau); the
                # class-pattern columns stay fully held. A bias (weight on the constant input 1, used by every earlier
                # row) keeps f / (f + mu).
                r_ = a.U.shape[1]
                mu_ = a.n / max(a.nlast, 1.0)
                f_ = max(a.floor, 1e-300)
                tau_ = a.tail / max(a.d - r_, 1)
                av_ = torch.zeros(a.W.shape[1], device=a.W.device, dtype=torch.float64)
                av_[:r_] = f_ / (f_ + mu_ * a.lam.to(av_.device))
                a.b = float(f_ / (f_ + mu_ * tau_))
                a.av = av_.float()
                a.sb = float(f_ / (f_ + mu_))
                a.dense = None
                if self.owmx and getattr(a, "Cfull", None) is not None:
                    # EXACT closed form on the full earlier moment (no isotropic-tail model), class patterns held
                    ev_, V_ = torch.linalg.eigh(a.Cfull.double())
                    sh_ = f_ / (f_ + mu_ * ev_.clamp(min=0.0))
                    Tx_ = (V_ * sh_.unsqueeze(0)) @ V_.t()
                    if a.Q is not None and a.Q.shape[1] > 0:
                        Qp_ = a.Q.double()
                        Tx_ = Tx_ - Qp_ @ (Qp_.t() @ Tx_)
                        Tx_ = Tx_ - (Tx_ @ Qp_) @ Qp_.t()
                    a.dense = (0.5 * (Tx_ + Tx_.t())).float().contiguous()
                    a.dshare = (float(sh_.mean()), float(sh_.min()), float(sh_.max()))
                a.daug = None
                if self.owmx and self.biasjoint and getattr(a, "Caug", None) is not None and a.kind != "ro" \
                        and any(getattr(m_, "bias", None) is not None for m_ in a.mods):
                    # (biasjoint) [W, b] jointly: (I + mu C_aug / f)^-1, class patterns held as [pattern; 1]
                    ev_, V_ = torch.linalg.eigh(a.Caug.double())
                    sh_ = f_ / (f_ + mu_ * ev_.clamp(min=0.0))
                    Ta_ = (V_ * sh_.unsqueeze(0)) @ V_.t()
                    Pp_ = self.pat.get(a.name)
                    if Pp_ is not None:
                        ks_ = self.sc["keys"]
                        Pm_ = (Pp_[1][ks_] / Pp_[0][ks_].unsqueeze(1)).double()
                        Qa_ = orth(torch.cat([Pm_, torch.ones(Pm_.shape[0], 1, dtype=Pm_.dtype, device=Pm_.device)], 1).t())
                        Ta_ = Ta_ - Qa_ @ (Qa_.t() @ Ta_)
                        Ta_ = Ta_ - (Ta_ @ Qa_) @ Qa_.t()
                    a.daug = (0.5 * (Ta_ + Ta_.t())).float().contiguous()
                    a.dense = a.daug[:-1, :-1].contiguous()          # (weight-only block: comparator, bias-free maps)
        if self.owm:
            sh = {}
            for a in self.all_areas:
                if a.av is not None:
                    r_ = a.U.shape[1]
                    sh.setdefault(a.tag, []).append((a.b, float(a.av[:r_].mean()) if r_ else 0.0, a.sb,
                                                     a.n / max(a.nlast, 1.0)))
            dx = {}
            for a in self.all_areas:
                if getattr(a, "dense", None) is not None:
                    dx.setdefault(a.tag, []).append(a.dshare)
            if dx:
                log.info("[HIPPO-VIS] task %d EXACT operator eigen-shares by kind (mean / min / max): %s" % (
                    t, " ".join("%s %.3f/%.4f/%.3f" % (k, *[sum(x[i] for x in v) / len(v) for i in range(3)]) for k, v in dx.items())))
            log.info("[HIPPO-VIS] task %d OWM shares by kind (tail b / held av mean / bias / mu): %s" % (
                t, " ".join("%s %.3f/%.4f/%.3f/%.1f" % (k, *[sum(x[i] for x in v) / len(v) for i in range(4)])
                            for k, v in sh.items())))
        log.info("[HIPPO-VIS] task %d start: %d records, radius mean %.3f (min %.3f max %.3f); hold rank/dim %s" % (
            t, len(ks), float(self.sc["r"].mean()), float(self.sc["r"].min()), float(self.sc["r"].max()),
            self._rank_summary()))

    def _rank_summary(self):
        out = {}
        for a in self.all_areas:
            if a.W is None:
                continue
            out.setdefault(a.tag, []).append((a.U.shape[1], a.W.shape[1], a.d))
        return " ".join("%s %.0f+%.0f/%.0f" % (k, sum(v[0] for v in vs) / len(vs),
                                               sum(v[1] - v[0] for v in vs) / len(vs), sum(v[2] for v in vs) / len(vs))
                        for k, vs in out.items())

    # ------------------------------------------------------------------ wake: readout (conflict, gate, comparator)
    def readout(self, h, z, y):
        """After the forward: h (B, d) decision features, z (B, C) logits over the seen classes, y gold columns."""
        self.armed = None
        self.dWA, self.cmpA = {}, {}
        self._cmpdone = False
        if not self.active:
            return
        self.xin["readout"] = h
        if self._energy_step():
            with torch.no_grad():
                if self.dw:
                    e = self._energy_w(self.ro, h.detach(), self._dz2(z.detach(), y))
                else:
                    e = self._energy(self.ro, h.detach(), None)
                o = self.now.get("readout")
                self.now["readout"] = e if o is None else [o[0] + e[0], o[1] + e[1], o[2] + e[2]]
        with torch.no_grad():
            hf = h.detach().float()
            zf = z.detach().float()
            sc = self.sc
            if self.pdet:
                Sp, own = self._pdet_scores(hf, y, update=True)
                inside = Sp > own.unsqueeze(1)
                conf = inside.any(1)
                jn = (Sp - own.unsqueeze(1)).argmax(1)
            else:
                hs = F.normalize(hf - sc["c"], dim=1)
                Sp = hs @ sc["M"].t()
                inside = Sp > sc["r"].unsqueeze(0)
                conf = inside.any(1)
                jn = (Sp - sc["r"].unsqueeze(0)).argmax(1)
            zg = zf.gather(1, y.unsqueeze(1)).squeeze(1)
            z2 = zf.scatter(1, y.unsqueeze(1), float("-inf"))
            zr = z2.max(1).values
            ncls = self.cls_cnt[y]
            nr = ncls / max(self.ep_prog, 1.0)
            Nj = sc["N"][jn]
            rp = torch.where(conf, Nj / (Nj + nr).clamp(min=1e-6), torch.zeros_like(Nj))
            self.plast = float((1.0 - rp).mean().clamp(0.0, 1.0))
            A = (conf & (zg <= zr)).nonzero(as_tuple=True)[0]
            if _DIAG.get("cmpoff"):
                A = A[:0]
            self.armed = A
            self.stat["conf"].append(conf.float().mean())
            self.stat["armed"].append(float(A.numel()) / max(y.numel(), 1))
            self.stat["plast"].append(self.plast)
        if A.numel() and not self.dw:
            # the operators and the comparator's step basis need only the forward: formed here, where the armed rows
            # are already read on the host, so no further sync is needed between the backward and the step
            self._advance()
            self._cmp_basis(A)
        if A.numel() and z.requires_grad:
            bounds = []
            o = 0
            for hd in self.heads:
                if hd.weight.requires_grad and o < z.shape[1]:
                    bounds.append((hd, o, o + hd.out_features))
                o += hd.out_features

            def gz(g, _h=h, _A=A, _b=bounds):
                with torch.no_grad():
                    hA = _h[_A].float()
                    # one product for every head: head rows are contiguous column slices
                    bb_ = [(hd, s, e) for hd, s, e in _b if e <= g.shape[1]]
                    if bb_:
                        s0_, e0_ = bb_[0][1], bb_[-1][2]
                        Gall_ = g[_A, s0_:e0_].float().t() @ hA
                        for hd, s, e in bb_:
                            self.dWA[id(hd)] = Gall_[s - s0_:e - s0_]
                return None
            z.register_hook(gz)

    @torch.no_grad()
    def _cmp_basis(self, A):
        """The comparator's step basis: every armed image's mismatch with the patterns per area, orthonormalised
        (Q~), and M = Q~ diag(s) - T Q~. Areas of one width are stacked: the mismatch, the Gram matrices and, for
        the low-rank operator forms, T Q~ and s_q are batched products."""
        self._cmpdone = True
        ents = []
        for a in self.all_areas:
            if a.Q is None or a.W is None or (a.kind == "conv" and self.cmp_conv_off):
                continue
            x = self.xin.get(a.name)
            if x is None:
                continue
            # conv areas: the armed image's mean input lies largely inside the held span (within-class spread is
            # not in the class means), so its mismatch is taken with the whole held span W = [U, patterns]
            B_ = a.W if ((a.kind == "conv" and CMP_CONV_HELD) or self.cmp_held_all) else a.Q
            if B_ is a.W and getattr(a, "dense", None) is None and getattr(a, "pd", None) is None and self.cmpowm \
                    and a.floor is not None and a.U is not None and (a.av is not None or self.cmpcons):
                # mismatch taken off the whole held span W (U inside it): Q~ is orthogonal to W, so T Q~ is the
                # tail share times Q~ (c, or b = f / (f + mu tau)) and the give-back share is the same number (s = c
                # under cmpcons; the objective gives q^T C q = tau, s = b) -- M = s Q~ - T Q~ = 0 exactly: skipped
                continue
            X = self._ans_mean(a, x[A].detach()) if a.kind != "ro" else x[A].detach().float()
            ents.append((a, B_, X))
        if not ents:
            return
        grp = {}
        for i_, (a, B_, X) in enumerate(ents):
            grp.setdefault(B_.shape[0], []).append(i_)
        gl_ = []
        for d_, ix in grp.items():
            ars = [ents[i_][0] for i_ in ix]
            Bs_ = tuple(ents[i_][1] for i_ in ix)
            key_ = ("B",) + tuple(a.name for a in ars)
            cb_ = self._cmpcache.get(key_)
            if cb_ is None or any(x_ is not y_ for x_, y_ in zip(cb_[0], Bs_)):
                rm_ = max(1, max(B_.shape[1] for B_ in Bs_))
                Bp_ = torch.zeros(len(Bs_), d_, rm_, device=Bs_[0].device)
                for j_, B_ in enumerate(Bs_):
                    Bp_[j_, :, :B_.shape[1]] = B_
                cb_ = self._cmpcache[key_] = (Bs_, Bp_)
            Xg = torch.stack([ents[i_][2] for i_ in ix])                               # (G, n, d)
            Xp = Xg - torch.bmm(torch.bmm(Xg, cb_[1]), cb_[1].transpose(1, 2))
            gl_.append((ars, Xp, Xg.pow(2).sum(2).mean(1)))
        # orthonormal row-space basis of every area's mismatch rows at once (Gram eigh, batched)
        Gs = torch.cat([torch.bmm(Xp, Xp.transpose(1, 2)) for _, Xp, _ in gl_]).double().cpu()
        ev, V = torch.linalg.eigh(Gs)                 # (on the host: batched CUDA eigh is slow for n > 32)
        dev_ = gl_[0][1].device
        ev, V = ev.float().to(dev_), V.float().to(dev_)
        # a mismatch direction must carry a real share of an armed row's energy (>= 1e-3 of the mean row
        # energy): normalising numerical residue to unit length would open random directions (held mode)
        thr = torch.cat([rE_ for _, _, rE_ in gl_]).float().unsqueeze(1) * 1e-3
        ok_ = (ev > 1e-6 * ev[:, -1:]) & (ev > thr)
        Wn = V * (ok_.float() / ev.clamp(min=1e-20).sqrt()).unsqueeze(1)
        o_ = 0
        for ars, Xp, _ in gl_:
            G_ = len(ars)
            Qg = torch.bmm(Xp.transpose(1, 2), Wn[o_:o_ + G_])                         # (G, d, n) orthonormal or 0
            o_ += G_
            ops_ = self._cmp_ops(ars)
            if ops_ is None:
                for j_, a in enumerate(ars):
                    self._cmp_area(a, Qg[j_])
                continue
            kind, W_, wv_, b_, c_, sm_, sv_, Uo_, lam_, tau_, mu_, f_, anyo_ = ops_
            Xr = Qg.transpose(1, 2)                                                    # (G, n, d)
            if kind == "av":
                TQ = torch.bmm(torch.bmm(Xr, W_) * wv_, W_.transpose(1, 2)) + Xr * b_
            else:
                TQ = (Xr - torch.bmm(torch.bmm(Xr, W_), W_.transpose(1, 2))) * c_
            TQ = TQ.transpose(1, 2)                                                    # (G, d, n)
            if anyo_:
                pq = torch.bmm(Uo_.transpose(1, 2), Qg)                                # (G, r, n)
                p2 = pq.pow(2)
                qcq = (lam_ * p2).sum(1) + tau_ * (1.0 - p2.sum(1)).clamp(min=0.0)
                so_ = f_ / (f_ + mu_ * qcq)                                            # (G, n)
                s_ = torch.where(sm_ == 1, so_, torch.where(sm_ == 2, sv_.expand_as(so_), torch.ones_like(so_)))
                M = Qg * s_.unsqueeze(1) - TQ
                v_ = (Qg.pow(2).sum(1) > 0).float() * (sm_ > 0).float()
                self.cmpS.append(torch.stack([(s_ * v_).sum(), v_.sum()]))
            else:
                M = Qg - TQ
            for j_, a in enumerate(ars):
                self.cmpA[a.name] = (Qg[j_], M[j_])

    @torch.no_grad()
    def _cmp_ops(self, ars):
        """A group's stacked operator (zero-padded spans) and give-back-share terms; None when an area's
        operator has a form that is applied per area (dense, per-direction share) or the forms are mixed.
        s mode per area: 0 none (s = 1), 1 the objective f / (f + mu q^T C q), 2 the tail share c (cmpcons)."""
        sig_ = tuple((a.W, a.av, a.b, a.c, getattr(a, "dense", None), getattr(a, "pd", None), a.U, a.floor) for a in ars)
        key_ = ("T",) + tuple(a.name for a in ars)
        co_ = self._cmpcache.get(key_)
        if co_ is not None and len(co_[0]) == len(sig_) and all(
                all((x_ is y_) if (torch.is_tensor(x_) or x_ is None) else (x_ == y_) for x_, y_ in zip(e0_, e1_))
                for e0_, e1_ in zip(co_[0], sig_)):
            return co_[1]
        kinds = set()
        for a in ars:
            if getattr(a, "dense", None) is not None or getattr(a, "pd", None) is not None:
                kinds.add("x")
            elif a.av is not None:
                kinds.add("av" if a.av.shape[0] == a.W.shape[1] else "x")
            else:
                kinds.add("c")
            if self.cmpowm and a.floor is not None and a.U is not None and self.cmpcons \
                    and getattr(a, "dense", None) is not None and getattr(a, "Cfull", None) is not None:
                kinds.add("x")
        if "x" in kinds or len(kinds) != 1:
            self._cmpcache[key_] = (sig_, None)
            return None
        kind = kinds.pop()
        G_, d_ = len(ars), ars[0].W.shape[0]
        dev = ars[0].W.device
        rm_ = max(1, max(a.W.shape[1] for a in ars))
        W_ = torch.zeros(G_, d_, rm_, device=dev)
        wv_ = torch.zeros(G_, 1, rm_, device=dev)
        b_ = torch.zeros(G_, 1, 1, device=dev)
        c_ = torch.ones(G_, 1, 1, device=dev)
        sm_l, sv_l = [0] * G_, [1.0] * G_
        ro_ = 1
        for j_, a in enumerate(ars):
            W_[j_, :, :a.W.shape[1]] = a.W
            if kind == "av":
                wv_[j_, 0, :a.W.shape[1]] = a.av - a.b
                b_[j_] = float(a.b)
            else:
                c_[j_] = float(a.c)
            if self.cmpowm and a.floor is not None and a.U is not None:
                if self.cmpcons and a.av is None:
                    sm_l[j_], sv_l[j_] = 2, float(a.c)
                else:
                    sm_l[j_] = 1
                    ro_ = max(ro_, a.U.shape[1])
        Uo_ = torch.zeros(G_, d_, ro_, device=dev)
        lam_ = torch.zeros(G_, ro_, 1, device=dev)
        tau_ = torch.zeros(G_, 1, device=dev)
        mu_ = torch.zeros(G_, 1, device=dev)
        f_ = torch.ones(G_, 1, device=dev)
        for j_, a in enumerate(ars):
            if sm_l[j_] != 1:
                continue
            r_ = a.U.shape[1]
            Uo_[j_, :, :r_] = a.U.float()
            lam_[j_, :r_, 0] = a.lam.float().to(dev)
            tau_[j_] = a.tail / max(a.d - r_, 1)
            mu_[j_] = a.n / max(a.nlast, 1.0)
            f_[j_] = a.floor
        sm_ = torch.tensor(sm_l, device=dev, dtype=torch.float32).unsqueeze(1)
        sv_ = torch.tensor(sv_l, device=dev, dtype=torch.float32).unsqueeze(1)
        ops_ = (kind, W_, wv_, b_, c_, sm_, sv_, Uo_, lam_, tau_, mu_, f_, any(m_ > 0 for m_ in sm_l))
        self._cmpcache[key_] = (sig_, ops_)
        return ops_

    @torch.no_grad()
    def _cmp_area(self, a, Qt):
        """the per-area comparator step basis (operator forms not batched)"""
        TQ = self._T(a, Qt.t()).t()
        M = Qt - TQ
        if self.cmpowm and a.floor is not None and a.U is not None:
            # give-back restored only to the minimum-interference share along each mismatch direction
            mu = a.n / max(a.nlast, 1.0)
            if self.cmpcons and getattr(a, "dense", None) is not None and getattr(a, "Cfull", None) is not None:
                qcq = (Qt * (a.Cfull @ Qt)).sum(0)                     # (cmpcons) the exact moment
                s_ = a.floor / (a.floor + mu * qcq)
            elif self.cmpcons and a.av is None:
                s_ = torch.full((Qt.shape[1],), float(a.c), device=Qt.device)   # (cmpcons) the tail share c
            else:
                cc = self.cmpc.get(a.name)
                if cc is None or cc[0] is not a.U:
                    r_ = a.U.shape[1]
                    cc = self.cmpc[a.name] = (a.U, a.U.float(), a.lam.float(), a.tail / max(a.d - r_, 1))
                _, Uf, lamf, tau = cc
                pq = Uf.t() @ Qt                                         # (r, n)
                qcq = (lamf.unsqueeze(1) * pq.pow(2)).sum(0) + tau * (1.0 - pq.pow(2).sum(0)).clamp(min=0.0)
                s_ = a.floor / (a.floor + mu * qcq)
            M = Qt * s_.unsqueeze(0) - TQ
            v_ = (Qt.pow(2).sum(0) > 0).float()
            self.cmpS.append(torch.stack([(s_ * v_).sum(), v_.sum()]))
        self.cmpA[a.name] = (Qt, M)

    @torch.no_grad()
    def _advance(self):
        """the step count and, on build steps, the operators (pre_backward's work), done once per step: from
        readout() when the comparator is armed (the operators only read the forward's statistics, all collected by
        then), else from pre_backward()"""
        if self._adv:
            return
        self._adv = True
        self.steps += 1
        if self.steps >= self.next_build:
            if self.dw:
                self._bpend = True               # (the decision weights arrive in the backward: built in apply())
            else:
                self._build()
            self.next_build *= 2

    def pre_backward(self):
        if not self.active:
            return
        self._advance()
        self._adv = False

    @torch.no_grad()
    def _fresh_basis(self, Aa):
        """the fresh head's hold basis: the earlier classes' [mean; 1] (option fresh_common: with the component common to
        ALL classes -- earlier records and the present classes' running means -- taken out: the hold keeps the new
        class's pattern across the earlier classes, not its level shared by every class)"""
        if not self.fresh_common:
            return orth(Aa)
        cols = [Aa]
        if self.pcm is not None:
            cnt, sm = self.pcm
            has = cnt > 0
            if bool(has.any()):
                Pp = sm[has] / cnt[has].unsqueeze(1)
                cols.append(torch.cat([Pp, torch.ones(Pp.shape[0], 1, device=Pp.device)], 1).t())
        Al = torch.cat(cols, 1)
        ab = Al.mean(1, keepdim=True)
        ab = ab / ab.norm().clamp(min=1e-30)                       # the common direction of all classes
        self._fcommon = ab
        Ar = Aa - ab @ (ab.t() @ Aa)
        B = orth(Ar)
        return B - ab @ (ab.t() @ B)

    def _build(self):
        cs = {}
        if self.fresh_common and getattr(self, "_faugA", None) is not None:
            self.faug = self._fresh_basis(self._faugA)             # (the present classes' means move: rebuilt)
        for a in self.all_areas:
            e = self.now.get(a.name)
            if a.U is None or e is None:
                continue
            tn = float(e[1]) / e[2]                                     # present per-position tail energy
            a.c = max(0.0, min(1.0, 1.0 - a.tail / tn)) if tn > 0 else 1.0
            if self.trunk_c1 and a.kind != "ro":
                a.c = 1.0
            tm = self.tmom.get(a.name) if self.pdir else None
            if tm is not None and a.kind != "ro":
                # PER-DIRECTION NOVELTY SHARE: the scalar c compares the present tail energy with the stored one as a
                # whole; here each eigen-direction v of the present tail moment is compared with the stored tail's
                # energy per tail direction (the stored tail spread over the d - r tail dims): D_v = 1 - floor / e_v
                # (clipped), the scalar rule applied per direction (as the gains' D per coordinate). A direction where
                # the present input carries more energy than the earlier inputs did is novel and passes.
                C = tm[0] / max(tm[1], 1.0)
                ev, V = torch.linalg.eigh(0.5 * (C + C.t()))
                ev = ev.clamp(min=0)
                fl = float(a.tail) / max(a.d - a.U.shape[1], 1)
                Dv = (1.0 - fl / ev.clamp(min=1e-30)).clamp(0.0, 1.0)
                k = Dv > 0
                a.pd = (V[:, k].contiguous(), Dv[k].contiguous()) if bool(k.any()) else None
                if a.pd is None:
                    a.c = 0.0
            if _DIAG.get("convhold") and a.kind == "conv":
                a.c = 0.0
            if _DIAG.get("rohold") and a.kind == "ro":
                a.c = 0.0
            cs.setdefault(a.tag, []).append(a.c)
        for g, e in self.gnow.items():
            if g not in self.gold:
                continue
            ln = e[0] / max(e[1], 1.0)
            ref = self.gold[g][0] / max(self.gold[g][1], 1.0)
            D = torch.where(ln > 0, ((ln - ref) / ln.clamp(min=1e-30)).clamp(0.0, 1.0), torch.ones_like(ln))
            D.masked_fill_(self.gown[g], 0.0)
            if self.owm and self.owm_coord and g in self.gfm:
                # MINIMUM-INTERFERENCE STEP for a gain g_i: E_old (dg_i x_i)^2 = dg_i^2 e_i, e_i the coordinate's earlier
                # energy: share f / (f + mu e_i) in place of the novelty share and the owned-coordinate mask
                f_, mu_ = self.gfm[g]
                D = f_ / (f_ + mu_ * ref.float())
            if _DIAG.get("gainhold"):
                D.zero_()
            self.D[g] = D
        if self.steps in (1, 8, 64, 512):
            log.info("[HIPPO-VIS] step %d: tail share c by kind %s | gain share D mean %.3f" % (
                self.steps, " ".join("%s %.3f[%.2f,%.2f]" % (k, sum(v) / len(v), min(v), max(v)) for k, v in cs.items()),
                float(torch.cat(list(self.D.values())).mean()) if self.D else -1))

    # ------------------------------------------------------------------ wake: gradient hold (after the backward)
    @torch.no_grad()
    def apply(self):
        if not self.active:
            return
        if self._bpend:
            self._build()
            self._bpend = False
        A = self.armed
        pf_ = getattr(self, "_prof", None)
        pf_ = pf_ if (pf_ is not None and pf_.live) else None
        if pf_ is not None:
            ev0_ = torch.cuda.Event(enable_timing=True); ev0_.record()
        # comparator step basis: the armed rows' mismatch with the patterns, per area (formed in readout() when the
        # operators could be built there; here only on the decision-weighted path, whose build waits for the backward)
        if A is not None and A.numel() and not self._cmpdone:
            self._cmp_basis(A)
        if pf_ is not None:
            ev1_ = torch.cuda.Event(enable_timing=True); ev1_.record(); pf_.mark("apply:comparator", ev0_, ev1_)
        oldg_ = []
        bj_ = set()
        pend_ = {}                                       # plain held gradients g T, batched per shape below
        cg_, cs_ = [], []                                # constants' gradients and their shares
        for a in self.all_areas:
            if a.W is None:
                continue
            for m in a.mods:
                p = m.weight
                if p.grad is None:
                    continue
                if self.oldrow and a.kind == "ro" and id(p) not in self.fresh:
                    oldg_.append((m, p.grad, m.bias.grad if m.bias is not None else None))
                    continue
                g = p.grad.reshape(p.shape[0], -1)
                g0 = g.pow(2).sum() if self._logstep else None        # read only on logged steps
                dWA = self.dWA.get(id(m))
                cm = self.cmpA.get(a.name)
                if id(p) in self.fresh and self.fresh_exact and self.Tfresh is not None and m.bias is not None:
                    gb_ = m.bias.grad if m.bias.grad is not None else torch.zeros(g.shape[0], device=g.device)
                    G_ = torch.cat([g, gb_.unsqueeze(1)], 1) @ self.Tfresh      # (fresh_exact) every readout row passes T
                    gh = G_[:, :-1].contiguous()
                    if m.bias.grad is not None:
                        m.bias.grad.copy_(G_[:, -1])
                elif id(p) in self.fresh and self.fresh_aug and self.faug is not None and m.bias is not None:
                    # (fresh_aug) fresh head: [w; b] held jointly off the earlier classes' [mean; 1]
                    gb_ = m.bias.grad if m.bias.grad is not None else torch.zeros(g.shape[0], device=g.device)
                    G_ = torch.cat([g, gb_.unsqueeze(1)], 1)
                    G_ = G_ - (G_ @ self.faug) @ self.faug.t()
                    gh = G_[:, :-1].contiguous()
                    if m.bias.grad is not None:
                        m.bias.grad.copy_(G_[:, -1])
                elif id(p) in self.fresh:
                    # FRESH HEAD (no memory yet): held only off the earlier memories' class-mean patterns (option
                    # fresh_span: off the readout's whole held span W = [U, patterns], at full scale)
                    B_ = a.W if self.fresh_span else a.Q
                    gh = g - (g @ B_) @ B_.t()
                elif getattr(a, "daug", None) is not None and m.bias is not None and m.bias.grad is not None:
                    # (biasjoint) weight and bias held together by the augmented moment
                    g_in = (g - dWA) if (dWA is not None and a.Q is not None and a.kind == "tok") else g
                    G_ = torch.cat([g_in, m.bias.grad.reshape(-1, 1)], 1) @ a.daug
                    gh = G_[:, :-1].contiguous()
                    m.bias.grad.copy_(G_[:, -1])
                    bj_.add(id(m.bias))
                    if dWA is not None and a.Q is not None and a.kind == "tok":
                        B_ = a.W if self.cmp_held_all else a.Q
                        gh = gh + (dWA - (dWA @ B_) @ B_.t())
                    elif a.kind == "conv" and cm is not None:
                        gh = gh + (g @ cm[1]) @ cm[0].t()
                elif a.kind == "conv" and cm is not None:
                    # conv areas: every position of an armed image is an answer row, so substituting its rows would
                    # open every direction; the give-back is taken along the armed images' mismatch directions only
                    # (the step's rule): g T + (g M) Q~^T
                    gh = self._T(a, g)
                    gh += (g @ cm[1]) @ cm[0].t()
                elif dWA is not None and a.Q is not None and self._hform(a) is not None:
                    # (batched below) (g - dWA) T + (dWA - (dWA B) B^T), B = Q (cmp_held_all: W)
                    pend_.setdefault((g.shape[0], g.shape[1], self._hform(a), True), []).append((a, p, g, dWA))
                    continue
                elif dWA is not None and a.Q is not None:
                    B_ = a.W if self.cmp_held_all else a.Q
                    gh = self._T(a, g - dWA) + (dWA - (dWA @ B_) @ B_.t())
                elif self._hform(a) is not None:
                    pend_.setdefault((g.shape[0], g.shape[1], self._hform(a), False), []).append((a, p, g, None))
                    continue
                else:
                    gh = self._T(a, g)
                if self._logstep:
                    st = self.stat["pass"].setdefault(a.tag, [0.0, 0.0])
                    st[0] = st[0] + gh.pow(2).sum(); st[1] = st[1] + g0
                p.grad.copy_(gh.reshape(p.shape))
            for p in a.consts:
                if self.oldrow and a.kind == "ro":
                    continue                     # (old heads' biases: held with their rows, _rowhold)
                if id(p) in bj_:
                    continue                     # (biasjoint: held with its weight)
                if p.grad is not None and _DIAG.get("consthold"):
                    p.grad.zero_()
                elif p.grad is not None and id(p) in self.tabconst and self.tab_share is not None:
                    cg_.append(p.grad); cs_.append(float(self.tab_share))     # (tabln) stream units
                elif p.grad is not None and a.av is not None and id(p) not in self.fresh:
                    cg_.append(p.grad); cs_.append(float(a.sb))
                elif p.grad is not None and a.c != 1.0 and id(p) not in self.fresh:
                    cg_.append(p.grad); cs_.append(float(a.c))
        if cg_:
            torch._foreach_mul_(cg_, cs_)        # every constant's share in one multi-tensor launch
        for key_, ents_ in pend_.items():
            self._hold_batched(key_, ents_)
        if pf_ is not None:
            ev2_ = torch.cuda.Event(enable_timing=True); ev2_.record(); pf_.mark("apply:hold", ev1_, ev2_)
        if oldg_:
            g0_ = sum(e[1].pow(2).sum() for e in oldg_)
            self._oldrows(oldg_)
            if self._logstep:
                st = self.stat["pass"].setdefault("ro", [0.0, 0.0])
                st[0] = st[0] + sum(e[1].pow(2).sum() for e in oldg_); st[1] = st[1] + g0_
        gg_, gd_ = [], []
        for g, mod in self.gains.items():
            D = self.D.get(g)
            if D is None or g in self.bn_skip:
                continue
            for p in (mod.weight, getattr(mod, "bias", None)):
                if p is not None and p.grad is not None:
                    gg_.append(p.grad); gd_.append(D)
        if gg_:
            torch._foreach_mul_(gg_, gd_)        # every gain's per-coordinate share in one multi-tensor launch
        self.xin = {}

    @staticmethod
    def _hform(a):
        """the operator form of an area's hold when it is batchable ('av' minimum-interference, 'c' tail share)"""
        if getattr(a, "dense", None) is not None or getattr(a, "pd", None) is not None:
            return None
        if a.av is not None:
            return "av" if a.av.shape[0] == a.W.shape[1] else None
        return "c"

    @torch.no_grad()
    def _Tb(self, key_, ents_):
        """X T_a for a stack of same-shape rows X of areas a at once: the areas' spans zero-padded and stacked once
        per operator build; bf16 operands, fp32 accumulation (the products are compute-bound). ents_: [(a, X (n, d))]."""
        n_, d_, form = key_
        sig_ = tuple((a.W, a.av, a.b, a.c) for a, _ in ents_)
        ck_ = ("H", n_, d_, form) + tuple(a.name for a, _ in ents_)
        co_ = self._cmpcache.get(ck_)
        if co_ is None or not all((e0[0] is e1[0]) and (e0[1] is e1[1]) and e0[2] == e1[2] and e0[3] == e1[3]
                                  for e0, e1 in zip(co_[0], sig_)):
            rm_ = max(1, max(a.W.shape[1] for a, _ in ents_))
            dev = ents_[0][1].device
            W_ = torch.zeros(len(ents_), d_, rm_, device=dev)
            wv_ = torch.zeros(len(ents_), 1, rm_, device=dev)
            sc_ = torch.ones(len(ents_), 1, 1, device=dev)
            for j_, (a, _) in enumerate(ents_):
                W_[j_, :, :a.W.shape[1]] = a.W
                if form == "av":
                    wv_[j_, 0, :a.W.shape[1]] = a.av - a.b
                    sc_[j_] = float(a.b)
                else:
                    sc_[j_] = float(a.c)
            co_ = self._cmpcache[ck_] = (sig_, (W_.to(torch.bfloat16), wv_, sc_))
        W_, wv_, sc_ = co_[1]
        G = torch.stack([x for _, x in ents_])                                       # (E, n, d)
        y_ = torch.bmm(G.to(torch.bfloat16), W_, out_dtype=torch.float32)
        if form == "av":
            y_.mul_(wv_)
            Y = torch.bmm(y_.to(torch.bfloat16), W_.transpose(1, 2), out_dtype=torch.float32)
            Y.add_(G * sc_)                                                          # b x + (x W diag(av - b)) W^T
        else:
            Y = torch.bmm(y_.to(torch.bfloat16), W_.transpose(1, 2), out_dtype=torch.float32)
            Y = torch.sub(G, Y, out=Y).mul_(sc_)                                     # c (x - x W W^T)
        return G, Y

    @torch.no_grad()
    def _hold_batched(self, key_, ents_):
        """the held weight gradients of one shape at once: g T, or with the comparator's armed rows (dWA, their part
        of the gradient) (g - dWA) T + (dWA - (dWA B) B^T); ents_: [(a, p, g, dWA or None)]"""
        n_, d_, form, wa_ = key_
        if wa_:
            G0 = torch.stack([g for _, _, g, _ in ents_])
            DA = torch.stack([w for _, _, _, w in ents_])
            _, Y = self._Tb((n_, d_, form), [(a, g - w) for a, _, g, w in ents_])
            Bs_ = tuple((a.W if self.cmp_held_all else a.Q) for a, _, _, _ in ents_)
            ck_ = ("QB", n_, d_) + tuple(a.name for a, _, _, _ in ents_)
            cb_ = self._cmpcache.get(ck_)
            if cb_ is None or any(x_ is not y_ for x_, y_ in zip(cb_[0], Bs_)):
                rm_ = max(1, max(B_.shape[1] for B_ in Bs_))
                Bp_ = torch.zeros(len(Bs_), d_, rm_, device=G0.device)
                for j_, B_ in enumerate(Bs_):
                    Bp_[j_, :, :B_.shape[1]] = B_
                cb_ = self._cmpcache[ck_] = (Bs_, Bp_)
            Y.add_(DA - torch.bmm(torch.bmm(DA, cb_[1]), cb_[1].transpose(1, 2)))
        else:
            G0, Y = self._Tb((n_, d_, form), [(a, g) for a, _, g, _ in ents_])
        if self._logstep:
            ge_ = G0.pow(2).sum((1, 2))
            ye_ = Y.pow(2).sum((1, 2))
            for j_, (a, p, g, _) in enumerate(ents_):
                st = self.stat["pass"].setdefault(a.tag, [0.0, 0.0])
                st[0] = st[0] + ye_[j_]; st[1] = st[1] + ge_[j_]
        torch._foreach_copy_([p.grad for _, p, _, _ in ents_], [Y[j_].view(p.shape) for j_, (_, p, _, _) in enumerate(ents_)])

    @torch.no_grad()
    def _step_batched(self, key_, ents_):
        """the held Adam steps u T (+ the comparator's give-back) of one shape at once, written; ents_: [(a, p, u2, lr)]"""
        _, Y = self._Tb(key_, [(a, u2) for a, _, u2, _ in ents_])
        for j_, (a, p, u2, lr) in enumerate(ents_):
            uh = Y[j_]
            cm = self.cmpA.get(a.name)
            if cm is not None:
                uh += (u2 @ cm[1]) @ cm[0].t()
            if self._logstep:
                # (logging only) the applied step's energy, and its part inside the held span U (old inputs see it)
                st = self.stat["step"].setdefault(a.tag, [0.0, 0.0, 0.0])
                U32 = a.W[:, :a.U.shape[1]]
                st[0] = st[0] + (uh @ U32).pow(2).sum(); st[1] = st[1] + uh.pow(2).sum()
                st[2] = st[2] + u2.pow(2).sum()
            p.add_(uh.reshape(p.shape), alpha=-lr)

    def step_audit(self, opt):
        """Logging only: which rule acts on each trainable parameter's post-Adam step, from the same maps step()
        reads: operator (T), fresh-head pattern hold (Q), region row hold (ROW), constant share (B: c or OWM
        f/(f+mu)), gain share (D), or none."""
        import re as _re
        nm = getattr(self, "pnames", {})
        cat = {}
        for grp in opt.param_groups:
            for p in grp["params"]:
                if not p.requires_grad:
                    continue
                k = self.pmap.get(id(p))
                if k is None:
                    c = "NONE"
                elif id(p) in self.fresh:
                    c = ("Q (fresh)" if k[0] == "w" else "NONE (fresh bias)") if not (self.fresh_aug and self.faug is not None) \
                        else "Qaug (fresh, joint [w; b])"
                elif self.oldrow and k[1] is self.ro:
                    c = "ROW"
                elif k[0] == "w":
                    c = "T" if k[1].W is not None else "NONE (no memory)"
                elif k[0] == "b":
                    c = ("B=%s" % ("owm" if k[1].av is not None else "c")) if k[1].W is not None else "NONE (no memory)"
                else:
                    c = "D" if (k[1] in self.D and k[1] not in self.bn_skip) else "NONE (gain, no D)"
                cat.setdefault(c, []).append(nm.get(id(p), "?"))
        log.info("[STEP-RULE] task %s: post-Adam step rule per trainable parameter: %s" % (
            getattr(self, "_tcur", "?"), " | ".join("%s %d" % (k, len(v)) for k, v in sorted(cat.items()))))
        for k, v in sorted(cat.items()):
            log.info("[STEP-RULE] %s: %s" % (k, ", ".join(sorted({_re.sub(r"\.(\d+)\.", ".*.", n) for n in v}))))

    @torch.no_grad()
    def _pdet_scores(self, h, y, update):
        """(pdet) centred cosine of each row to every earlier record and to its own present class's running mean;
        centre = mean of the earlier records' means and the present classes' means (class-level statistics)"""
        if self.pcm is None:
            self.pcm = [torch.zeros(self.K, device=h.device), torch.zeros(self.K, h.shape[1], device=h.device)]
        if update:
            self.pcm[0].index_add_(0, y, torch.ones_like(y, dtype=torch.float32))
            self.pcm[1].index_add_(0, y, h)
        cnt, sm = self.pcm
        hasf = (cnt > 0).float()
        Pmean = sm / cnt.clamp(min=1.0).unsqueeze(1)
        if getattr(self, "_pdH", None) is None or self._pdH[0] is not self.sc:
            ks = self.sc["keys"]
            self._pdH = (self.sc, self.recs[1][ks] / self.recs[0][ks].unsqueeze(1))   # once per task
        H = self._pdH[1]
        # masked sums instead of boolean indexing: no host sync per step
        c = ((H.sum(0) + (Pmean * hasf.unsqueeze(1)).sum(0)) / (H.shape[0] + hasf.sum())).unsqueeze(0)
        M = F.normalize(H - c, dim=1)
        Pall = F.normalize(Pmean - c, dim=1) * hasf.unsqueeze(1)
        hs = F.normalize(h - c, dim=1)
        Sp = hs @ M.t()
        own = (hs * Pall[y]).sum(1)
        own = torch.where(cnt[y] > 0, own, torch.full_like(own, 2.0))
        return Sp, own

    @torch.no_grad()
    def _sink_w(self, a, x):
        """(sink) per image w_0 = 1 + mean_h sum_i a_i0^2 for an attention-input area (q / k / v), else None"""
        if not self.sink or a.kind != "tok" or len(a.mods) < 3 or a.tag != "qkv":
            return None
        mq, mk = a.mods[0], a.mods[1]
        H = self.nheads
        B, L, d = x.shape
        with torch.autocast("cuda", enabled=False):
            xf = x.float()
            q = F.linear(xf, mq.weight.float(), mq.bias.float() if mq.bias is not None else None)
            k = F.linear(xf, mk.weight.float(), mk.bias.float() if mk.bias is not None else None)
            dh = q.shape[-1] // H
            q = q.view(B, L, H, dh).transpose(1, 2)
            k = k.view(B, L, H, dh).transpose(1, 2)
            a0 = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(dh), dim=-1)[..., 0]     # (B, H, L)
            w = 1.0 + a0.pow(2).sum(-1).mean(1)
        e = self.sinkw.get(a.name)
        self.sinkw[a.name] = [float(w.sum()), float(B)] if e is None else [e[0] + float(w.sum()), e[1] + float(B)]
        return w

    @torch.no_grad()
    def _oldrows(self, items):
        """items: [(head, gw, gb)] of every old head. In place: each row's [w; b] held off its class region; with
        ro_common the step's component common to all old rows (their mean) instead passes the augmented exact readout
        operator and only the relative component keeps the region hold."""
        if not items:
            return
        if not (self.ro_common and self.Taug is not None):
            self._rowhold_all(items)
            return
        Gs = [torch.cat([gw, (gb if gb is not None else gw.new_zeros(gw.shape[0])).unsqueeze(1)], 1) for _, gw, gb in items]
        G = torch.cat(Gs, 0)
        gbar = G.mean(0, keepdim=True)
        gc = gbar @ self.Taug                                    # the common move, held by the old images' readout energy
        o = 0
        for hd, gw, gb in items:
            n = gw.shape[0]
            R = G[o:o + n] - gbar
            Rw, Rb = R[:, :-1].contiguous(), R[:, -1].contiguous()
            self._rowhold(hd, Rw, Rb)
            gw.copy_(Rw + gc[:, :-1])
            if gb is not None:
                gb.copy_(Rb + gc[0, -1])
            o += n

    @torch.no_grad()
    def _rowhold_all(self, items):
        """Every old head at once: one batched projection off the class regions (old rows are the first class
        slots, so their bases are one contiguous slice)"""
        if not items:
            return
        if not (self.oldrow_span and getattr(self, "rbas", None) is not None):
            for hd, gw, gb in items:
                self._rowhold(hd, gw, gb)
            return
        Gs = [torch.cat([gw, (gb if gb is not None else gw.new_zeros(gw.shape[0])).unsqueeze(1)], 1) for _, gw, gb in items]
        G = torch.cat(Gs, 0)
        offs = [self.hoff[id(hd.weight)][1] for hd, _, _ in items]          # python ints: no sync
        o0, cont, oc = offs[0], True, offs[0]
        for (_, gw, _), of in zip(items, offs):
            cont, oc = cont and of == oc, of + gw.shape[0]
        if cont:
            B = self.rbas[o0:o0 + G.shape[0]]
        else:
            ix = torch.cat([torch.arange(of, of + gw.shape[0]) for (_, gw, _), of in zip(items, offs)]).to(G.device)
            B = self.rbas.index_select(0, ix)
        G = G - torch.einsum("nr,ndr->nd", torch.einsum("nd,ndr->nr", G, B), B)
        o = 0
        for _, gw, gb in items:
            n = gw.shape[0]
            gw.copy_(G[o:o + n, :-1])
            if gb is not None:
                gb.copy_(G[o:o + n, -1])
            o += n

    @torch.no_grad()
    def _rowhold(self, m, gw, gb):
        """in place: [gw_r, gb_r] of an old head's row r projected off its own class record [mean_k; 1] (unit)"""
        hd, o = self.hoff[id(m.weight)]
        G = torch.cat([gw, gb.unsqueeze(1) if gb is not None else gw.new_zeros(gw.shape[0], 1)], 1)
        if self.oldrow_span and getattr(self, "rbas", None) is not None:
            B = self.rbas[o:o + gw.shape[0]]                         # (rows, d + 1, r): the row's class region
            G = G - torch.einsum("nr,ndr->nd", torch.einsum("nd,ndr->nr", G, B), B)
        else:
            A = self.arec[o:o + gw.shape[0]]
            G = G - (G * A).sum(1, keepdim=True) * A
        gw.copy_(G[:, :-1])
        if gb is not None:
            gb.copy_(G[:, -1])

    # ------------------------------------------------------------------ wake: the step (AdamW with the held update)
    @torch.no_grad()
    def step(self, opt, write):
        """AdamW (torch's arithmetic, multi-tensor) with the update held: weights u T + (u M) Q~^T, constants u c,
        gains u D. An unwritten step moves only the moments."""
        b1, b2 = opt.param_groups[0]["betas"]
        eps, wd = opt.param_groups[0]["eps"], opt.param_groups[0]["weight_decay"]
        if getattr(self, "_auditlog", None) != getattr(self, "_tcur", None):
            self._auditlog = getattr(self, "_tcur", None)
            self.step_audit(opt)
        buckets = {}
        for grp in opt.param_groups:
            for p in grp["params"]:
                if p.grad is None:
                    continue
                st = opt.state[p]
                if len(st) == 0:
                    st["step"] = torch.tensor(0.0)
                    st["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    st["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                st["step"] += 1
                buckets.setdefault(float(st["step"]), []).append((p, grp["lr"]))
        rowd = {}
        bjd = {}
        spend_ = {}
        if self.biasjoint and not hasattr(self, "_bjb"):
            self._bjb = {id(m_.bias) for a_ in self.areas.values() for m_ in a_.mods if getattr(m_, "bias", None) is not None}
        for t, bp in buckets.items():
            ps = [e[0] for e in bp]
            gs = [p.grad for p in ps]
            ms = [opt.state[p]["exp_avg"] for p in ps]
            vs = [opt.state[p]["exp_avg_sq"] for p in ps]
            torch._foreach_lerp_(ms, gs, 1 - b1)
            torch._foreach_mul_(vs, b2)
            torch._foreach_addcmul_(vs, gs, gs, value=1 - b2)
            if not write:
                continue
            if not self.wdfold:
                torch._foreach_mul_(ps, [1 - lr * wd for _, lr in bp])
            den = torch._foreach_sqrt(vs)
            torch._foreach_div_(den, math.sqrt(1 - b2 ** t))
            torch._foreach_add_(den, eps)
            us = torch._foreach_div(ms, den)
            torch._foreach_mul_(us, 1.0 / (1 - b1 ** t))
            for (p, lr), u in zip(bp, us):
                k = self.pmap.get(id(p))
                if self.wdfold and wd:
                    u.add_(p, alpha=wd)                  # (wdfold) the decay passes the same hold as the step
                if self.biasjoint and k is not None and k[1] is not self.ro and getattr(k[1], "daug", None) is not None \
                        and (k[0] == "w" or id(p) in self._bjb):
                    bjd[id(p)] = (p, lr, u)              # (biasjoint: weight and bias held together below)
                    continue
                if (self.fresh_aug or self.fresh_exact) and self.faug is not None and id(p) in self.fresh:
                    rowd[id(p)] = (p, lr, u)             # (fresh head: weight and bias held together below)
                    continue
                if self.oldrow and k is not None and k[1] is self.ro and id(p) not in self.fresh:
                    rowd[id(p)] = (p, lr, u)             # (old heads: weight and bias held together below)
                    continue
                if k is not None:
                    if id(p) in self.fresh:
                        if k[0] == "w":
                            B_ = k[1].W if self.fresh_span else k[1].Q
                            u = u - (u @ B_) @ B_.t()
                    elif k[0] == "w" and k[1].W is not None and self._hform(k[1]) is not None:
                        u2 = u.reshape(u.shape[0], -1)
                        spend_.setdefault((u2.shape[0], u2.shape[1], self._hform(k[1])), []).append((k[1], p, u2, lr))
                        continue                         # written below, batched per shape
                    elif k[0] == "w" and k[1].W is not None:
                        a = k[1]
                        u2 = u.reshape(u.shape[0], -1)
                        uh = self._T(a, u2)
                        cm = self.cmpA.get(a.name)
                        if cm is not None:
                            uh += (u2 @ cm[1]) @ cm[0].t()
                        if self._logstep:
                            # (logging only) the applied step's energy, and its part inside the held span U (old inputs see it)
                            st = self.stat["step"].setdefault(a.tag, [0.0, 0.0, 0.0])
                            U32 = a.W[:, :a.U.shape[1]]
                            st[0] = st[0] + (uh @ U32).pow(2).sum(); st[1] = st[1] + uh.pow(2).sum()
                            st[2] = st[2] + u2.pow(2).sum()
                        u = uh.reshape(p.shape)
                    elif k[0] == "b" and id(p) in self.tabconst and self.tab_share is not None:
                        u = u * self.tab_share                   # (tabln) stream units
                    elif k[0] == "b" and k[1].W is not None:
                        u = u * (0.0 if _DIAG.get("consthold") else (k[1].sb if k[1].av is not None else k[1].c))
                    elif k[0] == "g" and k[1] in self.D:
                        u = u * self.D[k[1]]
                p.add_(u, alpha=-lr)
            del us, den
        for key_, ents_ in spend_.items():
            self._step_batched(key_, ents_)
        if bjd:
            for a_ in self.areas.values():
                if getattr(a_, "daug", None) is None:
                    continue
                for m_ in a_.mods:
                    ew = bjd.get(id(m_.weight))
                    if ew is None:
                        continue
                    eb = bjd.get(id(m_.bias)) if getattr(m_, "bias", None) is not None else None
                    p_, lr_, u_ = ew
                    u2 = u_.reshape(u_.shape[0], -1)
                    if eb is not None:
                        G_ = torch.cat([u2, eb[2].reshape(-1, 1)], 1) @ a_.daug
                        uw, ub = G_[:, :-1], G_[:, -1]
                    else:
                        uw, ub = u2 @ a_.dense, None
                    cm = self.cmpA.get(a_.name)
                    if cm is not None:
                        uw = uw + (u2 @ cm[1]) @ cm[0].t()
                    p_.add_(uw.reshape(p_.shape), alpha=-lr_)
                    if eb is not None:
                        eb[0].add_(ub.reshape(eb[0].shape), alpha=-eb[1])
        olds_ = []
        for hd in self.heads:
            e = rowd.get(id(hd.weight))
            if e is None:
                continue
            eb = rowd.get(id(hd.bias)) if hd.bias is not None else None
            if id(hd.weight) not in self.fresh and not (self.ro_common and self.Taug is not None):
                olds_.append((hd, e, eb))
                continue
            if id(hd.weight) in self.fresh:
                G_ = torch.cat([e[2], (eb[2] if eb is not None else e[2].new_zeros(e[2].shape[0])).unsqueeze(1)], 1)
                if self.fresh_exact and self.Tfresh is not None:
                    G_ = G_ @ self.Tfresh
                else:
                    G_ = G_ - (G_ @ self.faug) @ self.faug.t()
                e[2].copy_(G_[:, :-1])
                if eb is not None:
                    eb[2].copy_(G_[:, -1])
                e[0].add_(e[2], alpha=-e[1])
                if eb is not None:
                    eb[0].add_(eb[2], alpha=-eb[1])
                continue
            self._rowhold(hd, e[2], eb[2] if eb is not None else None)
            e[0].add_(e[2], alpha=-e[1])
            if eb is not None:
                eb[0].add_(eb[2], alpha=-eb[1])
        if olds_:
            self._rowhold_all([(hd, e[2], eb[2] if eb is not None else None) for hd, e, eb in olds_])
            for hd, e, eb in olds_:
                e[0].add_(e[2], alpha=-e[1])
                if eb is not None:
                    eb[0].add_(eb[2], alpha=-eb[1])
        self.cmpA, self.dWA, self.armed = {}, {}, None

    def log_wake(self, prefix=""):
        if self.sinkw:
            import re as _re
            ws = sorted(((int(_re.search(r"layer\.(\d+)\.", k).group(1)), v[0] / max(v[1], 1.0)) for k, v in self.sinkw.items()
                         if _re.search(r"layer\.(\d+)\.", k)))
            log.info("[HIPPO-VIS] %sdecision-row sink weight w_0 = 1 + sum_i a_i0^2 per layer: %s" % (
                prefix, " ".join("L%d %.2f" % (l + 1, w) for l, w in ws)))
            self.sinkw = {}
        if self.cmpS:
            t_ = torch.stack(self.cmpS).sum(0)
            log.info("[HIPPO-VIS] %scomparator give-back share s_q (objective) mean %.4f" % (prefix, float(t_[0] / t_[1].clamp(min=1.0))))
            self.cmpS = []
        s = self.stat
        if not s["plast"]:
            return
        pas = " ".join("%s %.3f" % (k, float(v[0]) / max(float(v[1]), 1e-30)) for k, v in s["pass"].items())
        log.info("[HIPPO-VIS] %sconflict %.3f armed %.3f plast %.3f | passed gradient energy by kind: %s" % (
            prefix, float(sum(s["conf"])) / len(s["conf"]), sum(s["armed"]) / len(s["armed"]),
            sum(s["plast"]) / len(s["plast"]), pas))
        if s["step"]:
            log.info("[HIPPO-VIS] %sapplied step energy / Adam's (inside U share): %s" % (prefix, " ".join(
                "%s %.4f (%.4f)" % (k, float(v[1]) / max(float(v[2]), 1e-30), float(v[0]) / max(float(v[1]), 1e-30))
                for k, v in s["step"].items())))
        self._zstat()

    # ------------------------------------------------------------------ sleep
    @torch.no_grad()
    def sleep(self, t, batches, feat_fn, logit_fn=None):
        """batches(): iterator of (x, y) over the present task's training images (no augmentation);
        feat_fn(x): the decision features (backbone forward, eval mode, hooks reading)."""
        self.active = False
        self._t0 = time.time()
        self.mode = "rec"
        self.rec, self.grec = {}, {}
        Cro, nro, swr = None, 0.0, 0.0
        cm_ = {}                                         # (oldrow_span) class -> [sum [h;1][h;1]^T, n]
        for x, y in batches():
            self._ylab = y
            if self.dw:
                # decision weights: the present task's own CE on its own training images, backpropagated
                with torch.enable_grad():
                    x = x.detach().requires_grad_(True)
                    hg = feat_fn(x).float()
                    zg = logit_fn(hg)
                    w = self._dz2(zg.detach(), y)
                    F.cross_entropy(zg, y, reduction="sum").backward()
                h = hg.detach()
            else:
                h = feat_fn(x).float()
                w = torch.ones(h.shape[0], device=h.device)
            hd = h.double()
            C = (hd * w.double().unsqueeze(1)).t() @ hd
            Cro = C if Cro is None else Cro + C
            hsum_ = h.double().sum(0)
            self._hsum = hsum_ if nro == 0.0 else self._hsum + hsum_
            nro += float(h.shape[0])
            swr += float(w.sum())
            self._pat_add("readout", h, y)
            if self.oldrow_span:
                ha = torch.cat([h.double(), h.new_ones(h.shape[0], 1).double()], 1)
                for k in torch.unique(y).tolist():
                    Hk = ha[y == k]
                    e = cm_.get(k)
                    C = Hk.t() @ Hk
                    cm_[k] = [C, float(Hk.shape[0])] if e is None else [e[0] + C, e[1] + float(Hk.shape[0])]
            if self.recs is None:
                self.recs = [torch.zeros(self.K, device=h.device), torch.zeros(self.K, h.shape[1], device=h.device)]
            self.recs[0].index_add_(0, y, torch.ones_like(y, dtype=torch.float32))
            self.recs[1].index_add_(0, y, h)
        self.mode = None
        for k, (C, n) in cm_.items():
            # the class's REGION: Gavish-Donoho top directions of its augmented second moment (class-level statistic)
            ev, V = torch.linalg.eigh(C / n)
            r, _ = gd_rank(ev.clamp(min=0), n)
            self.cbas[k] = V[:, V.shape[1] - max(r, 1):].float().contiguous()
        if cm_:
            log.info("[HIPPO-VIS] class regions after task %d: rank mean %.1f (min %d max %d) of %d" % (
                t, sum(self.cbas[k].shape[1] for k in cm_) / len(cm_), min(self.cbas[k].shape[1] for k in cm_),
                max(self.cbas[k].shape[1] for k in cm_), next(iter(cm_.values()))[0].shape[0]))
        torch.cuda.synchronize()
        self._tm = [time.time()]
        self.rec["readout"] = [Cro, nro, swr]
        if (self.ro_common or self.fresh_exact) and Cro is not None:
            d_ = Cro.shape[0]
            A_ = torch.zeros(d_ + 1, d_ + 1, dtype=torch.float64, device=Cro.device)
            A_[:d_, :d_] = Cro / nro
            sh_ = self.recs[1].double().sum(0) if self.recs is not None else None
            hs_ = self._hsum.double() / nro
            A_[:d_, d_] = hs_; A_[d_, :d_] = hs_; A_[d_, d_] = 1.0
            bx_ = nro / (self.caug_n + nro)
            self.Caug = A_ if self.Caug is None else (1.0 - bx_) * self.Caug + bx_ * A_
            self.caug_n += nro
        # keys per area
        self.kc = {}
        for name, ent in self.rec.items():
            C, n = ent[0], ent[1]
            sw = ent[2] if len(ent) > 2 else n                 # (decision-weighted: the weighted mean moment)
            # (float64: the low end of the spectrum -- the tail -- is below fp32 eigh accuracy for smooth inputs)
            Ce = C.double() / max(sw, 1e-300)
            Ce = 0.5 * (Ce + Ce.t())
            ev, V = torch.linalg.eigh(Ce)
            ev = ev.clamp(min=0)
            k, tau = gd_rank(ev, n)
            self.kc[name] = {"K": V[:, ev.shape[0] - k:].float().contiguous(), "Kd": V[:, ev.shape[0] - k:].contiguous(),
                             "lk": ev[ev.shape[0] - k:].clone(),
                             "floor": tau * tau, "tr": float(ev.sum()), "Ce": Ce, "n": n}
            if name in self.rsum:
                self.kc[name]["m"] = self.rsum[name][0] / max(sw, 1e-300)
                self.kc[name]["w"] = self.rsum[name][1] / max(sw, 1e-300)
        self.rec = {}
        self.rsum = {}
        torch.cuda.synchronize()
        self._tm.append(time.time())
        # second read: conflict share rho per key (only when earlier records exist)
        rho, conf_frac = {}, None
        if self.sc is not None and t > 0:
            self.mode = "g2s"
            E = {}
            cf = []
            for x, y in batches():
                self.g2P = {}
                if self.dw:
                    with torch.enable_grad():
                        x = x.detach().requires_grad_(True)
                        hg = feat_fn(x).float()
                        zg = logit_fn(hg)
                        wr = self._dz2(zg.detach(), y)
                        F.cross_entropy(zg, y, reduction="sum").backward()
                    h = hg.detach()
                else:
                    h = feat_fn(x).float()
                    wr = torch.ones(h.shape[0], device=h.device)
                if self.pdet and self.pcm is not None:
                    Sp_, own_ = self._pdet_scores(h.float(), y, update=False)
                    conf = (Sp_ > own_.unsqueeze(1)).any(1).float()
                else:
                    hs = F.normalize(h - self.sc["c"], dim=1)
                    conf = ((hs @ self.sc["M"].t()) > self.sc["r"].unsqueeze(0)).any(1).float()
                cf.append(float(conf.mean()))
                self.g2P["readout"] = ((h @ self.kc["readout"]["K"]).pow(2) * wr.unsqueeze(1), None)
                for name, (P, sid) in self.g2P.items():
                    w = conf if sid is None else conf[sid]
                    e = E.get(name)
                    s0, s1 = P.sum(0), (P * w.unsqueeze(1)).sum(0)
                    E[name] = [s0, s1] if e is None else [e[0] + s0, e[1] + s1]
            self.mode = None
            self.g2P = {}
            rho = {k: (v[1] / v[0].clamp(min=1e-30)).clamp(0, 1) for k, v in E.items()}
            conf_frac = sum(cf) / max(len(cf), 1)
        torch.cuda.synchronize()
        self._tm.append(time.time())
        # write
        rk = {}
        for a in self.all_areas:
            kc = self.kc.pop(a.name, None)
            if kc is None:
                continue
            self._write(a, kc, rho.get(a.name))
            rk.setdefault(a.tag, []).append((a.U.shape[1], a.d, a.tail / max(a.tr, 1e-30)))
        self.kc = {}
        # gains
        for g, (S, n) in self.grec.items():
            e = self.gold.get(g)
            self.gold[g] = [S, n] if e is None else [e[0] + S, e[1] + n]
            ev = self.gold[g][0] / max(self.gold[g][1], 1.0)
            srt = torch.sort(ev[ev > 0]).values
            kp = gd_rank(srt, self.gold[g][1])[0] if srt.numel() else 0
            own = torch.zeros_like(ev, dtype=torch.bool)
            if kp > 0:
                own[torch.topk(ev, kp).indices] = True
            self.gown[g] = own
            # (minimum-interference step) the per-coordinate noise floor: the GD cut of the coordinate energies (the
            # smallest owned one); mu: all earlier samples over the episode just written
            fl_ = float(srt[-kp]) if kp > 0 else (float(srt[-1]) if srt.numel() else 1.0)
            self.gfm[g] = (max(fl_, 1e-30), float(self.gold[g][1]) / max(float(n), 1.0))
        self.grec = {}
        torch.cuda.synchronize()
        self._tm.append(time.time())
        log.info("[HIPPO-VIS] sleep timing: read %.0fs keys %.0fs second read %.0fs write %.0fs" % (
            self._tm[0] - self._t0, self._tm[1] - self._tm[0], self._tm[2] - self._tm[1], self._tm[3] - self._tm[2]))
        log.info("[HIPPO-VIS] sleep after task %d: conflict at sleep read %s | rho mean %s | held rank/dim by kind %s | "
                 "gains owned %.3f" % (
                     t, "%.3f" % conf_frac if conf_frac is not None else "-",
                     "%.3f" % (sum(float(r.mean()) for r in rho.values()) / len(rho)) if rho else "-",
                     " ".join("%s %.0f/%.0f (tail %.2f)" % (k, sum(r for r, _, _ in v) / len(v), sum(d for _, d, _ in v) / len(v),
                                                          sum(f for _, _, f in v) / len(v))
                              for k, v in rk.items()),
                     float(torch.cat([o.float() for o in self.gown.values()]).mean()) if self.gown else -1))

    @torch.no_grad()
    def _write(self, a, kc, rho):
        K, lk, floor, tr_e, Ce, n = kc["Kd"], kc["lk"], kc["floor"], kc["tr"], kc["Ce"], kc["n"]
        if self.owmx:
            # (owmx) the objective's C: the per-sample mean moment of every earlier input (running mean over episodes)
            Cf = getattr(a, "Cfull", None)
            bx = n / ((a.n if a.U is not None else 0.0) + n)
            a.Cfull = Ce.float().clone() if Cf is None else ((1.0 - bx) * Cf + bx * Ce.float())
            if self.biasjoint and "m" in kc:
                d_ = Ce.shape[0]
                A_ = torch.empty(d_ + 1, d_ + 1, dtype=torch.float32, device=Ce.device)
                A_[:d_, :d_] = Ce.float(); A_[:d_, d_] = kc["m"].float(); A_[d_, :d_] = kc["m"].float(); A_[d_, d_] = kc["w"]
                Ca = getattr(a, "Caug", None)
                a.Caug = A_ if Ca is None else ((1.0 - bx) * Ca + bx * A_)
        dt = torch.float64                                   # the write in float64
        K, lk = K.to(dt), lk.to(dt)
        if a.U is None:
            U, lam, tr, nn_, beta = K, lk, tr_e, n, 1.0
        else:
            beta = n / (a.n + n)
            Uo, lo = a.U.to(dt), a.lam.to(dt)
            B = torch.linalg.qr(torch.cat([Uo, K], 1)).Q
            Ab = B.t() @ Uo
            S = (Ab * lo) @ Ab.t()
            Kb = B.t() @ K
            er = beta * (1.0 - rho.to(dt)) if rho is not None else beta * torch.ones(Kb.shape[1], device=K.device, dtype=dt)
            Kw = Kb * (1.0 - (1.0 - er).clamp(min=0.0).sqrt()).unsqueeze(0)
            G = torch.eye(Kb.shape[0], device=K.device, dtype=dt) - Kw @ Kb.t()
            S = G @ S @ G.t() + beta * ((Kb * lk) @ Kb.t())
            S = 0.5 * (S + S.t())
            lw, Wv = torch.linalg.eigh(S)
            keep = lw > floor
            U, lam = (B @ Wv[:, keep]).contiguous(), lw[keep]
            tr, nn_ = (1.0 - beta) * a.tr + beta * tr_e, a.n + n
        # (float64: the tail is a small difference of two large energies)
        Ud = U.double()
        tail_e = max(float(Ce.diagonal().sum()) - float(((Ce @ Ud) * Ud).sum()), 0.0)
        a.tail = tail_e if a.tail is None else (1.0 - beta) * a.tail + beta * tail_e
        a.U, a.lam, a.tr, a.n = U.contiguous(), lam, tr, nn_          # float64 master (the tail is tiny)
        a.floor, a.nlast = float(floor), float(n)
        a.W = None
