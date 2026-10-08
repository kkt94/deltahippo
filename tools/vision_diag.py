"""Diagnostics (evaluation only) for tools/vision_hippo.py: margin decomposition, per-block drift, acquisition
measures and downstream gain.

Nothing here reaches training: every pass runs under no_grad except the acquisition gradient pass, which only fills
.grad (zeroed afterwards, the optimiser state is read, never written) on the present task's training images in eval
mode, after the task's last step and before the sleep. No RNG is drawn.

1. MARGIN DECOMPOSITION (per earlier task e, between consecutive evaluations t-1 -> t), per test image:
     margin m = z_gold - max_{other seen class} z.
     m(t) - m(t-1) = [m(heads_t restricted to the old classes, h_{t-1}) - m(t-1)]          readout, old heads
                   + [m(heads_t, h_{t-1}) - m(heads_t restricted to the old classes, h_{t-1})]   readout, new head entry
                   + [m(heads_t, h_t) - m(heads_t, h_{t-1})]                                 trunk
   and the accuracies prev / mid (= the current heads on the previous states: readout change only) / now.
2. PER-BLOCK DRIFT of the decision position (ViT [CLS] row after each block; ResNet global pool after each block):
   per image |dh|/|h|; class-mean differential drift and common-mode drift relative to the between-class spread
   (rms distance of the class means from their mean) of that block.
3. ACQUISITION (after training task t, before its sleep): eval-mode CE/acc of the present task's training images
   (all seen heads, and within its own head), the full gradient G over those images, per kind:
     gshare  = |held(G)|^2 / |G|^2
     fo_seq  = sum lr <G, P(G)>          (first-order loss decrease of an unheld Adam step with the present v)
     fo_held = sum lr <G, held(P(held(G)))>   (the same for the held step: held gradient into Adam, held update)
     keep    = fo_held / fo_seq           (share of the needed first-order decrease the hold lets through)
   P(x) = x / (sqrt(v / (1 - b2^k)) + eps) (Adam's preconditioner, momentum ignored). For SEQ held = identity, so
   fo_seq is the reference size of a step.
4. DOWNSTREAM GAIN (opt-in, env VIS_GAIN=1): see downstream_gain().
"""
import logging

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger("hippo_vis")


def _margin(Z, y):
    g = Z.gather(1, y.unsqueeze(1)).squeeze(1)
    o = Z.scatter(1, y.unsqueeze(1), float("-inf")).max(1).values
    return g - o


def _blocks(net):
    """(name, module, extractor) per block, extractor(out) -> (B, d) decision-position state"""
    out = []
    if net.kind == "vit":
        cls = lambda o: (o[0] if isinstance(o, tuple) else o)[:, 0]
        out.append(("emb", net.bb.embeddings, cls))
        for i, l in enumerate(net.bb.encoder.layer):
            out.append(("L%d" % (i + 1), l, cls))
        out.append(("lnF", net.bb.layernorm, cls))
    else:
        pool = lambda o: o.mean((2, 3))
        out.append(("stem", net.bb.maxpool, pool))
        for li in range(1, 5):
            for bi, b in enumerate(getattr(net.bb, "layer%d" % li)):
                out.append(("%d.%d" % (li, bi), b, pool))
    return out


class EvalDiag:
    def __init__(self, net, data):
        self.net, self.data = net, data
        self.blocks = _blocks(net)
        self.prev = {}          # e -> dict(H, Z, Bk, y) at the previous evaluation
        self.learnt = {}        # e -> block states at the evaluation right after e was learnt
        self.cum = {}           # e -> cumulative margin parts since learnt [ro_old, ro_new, trunk]

    # ------------------------------------------------------------------ evaluation snapshot
    @torch.no_grad()
    def collect(self, e):
        """test images of task e: final feature H, logits Z, per-block states, labels"""
        net = self.net
        cap, hs = {}, []
        for i, (_, m, ex) in enumerate(self.blocks):
            def f(mod, inp, out, _i=i, _ex=ex):
                cap.setdefault(_i, []).append(_ex(out).float().cpu())
            hs.append(m.register_forward_hook(f))
        H, Z, Y = [], [], []
        try:
            for x, y in self.data.plain_batches(e, 256, test=True):
                h = net.features(x).float()
                H.append(h.cpu()); Z.append(net.logits(h).cpu()); Y.append(y.cpu())
        finally:
            for h_ in hs:
                h_.remove()
        return {"H": torch.cat(H), "Z": torch.cat(Z), "y": torch.cat(Y),
                "B": [torch.cat(cap[i]) for i in range(len(self.blocks))]}

    @torch.no_grad()
    def heads_on(self, H):
        dev = next(self.net.parameters()).device
        out = []
        for i in range(0, H.shape[0], 1024):
            out.append(self.net.logits(H[i:i + 1024].to(dev)).cpu())
        return torch.cat(out)

    @torch.no_grad()
    def after_eval(self, t):
        """called after evaluate() at task t (net.n = t + 1)"""
        per = self.data.per
        tot = {"n": 0, "dm": 0.0, "ro_old": 0.0, "ro_new": 0.0, "tr": 0.0, "acc_p": 0.0, "acc_m": 0.0, "acc_n": 0.0,
               "lost": 0, "lost_tr": 0, "lost_ro": 0, "gained": 0}
        blk_step, blk_cum = [], []
        for e in range(t + 1):
            cur = self.collect(e)
            if e == t:
                self.learnt[e] = cur["B"]
                self.cum[e] = [0.0, 0.0, 0.0]
                self.prev[e] = cur
                continue
            pv = self.prev[e]
            y = cur["y"].long()
            C0 = t * per                                       # classes seen at the previous evaluation
            Zm = self.heads_on(pv["H"])                         # current heads on the previous states
            m_p = _margin(pv["Z"][:, :C0], y)
            m_mo = _margin(Zm[:, :C0], y)
            m_m = _margin(Zm, y)
            m_n = _margin(cur["Z"], y)
            ro_old, ro_new, tr = (m_mo - m_p), (m_m - m_mo), (m_n - m_m)
            ok_p, ok_m, ok_n = m_p > 0, m_m > 0, m_n > 0
            lost = ok_p & ~ok_n
            g_tr = (cur["Z"].gather(1, y[:, None]) - Zm.gather(1, y[:, None])).squeeze(1)
            cm = self.cum[e]
            cm[0] += float(ro_old.mean()); cm[1] += float(ro_new.mean()); cm[2] += float(tr.mean())
            dh = (cur["H"] - pv["H"]).norm(dim=1) / pv["H"].norm(dim=1).clamp(min=1e-12)
            try:
                # the newest head's top logit on these (current) states, split at its argmax column into bias,
                # class-mean part (its row on the image's class-mean state) and within-class part (row on h - mean)
                hd = self.net.heads[t]
                Wn = hd.weight.detach().float().cpu()
                bn = hd.bias.detach().float().cpu() if getattr(hd, "bias", None) is not None else torch.zeros(Wn.shape[0])
                Hc = cur["H"]
                Mc = torch.zeros_like(Hc)
                for k in torch.unique(y):
                    Mc[y == k] = Hc[y == k].mean(0)
                if Wn.shape[1] == Hc.shape[1] and bn.numel() == Wn.shape[0] and not hasattr(hd, "s"):
                    zn = Hc @ Wn.t() + bn
                    j = zn.argmax(1)
                    pb, pm = bn[j], (Mc * Wn[j]).sum(1)
                    ps = ((Hc - Mc) * Wn[j]).sum(1)
                    zo_ = cur["Z"][:, :C0].max(1).values
                    log.info("[DIAG-N] T%d task %d: newest head top logit %.3f = bias %+.3f + class-mean part %+.3f + "
                             "within-class part %+.3f | old heads' top logit %.3f | newest wins %.3f" % (
                                 t, e, float(zn.max(1).values.mean()), float(pb.mean()), float(pm.mean()),
                                 float(ps.mean()), float(zo_.mean()), float((zn.max(1).values > zo_).float().mean())))
            except Exception as ex_:                                   # diagnostics must never stop a run
                log.info("[DIAG-N] skipped: %s" % ex_)
            log.info("[DIAG-M] T%d task %d: acc prev %.1f mid(readout only) %.1f now %.1f | margin %+.3f -> %+.3f: "
                     "readout old-heads %+.3f, new-head entry %+.3f, trunk %+.3f (gold logit trunk %+.3f) | "
                     "lost %d (trunk-caused %d, readout-caused %d), gained %d | lost images: ro_old %+.3f ro_new %+.3f "
                     "trunk %+.3f | |dh|/|h| %.4f | since learnt: ro_old %+.3f ro_new %+.3f trunk %+.3f" % (
                         t, e, 100 * float(ok_p.float().mean()), 100 * float(ok_m.float().mean()),
                         100 * float(ok_n.float().mean()), float(m_p.mean()), float(m_n.mean()),
                         float(ro_old.mean()), float(ro_new.mean()), float(tr.mean()), float(g_tr.mean()),
                         int(lost.sum()), int((lost & ok_m).sum()), int((lost & ~ok_m).sum()),
                         int((~ok_p & ok_n).sum()),
                         float(ro_old[lost].mean()) if lost.any() else 0.0,
                         float(ro_new[lost].mean()) if lost.any() else 0.0,
                         float(tr[lost].mean()) if lost.any() else 0.0, float(dh.mean()), cm[0], cm[1], cm[2]))
            n = y.numel()
            tot["n"] += n
            for k_, v_ in (("dm", m_n - m_p), ("ro_old", ro_old), ("ro_new", ro_new), ("tr", tr)):
                tot[k_] += float(v_.sum())
            tot["acc_p"] += float(ok_p.sum()); tot["acc_m"] += float(ok_m.sum()); tot["acc_n"] += float(ok_n.sum())
            tot["lost"] += int(lost.sum()); tot["lost_tr"] += int((lost & ok_m).sum())
            tot["lost_ro"] += int((lost & ~ok_m).sum()); tot["gained"] += int((~ok_p & ok_n).sum())
            blk_step.append(self._blk(pv["B"], cur["B"], y))
            blk_cum.append(self._blk(self.learnt[e], cur["B"], y))
            self.prev[e] = cur
        if tot["n"]:
            n = tot["n"]
            log.info("[DIAG-M] T%d ALL OLD: acc prev %.2f mid %.2f now %.2f | dmargin %+.3f = ro_old %+.3f + ro_new %+.3f "
                     "+ trunk %+.3f | lost %d (trunk %d, readout %d) gained %d" % (
                         t, 100 * tot["acc_p"] / n, 100 * tot["acc_m"] / n, 100 * tot["acc_n"] / n, tot["dm"] / n,
                         tot["ro_old"] / n, tot["ro_new"] / n, tot["tr"] / n, tot["lost"], tot["lost_tr"],
                         tot["lost_ro"], tot["gained"]))
            for nm, bl in (("step", blk_step), ("since learnt", blk_cum)):
                A = np.mean(np.array(bl), 0)                    # (blocks, 3)
                log.info("[DIAG-B] T%d old tasks, drift %s per block (|dh|/|h| / differential/spread / common/spread): %s"
                         % (t, nm, " ".join("%s %.3f/%.3f/%.3f" % (self.blocks[i][0], A[i, 0], A[i, 1], A[i, 2])
                                            for i in range(A.shape[0]))))

    @staticmethod
    def _blk(B0, B1, y):
        ks = torch.unique(y)
        out = []
        for a0, a1 in zip(B0, B1):
            rel = float(((a1 - a0).norm(dim=1) / a0.norm(dim=1).clamp(min=1e-12)).mean())
            M0 = torch.stack([a0[y == k].mean(0) for k in ks])
            M1 = torch.stack([a1[y == k].mean(0) for k in ks])
            sp = float((M0 - M0.mean(0)).norm(dim=1).pow(2).mean().sqrt())
            D = M1 - M0
            cmn = D.mean(0)
            dif = float((D - cmn).norm(dim=1).pow(2).mean().sqrt())
            out.append((rel, dif / max(sp, 1e-12), float(cmn.norm()) / max(sp, 1e-12)))
        return out


# ---------------------------------------------------------------------- acquisition
def _kind_of(name, kind):
    if kind == "vit":
        for k_, v_ in (("patch", "patch"), ("query", "qkv"), ("key", "qkv"), ("value", "qkv"),
                       ("attention.output", "o"), ("intermediate", "fc1"), ("output.dense", "fc2"),
                       ("layernorm", "LN"), ("cls_token", "const"), ("position", "const")):
            if k_ in name:
                return v_
        return "other"
    for k_ in ("layer1", "layer2", "layer3", "layer4"):
        if name.startswith(k_):
            return ("L" + k_[-1]) + ("-bn" if ".bn" in name or "downsample.1" in name else "")
    return "stem-bn" if "bn" in name else "stem"


def acquisition(net, data, t, opt, hx=None, n_img=1024, tag=""):
    """after training task t, before the sleep (see the module docstring)"""
    was = net.training
    net.eval()
    net.n = t + 1
    per = data.per
    dev = next(net.parameters()).device
    rg = {id(p): p.requires_grad for p in net.parameters()}
    for p in net.parameters():
        p.requires_grad_(True)
        p.grad = None
    mode0 = hx.mode if hx is not None else None
    if hx is not None:
        hx.mode = None
    idx = data.tr_idx[t][:n_img]
    ce_all, ce_own, acc_all, acc_own, n = 0.0, 0.0, 0.0, 0.0, 0
    # the present task's tail moment per trunk area: input rows outside the held span W, for the per-direction
    # novelty measure below
    tm, hk = {}, []
    if hx is not None and hx.active:
        for a in hx.areas.values():
            if a.W is None:
                continue
            def pre(mod, inp, _a=a):
                with torch.no_grad(), torch.autocast("cuda", enabled=False):
                    X = hx._rows(_a, inp[0].detach().float())[0]
                    Xt = X - (X @ _a.W) @ _a.W.t()
                    e = tm.get(_a.name)
                    C = Xt.t() @ Xt
                    tm[_a.name] = [C, float(X.shape[0]), float(X.pow(2).sum())] if e is None else \
                        [e[0] + C, e[1] + float(X.shape[0]), e[2] + float(X.pow(2).sum())]
            hk.append(a.mods[0].register_forward_pre_hook(pre))
    try:
        for i in range(0, len(idx), 64):
            x, y = data.batch(data.xtr, data.ytr_m, idx[i:i + 64])
            h = net.features(x).float()
            z = net.logits(h)
            loss = F.cross_entropy(z, y, reduction="sum")
            (loss / len(idx)).backward()
            with torch.no_grad():
                if hk and hx.ro.W is not None:
                    hf = h.detach().float()
                    Xt = hf - (hf @ hx.ro.W) @ hx.ro.W.t()
                    e = tm.get("readout")
                    C = Xt.t() @ Xt
                    tm["readout"] = [C, float(hf.shape[0]), float(hf.pow(2).sum())] if e is None else \
                        [e[0] + C, e[1] + float(hf.shape[0]), e[2] + float(hf.pow(2).sum())]
                zo = z[:, t * per:(t + 1) * per]
                ce_all += float(loss); ce_own += float(F.cross_entropy(zo, y - t * per, reduction="sum"))
                acc_all += float((z.argmax(1) == y).sum()); acc_own += float((zo.argmax(1) == y - t * per).sum())
                n += y.numel()
    finally:
        for h_ in hk:
            h_.remove()
    pdir = {}
    if tm:
        # PER-DIRECTION NOVELTY (measure only): eigen-directions v of the present tail moment, old energy per tail
        # direction taken as the stored tail spread evenly over the tail's dims (floor = tail / (d - rank W));
        # D_v = clip(1 - floor / e_v); reported: the scalar c, the tail share of the present input energy, the share
        # of the tail energy on directions with D_v > 0 weighted by D_v, and the gradient share passing
        # (I - W W^T) V D V^T (in place of c (I - W W^T))
        with torch.no_grad():
            for a in list(hx.areas.values()) + [hx.ro]:
                e = tm.get(a.name)
                if e is None:
                    continue
                C = e[0] / e[1]
                ev, V = torch.linalg.eigh(0.5 * (C + C.t()))
                ev = ev.clamp(min=0)
                fl = float(a.tail) / max(a.d - a.W.shape[1], 1)
                D = ((1.0 - fl / ev.clamp(min=1e-30)).clamp(0, 1))
                tail_sh = float(ev.sum()) / max(e[2] / e[1], 1e-30)
                pas = float((ev * D).sum() / ev.sum().clamp(min=1e-30))
                gs = {}
                for m in a.mods:
                    if m.weight.grad is None:
                        continue
                    G = m.weight.grad.reshape(m.weight.shape[0], -1).float()
                    Gp = ((G @ V) * D) @ V.t()
                    tg = a.tag if a.kind != "ro" else ("ro-fresh" if id(m.weight) in hx.fresh else "ro-old")
                    g_ = gs.setdefault(tg, [0.0, 0.0])
                    g_[0] += float(G.pow(2).sum()); g_[1] += float(Gp.pow(2).sum())
                for tg, g_ in gs.items():
                    r = pdir.setdefault(tg, [])
                    r.append((a.c, tail_sh, pas, g_[1] / max(g_[0], 1e-30), int((D > 0.5).sum())))
        log.info("[DIAG-A] %sT%d per-direction novelty by kind (c / present tail share / D-weighted tail share / "
                 "gradient share through V D V^T / dirs with D>0.5): %s" % (
                     tag, t, " ".join("%s %.3f/%.3f/%.3f/%.4f/%.0f" % (k, *[sum(x[i] for x in v) / len(v) for i in range(5)])
                                      for k, v in pdir.items())))
    lr_of = {}
    for g in opt.param_groups:
        for p in g["params"]:
            lr_of[id(p)] = g["lr"]
    b2, eps = opt.param_groups[0]["betas"][1], opt.param_groups[0]["eps"]
    names = {id(p): nm for nm, p in net.named_parameters()}
    head_ids = {id(p): i for i, hd in enumerate(net.heads) for p in hd.parameters()}
    st = {}
    joint = {}
    cap_ = [0.0] * 6
    with torch.no_grad():
        if hx is not None and hx.active and getattr(hx, "oldrow", False):
            # old heads under the per-row hold: weight and bias held together, exactly as HippoVis does
            for i_, hd in enumerate(net.heads):
                if i_ >= t or hd.weight.grad is None:
                    continue
                Pw = _prec(opt, hd.weight, b2, eps); Pb = _prec(opt, hd.bias, b2, eps)
                gw, gb = hd.weight.grad.clone(), hd.bias.grad.clone()
                g0w_, g0b_ = gw.clone(), gb.clone()
                hx._rowhold(hd, gw, gb)
                uw, ub = Pw(gw), Pb(gb)
                hx._rowhold(hd, uw, ub)
                joint[id(hd.weight)] = (gw, uw); joint[id(hd.bias)] = (gb, ub)
                # capture: what the per-row region removes from each old row's present-task gradient, split
                # into the part along the row's own class mean [mean_k; 1] and the rest of its region (spread)
                A_ = hx.arec[hx.hoff[id(hd.weight)][1]:hx.hoff[id(hd.weight)][1] + gw.shape[0]]
                G0_ = torch.cat([g0w_, g0b_.unsqueeze(1)], 1).double()
                Gr_ = torch.cat([gw, gb.unsqueeze(1)], 1).double()
                cap_[0] += float(G0_.pow(2).sum()); cap_[1] += float((G0_ - Gr_).pow(2).sum())
                cap_[2] += float((G0_ * A_.double()).sum(1).pow(2).sum())
        if hx is not None and hx.active and getattr(hx, "faug", None) is not None:
            # capture: the fresh head's present-task gradient [w; b] against the earlier classes' [mean; 1] span:
            # removed share, and the part of it along the earlier classes' common direction (their mean [mean; 1])
            hd_ = net.heads[t]
            if hd_.weight.grad is not None and hd_.bias is not None and hd_.bias.grad is not None:
                Gf_ = torch.cat([hd_.weight.grad, hd_.bias.grad.unsqueeze(1)], 1).double()
                Fa_ = hx.faug.double()
                ks_ = hx.sc["keys"]
                Hm_ = (hx.recs[1][ks_] / hx.recs[0][ks_].unsqueeze(1)).double()
                cm_ = torch.cat([Hm_.mean(0), Hm_.new_ones(1)])
                cm_ = cm_ / cm_.norm()
                cap_[3] += float(Gf_.pow(2).sum()); cap_[4] += float((Gf_ @ Fa_).pow(2).sum())
                cap_[5] += float((Gf_ @ cm_).pow(2).sum())
        for p in net.parameters():
            G = p.grad
            if G is None or id(p) not in lr_of or G.abs().sum() == 0:
                continue
            if id(p) in head_ids:
                k = "head-new" if head_ids[id(p)] == t else "head-old"
            else:
                k = _kind_of(names[id(p)][3:], net.kind)
            s = opt.state.get(p, {})
            if "exp_avg_sq" in s:
                den = (s["exp_avg_sq"] / (1 - b2 ** float(s["step"]))).sqrt() + eps
            else:
                den = None
            P = (lambda X: X / den) if den is not None else (lambda X: X)
            Gh = _held(hx, p, G) if hx is not None and hx.active else G
            u = P(G)
            uh = _held(hx, p, P(Gh)) if hx is not None and hx.active else u
            if id(p) in joint:
                Gh, uh = joint[id(p)]
            # the subspace alone: the same hold with the tail share c = 1, the fresh head held by the readout span
            G1 = _held(hx, p, G, c1=True) if hx is not None and hx.active else G
            lr = lr_of[id(p)]
            e = st.setdefault(k, [0.0, 0.0, 0.0, 0.0, 0.0])
            e[0] += float(G.double().pow(2).sum()); e[1] += float(Gh.double().pow(2).sum())
            e[2] += lr * float((G.double() * u.double()).sum()); e[3] += lr * float((G.double() * uh.double()).sum())
            e[4] += float(G1.double().pow(2).sum())
        for p in net.parameters():
            p.grad = None
    for p in net.parameters():
        p.requires_grad_(rg[id(p)])
    if hx is not None:
        hx.mode = mode0
    net.train(was)
    if cap_[0] > 0 or cap_[3] > 0:
        log.info("[DIAG-C] %sT%d capture: old rows' present-task gradient removed by their regions %.3f of its energy "
                 "(along the row's own [mean; 1] %.3f, the region's spread part %.3f) | fresh head's gradient removed by "
                 "the earlier [mean; 1] span %.3f (along the earlier classes' common [mean; 1] alone %.3f)" % (
                     tag, t, cap_[1] / max(cap_[0], 1e-30), cap_[2] / max(cap_[0], 1e-30),
                     (cap_[1] - cap_[2]) / max(cap_[0], 1e-30), cap_[4] / max(cap_[3], 1e-30),
                     cap_[5] / max(cap_[3], 1e-30)))
    tot0 = sum(v[2] for v in st.values()); tot1 = sum(v[3] for v in st.values())
    log.info("[DIAG-A] %sT%d train fit (eval mode, %d images): CE all-heads %.4f own-head %.4f | acc all %.2f own %.2f | "
             "first-order decrease per step: unheld %.3e held %.3e (keep %.3f)" % (
                 tag, t, n, ce_all / n, ce_own / n, 100 * acc_all / n, 100 * acc_own / n, tot0, tot1,
                 tot1 / max(tot0, 1e-30)))
    log.info("[DIAG-A] %sT%d per kind: gshare / fo_unheld / fo_held / keep / gshare at c=1: %s" % (
        tag, t, " ".join("%s %.4f/%.2e/%.2e/%.3f/%.4f" % (k, v[1] / max(v[0], 1e-30), v[2], v[3],
                                                          v[3] / max(v[2], 1e-30), v[4] / max(v[0], 1e-30))
                         for k, v in sorted(st.items()))))
    return ce_all / n


def _prec(opt, p, b2, eps):
    s = opt.state.get(p, {})
    if "exp_avg_sq" not in s:
        return lambda X: X
    den = (s["exp_avg_sq"] / (1 - b2 ** float(s["step"]))).sqrt() + eps
    return lambda X: X / den


@torch.no_grad()
def _held(hx, p, X, c1=False):
    """the hold HippoVis applies (gradient and step alike), without the comparator's give-back; c1: the same span with
    the tail share c = 1 (and the fresh head held by the readout's whole held span, as the old heads)"""
    k = hx.pmap.get(id(p))
    if k is None:
        return X
    if id(p) in hx.fresh and not c1:
        if k[0] == "w":
            Q = k[1].W if getattr(hx, "fresh_span", False) else k[1].Q
            return X - (X @ Q) @ Q.t()
        return X
    if getattr(hx, "oldrow", False) and k[1] is hx.ro and not c1 and id(p) not in hx.fresh:
        # old heads under the per-row hold: weight and bias are held together; here each on its own row record part
        hd = next((h for h in hx.heads if h.weight is p or h.bias is p), None)
        if hd is not None:
            o = hx.hoff[id(hd.weight)][1]
            A = hx.arec[o:o + X.shape[0]]
            if X.dim() == 2:
                Aw = A[:, :-1]
                return X - (X * Aw).sum(1, keepdim=True) * Aw
            return X - X * A[:, -1] * A[:, -1]
    if k[0] == "w":
        a = k[1]
        if a.W is None:
            return X
        X2 = X.reshape(X.shape[0], -1).float()
        if c1:
            return (X2 - (X2 @ a.W) @ a.W.t()).reshape(X.shape)
        return hx._T(a, X2).reshape(X.shape)
    if k[0] == "b":
        a = k[1]
        if c1:
            return X
        if a.W is not None and getattr(a, "av", None) is not None:
            return X * a.sb
        return X * a.c if a.W is not None else X
    if k[0] == "g":
        D = hx.D.get(k[1])
        if D is None or k[1] in hx.bn_skip:
            return X
        return X * D
    return X


@torch.no_grad()
def downstream_gain(net, data, t, hx, n_img=128, rel=1e-2, tag=""):
    """Diagnostic (evaluation only; does not affect training): DOWNSTREAM GAIN per trunk area: how much of a
    perturbation of the area's OUTPUT reaches the decision. For each synapse group m of the area, a random
    perturbation eps (Gaussian, rms = rel x rms of the output) is added to m's output on the present task's training
    images (eval mode, no augmentation), and
    gamma_h = E|dh|^2 / E|eps|^2 (h the decision feature), gamma_z = E|dz|^2 / E|eps|^2 (z the logits of every head)
    are averaged over the area's groups. The minimum-interference objective counts |dW x|^2 at the area's output;
    the decision sees gamma times it."""
    if hx is None or not hx.areas:
        return
    was = net.training
    net.eval()
    net.n = t + 1
    mode0 = hx.mode
    hx.mode = None
    idx = data.tr_idx[t][:n_img]
    batches = [data.batch(data.xtr, data.ytr_m, idx[i:i + 64])[0] for i in range(0, len(idx), 64)]
    g = torch.Generator(device=batches[0].device).manual_seed(4321)
    base = [net.features(x).float() for x in batches]
    zb = [net.logits(h) for h in base]
    res = {}
    try:
        for a in hx.areas.values():
            gh_, gz_ = [], []
            for m in a.mods:
                acc = [0.0, 0.0, 0.0]
                for x, h0, z0 in zip(batches, base, zb):
                    st_ = {}

                    def hk(mod, inp, out, _st=st_):
                        o = out.float()
                        e = torch.randn(o.shape, generator=g, device=o.device) * (rel * o.pow(2).mean().sqrt())
                        _st["e"] = float(e.pow(2).sum())
                        return (o + e).to(out.dtype)
                    hd = m.register_forward_hook(hk)
                    try:
                        h1 = net.features(x).float()
                    finally:
                        hd.remove()
                    z1 = net.logits(h1)
                    acc[0] += float((h1 - h0).pow(2).sum()); acc[1] += float((z1 - z0).pow(2).sum()); acc[2] += st_["e"]
                gh_.append(acc[0] / max(acc[2], 1e-30)); gz_.append(acc[1] / max(acc[2], 1e-30))
            res[a.name] = (sum(gh_) / len(gh_), sum(gz_) / len(gz_))
    finally:
        hx.mode = mode0
        net.train(was)
    # ALONG THE UPDATE: the same gain for the weight change the method actually makes (the present task's held
    # gradient direction), dW = -alpha Gh with alpha small: gamma_step = E|dz|^2 / E|dW x|^2 (output change of the
    # group itself measured by a hook). Random output perturbations spread over every output direction; the update
    # is aligned with the directions the loss moves the logits along.
    gst = {}
    net.eval()
    rg_ = {id(p_): p_.requires_grad for p_ in net.parameters()}
    mode1_ = hx.mode
    hx.mode = None
    try:
        net.zero_grad(set_to_none=True)
        for p_ in net.parameters():
            p_.requires_grad_(True)
        with torch.enable_grad():
            for x, y in [data.batch(data.xtr, data.ytr_m, idx[i:i + 64]) for i in range(0, len(idx), 64)]:
                z_ = net.logits(net.features(x).float())
                (F.cross_entropy(z_, y, reduction="sum") / len(idx)).backward()
        for a in hx.areas.values():
            rs_ = []
            for m in a.mods:
                if m.weight.grad is None:
                    continue
                G_ = m.weight.grad.reshape(m.weight.shape[0], -1).float()
                Gh_ = hx._T(a, G_) if a.W is not None else G_
                D_ = Gh_.reshape(m.weight.shape)
                alpha = 1e-3 * float(m.weight.detach().float().norm()) / max(float(D_.norm()), 1e-30)
                acc = [0.0, 0.0]
                for x, h0, z0 in zip(batches, base, zb):
                    st_ = {}

                    def hk2(mod, inp, out, _st=st_):
                        _st["y"] = out.detach().float()
                    hd = m.register_forward_hook(hk2)
                    try:
                        net.features(x)
                        y0 = st_["y"]
                        w0 = m.weight.data.clone()
                        m.weight.data.add_(D_.to(m.weight.dtype), alpha=-alpha)
                        h1 = net.features(x).float()
                        y1 = st_["y"]
                        m.weight.data.copy_(w0)
                    finally:
                        hd.remove()
                    z1 = net.logits(h1)
                    acc[0] += float((z1 - z0).pow(2).sum()); acc[1] += float((y1 - y0).pow(2).sum())
                rs_.append(acc[0] / max(acc[1], 1e-30))
            if rs_:
                gst[a.name] = sum(rs_) / len(rs_)
    finally:
        net.zero_grad(set_to_none=True)
        for p_ in net.parameters():
            p_.requires_grad_(rg_[id(p_)])
        hx.mode = mode1_
        net.train(was)
    if gst:
        log.info("[DIAG-J] %sT%d gain ALONG THE HELD UPDATE / random-output gain (logits), per area: %s" % (tag, t, " ".join(
            "%s:%.3g/%.3g" % (n.replace("bb.encoder.layer.", "L").replace("encoder.layer.", "L").replace(".attention.attention", "").replace(
                ".attention.output.dense", ".o").replace(".intermediate.dense", ".fc1").replace(".output.dense", ".fc2"),
                v, res[n][1]) for n, v in gst.items() if n in res)))
    import re as _re
    def _blk(n):
        m_ = _re.search(r"layer\.?(\d+)", n) or _re.search(r"layer(\d)", n)
        return m_.group(0) if m_ else n
    by = {}
    for n, (gh, gz) in res.items():
        by.setdefault(_kind_of(n, getattr(net, "kind", "")) if False else hx.areas[n].tag, []).append((n, gh, gz))
    log.info("[DIAG-J] %sT%d downstream gain of an area's output perturbation (decision feature / logits, per unit "
             "output energy), per area in order: %s" % (tag, t, " ".join(
                 "%s:%.3g/%.3g" % (n.replace("bb.encoder.layer.", "L").replace(".attention.attention", "").replace(
                     ".attention.output.dense", ".o").replace(".intermediate.dense", ".fc1").replace(".output.dense", ".fc2"),
                     gh, gz) for n, (gh, gz) in res.items())))
    log.info("[DIAG-J] %sT%d by kind (mean gamma_h / gamma_z, min, max): %s" % (tag, t, " ".join(
        "%s %.3g/%.3g [%.3g, %.3g]" % (k, sum(v[1] for v in vs) / len(vs), sum(v[2] for v in vs) / len(vs),
                                        min(v[1] for v in vs), max(v[1] for v in vs)) for k, vs in by.items())))
