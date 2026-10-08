"""Diagnostics (evaluation only): margin decomposition, per-layer drift and acquisition measures for HippoLiteEnc.

Nothing here reaches training. The margin / drift part only reads the test-set states the evaluation already
computes. The acquisition pass (after the task's last step, before its sleep) runs the present task's training
sentences in eval mode through a sequential loader with every hippocampus hook detached, fills .grad (cleared
afterwards; the optimiser state is only read) and restores every RNG state, so the training that follows is unchanged.

1. MARGIN (per earlier task e, evaluations t-1 -> t), per test sentence, logits recomputed in fp32 from the captured
   decision state h (position 0, last layer):  m = z_gold - max_other z;
     m(t) - m(t-1) = readout old heads  [m(heads_t on old classes, h_{t-1}) - m(t-1)]
                   + new-head entry     [m(heads_t, h_{t-1}) - m(heads_t on old classes, h_{t-1})]
                   + trunk              [m(heads_t, h_t) - m(heads_t, h_{t-1})]
2. PER-LAYER DRIFT of the decision position (emb, L1..L12): per sentence |dh|/|h|; differential (class-mean, minus
   their common motion) and common-mode drift relative to the between-class spread of that layer.
3. ACQUISITION: eval-mode CE / acc on the present task's training sentences (all heads, own head); gradient G; per
   kind gshare = |held(G)|^2/|G|^2, first-order decrease of an unheld Adam step lr <G, P(G)> and of the held one
   lr <G, held(P(held(G)))>, keep = their ratio (comparator give-back not included).
"""
import logging
import random

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger()


def _margin(Z, y):
    g = Z.gather(1, y.unsqueeze(1)).squeeze(1)
    o = Z.scatter(1, y.unsqueeze(1), float("-inf")).max(1).values
    return g - o


class EncDiag:
    def __init__(self, learner):
        self.L = learner
        self.prev, self.learnt, self.cum = {}, {}, {}
        self.tot = None

    def _heads(self, cur):
        m = self.L._unwrap(self.L.wrap_model).model
        hd = m.readout.heads
        W = torch.cat([hd[h].weight.detach().float().cpu() for h in range(cur + 1)])
        b = torch.cat([hd[h].bias.detach().float().cpu() for h in range(cur + 1)])
        return W, b

    @torch.no_grad()
    def margin(self, e, cur, HL, Y):
        """HL (N, layers+1, d) decision-position states of task e's test sentences after task cur; Y gold columns."""
        nc = self.L.CL_dataset.continual_config["CUR_NUM_CLASS"]
        bd = np.cumsum([0] + list(nc))
        W, b = self._heads(cur)
        # computed on the device: CPU tensor math can stall when many host threads are in use
        dev = next(self.L._unwrap(self.L.wrap_model).model.parameters()).device
        W, b, HL, Y = W.to(dev), b.to(dev), HL.to(dev), Y.to(dev).long()
        H = HL[:, -1]
        Z = H @ W.t() + b
        if self.tot is None or self.tot.get("t") != cur:
            self.tot = {"t": cur, "n": 0, "dm": 0.0, "ro_old": 0.0, "ro_new": 0.0, "tr": 0.0, "ap": 0.0, "am": 0.0,
                        "an": 0.0, "lost": 0, "lt": 0, "lr": 0, "gain": 0, "bs": [], "bc": []}
        cur_st = {"H": H, "Z": Z, "HL": HL, "y": Y}
        if e == cur or e not in self.prev:
            self.prev[e] = cur_st
            self.learnt[e] = HL
            self.cum[e] = [0.0, 0.0, 0.0]
            return
        pv = self.prev[e]
        if pv["y"].shape != Y.shape or not bool((pv["y"] == Y).all()):
            logger.info("[DIAG-M] task %d: test order changed, skipped" % e)
            self.prev[e] = cur_st
            return
        C0 = int(bd[cur])
        Zm = pv["H"] @ W.t() + b
        m_p = _margin(pv["Z"][:, :C0], Y)
        m_mo = _margin(Zm[:, :C0], Y)
        m_m = _margin(Zm, Y)
        m_n = _margin(Z, Y)
        ro_old, ro_new, tr = m_mo - m_p, m_m - m_mo, m_n - m_m
        ok_p, ok_m, ok_n = m_p > 0, m_m > 0, m_n > 0
        lost = ok_p & ~ok_n
        cm = self.cum[e]
        cm[0] += float(ro_old.mean()); cm[1] += float(ro_new.mean()); cm[2] += float(tr.mean())
        g_tr = (Z.gather(1, Y[:, None]) - Zm.gather(1, Y[:, None])).squeeze(1)
        dh = (H - pv["H"]).norm(dim=1) / pv["H"].norm(dim=1).clamp(min=1e-12)
        lm = lambda v: float(v[lost].mean()) if bool(lost.any()) else 0.0
        logger.info("[DIAG-M] T%d task %d: acc prev %.1f mid(readout only) %.1f now %.1f | margin %+.3f -> %+.3f: "
                    "readout old-heads %+.3f, new-head entry %+.3f, trunk %+.3f (gold logit trunk %+.3f) | lost %d "
                    "(trunk-caused %d, readout-caused %d), gained %d | lost: ro_old %+.3f ro_new %+.3f trunk %+.3f | "
                    "|dh|/|h| %.4f | since learnt: ro_old %+.3f ro_new %+.3f trunk %+.3f" % (
                        cur, e, 100 * float(ok_p.float().mean()), 100 * float(ok_m.float().mean()),
                        100 * float(ok_n.float().mean()), float(m_p.mean()), float(m_n.mean()), float(ro_old.mean()),
                        float(ro_new.mean()), float(tr.mean()), float(g_tr.mean()), int(lost.sum()),
                        int((lost & ok_m).sum()), int((lost & ~ok_m).sum()), int((~ok_p & ok_n).sum()),
                        lm(ro_old), lm(ro_new), lm(tr), float(dh.mean()), cm[0], cm[1], cm[2]))
        if bool(lost.any()):
            # where the lost sentences go: the predicted class's task (the newest task, the sentence's own
            # task = a within-task counterpart, another earlier task) and the most frequent (gold -> predicted) pairs
            pr_ = Z.argmax(1)[lost]
            yl_ = Y[lost]
            bdt_ = torch.tensor(bd[1:], device=pr_.device)
            tp_ = torch.searchsorted(bdt_, pr_, right=True)
            nl_ = float(lost.sum())
            pairs_ = {}
            for g_, q_ in zip(yl_.tolist(), pr_.tolist()):
                pairs_[(g_, q_)] = pairs_.get((g_, q_), 0) + 1
            top_ = sorted(pairs_.items(), key=lambda kv: -kv[1])[:4]
            try:
                cc_ = self.L.CL_dataset.continual_config
                nm_ = [str(cc_["idx2label"][ci_]) for t_ in range(len(cc_["CUR_CLASS"])) for ci_ in cc_["CUR_CLASS"][t_]]
            except Exception:
                nm_ = None
            fmt_ = (lambda c: "%d:%s" % (c, nm_[c][:18])) if nm_ else (lambda c: str(c))
            logger.info("[DIAG-P] T%d task %d: %d lost -> newest task %.2f, own task %.2f, other earlier %.2f | "
                        "distinct pairs %d, top: %s" % (
                            cur, e, int(nl_), float((tp_ == cur).sum()) / nl_, float((tp_ == e).sum()) / nl_,
                            float(((tp_ != cur) & (tp_ != e)).sum()) / nl_, len(pairs_),
                            ", ".join("%s->%s x%d" % (fmt_(g_), fmt_(q_), c_) for (g_, q_), c_ in top_)))
        T = self.tot
        T["n"] += Y.numel()
        for k_, v_ in (("dm", m_n - m_p), ("ro_old", ro_old), ("ro_new", ro_new), ("tr", tr)):
            T[k_] += float(v_.sum())
        T["ap"] += float(ok_p.sum()); T["am"] += float(ok_m.sum()); T["an"] += float(ok_n.sum())
        T["lost"] += int(lost.sum()); T["lt"] += int((lost & ok_m).sum()); T["lr"] += int((lost & ~ok_m).sum())
        T["gain"] += int((~ok_p & ok_n).sum())
        T["bs"].append(self._lay(pv["HL"], HL, Y))
        T["bc"].append(self._lay(self.learnt[e], HL, Y))
        self.prev[e] = cur_st

    def summary(self, cur):
        T = self.tot
        if T is None or T.get("t") != cur or not T["n"]:
            return
        n = T["n"]
        logger.info("[DIAG-M] T%d ALL OLD: acc prev %.2f mid %.2f now %.2f | dmargin %+.3f = ro_old %+.3f + ro_new %+.3f "
                    "+ trunk %+.3f | lost %d (trunk %d, readout %d) gained %d" % (
                        cur, 100 * T["ap"] / n, 100 * T["am"] / n, 100 * T["an"] / n, T["dm"] / n, T["ro_old"] / n,
                        T["ro_new"] / n, T["tr"] / n, T["lost"], T["lt"], T["lr"], T["gain"]))
        for nm, bl in (("step", T["bs"]), ("since learnt", T["bc"])):
            A = np.mean(np.array(bl), 0)
            logger.info("[DIAG-B] T%d old tasks, drift %s per layer (|dh|/|h| / differential/spread / common/spread): %s"
                        % (cur, nm, " ".join("%s %.3f/%.3f/%.3f" % ("emb" if i == 0 else "L%d" % i, A[i, 0], A[i, 1],
                                                                     A[i, 2]) for i in range(A.shape[0]))))

    @staticmethod
    def _lay(H0, H1, y):
        ks = torch.unique(y)
        out = []
        for l in range(H0.shape[1]):
            a0, a1 = H0[:, l], H1[:, l]
            rel = float(((a1 - a0).norm(dim=1) / a0.norm(dim=1).clamp(min=1e-12)).mean())
            M0 = torch.stack([a0[y == k].mean(0) for k in ks])
            M1 = torch.stack([a1[y == k].mean(0) for k in ks])
            sp = float((M0 - M0.mean(0)).norm(dim=1).pow(2).mean().sqrt())
            D = M1 - M0
            c = D.mean(0)
            dif = float((D - c).norm(dim=1).pow(2).mean().sqrt())
            out.append((rel, dif / max(sp, 1e-12), float(c.norm()) / max(sp, 1e-12)))
        return out

    # ------------------------------------------------------------------ acquisition
    def acquisition(self, task_id, n_max=1024, bs=16):
        L = self.L
        hx = L.hidx
        model = L._unwrap(L.wrap_model).model
        base = L.train_loader_list[task_id].loader
        rs = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), random.getstate(), np.random.get_state())
        was = model.training
        col0 = hx.collect
        hx.pause_hooks()
        hx.collect = None
        mc0 = L._mcache
        L._mcache = None
        model.eval()
        nc = L.CL_dataset.continual_config["CUR_NUM_CLASS"]
        bd = np.cumsum([0] + list(nc))
        lo, hi = int(bd[task_id]), int(bd[task_id + 1])
        dev = next(model.parameters()).device
        for p in model.parameters():
            p.grad = None
        ds = base.dataset
        idx = list(range(min(n_max, len(ds))))
        dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=bs, shuffle=False,
                                         collate_fn=base.collate_fn, num_workers=0)
        ce_a = ce_o = acc_a = acc_o = 0.0
        n = 0
        try:
            for b in dl:
                ids = b["input_ids"].to(dev); am = b["attention_mask"].to(dev)
                y = torch.as_tensor(b["label_idx_cil"]).to(dev).long()
                o = model(input_ids=ids, attention_mask=am)
                z = o.logits[:, 0].float()
                loss = F.cross_entropy(z, y, reduction="sum")
                (loss / len(idx)).backward()
                with torch.no_grad():
                    zo = z[:, lo:hi]
                    ce_a += float(loss); ce_o += float(F.cross_entropy(zo, y - lo, reduction="sum"))
                    acc_a += float((z.argmax(1) == y).sum()); acc_o += float((zo.argmax(1) == y - lo).sum())
                    n += y.numel()
            st = self._shares(task_id, model)
        finally:
            for p in model.parameters():
                p.grad = None
            hx.collect = col0
            hx.resume_hooks()
            L._mcache = mc0
            model.train(was)
            torch.set_rng_state(rs[0]); torch.cuda.set_rng_state_all(rs[1])
            random.setstate(rs[2]); np.random.set_state(rs[3])
        t0 = sum(v[2] for v in st.values()); t1 = sum(v[3] for v in st.values())
        logger.info("[DIAG-A] T%d train fit (eval mode, %d sentences): CE all-heads %.4f own-head %.4f | acc all %.2f "
                    "own %.2f | first-order decrease per step: unheld %.3e held %.3e (keep %.3f)" % (
                        task_id, n, ce_a / n, ce_o / n, 100 * acc_a / n, 100 * acc_o / n, t0, t1, t1 / max(t0, 1e-30)))
        logger.info("[DIAG-A] T%d per kind: gshare / fo_unheld / fo_held / keep / gshare at c=1: %s" % (
            task_id, " ".join("%s %.4f/%.2e/%.2e/%.3f/%.4f" % (k, v[1] / max(v[0], 1e-30), v[2], v[3],
                                                              v[3] / max(v[2], 1e-30), v[4] / max(v[0], 1e-30))
                              for k, v in sorted(st.items()))))

    @torch.no_grad()
    def _shares(self, task_id, model):
        from utils.hippo_enc import _LowT, HippoIndexEnc
        L, hx = self.L, self.L.hidx
        opt = L.optimizer
        lr_of = {}
        for g in opt.param_groups:
            for p in g["params"]:
                lr_of[id(p)] = (g["lr"], g["betas"][1], g["eps"])
        on = bool(hx.eig)
        Tm, Bm, Dm, Rg, Rs = {}, {}, {}, {}, {}
        if on:
            for area, mods in hx.areas.items():
                T = hx.T.get(area)
                for mod in mods:
                    Tq = hx._Tf if (hx._Tf is not None and id(mod) in hx._fresh) else T
                    if Tq is not None:
                        Tm[id(mod.weight)] = Tq
                    if mod.bias is not None and isinstance(T, _LowT) and not (hx._Tf is not None and id(mod) in hx._fresh):
                        Bm[id(mod.bias)] = (T.sb if T.av is not None else T.c) * T.m   # (OWM: f / (f + mu), as step())
            for name, mod in hx.diag.items():
                D = hx.dD.get("g:" + name)
                if D is not None:
                    Dm[id(mod.weight)] = D
                    if isinstance(getattr(mod, "bias", None), torch.nn.Parameter):
                        Dm[id(mod.bias)] = D
            for tn, mod in hx.tabs.items():
                D = hx.dD.get(tn)
                if D is not None:
                    Rg[id(mod.weight)] = D
                    if bool(getattr(L, "table_step_share", False)):
                        Rs[id(mod.weight)] = D
        fresh_w = {id(m_.weight) for m_ in hx.areas["readout"] if id(m_) in hx._fresh}
        fresh_b = {id(m_.bias) for m_ in hx.areas["readout"] if id(m_) in hx._fresh and m_.bias is not None}
        heads_w = {id(m_.weight) for m_ in hx.areas["readout"]} | {id(m_.bias) for m_ in hx.areas["readout"]}

        def hold_w(X, T):
            return HippoIndexEnc._tmm(X.float().contiguous(), T)

        def held_g(p, G):
            i = id(p)
            if i in Tm and G.dim() == 2:
                return hold_w(G, Tm[i])
            if i in Bm:
                return G * Bm[i]
            if i in Dm and G.dim() == 1:
                return G * Dm[i].to(G.device)
            if i in Rg and G.dim() == 2:
                return G * Rg[i].to(G.device, G.dtype).unsqueeze(1)
            return G

        def held_s(p, U):
            i = id(p)
            if i in Tm and U.dim() == 2:
                return hold_w(U, Tm[i])
            if i in Bm:
                return U * Bm[i]
            if i in Dm and U.dim() == 1:
                return U * Dm[i].to(U.device)
            if i in Rs and U.dim() == 2:
                return U * Rs[i].to(U.device, U.dtype).unsqueeze(1)
            return U

        st = {}
        joint = {}
        if on and getattr(hx, "_rhold", None):
            for m_ in hx.areas["readout"]:
                if id(m_) not in hx._rhold or m_.weight.grad is None:
                    continue
                def _P(p_):
                    s_ = opt.state.get(p_, {})
                    lr_, b2_, eps_ = lr_of[id(p_)]
                    if "exp_avg_sq" not in s_:
                        return lambda X: X
                    den_ = (s_["exp_avg_sq"].float() / (1 - b2_ ** float(s_["step"]))).sqrt() + eps_
                    return lambda X: X / den_
                gw, gb = m_.weight.grad.float().clone(), m_.bias.grad.float().clone()
                hx._rowproj(m_, gw, gb)
                uw, ub = _P(m_.weight)(gw), _P(m_.bias)(gb)
                hx._rowproj(m_, uw, ub)
                joint[id(m_.weight)] = (gw, uw); joint[id(m_.bias)] = (gb, ub)
        for nm, p in model.named_parameters():
            G = p.grad
            if G is None or id(p) not in lr_of:
                continue
            G = G.float()
            if float(G.abs().sum()) == 0:
                continue
            i = id(p)
            if i in heads_w:
                k = "head-new" if (i in fresh_w or i in fresh_b) else "head-old"
            elif i in hx._pk:
                k = hx._pk[i][0]
            else:
                k = "other"
            lr, b2, eps = lr_of[i]
            s = opt.state.get(p, {})
            den = (s["exp_avg_sq"].float() / (1 - b2 ** float(s["step"]))).sqrt() + eps if "exp_avg_sq" in s else None
            P = (lambda X: X / den) if den is not None else (lambda X: X)
            Gh = held_g(p, G) if on else G
            u = P(G)
            uh = held_s(p, P(Gh)) if on else u
            if i in joint:
                Gh, uh = joint[i]
            # (the span alone: the trunk operators with the tail share c = 1)
            G1 = G
            if on and i in Tm and G.dim() == 2 and isinstance(Tm[i], _LowT):
                T1 = _LowT(Tm[i].Ub, 1.0)
                G1 = hold_w(G, T1)
            e = st.setdefault(k, [0.0, 0.0, 0.0, 0.0, 0.0])
            e[0] += float(G.double().pow(2).sum()); e[1] += float(Gh.double().pow(2).sum())
            e[2] += lr * float((G.double() * u.double()).sum()); e[3] += lr * float((G.double() * uh.double()).sum())
            e[4] += float(G1.double().pow(2).sum())
        return st


class LayerOwn:
    """Diagnostic (evaluation only): where the drift of old sentences starts: on the first N test sentences of task 0, each
    encoder layer's input h, attention-sublayer output a and layer output y (all positions) are kept from one
    evaluation to the next (test activations only; no weights are copied). At the next evaluation the CURRENT
    sublayers are run on the PREVIOUS inputs:
        attention own change = |attn_now(h_prev) - a_prev| / |a_prev|      (that sublayer's parameter change alone)
        FFN own change       = |ffn_now(a_prev) - y_prev| / |y_prev|        (FFN + its norm, parameter change alone)
        total                = |y_now - y_prev| / |y_prev|,  input = |h_now - h_prev| / |h_prev|
    at the decision position (0) and over the content positions (attended, other than 0)."""
    def __init__(self, n_max=256):
        self.n_max = n_max
        self.prev = None
        self.cur = None
        self.hs = []

    def hooks(self, model):
        enc = model.enc
        layers = list(enc.encoder.layer)
        self.cur = {"m": [], "xm": [], "L": [[[], [], [], None] for _ in layers], "n": 0}
        cur = self.cur

        def enc_pre(mod, args, kwargs):
            am = kwargs.get("attention_mask")
            if am is not None and cur["n"] < self.n_max:
                cur["m"].append(am.detach().cpu())
            return None
        self.hs.append(enc.register_forward_pre_hook(enc_pre, with_kwargs=True))
        for i, l in enumerate(layers):
            def lpre(mod, args, kwargs, _i=i):
                if cur["n"] >= self.n_max:
                    return None
                cur["L"][_i][0].append(args[0].detach().to(torch.bfloat16).cpu())
                if _i == 0:
                    # (the layer's own mask argument, as the encoder built it for this batch)
                    xm = args[1] if len(args) > 1 else kwargs.get("attention_mask")
                    cur["xm"].append(xm.detach().cpu() if torch.is_tensor(xm) else None)
                return None

            def lpost(mod, args, out, _i=i):
                if cur["n"] >= self.n_max:
                    return None
                o = out[0] if isinstance(out, tuple) else out
                cur["L"][_i][2].append(o.detach().to(torch.bfloat16).cpu())
                if _i == len(layers) - 1:
                    cur["n"] += o.shape[0]
                return None

            def apost(mod, args, out, _i=i):
                if cur["n"] >= self.n_max:
                    return None
                o = out[0] if isinstance(out, tuple) else out
                cur["L"][_i][1].append(o.detach().to(torch.bfloat16).cpu())
                return None
            self.hs.append(l.register_forward_pre_hook(lpre, with_kwargs=True))
            self.hs.append(l.register_forward_hook(lpost))
            self.hs.append(l.attention.register_forward_hook(apost))

    def remove(self):
        for h in self.hs:
            h.remove()
        self.hs = []

    @torch.no_grad()
    def report(self, model, cur_t):
        if self.cur is None:
            return
        cur, prev = self.cur, self.prev
        self.prev, self.cur = cur, None
        if prev is None or len(prev["m"]) != len(cur["m"]):
            return
        dev = next(model.parameters()).device
        layers = list(model.enc.encoder.layer)
        was = model.training
        model.eval()
        rows = []
        try:
            for i, l in enumerate(layers):
                acc = np.zeros((2, 5))                    # [pos0, content] x [own_attn, own_ffn, total, input, n]
                for bi, m2 in enumerate(prev["m"]):
                    hp, ap, yp = prev["L"][i][0][bi], prev["L"][i][1][bi], prev["L"][i][2][bi]
                    hn, yn = cur["L"][i][0][bi], cur["L"][i][2][bi]
                    # the layer's call arguments (extended mask) are rebuilt by the model for each batch: re-derive
                    # the additive mask from the 2D mask in the encoder's own dtype convention
                    xm = prev["xm"][bi]
                    ext = xm.to(dev) if xm is not None else None
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        hpd, apd = hp.to(dev).float(), ap.to(dev).float()
                        a_own = l.attention(hpd, ext)[0]
                        y_own = l.output(l.intermediate(apd), apd)
                    # the comparison runs on the device: CPU tensor math can stall when many host threads are in use
                    a_own, y_own = a_own.float(), y_own.float()
                    hp, ap, yp = hpd, apd, yp.to(dev).float()
                    hn, yn = hn.to(dev).float(), yn.to(dev).float()
                    msk = m2.to(dev).bool().clone()
                    for k_, sel in ((0, None), (1, msk)):
                        if k_ == 0:
                            f = lambda T: T[:, 0]
                        else:
                            sel = sel.clone(); sel[:, 0] = False
                            f = lambda T, _s=sel: T[_s]
                        acc[k_, 0] += float(((f(a_own) - f(ap)).norm(dim=-1) / f(ap).norm(dim=-1).clamp(min=1e-6)).sum())
                        acc[k_, 1] += float(((f(y_own) - f(yp)).norm(dim=-1) / f(yp).norm(dim=-1).clamp(min=1e-6)).sum())
                        acc[k_, 2] += float(((f(yn) - f(yp)).norm(dim=-1) / f(yp).norm(dim=-1).clamp(min=1e-6)).sum())
                        acc[k_, 3] += float(((f(hn) - f(hp)).norm(dim=-1) / f(hp).norm(dim=-1).clamp(min=1e-6)).sum())
                        acc[k_, 4] += float(f(yp).shape[0])
                a = acc[:, :4] / np.maximum(acc[:, 4:5], 1)
                rows.append("L%d pos0 %.4f/%.4f/%.4f/%.4f content %.4f/%.4f/%.4f/%.4f" % (i + 1, *a[0], *a[1]))
        finally:
            model.train(was)
        for k0 in range(0, len(rows), 4):
            logger.info("[DIAG-L] T%d task 0 layer-own drift since T%d (attention own / FFN own / layer total / layer "
                        "input, relative): %s" % (cur_t, cur_t - 1, " | ".join(rows[k0:k0 + 4])))
