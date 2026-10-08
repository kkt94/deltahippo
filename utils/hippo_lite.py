"""HippoLite (DeltaHippo): one hippocampus for every synapse group of a full fine-tuned model.

Areas: linear modules of a layer that read the same input (q/k/v; gate/up; o; down) and the readout (the output
table); norm gains ("g:<name>") and the input table ("tok") are per-coordinate areas. Every linear area is treated
alike; its moment is taken over the positions where it receives a learning signal (trunk: all positions, readout:
answer steps).

Sleep (end of every task but the last; no learning, no graph, eval mode): the finished model re-reads the task's
training set.
  * every linear area (trunk and readout): gated-delta subspace memory (U, lam, n, nlast, floor, tail). New keys K =
    Gavish-Donoho top of the episode's moment; in B = [U, K] the old content along each key is erased by
    beta (1 - rho_i), rho_i the share of the answer rows' energy along k_i coming from rows in state conflict with an
    earlier record (a second read, RNG state replayed); the episode's energy is written by beta; re-diagonalised above
    the episode's noise floor f. Tail = running mean of each episode's energy outside the span. The d x d moments are
    taken group by group below the training peak.
  * gains / input table: accumulated per-coordinate energy, GD owned coordinates.
  * class-level answer records per (class id, answer step) and every area's mean answer-step input per memory.
Wake (task >= 1): operators rebuilt at steps 1, 2, 4, ...: the MINIMUM-INTERFERENCE operator per linear area,
T = b I + U diag(a - b) U^T, a_i = f / (f + mu lam_i), b = f / (f + mu tau) (mu = earlier samples over the last
episode's, tau = tail energy per direction), never materialised; U also spans the earlier memories' class-level mean
patterns, which take share 0 (PATTERN HOLD). Gains / input table by share vectors; each row of a tied table also by
f / (f + mu u_v), u_v its earlier use as an input. The gradient is formed from the HELD INPUT (dW = dy^T (x T), the
comparator's answer rows entering as their mismatch Xp); Adam (fused kernel) steps through T again (its elementwise
normalisation re-rotates the held gradient). The readout hook detects state conflicts (answer state inside an earlier
record's neighbourhood), sets the write gate, sets up the CA1 comparator (give-back along the mismatch with all memories'
mean patterns) and the joint-share target at parting steps.
"""
import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger()

_GROUP = {"q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv", "gate_proj": "gu", "up_proj": "gu",
          "o_proj": "o", "down_proj": "down"}


def _gd_rank(ev, n):
    """Gavish-Donoho optimal hard threshold on the singular values sqrt(ev) of a second moment from n patterns."""
    sv = ev.clamp(min=0).sqrt().flip(0)
    D = float(ev.shape[0])
    bt = min(float(n), D) / max(float(n), D)
    om = 0.56 * bt ** 3 - 0.95 * bt ** 2 + 1.82 * bt + 1.43
    tau = om * float(sv[:int(min(n, D))].median())
    return int((sv > tau).sum())


def _eigh(A):
    """torch.linalg.eigh; only when the fp32 GPU solver fails to converge (ill-conditioned or many repeated
    eigenvalues), the same decomposition in fp64, cast back."""
    try:
        return torch.linalg.eigh(A)
    except torch._C._LinAlgError:
        ev, V = torch.linalg.eigh(A.double())
        return ev.to(A.dtype), V.to(A.dtype)


def _pad64(U):
    """The basis with zero columns appended up to a multiple of 64: (x U) U^T is unchanged, and the GEMMs stay on the
    aligned kernels."""
    r = U.shape[1]
    r2 = (r + 63) // 64 * 64
    if r2 == r:
        return U.contiguous()
    return torch.cat([U, U.new_zeros(U.shape[0], r2 - r)], 1).contiguous()


class _HoldLinear(torch.autograd.Function):
    """A linear synapse group whose weight gradient is formed from its HELD input: dW = dy^T (x T), the comparator's
    answer rows entering as their mismatch Xp. The same as taking dW through T afterwards (dW T = dy^T (x T)), at
    the cost of the batch's rows instead of the weight's rows. Forward and input gradient are autocast's linear."""
    @staticmethod
    def forward(ctx, x, W, b, hx, mod, area):
        with torch.autocast("cuda", enabled=False):
            xb = hx._shared_cast(area, x)          # q/k/v and gate/up save ONE bf16 copy of their common input
            Wb = W.to(torch.bfloat16)
            y = F.linear(xb, Wb, b.to(torch.bfloat16) if b is not None else None)
        ctx.save_for_backward(xb, Wb)
        ctx.hx, ctx.mod, ctx.area, ctx.xdt, ctx.hasb = hx, mod, area, x.dtype, b is not None
        return y

    @staticmethod
    def backward(ctx, dy):
        xb, Wb = ctx.saved_tensors
        with torch.autocast("cuda", enabled=False):
            dyb = dy.to(torch.bfloat16)
            dx = (dyb @ Wb).to(ctx.xdt)
            dy2 = dyb.reshape(-1, dyb.shape[-1])
            Xt = ctx.hx._held_input(ctx.mod, ctx.area, xb)
            dW = torch.mm(dy2.t(), Xt, out_dtype=torch.float32)
            db = dy2.float().sum(0) if ctx.hasb else None
        return dx, dW, db, None, None, None


class _LowT:
    """Minimum-interference operator T = b I + U diag(av - b) U^T, U (d x r) the held span (bf16, padded to 64
    columns; av is 0 on the pattern-hold and padding columns), never materialised."""
    def __init__(self, Ub, av, b):
        self.Ub, self.av, self.b = Ub, av, float(b)
        self.shape = (Ub.shape[0], Ub.shape[0])


class HippoIndexLite:
    _CMOFF = 32                              # answer-step offsets per co-movement class slot

    def __init__(self, learner, model):
        self.L = learner
        self.areas = {}                      # area -> [modules]
        self.first = {}                      # area -> the module whose input is read
        for name, mod in model.named_modules():
            if not isinstance(mod, torch.nn.Linear):
                continue
            leaf = name.rsplit(".", 1)[-1]
            if leaf not in _GROUP or ".layers." not in "." + name:
                continue
            area = name.rsplit(".", 1)[0] + "." + _GROUP[leaf]
            self.areas.setdefault(area, []).append(mod)
        head = model.get_output_embeddings()
        self.areas["readout"] = [head]
        self._tied = model.get_input_embeddings().weight is head.weight
        self._tieid = id(model.get_input_embeddings().weight)   # the input table (tied: also the readout)
        self._tieR = None                    # (tiedrow) per-row share of the table's step, from its input use
        # every hook is kept as (module, kind, fn) so it can be detached during evaluation, where all of them
        # return without effect (collect is None, _mcache is None, no gradient)
        self._hookspec, self._handles = [], []
        for area, mods in self.areas.items():
            self.first[area] = mods[0]
            self._hookspec.append((mods[0], "pre", self._make_hook(area)))
            if area != "readout":
                self._hookspec.append((mods[0], "fwd", self._make_cmp_hook(area)))
        self.diag = {}
        for name, mod in model.named_modules():
            if name.endswith("norm") and isinstance(getattr(mod, "weight", None), torch.nn.Parameter) \
                    and mod.weight.dim() == 1:
                self.diag[name] = mod
                self._hookspec.append((mod, "pre", self._make_diag_hook(name, float(getattr(mod, "eps", getattr(mod, "variance_epsilon", 1e-6))))))
        for area, mods in self.areas.items():
            for mod in mods:
                self._install(mod, area)
        self._xt = {}
        self._xbc = None                     # one-entry cache of the present area's bf16 input (forward)
        self._tmdm = None                    # step(): param -> operator / share vector, rebuilt only when T changes
        self.emb = model.get_input_embeddings()
        self._hookspec.append((self.emb, "fwd", self._emb_hook))
        self._hookspec.append((head, "fwd", self._conflict_hook))
        self.resume_hooks()

        self.collect = None                  # None | "now" (training, task >= 1) | "rec" (sleep) | "g2s" (2nd sleep read)
        self.now, self.rec, self.old = {}, {}, {}
        self.eig = {}                        # gains/tok: (ev, None, owned) | linear areas (trunk, readout): ("subspace",)
        self.T = {}                          # linear area -> _LowT
        self.dD = {}                         # gain / token areas -> share vector
        self._mem = {}                       # linear area -> {"U", "Ub", "lam", "n", "nlast", "floor", "tail"}
        self.steps = 0
        self.next_build = 1
        self.n_eps = 0
        self.plast = 1.0                     # write gate of the present step (set by the conflict hook)
        self._tid = 0
        self._sk = None                      # sleep: the batch's masks, row indices and record keys
        self.recs = {}                       # (cid, off) -> [n, sum_h]
        self.cids = {}
        self.next_cid = 0
        self._sc_M = None
        # CA1 comparator: per memory (class, answer step) every area's summed answer-step input
        self._pat, self._pkey, self._pc_n = {}, {}, None
        self._psk, self._pom, self._rgQ = {}, {}, {}
        self._cmpB, self._j2p, self._cmpA = {}, None, {}
        self._cxs, self._cxa, self._cmpX, self._cmpsel, self._cmpd = {}, {}, {}, None, {}
        self._ansk, self._ansix, self._ansfl = None, None, None
        # co-movement (present task only)
        self._cm, self._cmB, self._cm_rowfac, self._cm_sl = None, None, None, None
        # conflict-weighted erase (second sleep read)
        self._kc, self._g2P, self._g2E, self._sk2, self._g2st, self._cest = {}, {}, {}, None, [], []
        self._scs = []
        self._grp = set()                    # sleep: the trunk areas whose moment the present read accumulates
        self._pha = {}                       # per task: the held span with the earlier memories' patterns (pattern hold)
        self._gfm = {}                       # input table: (noise floor, earlier-sample weight) for the tied rows' share
        self.tiedrow = True                  # tied table: each row's step also weighed by its earlier use as an input
        self.orthobasis = False              # (hippo_orthobasis) held span + pattern columns as one fp64 QR([U, P])
        self.train_peak = None               # the finished task's training peak (bytes): the sleep stays below it
        logger.info("[HIPPO-LITE] one hippocampus indexing %d cortical areas (%d synapse groups), %d gain areas and the "
                    "input table" % (len(self.areas), sum(len(v) for v in self.areas.values()), len(self.diag)))

    def _install(self, mod, area):
        orig = mod.forward

        def fwd(x, *a, **k):
            # (the path depends only on the phase -- training after the first episode -- so a recomputed checkpoint
            # segment takes the same path as its forward even on a build step, where T is made before the backward)
            if self.collect == "now" and self.eig and torch.is_grad_enabled() and mod.weight.requires_grad and not a and not k:
                return _HoldLinear.apply(x, mod.weight, mod.bias, self, mod, area)
            return orig(x, *a, **k)
        mod.forward = fwd

    def _shared_cast(self, area, x):
        """The bf16 copy of an area's input, made once for all of the area's synapse groups (they are called in a
        row on the same x), so that _held_input's cache (keyed on the saved copy's address) forms x T once per area.
        Cleared in pre_backward."""
        if x.dtype == torch.bfloat16:
            return x
        k_ = (area, x.data_ptr(), x._version, tuple(x.shape))
        c_ = self._xbc
        if c_ is not None and c_[0] == k_:
            return c_[1]
        xb = x.to(torch.bfloat16)
        self._xbc = (k_, xb)
        return xb

    @torch.no_grad()
    def _held_input(self, mod, area, xb):
        """The area's input rows as the index holds them (x T), the comparator's answer rows replaced by their
        mismatch; computed once per area and input (q/k/v and gate/up share it), bf16."""
        c_ = self._cmpX.get(id(mod)) if self._cmpsel is not None else None
        # the comparator's rows are the same tensor for every group of the area (_cmp_arm shares them): cached too
        key = (area, xb.data_ptr(), tuple(xb.shape), c_ is not None)
        Xt = self._xt.get(key)
        if Xt is not None:
            return Xt
        X = xb.reshape(-1, xb.shape[-1])
        T = self.T.get(area)
        Xf = self._tmm(X, T) if T is not None else X.float()
        if c_ is not None:
            bs, ps = self._cmpsel
            Xf[bs * xb.shape[1] + ps] = c_[1].to(Xf.dtype)
        Xt = Xf.to(torch.bfloat16)
        if len(self._xt) >= 2:
            self._xt.pop(next(iter(self._xt)))
        self._xt[key] = Xt
        return Xt

    def pre_backward(self):
        """Between the forward and the backward: the step count and, on build steps, the operators (from this step's
        forward energies) -- the backward forms every held gradient with them."""
        self._xt = {}
        self._xbc = None
        if not self.eig:
            return
        self.steps += 1
        if self.steps >= self.next_build:
            self._build()
            self.next_build *= 2

    # ------------------------------------------------------------------ hooks on / off
    def resume_hooks(self):
        """(Re-)register every hook (each module holds at most one hook of each kind, so the order is immaterial)."""
        if self._handles:
            return
        for mod, kind, fn in self._hookspec:
            self._handles.append(mod.register_forward_pre_hook(fn) if kind == "pre" else mod.register_forward_hook(fn))

    def pause_hooks(self):
        """Detach every hook (evaluation only: there each hook returns None and the only state any of them touches,
        _cxs / _cxa, is already empty after the sleep)."""
        for h in self._handles:
            h.remove()
        self._handles = []

    # ------------------------------------------------------------------ patterns
    def _store(self, area, C, n):
        st = self.now if self.collect == "now" else self.rec
        e = st.get(area)
        if e is None:
            st[area] = [C, n]
        else:
            e[0] += C
            e[1] += n

    def _energy_needed(self):
        """True only on the batches of the operator build steps (1, 2, 4, ... <= the task's steps)."""
        spe = self.L._spe
        if not spe:
            return True
        tot = int(spe) * int(self.L.params.training_epochs)
        last = 1 << (tot.bit_length() - 1) if tot > 0 else 0
        s_ = self.steps + 1
        return s_ <= last and (s_ & (s_ - 1)) == 0

    def _sleep_batch(self, ids, am, lb, tg):
        """SLEEP, once per batch before its forward: masks, flat row indices, counts and the answer rows' keys."""
        B, Lq = lb.shape
        ma = torch.zeros((B, Lq), dtype=torch.bool, device=lb.device)
        ma[:, :-1] = lb[:, 1:] != -100
        mt = am.bool()
        ix_all = mt.reshape(-1).nonzero(as_tuple=True)[0]
        ix_ans = ma.reshape(-1).nonzero(as_tuple=True)[0]
        bi, pi = ma.nonzero(as_tuple=True)
        bl, pl = bi.tolist(), pi.tolist()
        first, cnt = {}, {}
        rk, pk = [], []
        for b, p in zip(bl, pl):
            f_ = first.setdefault(b, p)
            o_ = cnt.get(b, 0); cnt[b] = o_ + 1
            cid = self.cids[tg[b]]
            rk.append((b, p, (cid, o_)))
            pk.append((cid, p - f_))
        rows = []
        for k in pk:
            r = self._pkey.get(k)
            if r is None:
                r = self._pkey[k] = len(self._pkey)
            rows.append(r)
        rows = torch.tensor(rows, device=lb.device)
        n = len(self._pkey)
        if self._pc_n is None or self._pc_n.shape[0] < n:
            c2 = torch.zeros(n + 64, device=lb.device)
            if self._pc_n is not None:
                c2[:self._pc_n.shape[0]] = self._pc_n
            self._pc_n = c2
        self._pc_n.index_add_(0, rows, torch.ones_like(rows, dtype=torch.float32))
        self._sk = {"shape": (B, Lq), "mt": mt, "ma": ma, "ix_all": ix_all, "ix_ans": ix_ans,
                    "n_all": int(ix_all.numel()), "n_ans": int(ix_ans.numel()), "bi": bi, "pi": pi, "rows": rows,
                    "rk": rk, "ids_p": ids[mt & (lb == -100)]}

    def _make_hook(self, area):
        ans = area == "readout"

        def hook(mod, inp):
            if self.collect == "g2s":
                # second sleep read: each answer row's energy along the area's new keys
                kc_, sk2_ = self._kc.get(area), self._sk2
                x = inp[0] if isinstance(inp, tuple) else inp
                if kc_ is None or sk2_ is None or not torch.is_tensor(x) or tuple(x.shape[:2]) != sk2_["shape"]:
                    return None
                with torch.no_grad():
                    X = x.detach().reshape(-1, x.shape[-1]).index_select(0, sk2_["ix_ans"]).float()
                    Kp_ = kc_.get("Kp")
                    if Kp_ is None:
                        Kp_ = kc_["Kp"] = _pad64(kc_["K"])
                    self._g2P[area] = (X @ Kp_)[:, :kc_["K"].shape[1]].pow(2)
                return None
            if self.collect is None or self.collect == "now":
                # (wake: a linear area's operator needs only its memory, nothing of the present batch)
                return None
            x = inp[0] if isinstance(inp, tuple) else inp
            if not torch.is_tensor(x) or x.dim() != 3:
                return None
            # SLEEP ("rec"): the area's full second moment, and its answer-step patterns per memory; "recC": a later
            # read of the same episode for the next group of trunk areas (moments only)
            sk = self._sk
            if sk is None or sk["shape"] != tuple(x.shape[:2]):
                return None
            if (sk["n_ans"] if ans else sk["n_all"]) <= 0:
                return None
            if self.collect == "rec":
                self._pat_rec(area, x)
            if not ans and area not in self._grp:
                return None
            if ans and self.collect == "recC":
                return None                                  # the readout moment was taken in the "rec" read
            with torch.no_grad():
                xf = x.detach().reshape(-1, x.shape[-1])
                X = xf.index_select(0, sk["ix_ans"] if ans else sk["ix_all"])
                if X.dtype != torch.bfloat16:
                    X = X.to(torch.bfloat16)
                C = torch.mm(X.t(), X, out_dtype=torch.float32)
                if not ans and sk["n_ans"] > 0:
                    Xa = xf.index_select(0, sk["ix_ans"]).to(torch.bfloat16)
                    Ca = torch.mm(Xa.t(), Xa, out_dtype=torch.float32)
                    # the same two roundings as C + Ca * s, without two more d x d fp32 temporaries
                    Ca.mul_(float(sk["n_all"]) / float(sk["n_ans"]))
                    C.add_(Ca)
                    del Ca
                e = self.rec.get(area)
                if e is None:
                    self.rec[area] = [C, float(X.shape[0])]
                else:
                    e[0] += C
                    e[1] += float(X.shape[0])
            return None
        return hook

    def _make_diag_hook(self, name, eps):
        def hook(mod, inp):
            if self.collect is None or self.collect == "recC":
                return None
            x = inp[0] if isinstance(inp, tuple) else inp
            if not torch.is_tensor(x) or x.dim() < 3:
                return None
            if self.collect == "now":
                if torch._C._current_graph_task_id() != -1:
                    return None                                          # checkpoint recomputation: once per step
                ck = self.L._mcache
                if ck is None or ck[0] != tuple(x.shape[:2]):
                    return None
                n_ = ck[3]
                if n_ <= 0 or not self._energy_needed():
                    return None
                m = ck[2]
                with torch.no_grad():
                    xf = x.detach().float()
                    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
                    mk = m.reshape(m.shape + (1,) * (x.dim() - 2)).float()
                    S_ = (xf.pow(2) * mk).reshape(-1, x.shape[-1]).sum(0)
                    rows_ = 1
                    for d_ in x.shape[2:-1]:
                        rows_ *= int(d_)
                    self._store("g:" + name, S_, float(n_ * rows_))
                return None
            sk = self._sk
            if sk is None or sk["shape"] != tuple(x.shape[:2]) or sk["n_all"] <= 0:
                return None
            with torch.no_grad():
                X = x.detach().reshape((-1,) + tuple(x.shape[2:])).index_select(0, sk["ix_all"])
                X = X.float().reshape(-1, x.shape[-1])
                X = X * torch.rsqrt(X.pow(2).mean(-1, keepdim=True) + eps)
                self._store("g:" + name, X.pow(2).sum(0), float(X.shape[0]))
            return None
        return hook

    def _emb_hook(self, mod, inp, out):
        ids = inp[0] if isinstance(inp, tuple) else inp
        if not torch.is_tensor(ids) or ids.dim() != 2 or self.collect == "recC":
            return None
        if self.collect == "now":
            ck = self.L._mcache
            lb = self.L.cur_labels
            if ck is not None and ck[0] == tuple(ids.shape) and self._energy_needed():
                with torch.no_grad():
                    pm = ck[2] & (lb.to(ids.device) == -100)
                    cnt = torch.bincount(ids.reshape(-1), weights=pm.reshape(-1).float(), minlength=mod.weight.shape[0])
                    self._store("tok", cnt, float(ck[3] - ck[4]))
        elif self.collect == "rec":
            sk = self._sk
            if sk is not None and sk["shape"] == tuple(ids.shape):
                with torch.no_grad():
                    t = sk["ids_p"]
                    cnt = torch.bincount(t, minlength=mod.weight.shape[0]).float()
                    self._store("tok", cnt, float(t.numel()))
        D = self.dD.get("tok")
        # untied table: the readout's held span lives in the last layer's space, not the input table's
        Tr = self.T.get("readout") if self.collect == "now" and self._tied else None
        if (D is not None or Tr is not None) and torch.is_tensor(out) and out.requires_grad:
            sc = D.to(out.device)[ids].unsqueeze(-1) if D is not None else None

            def _h(g, _sc=sc, _Tr=Tr):
                gg = g.float() * _sc if _sc is not None else g.float()
                if _Tr is not None:
                    # (the tied table's embedding rows are held by the readout operator, as its readout rows are)
                    gg = self._tmm(gg.reshape(-1, gg.shape[-1]), _Tr).reshape(gg.shape)
                return gg.to(g.dtype)
            out.register_hook(_h)
        return None

    # ------------------------------------------------------------------ products with the operator
    @staticmethod
    def _tmm(x, T):
        """x @ T in fp32 from bf16 operands: b x + ((x U) (av - b)) U^T."""
        y = torch.mm(x.to(torch.bfloat16), T.Ub, out_dtype=torch.float32)
        y.mul_(T.av - T.b)
        out = torch.mm(y.to(torch.bfloat16), T.Ub.t(), out_dtype=torch.float32)
        out.add_(x.to(torch.float32), alpha=T.b)
        return out

    @staticmethod
    def _tmm_(x, T):
        """x <- x @ T in place for an fp32 scratch x (the step's update): the same arithmetic as _tmm, without
        the result tensor and the copy back."""
        if x.dtype != torch.float32:
            return x.copy_(HippoIndexLite._tmm(x, T))
        y = torch.mm(x.to(torch.bfloat16), T.Ub, out_dtype=torch.float32)
        y.mul_(T.av - T.b)
        x.mul_(T.b)
        x.add_(torch.mm(y.to(torch.bfloat16), T.Ub.t(), out_dtype=torch.float32))
        return x

    @staticmethod
    def _mm_rows(x, T, rows=32768):
        """x <- x @ T written back into x block by block of rows. Callers hold no_grad."""
        for i0 in range(0, x.shape[0], rows):
            HippoIndexLite._tmm_(x[i0:i0 + rows], T)
        return x

    @staticmethod
    def _proj(x, T, rows=32768):
        """x @ T exactly as _mm_rows computes it. A tensor of at most `rows` rows is one block, projected in place in
        one call; taller tensors (the readout table) are projected in place block by block."""
        if x.shape[0] <= rows:
            return HippoIndexLite._tmm_(x, T)       # in place: no result tensor (one fp32 param-sized buffer fewer)
        return HippoIndexLite._mm_rows(x, T, rows)

    # ------------------------------------------------------------------ write through the index (the optimiser step)
    @torch.no_grad()
    def step(self, opt):
        """AdamW with Adam's step taken through the index; the comparator gives back, along the mismatch directions,
        what the index removed from Adam's step. The moments and the update u = m^/(sqrt(v^)+eps) come from torch's
        fused AdamW kernel (written into a zero buffer at lr -1); unwritten steps (lr 0) only move the moments."""
        if self._tmdm is None:
            # rebuilt only after _build / begin_task / sleep (T and dD change only there)
            Tm = {}
            for area, mods in self.areas.items():
                T = self.T.get(area)
                if T is not None:
                    for mod in mods:
                        Tm[id(mod.weight)] = T
            Dm = {}
            for name, mod in self.diag.items():
                D = self.dD.get("g:" + name)
                if D is not None:
                    Dm[id(mod.weight)] = D
            self._tieR = None
            if self.tiedrow and "tok" in self._gfm and self.eig.get("tok") is not None:
                # TIED TABLE: a row v is both a readout row and token v's input embedding. Its change moves every
                # earlier output through both uses: through the readout (held by the readout operator) and as the
                # input of every earlier occurrence of v, E_old |dv|^2 u_v with u_v the token's earlier use. The same
                # minimum-interference objective gives the row the share f / (f + mu u_v) on top of the readout's
                # operator (product form). Without it the label tokens' rows -- words the earlier inputs are full of --
                # move as readout rows with no regard for their input role. An untied table has only the input role;
                # a share on its gradient would be undone by Adam's normalisation, so the share is applied to its
                # step.
                f_t, mu_t = self._gfm["tok"]
                ev_t = self.eig["tok"][0]
                self._tieR = f_t / (f_t + mu_t * ev_t.float())                # (moved to the step's device once, below)
                if getattr(self, "_tielog", None) != self._tid:
                    self._tielog = self._tid
                    logger.info("[HIPPO-TIEDROW] task %d: row share mean %.3f | rows below 0.5: %d of %d | f %.3g mu %.2f"
                                % (int(self._tid), float(self._tieR.mean()), int((self._tieR < 0.5).sum()),
                                   self._tieR.numel(), f_t, mu_t))
            self._tmdm = (Tm, Dm)
        Tm, Dm = self._tmdm
        tdev_ = {}
        idle_, chunk_, nb_ = [], [], [0]

        def flush():
            # one fused kernel launch for the chunk's moments and updates, then each group through its operator
            if not chunk_:
                return
            # one zero buffer per chunk (one allocation and one memset instead of one per parameter)
            buf_ = torch.zeros(sum(e[0].numel() for e in chunk_), device=chunk_[0][0].device, dtype=chunk_[0][0].dtype)
            us, o_ = [], 0
            for e in chunk_:
                us.append(buf_[o_:o_ + e[0].numel()].view_as(e[0])); o_ += e[0].numel()
            c0 = chunk_[0]
            torch._fused_adamw_(us, [e[1] for e in chunk_], [e[2] for e in chunk_], [e[3] for e in chunk_], [],
                                [e[4] for e in chunk_], lr=-1.0, beta1=c0[5], beta2=c0[6], weight_decay=0.0, eps=c0[7],
                                amsgrad=False, maximize=False)
            for (p, g, m, v, tt, b1_, b2_, eps_, lr_), u in zip(chunk_, us):
                p.grad = None
                T = Tm.get(id(p))
                cq_ = self._cmpd.get(id(p)) if self._cmpd else None
                uM_ = None
                if cq_ is not None and len(cq_) == 1 and u.dim() == 2 and cq_[0] in self._cmpA:
                    Qa_, Ma_ = self._cmpA[cq_[0]]
                    uM_ = u @ Ma_
                if T is not None and u.dim() == 2 and u.shape[1] == T.shape[0]:
                    u = self._proj(u, T)
                elif id(p) in Dm and u.dim() == 1:
                    u.mul_(Dm[id(p)].to(u.device))
                if self._tieR is not None and id(p) == self._tieid \
                        and u.dim() == 2 and u.shape[0] == self._tieR.shape[0]:
                    if self._tieR.device != u.device or self._tieR.dtype != u.dtype:
                        self._tieR = self._tieR.to(device=u.device, dtype=u.dtype)
                    u.mul_(self._tieR.unsqueeze(1))
                p.add_(u, alpha=-lr_)
                del u
                if uM_ is not None:
                    p.addmm_(uM_, Qa_.t(), alpha=-lr_)            # the give-back, straight into the weight
                    del uM_
            us.clear(); chunk_.clear(); nb_[0] = 0
            del buf_

        for grp in list(opt.param_groups)[::-1]:
            lr, wd, eps = grp["lr"], grp["weight_decay"], grp["eps"]
            b1, b2 = grp["betas"]
            # decay p by (1 - lr wd) only when that can change p (for lr wd < 2^-25 it is p bit for bit)
            decay_ = not (lr * wd < 2.0 ** -25)
            for p in grp["params"]:
                if p.grad is None:
                    continue
                st = opt.state[p]
                if len(st) == 0:
                    st["step"] = torch.tensor(0.0)
                    st["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    st["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                st["step"] += 1
                t = float(st["step"])
                m, v = st["exp_avg"], st["exp_avg_sq"]
                if decay_:
                    p.mul_(1 - lr * wd)
                if lr == 0:
                    idle_.append((p, m, v, b1, b2))
                    continue
                key_ = (t, p.device)
                if key_ not in tdev_:
                    # torch.full fills on the device; torch.tensor(t, device=...) is a pageable copy + stream sync
                    tdev_[key_] = torch.full((), t, device=p.device, dtype=torch.float32)
                chunk_.append((p, p.grad, m, v, tdev_[key_], b1, b2, eps, lr))
                nb_[0] += p.numel() * 4
                if nb_[0] >= (1 << 30):
                    flush()
        flush()
        if idle_:
            # an unwritten step only moves the moments (multi-tensor, the same elementwise arithmetic)
            gs_ = [e[0].grad for e in idle_]
            torch._foreach_lerp_([e[1] for e in idle_], gs_, 1 - idle_[0][3])
            torch._foreach_mul_([e[2] for e in idle_], idle_[0][4])
            torch._foreach_addcmul_([e[2] for e in idle_], gs_, gs_, value=1 - idle_[0][4])
            for e in idle_:
                e[0].grad = None
        self._cmpd = {}
        # the comparator's rows and bases are consumed: free them now instead of at the next readout forward
        self._cmpX, self._cmpsel, self._cmpA = {}, None, {}

    # ------------------------------------------------------------------ co-movement (present class, class level)
    @torch.no_grad()
    def _cm_collect(self, x, tg, bi, pi, offs):
        """Per present (class, answer step): the share of its mean answer state's motion lying in each earlier memory's
        member space; the twin evidence (top - median) / top per slot (read by the joint-share target)."""
        self._cm_rowfac = None
        if self._cm is None:
            self._cm = {"S": None, "N": None, "m0": {}}
        V = self._cm
        if self._cmB is None:
            Q_ = self._rgQ.get("readout")
            if Q_ is None or self._j2p is None:
                return
            pr_ = self._j2p.tolist()                           # host list: no device->host sync per record
            H_ = (self._recH.to(x.device).float() - self._sc_c.to(x.device).float())
            B_ = []
            for j_ in range(H_.shape[0]):
                cols_ = [H_[j_:j_ + 1].t()]
                if pr_[j_] >= 0:
                    cols_.append(Q_[pr_[j_]].to(x.device).float())
                B_.append(torch.linalg.qr(torch.cat(cols_, 1)).Q)
            k_ = max(b_.shape[1] for b_ in B_)
            self._cmB = torch.stack([F.pad(b_, (0, k_ - b_.shape[1])) for b_ in B_])   # (R, d, k)
        cl_ = V.setdefault("cls", {})
        for t_ in tg:
            if str(t_) not in cl_:
                cl_[str(t_)] = len(cl_)
        n_ = len(cl_) * self._CMOFF
        if V["S"] is None or V["S"].shape[0] < n_:
            n2_ = (len(cl_) + 8) * self._CMOFF
            S2_ = torch.zeros(n2_, x.shape[-1], device=x.device); N2_ = torch.zeros(n2_, device=x.device)
            E2_ = torch.full((n2_,), 0.0, device=x.device)
            if V["S"] is not None:
                S2_[:V["S"].shape[0]] = V["S"]; N2_[:V["N"].shape[0]] = V["N"]; E2_[:V["E"].shape[0]] = V["E"]
            V["S"], V["N"], V["E"] = S2_, N2_, E2_
        ci_ = torch.tensor([cl_[str(t_)] for t_ in tg], dtype=torch.long).pin_memory().to(x.device, non_blocking=True)
        sl_ = ci_[bi] * self._CMOFF + offs.clamp(0, self._CMOFF - 1)
        self._cm_sl = sl_
        self._cm_rowfac = V["E"][sl_]
        V["S"].index_add_(0, sl_, x.detach()[bi, pi].float())
        V["N"].index_add_(0, sl_, torch.ones_like(sl_, dtype=torch.float32))
        if self.steps % 25 != 0:
            return
        offR = self._sc_off.to(x.device)
        D_ = {}
        nm_ = {v_: k_ for k_, v_ in V.get("cls", {}).items()}
        Nc_ = V["N"].tolist() if V["N"] is not None else []
        act_ = [((nm_[i_ // self._CMOFF], i_ % self._CMOFF), i_) for i_, c_ in enumerate(Nc_)
                if c_ != 0 and (i_ // self._CMOFF) in nm_]
        if act_:
            ix_ = torch.tensor([i_ for _, i_ in act_], device=x.device)
            mus_ = V["S"][ix_] / V["N"][ix_].unsqueeze(1)
            V["S"][ix_] = 0.0; V["N"][ix_] = 0.0
            for j_, (k_, i_) in enumerate(act_):
                if k_ not in V["m0"]:
                    V["m0"][k_] = mus_[j_].clone()
                    continue
                D_[k_] = mus_[j_] - V["m0"][k_]
        if D_:
            ks_ = list(D_)
            Dm_ = torch.stack([D_[k_] for k_ in ks_])
            dn_t = Dm_.norm(dim=1)
            selc_ = V.setdefault("sel", {})
            res_ = []
            for o_ in sorted({k_[1] for k_ in ks_}):
                if o_ not in selc_:
                    selc_[o_] = (offR == o_).nonzero(as_tuple=True)[0]
                sel_ = selc_[o_]
                if sel_.numel() == 0:
                    continue
                rw_ = [j_ for j_, k_ in enumerate(ks_) if k_[1] == o_]
                rwt_ = torch.tensor(rw_, device=x.device)
                sh_ = torch.einsum("md,sdk->msk", Dm_[rwt_], self._cmB[sel_]).pow(2).sum(2) \
                    / dn_t[rwt_].pow(2).clamp(min=1e-30).unsqueeze(1)
                tp_ = sh_.max(1).values
                exv_ = ((tp_ - sh_.median(1).values) / tp_.clamp(min=1e-12)).clamp(min=0.0)
                res_.append((rw_, sel_, sh_, exv_))
            if res_:
                hs_ = torch.cat([torch.stack([dn_t[torch.tensor(r_[0], device=x.device)], r_[3]], 1) for r_ in res_]).tolist()
                c_ = 0
                for rw_, sel_, sh_, exv_ in res_:
                    for t_, j_ in enumerate(rw_):
                        dnv_, exf_ = hs_[c_]; c_ += 1
                        if dnv_ == 0:
                            continue
                        V["E"][V["cls"][ks_[j_][0]] * self._CMOFF + ks_[j_][1]] = exf_

    # ------------------------------------------------------------------ conflict (readout forward hook)
    def _conflict_hook(self, mod, inp, out):
        if self.collect == "g2s":
            return self._g2s_conflict(inp)
        if self.collect == "recC":
            return None
        cxs_, self._cxs = self._cxs, {}
        self._cxa = {}
        if self.collect == "now":
            self._cmpX, self._cmpsel = {}, None
        x = inp[0] if isinstance(inp, tuple) else inp
        if not torch.is_tensor(x) or x.dim() != 3:
            return None
        B, Lq, d = x.shape
        if self.collect == "rec":
            # SLEEP: the finished model's answer states per (class id, answer step): count and sum
            sk = self._sk
            if sk is None or sk["shape"] != (B, Lq) or sk["n_ans"] <= 0:
                return None
            with torch.no_grad():
                # the answer rows gathered and cast once (rk is in (bi, pi) order): row j holds exactly the values
                # x[b, p].float() held
                H_ = x.detach()[sk["bi"], sk["pi"]].float()
                for j_, (b, p, k) in enumerate(sk["rk"]):
                    e = self.recs.get(k)
                    h = H_[j_]
                    if e is None:
                        self.recs[k] = [1.0, h.clone()]
                    else:
                        e[0] += 1.0; e[1] += h
            return None
        ck = self.L._mcache
        tg = self.L.cur_tg
        if ck is None or ck[0] != (B, Lq) or ck[4] <= 0:
            return None
        if self._sc_M is None or not torch.is_grad_enabled() or not x.requires_grad or tg is None:
            return None
        m = ck[1]
        lb = self.L.cur_labels.to(x.device)
        with torch.no_grad():
            if self._ansk is not ck:                                     # one nonzero per batch, shared
                self._ansk, self._ansix = ck, m.nonzero(as_tuple=True)
                self._ansfl = None
            bi, pi = self._ansix
            offs = pi - m.float().argmax(1)[bi]                              # answer positions are contiguous
            z = out.detach()[bi, pi].float()
            gold = lb[bi, pi + 1]
            zg = z.gather(1, gold.clamp(min=0).unsqueeze(1)).squeeze(1)
            z2 = z.clone(); z2.scatter_(1, gold.clamp(min=0).unsqueeze(1), float("-inf"))
            riv = z2.argmax(1)
            zr = z2.gather(1, riv.unsqueeze(1)).squeeze(1)                  # the top non-gold logit (its identity unread)
            # STATE CONFLICT: inside an earlier record's neighbourhood (centred cosine, same answer step)
            hs = F.normalize(x.detach()[bi, pi].float() - self._sc_c.to(x.device), dim=1)
            Sp = hs @ self._sc_M.to(x.device).t()
            same = offs.unsqueeze(1) == self._sc_off.to(x.device).unsqueeze(0)
            inside = (Sp > self._sc_nn.to(x.device).unsqueeze(0)) & same
            sc_ = inside.any(1)
            jn = (Sp - self._sc_nn.to(x.device).unsqueeze(0)).masked_fill(~same, -9.0).argmax(1)
            sc_nearest = self._rcnt.to(x.device)[jn]
            self._cm_collect(x, tg, bi, pi, offs)
            if self.steps % 10 == 0:
                self._scs.append(sc_.float().mean())                 # read on the host only when logged
            if len(self._scs) >= 20:
                logger.info("[HIPPO-LITE] last 200 steps (every 10th): answer steps in state conflict %.3f"
                            % (sum(float(v_) for v_ in self._scs) / len(self._scs)))
                self._scs = []
            seen = self.L._clsSeen or {}
            nn_ = torch.tensor([float(seen.get(str(t_).strip().lower(), 0)) for t_ in tg]).pin_memory() \
                .to(x.device, non_blocking=True)[bi]                     # no pageable copy (= no stream sync)
            # WRITE GATE: each conflicting answer step at its pair's joint share, the rest in full
            spe_ = self.L._spe
            prog_ = max(float(self.L.ep) + float(self.L._stepEp) / float(spe_ or 1), 1.0)
            nr_ = nn_ / prog_                                              # present class, per epoch
            rp = torch.zeros_like(zg)
            # no host test: with no conflicting row torch.where returns rp (zeros) unchanged
            rp = torch.where(sc_, sc_nearest / (sc_nearest + nr_).clamp(min=1e-6), rp)
            self.plast = (1.0 - rp).mean().clamp(0.0, 1.0)      # a device scalar: read after the backward, not mid-forward
            if out.requires_grad and self._cm_rowfac is not None and self._cm_rowfac.shape[0] == bi.shape[0]:
                # JOINT SHARE TARGET: (1 - a) gold + a q at parting steps inside the answer, a = co-movement x share
                a_ = (self._cm_rowfac.float() * rp.float()).clamp(0.0, 1.0)
                a_ = a_ * (offs > 0).float()
                sl_ = self._cm_sl
                n_ = self._cm["N"].shape[0]
                F_ = self._cm.get("jsf")
                if F_ is None or F_.shape[0] < n_:
                    F2_ = torch.zeros(n_ + 64, device=a_.device)
                    if F_ is not None:
                        F2_[:F_.shape[0]] = F_
                    F_ = self._cm["jsf"] = F2_
                F_.index_add_(0, sl_, (sc_ & (zg <= zr)).float())
                a_ = a_ * (F_[sl_] > 0).float()
                gd_ = gold.clamp(min=0)
                dj_ = torch.softmax(z, dim=1)
                pg_ = dj_.gather(1, gd_.unsqueeze(1)).squeeze(1)
                dj_.mul_((-a_.float() / (1.0 - pg_).clamp(min=1e-6)).unsqueeze(1))
                dj_.scatter_(1, gd_.unsqueeze(1), a_.float().unsqueeze(1))
                okj_ = (pg_ < 1.0 - 1e-4) & (a_ > 0)

                def _js(g, bj_=bi, pj_=pi, dj_=dj_, gd_=gd_, pg_=pg_, okj_=okj_):
                    gg_ = g[bj_, pj_].float()
                    s_ = (gg_.gather(1, gd_.unsqueeze(1)).squeeze(1) / (pg_ - 1.0).clamp(max=-1e-4))
                    s_ = torch.where(okj_, s_.clamp(min=0.0), torch.zeros_like(s_))
                    g.index_put_((bj_, pj_), (gg_ + s_.unsqueeze(1) * dj_).to(g.dtype))
                    return g
                out.register_hook(_js)
            if self._cmpB and x.requires_grad:
                # CA1 COMPARATOR: inside a memory's neighbourhood while gold is not yet above the top non-gold logit
                pr = self._j2p.to(x.device)[jn]
                ok = sc_ & (pr >= 0) & (zg <= zr)
                sel_ = ok.nonzero(as_tuple=True)[0]                     # one host sync
                if sel_.numel():
                    self._cmp_arm(bi[sel_], pi[sel_], cxs_, mod, x, out, sel_)
        return None

    def _g2s_conflict(self, inp):
        """Second sleep read: per answer row the state-conflict flag; per area the energy along its new keys, all rows
        and conflict rows, summed."""
        sk2_ = self._sk2
        x = inp[0] if isinstance(inp, tuple) else inp
        if sk2_ is None or not torch.is_tensor(x) or x.dim() != 3 or tuple(x.shape[:2]) != sk2_["shape"]:
            return None
        with torch.no_grad():
            bi, pi = sk2_["bi"], sk2_["pi"]
            offs = pi - sk2_["ma"].float().argmax(1)[bi]
            hs = torch.nn.functional.normalize(x.detach()[bi, pi].float() - self._sc_c.to(x.device), dim=1)
            Sp = hs @ self._sc_M.to(x.device).t()
            same = offs.unsqueeze(1) == self._sc_off.to(x.device).unsqueeze(0)
            nn_ = self._sc_nn.to(x.device).unsqueeze(0)
            sc_ = ((Sp > nn_) & same).any(1)
            jn = (Sp - nn_).masked_fill(~same, -9.0).argmax(1)
            wc_ = (sc_ & (self._j2p.to(x.device)[jn] >= 0)).float()
            self._g2st.append(float(wc_.mean()))
            for area, P in self._g2P.items():
                if P.shape[0] != wc_.shape[0]:
                    continue
                e_ = self._g2E.get(area)
                if e_ is None:
                    e_ = self._g2E[area] = [torch.zeros(P.shape[1], device=P.device), torch.zeros(P.shape[1], device=P.device)]
                e_[0] += P.sum(0)
                e_[1] += (P * wc_.unsqueeze(1)).sum(0)
            self._g2P = {}
        return None

    @torch.no_grad()
    def _recall_build(self):
        """Every earlier record's state neighbourhood: centred unit mean states; per record, its similarity to the
        nearest record of another class at the same answer step."""
        self._sc_M = None
        if not self.recs:
            return
        dev = self.first["readout"].weight.device
        keys = list(self.recs)
        H = torch.stack([self.recs[k][1] / self.recs[k][0] for k in keys]).to(dev)
        self._rcnt = torch.tensor([float(self.recs[k][0]) for k in keys], device=dev)
        self._sc_c = H.mean(0, keepdim=True)
        Mn = F.normalize(H - self._sc_c, dim=1)
        offs = torch.tensor([k[1] for k in keys], device=Mn.device)
        cls_ = torch.tensor([k[0] for k in keys], device=Mn.device)
        S_ = Mn @ Mn.t()
        bad = (offs.unsqueeze(0) != offs.unsqueeze(1)) | (cls_.unsqueeze(0) == cls_.unsqueeze(1))
        S_ = S_.masked_fill(bad, -2.0)
        self._sc_M, self._sc_off = Mn, offs
        self._recH = H
        self._sc_nn = S_.max(1).values.clamp(min=-1.0)

    # ------------------------------------------------------------------ task boundaries
    def begin_task(self, task_id):
        self._cm, self._cmB = None, None
        self._tid = task_id
        self.now, self.T, self.dD = {}, {}, {}
        self._tmdm = None
        self.steps, self.next_build = 0, 1
        self._pha = {}
        self.collect = "now" if self.eig else None
        self._recall_build()
        self._cmp_build()

    # ------------------------------------------------------------------ CA1 comparator
    @torch.no_grad()
    def _pat_rec(self, area, x):
        """SLEEP: the pattern this area receives at every answer step, summed per (class id, answer step); at the
        readout also a 16-column spread sketch per memory."""
        sk = self._sk
        if sk["n_ans"] <= 0:
            return
        bi, pi, rows = sk["bi"], sk["pi"], sk["rows"]
        n = len(self._pkey)
        P = self._pat.get(area)
        if P is None or P.shape[0] < n:
            P2 = torch.zeros(n + 64, x.shape[-1], device=x.device, dtype=torch.float32)
            if P is not None:
                P2[:P.shape[0]] = P
            P = self._pat[area] = P2
        Xr = x.detach()[bi, pi].float()
        P.index_add_(0, rows, Xr)
        if area == "readout":
            Om = self._pom.get(area)
            if Om is None:
                g_ = torch.Generator(device="cpu").manual_seed(1234 + Xr.shape[1])
                Om = self._pom[area] = (torch.randn(Xr.shape[1], 16, generator=g_) / Xr.shape[1] ** 0.5).to(x.device)
            Y = self._psk.get(area)
            if Y is not None and Y.device != x.device:
                Y = self._psk[area] = Y.to(x.device)
            if Y is None or Y.shape[0] < P.shape[0]:
                Y2 = torch.zeros(P.shape[0], Xr.shape[1], 16, device=x.device, dtype=torch.float32)
                if Y is not None:
                    Y2[:Y.shape[0]] = Y
                Y = self._psk[area] = Y2
            Z_ = Xr @ Om
            Y.index_add_(0, rows, Xr.unsqueeze(2) * Z_.unsqueeze(1))

    @torch.no_grad()
    def _cmp_build(self):
        """Task start: per area, an orthonormal basis of all recorded memories' mean patterns (readout: also each
        memory's spread basis); each recalled record's pattern row."""
        self._cmpB, self._j2p = {}, None
        if not self._pkey or not self.recs:
            return
        n = len(self._pkey)
        cnt = self._pc_n[:n].clamp(min=1).unsqueeze(1)
        for area, P in self._pat.items():
            M = P[:n] / cnt
            if area in self._psk:
                Y_ = self._psk[area][:n].to(M.device)
                Yc_ = Y_ - (cnt.unsqueeze(2) * M.unsqueeze(2)) * (M @ self._pom[area].to(M.device)).unsqueeze(1)
                self._rgQ[area] = torch.linalg.qr(Yc_).Q.to(torch.bfloat16)
                del Y_, Yc_
            Q, R = torch.linalg.qr(M.t())
            dg = R.diagonal().abs()
            keep = dg > 1e-4 * dg.max()
            self._cmpB[area] = Q[:, keep].contiguous()
        keys = list(self.recs)
        self._j2p = torch.tensor([self._pkey.get(k, -1) for k in keys])
        # the pattern sums wait on the host until the next sleep adds to them
        self._pat = {a_: P_.cpu() for a_, P_ in self._pat.items()}

    def _make_cmp_hook(self, area):
        def hook(mod, inp, out):
            if self.collect != "now" or area not in self._cmpB or not torch.is_tensor(out) \
                    or not out.requires_grad:
                return None
            if torch._C._current_graph_task_id() != -1:
                # (gradient checkpointing re-runs the layer inside the backward: the first pass already holds this
                # step's rows and gradient hooks; a second capture would leak into the next batch)
                return None
            x = inp[0] if isinstance(inp, tuple) else inp
            ck = self.L._mcache
            if ck is None or ck[0] != tuple(x.shape[:2]):
                return None
            if self._ansk is not ck:
                self._ansk, self._ansix = ck, ck[1].nonzero(as_tuple=True)
                self._ansfl = None
            ba, pa = self._ansix
            xa_ = self._cxa.get(area)
            if xa_ is None:
                if self._ansfl is None:
                    self._ansfl = ba * x.shape[1] + pa
                # the same rows as x[ba, pa], gathered by one index_select (cheaper than 2-D advanced indexing)
                xa_ = x.detach().reshape(-1, x.shape[-1]).index_select(0, self._ansfl)
                self._cxa[area] = xa_
            for m_ in self.areas[area]:                                # (the area's groups share this input)
                self._cxs[id(m_)] = (m_, area, xa_)
            return None
        return hook

    def _cmp_arm(self, b, p, cxs, head, xh, zh, sel):
        """Each conflicting answer step's input at every area and its mismatch with all memories' patterns; the
        learning signal at the area's output is caught in the backward pass."""
        self._cmpsel = (b, p)
        done_ = {}
        for mod, area, xin in list(cxs.values()) + [(head, "readout", None)]:
            B_ = self._cmpB.get(area)
            if B_ is None:
                continue
            if area in done_:
                Xp = done_[area]
                self._cmpX[id(mod)] = (None, Xp, area)
                continue
            with torch.no_grad():
                X = (xh.detach()[b, p] if xin is None else xin[sel]).float()
                Xp = X - (X @ B_) @ B_.t()
            del X                                                         # only Xp is kept until the step
            done_[area] = Xp
            self._cmpX[id(mod)] = (None, Xp, area)

    # ------------------------------------------------------------------ sleep: write the memory
    @torch.no_grad()
    def _write(self, area, C, n):
        """Delta-rule write of a linear area's subspace memory (C None: deferred write after the second sleep read)."""
        kc_ = self._kc.pop(area, None)
        if kc_ is None:
            kc_ = self._keys(C, n)
        kc_ = {k_: (v_.to(kc_["dev"]) if torch.is_tensor(v_) else v_) for k_, v_ in kc_.items()}
        dev, K, lk, floor, tr_e, nd_ = kc_["dev"], kc_["K"], kc_["lk"], kc_["floor"], kc_["tr"], kc_["nd"]
        Ce = kc_.get("Ce") if C is not None else None             # the moment _keys already divided
        m = self._mem.get(area)
        if m is None or m["U"].shape[1] == 0:
            U, lam, nn = K, lk, float(n)
            beta = 1.0
        else:
            beta = float(n) / (m["n"] + float(n))
            Uo = m["U"].to(dev)
            lo = m["lam"].to(dev)
            B = kc_.get("B")                                             # the basis _keys made from the same
            if B is None:                                                # [Uo, K]: bit for bit the same QR
                B = torch.linalg.qr(torch.cat([Uo, K], 1)).Q             # (d, r + k)
            Ab = B.t() @ Uo
            S = (Ab * lo) @ Ab.t()                                       # old memory in B
            Kb = B.t() @ K
            if area in self._g2E:
                # second sleep read: along k_i erase by beta (1 - rho_i), rho_i = conflict rows' share of the answer
                # rows' energy along k_i
                ea_, ec_ = self._g2E.pop(area)
                rho_ = (ec_.to(dev) / ea_.to(dev).clamp(min=1e-30)).clamp(0.0, 1.0)
                self._cest.append(float(rho_.mean()))
                er_ = beta * (1.0 - rho_)
            else:
                er_ = beta * torch.ones(Kb.shape[1], device=dev)
            Kw_ = Kb * (1.0 - (1.0 - er_).clamp(min=0.0).sqrt()).unsqueeze(0)
            G_ = torch.eye(Kb.shape[0], device=dev) - Kw_ @ Kb.t()
            S = G_ @ S @ G_.t()
            S = S + beta * ((Kb * lk) @ Kb.t())
            S = 0.5 * (S + S.t())
            lw, W = _eigh(S)
            keep = lw > floor
            U, lam = B @ W[:, keep], lw[keep]
            nn = m["n"] + float(n)
        # TAIL: the episode's own per-position energy outside the new span U, running mean over episodes
        Uall = U
        if Ce is not None:
            tail_e = max(tr_e - float(((Ce @ Uall) * Uall).sum()), 0.0)
        else:
            Ua_ = B.t() @ Uall                                           # (the new span lies inside B)
            tail_e = max(tr_e - float(((kc_["Ceb"] @ Ua_) * Ua_).sum()), 0.0)
        tail = tail_e if m is None else (1.0 - beta) * m["tail"] + beta * tail_e
        self._mem[area] = {"U": U.cpu(), "Ub": _pad64(U.to(torch.bfloat16)), "lam": lam.cpu(), "n": nn,
                           "tail": tail, "floor": float(floor), "nlast": float(n)}
        return U.shape[1], nd_

    @torch.no_grad()
    def _keys(self, C, n, area=None):
        """The episode's new keys: the Gavish-Donoho top of its per-position moment, their energies and its noise floor;
        for a deferred write also the merge basis B = [U_old, K] and the episode's moment inside it."""
        dev = C.device
        # C is the caller's own moment (popped from self.rec and dropped after this call): divided in place,
        # saving one d x d fp32 buffer; a non-deferred write reads Ce from the result
        Ce = C.div_(max(n, 1.0)) if C.dtype == torch.float32 else (C / max(n, 1.0)).float()
        ev, V = _eigh(Ce)
        ev = ev.clamp(min=0)
        k = _gd_rank(ev, n)
        sv = ev.sqrt().flip(0)
        D = float(ev.shape[0])
        bt = min(float(n), D) / max(float(n), D)
        tau = (0.56 * bt ** 3 - 0.95 * bt ** 2 + 1.82 * bt + 1.43) * float(sv[:int(min(n, D))].median())
        floor = tau * tau                                          # the episode's noise floor, in energy units
        out = {"dev": dev, "K": V[:, ev.shape[0] - k:].contiguous(), "lk": ev[ev.shape[0] - k:].clone(), "k": k,
               "floor": floor, "tr": float(ev.sum()), "nd": int(ev.shape[0])}
        m = self._mem.get(area) if area is not None else None
        if m is not None and m["U"].shape[1] > 0:
            Uo = m["U"].to(dev)
            B = torch.linalg.qr(torch.cat([Uo, out["K"]], 1)).Q
            out["Ceb"] = B.t() @ Ce @ B
            out["B"] = B.cpu()                     # _write reuses it instead of a second identical QR
            del B
        else:
            out["Ce"] = Ce
        return out

    @torch.no_grad()
    def _g2s_pass(self, loader, model):
        """Second sleep read: the episode just learnt is read once more (global RNG state saved and restored around the
        loader iteration); per area only (k,) sums along the new keys are kept."""
        import random as _rd
        import numpy as _np
        was = model.training
        model.eval()
        self.L._mcache = None
        rs_ = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), _rd.getstate(), _np.random.get_state())
        self._sk, self._g2E, self._g2st = None, {}, []
        for kc_ in self._kc.values():
            # only the aligned copy the hook multiplies with lives on the device during this read
            kc_["Kp"] = _pad64(kc_["K"].to(kc_["dev"]))
        self.collect = "g2s"
        try:
            for batch in loader:
                ids = batch["input_ids_with_ans"]
                am = batch["attention_mask_with_ans"]
                lb = batch["labels_with_ans"]
                B, Lq = lb.shape
                ma = torch.zeros((B, Lq), dtype=torch.bool, device=lb.device)
                ma[:, :-1] = lb[:, 1:] != -100
                ix_ans = ma.reshape(-1).nonzero(as_tuple=True)[0]
                if ix_ans.numel() == 0:
                    continue
                bi, pi = ma.nonzero(as_tuple=True)
                self._sk2 = {"shape": (B, Lq), "ix_ans": ix_ans, "bi": bi, "pi": pi, "ma": ma}
                self._g2P = {}
                model(input_ids=ids, attention_mask=am, use_cache=False, return_dict=True, output_hidden_states=False)
        finally:
            self.collect = None
            self._sk2, self._g2P = None, {}
            for kc_ in self._kc.values():
                kc_.pop("Kp", None)
            torch.set_rng_state(rs_[0]); torch.cuda.set_rng_state_all(rs_[1]); _rd.setstate(rs_[2]); _np.random.set_state(rs_[3])
            if was:
                model.train()

    def _groups(self, model):
        """Trunk areas split into consecutive groups whose d x d moments fit below the finished task's training peak
        (the sleep never needs more memory than learning did)."""
        areas = [a for a in self.areas if a != "readout"]
        need = {a: 4 * self.first[a].weight.shape[1] ** 2 for a in areas}
        free = None
        if torch.cuda.is_available() and self.train_peak:
            free = 0.5 * (self.train_peak - torch.cuda.memory_allocated())
        if free is None or free >= sum(need.values()):
            return [set(areas)]
        free = max(free, max(need.values()))
        out, cur, used = [], set(), 0
        for a in areas:
            if cur and used + need[a] > free:
                out.append(cur); cur, used = set(), 0
            cur.add(a); used += need[a]
        if cur:
            out.append(cur)
        return out

    def _recC_pass(self, loader, model):
        """A later read of the episode (same batches, same order: the loader's RNG draw is replayed) that accumulates
        the trunk moments of the present group only."""
        self.collect = "recC"
        for batch in loader:
            ids = batch["input_ids_with_ans"]
            am = batch["attention_mask_with_ans"]
            lb = batch["labels_with_ans"]
            B, Lq = lb.shape
            ma = torch.zeros((B, Lq), dtype=torch.bool, device=lb.device)
            ma[:, :-1] = lb[:, 1:] != -100
            mt = am.bool()
            ix_all = mt.reshape(-1).nonzero(as_tuple=True)[0]
            ix_ans = ma.reshape(-1).nonzero(as_tuple=True)[0]
            self._sk = {"shape": (B, Lq), "ix_all": ix_all, "ix_ans": ix_ans, "n_all": int(ix_all.numel()),
                        "n_ans": int(ix_ans.numel())}
            model(input_ids=ids, attention_mask=am, use_cache=False, return_dict=True, output_hidden_states=False)
        self.collect = None
        self._sk = None

    @torch.no_grad()
    def sleep(self, task_id, loader, model):
        """The finished model re-reads the episode just learnt (no learning, no graph; the loader iteration draws the
        global RNG) and the index is written."""
        import random as _rd
        import numpy as _np
        was = model.training
        model.eval()
        self.L._mcache = None
        self.T = {}
        self._tmdm = None
        self.n_eps += 1
        self.rec = {}
        groups = self._groups(model)
        self._grp = groups[0]
        rs0_ = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), _rd.getstate(), _np.random.get_state())
        if self._pat:
            self._pat = {a_: P_.to(self.first["readout"].weight.device) for a_, P_ in self._pat.items()}
        self.collect = "rec"
        for batch in loader:
            ids = batch["input_ids_with_ans"]
            self.L.cur_attn = batch["attention_mask_with_ans"]
            self.L.cur_labels = batch["labels_with_ans"]
            self.L.cur_tg = [str(t).strip().lower() for t in batch["target"]]
            for t_ in self.L.cur_tg:
                self.cids.setdefault(t_, self.next_cid + len(self.cids))
            self._sleep_batch(ids, self.L.cur_attn, self.L.cur_labels, self.L.cur_tg)
            model(input_ids=ids, attention_mask=self.L.cur_attn, use_cache=False, return_dict=True,
                  output_hidden_states=False)
        self.collect = None
        self._sk = None
        self.next_cid += len(self.cids)
        self.cids = {}
        ranks, dims = [], []
        g2s_ = self._sc_M is not None and self._j2p is not None
        g2d_ = []
        rs1_ = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), _rd.getstate(), _np.random.get_state())
        for gi_ in range(len(groups)):
          if gi_ > 0:
            torch.cuda.empty_cache()
            torch.set_rng_state(rs0_[0]); torch.cuda.set_rng_state_all(rs0_[1]); _rd.setstate(rs0_[2]); _np.random.set_state(rs0_[3])
            self._grp = groups[gi_]
            self._recC_pass(loader, model)
          for area in list(self.rec):
              C, n = self.rec.pop(area)
              if area == "tok" or area.startswith("g:"):
                  e = self.old.get(area)
                  if e is None:
                      self.old[area] = [C, n]
                  else:
                      e[0] += C
                      e[1] += n
                  ev = self.old[area][0] / max(self.old[area][1], 1.0)
                  srt = torch.sort(ev[ev > 0]).values
                  kp = _gd_rank(srt, self.old[area][1]) if srt.numel() else 0
                  own = torch.zeros_like(ev, dtype=torch.bool)
                  if kp > 0:
                      own[torch.topk(ev, kp).indices] = True
                  self.eig[area] = (ev, None, own)
                  if area == "tok":
                      # (tied table) the per-row noise floor is the GD cut of the rows' earlier use (the smallest owned
                      # one), and the earlier samples' weight is all of them over the episode just written
                      fl_ = float(srt[-kp]) if kp > 0 else (float(srt[-1]) if srt.numel() else 1.0)
                      self._gfm[area] = (max(fl_, 1e-30), float(self.old[area][1]) / max(float(n), 1.0))
                  continue
              if g2s_ and area in self._mem and self._mem[area]["U"].shape[1] > 0:
                  kc_ = self._keys(C, n, area)
                  if len(groups) > 1:
                      kc_ = {k_: (v_.cpu() if torch.is_tensor(v_) else v_) for k_, v_ in kc_.items()}
                  self._kc[area] = kc_
                  g2d_.append((area, n))
                  del C
                  continue
              r, d = self._write(area, C, n)
              del C
              self.eig[area] = ("subspace",)
              ranks.append(r)
              dims.append(d)
          self.rec = {}
        torch.set_rng_state(rs1_[0]); torch.cuda.set_rng_state_all(rs1_[1]); _rd.setstate(rs1_[2]); _np.random.set_state(rs1_[3])
        self._grp = set()
        if g2d_:
            torch.cuda.empty_cache()
            self._g2s_pass(loader, model)
            for area, n in g2d_:
                r, d = self._write(area, None, n)
                self.eig[area] = ("subspace",)
                ranks.append(r)
                dims.append(d)
            logger.info("[HIPPO-LITE] after task %d: answer rows in conflict at the sleep read %.3f | deferred areas %d | "
                        "new keys' conflict share rho mean %.3f"
                        % (int(task_id), sum(self._g2st) / max(len(self._g2st), 1), len(g2d_),
                           sum(self._cest) / max(len(self._cest), 1)))
            self._kc, self._g2E = {}, {}
        self._cest = []
        torch.cuda.empty_cache()
        if ranks:
            mb = sum(m["U"].numel() for m in self._mem.values()) * 6 / 2 ** 30
            tr_ = [(r, d) for (r, d) in zip(ranks, dims)]
            logger.info("[HIPPO-LITE] after task %d: %d areas written | rank / dim mean %.3f (rank %.1f of %.0f dims) | "
                        "readout owned %s | subspace memory %.2f GB (fp32 host + bf16 device)"
                        % (int(task_id), len(ranks), sum(r / d for r, d in tr_) / len(tr_),
                           sum(r for r, _ in tr_) / len(tr_), sum(d for _, d in tr_) / len(tr_),
                           ("%d of %d" % tuple(self._mem["readout"]["U"].shape[::-1])) if "readout" in self._mem else "-", mb))
        if was:
            model.train()

    def rest(self):
        """After the last task: nothing is learnt any more, so no sleep follows; the wake state is dropped as the
        sleep would drop it (evaluation then runs with the hooks detached)."""
        self.collect = None
        self.L._mcache = None
        self.T, self.now = {}, {}
        self._tmdm = None

    # ------------------------------------------------------------------ wake: build the operators
    @torch.no_grad()
    def _build(self):
        self._tmdm = None
        no = float(self.n_eps)
        for area, ent in self.eig.items():
            e = self.now.get(area)
            if ent[0] is not None and torch.is_tensor(ent[0]) and ent[1] is None:
                # gains / input table: per-coordinate novelty share, 0 on the owned coordinates
                ev, _, kp = ent
                ln = (e[0] / max(e[1], 1.0)) if e is not None else torch.zeros_like(ev)
                ev = ev * no
                ref_ = ev / no if no > 0 else ev
                D = torch.where(ln > 0, ((ln - ref_) / ln.clamp(min=1e-30)).clamp(0.0, 1.0), torch.ones_like(ln))
                D.masked_fill_(kp, 0.0)                                   # no nonzero / sync
                self.dD[area] = D
                continue
            # linear area (trunk or readout): the minimum-interference operator from its memory
            m = self._mem.get(area)
            if m is None:
                continue
            if self.orthobasis:
                # (hippo_orthobasis) the held span and the pattern-hold columns as ONE orthonormal factorisation in fp64:
                # B = QR([U, P]) with U first (span(U) kept; U's columns orthonormalised in their order, so each keeps its
                # share). A pattern column is kept where |R_ii| exceeds the numerical-rank tolerance max(d, n) eps_fp32
                # |column|. A single projection P - U(U^T P) leaves cancellation noise along U when the patterns lie
                # almost entirely inside U; the operator is then not a contraction. Stored as the area's one bf16 copy.
                Ua_ = self._pha.get(area)
                if Ua_ is None:
                    dv_ = self.first["readout"].weight.device
                    U_ = m["U"].to(dv_).double()
                    r0_ = U_.shape[1]
                    has_p_ = area in self._cmpB and self._cmpB[area].shape[1] > 0
                    M_ = torch.cat([U_, self._cmpB[area].to(dv_).double()], 1) if has_p_ else U_
                    Q_, R_ = torch.linalg.qr(M_)
                    tol_ = max(M_.shape[0], M_.shape[1]) * float(torch.finfo(torch.float32).eps)
                    k_ = R_.diagonal().shape[0]                 # min(d, r + n): with more columns than d, only d survive
                    kp_ = R_.diagonal().abs()[r0_:] > tol_ * M_.norm(dim=0)[r0_:k_]
                    Ua_ = self._pha[area] = _pad64(torch.cat([Q_[:, :r0_], Q_[:, r0_:][:, kp_]], 1).to(torch.bfloat16))
                    m["Ub"] = None                              # (one bf16 basis per area: U is its first r columns)
                    del U_, M_, Q_, R_
                Ub_ = Ua_
            elif area in self._cmpB and self._cmpB[area].shape[1] > 0:
                # PATTERN HOLD: the held span also covers the earlier memories' class-level mean patterns -- the part of
                # an earlier class's mean input outside U would otherwise move with the tail's share (a counterpart
                # class's row then grows on the earlier class's states). Built once per task.
                Ua_ = self._pha.get(area)
                if Ua_ is None:
                    U_ = m["U"].to(self._cmpB[area].device).float()
                    Pb_ = self._cmpB[area].float()
                    Pr_ = Pb_ - U_ @ (U_.t() @ Pb_)                            # the patterns' part outside U
                    Qr_, Rr_ = torch.linalg.qr(Pr_)
                    dg_ = Rr_.diagonal().abs()
                    Qr_ = Qr_[:, dg_ > 1e-3 * max(float(dg_.max()), 1e-30)]
                    Ua_ = self._pha[area] = _pad64(torch.cat([U_, Qr_], 1).to(torch.bfloat16))
                    m["Ub"] = None                              # (one bf16 basis per area: U is its first r columns)
                Ub_ = Ua_
            else:
                Ub_ = m["Ub"]
            # MINIMUM-INTERFERENCE STEP. The step minimises <G, dW> + (1/2eta)(|dW|^2 + mu/f tr(dW C dW^T)): the new
            # loss, the step's size and the earlier tasks' expected output change E_old |dW x|^2 = tr(dW C dW^T),
            # weighed by the earlier samples' share mu = n_old / n_new and measured against the area's noise floor f.
            # Closed form dW = -eta G (I + mu C / f)^-1; with C = U diag(lam) U^T + tau (I - U U^T) every held
            # direction keeps the share f / (f + mu lam_i), the tail f / (f + mu tau); the class patterns (the basis's
            # columns after the first r) take share 0 and stay held.
            r_ = m["U"].shape[1]
            d_ = m["U"].shape[0]
            mu_ = m["n"] / max(m["nlast"], 1.0)
            f_ = max(m["floor"], 1e-30)
            lam_ = m["lam"].to(Ub_.device).float()
            tau_ = m["tail"] / max(d_ - r_, 1)
            av_ = torch.zeros(Ub_.shape[1], device=Ub_.device)
            av_[:r_] = f_ / (f_ + mu_ * lam_)
            self.T[area] = _LowT(Ub_, av_, float(f_ / (f_ + mu_ * tau_)))

    @torch.no_grad()
    def apply(self):
        """After the backward (every held gradient already formed from held inputs): the gains' shares, and the
        comparator's bases for the step's give-back."""
        self._xt = {}
        self._xbc = None
        if not self.eig:
            return
        gs_, Ds_ = [], []
        for name, mod in self.diag.items():
            D = self.dD.get("g:" + name)
            g = mod.weight.grad
            if D is not None and g is not None:
                if g.dtype == torch.float32:
                    gs_.append(g); Ds_.append(D.to(g.device))
                else:
                    g.copy_((g.float() * D.to(g.device)).to(g.dtype))
        if gs_:
            torch._foreach_mul_(gs_, Ds_)                                 # fp32 g * D, written in place
        if not self.T or self._cmpsel is None or not self._cmpX:
            return
        xp_ = {}
        for area, mods in self.areas.items():
            for mod in mods:
                c_ = self._cmpX.get(id(mod))
                if c_ is None or mod.weight.grad is None:
                    continue
                X_, Xp_, ar_ = c_
                xp_[ar_] = Xp_
                self._cmpd[id(mod.weight)] = (ar_,)
        self._cmpA = {}
        if xp_:
            ars = list(xp_)
            with torch.autocast("cuda", enabled=False):
                Gs = torch.stack([xp_[a].float() @ xp_[a].float().t() for a in ars])   # (areas, n, n)
                ev_, V_ = _eigh(Gs)
                W_ = V_ * ((ev_ > 1e-6 * ev_[:, -1:]).float() / ev_.clamp(min=1e-20).sqrt()).unsqueeze(1)
                for i_, a in enumerate(ars):
                    Q_ = xp_[a].t() @ W_[i_]                                  # (d, n) orthonormal (or zero)
                    T_ = self.T.get(a)
                    M_ = Q_ - self._tmm(Q_.t(), T_).t() if T_ is not None else Q_
                    self._cmpA[a] = (Q_, M_)                                  # (I - T) Q: what the index removes
