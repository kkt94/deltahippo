"""HippoEnc: the DeltaHippo hippocampus of utils/hippo_lite.py with model-agnostic area detection, for encoders.

The same method as utils/hippo_lite.py (which holds the decoder code path). What differs:
  * AREAS are found by a probe forward, not by module names: an area is the set of nn.Linear modules (the readout
    excluded) that read the SAME input tensor object in one forward pass (Qwen3 / Llama: q/k/v, o, gate/up, down,
    the same grouping as the decoder name rule; BERT / RoBERTa: query/key/value, attention.output.dense, intermediate.dense,
    output.dense). Ordered by named_modules (layer order).
  * GAINS: every normalisation module (class name ending in "Norm": RMSNorm, LayerNorm, ...) with a 1-D weight that
    reads an input in the probe. LayerNorm's gain reads the CENTRED normalised input, RMSNorm's the uncentred one.
  * BIASES. Linear: b is the weight on a constant input 1. The held operator T = c (I - U U^T) acts on directions of
    the area's input; the constant input is not a direction of x, so it is treated as lying outside the held span U:
    its share is the area's tail share c. The bias gradient (db = c sum_t dy_t) and its Adam step are both scaled by
    c -- the same two places T acts on the weight. Norm biases: the gain's per-coordinate share D (gradient and step).
  * TABLES: every nn.Embedding that is read in the probe (token table "tok", absolute positions, token types) is a
    per-row area, counted over attended positions, with the token table's rule (share vector on the gradient).
  * READOUT: a list of linear maps reading the last hidden sequence (encoder: the per-task classifier heads; their
    concatenation is the readout forward that the conflict hook reads), and the comparator's give-back reaches every
    readout map.

Areas: linear modules of a layer that read the same input (q/k/v; gate/up; o; down) and the readout (the tied table);
norm gains ("g:<name>") and the input table ("tok") are per-coordinate areas. Every linear area is treated alike; its
moment is taken over the positions where it receives a learning signal (trunk: all positions, readout: answer steps).

Sleep (end of every task; no learning, no graph): the finished model re-reads the task's training set.
  * every linear area: gated-delta subspace memory (U, lam, tr, n, tail). New keys K = Gavish-Donoho top of the
    episode's moment; in B = [U, K] the old content along each key is erased by beta (1 - rho_i), rho_i the share of the
    answer rows' energy along k_i coming from rows in state conflict with an earlier record (a second read, RNG state
    replayed); the episode's energy is written by beta; re-diagonalised above the episode's noise floor. Tail = running
    mean of each episode's energy outside the span. The d x d moments are taken group by group below the training peak.
  * gains / input table: accumulated per-coordinate energy, GD owned coordinates.
  * class-level answer records per (class id, answer step) and every area's mean answer-step input per memory.
Wake (task >= 1): operators rebuilt at steps 1, 2, 4, ...: T = c (I - U U^T) (c = tail novelty share; U also spans the
earlier memories' class-level mean patterns: PATTERN HOLD), gains/table by share vectors. The gradient is formed from the HELD INPUT (dW = dy^T (x T), the comparator's answer rows entering as
their mismatch Xp); Adam (fused kernel) steps through T again (its elementwise normalisation re-rotates the held
gradient). The readout hook detects state conflicts (answer state inside an earlier record's neighbourhood), sets the
write gate, arms the CA1 comparator (give-back along the mismatch with all memories' mean patterns) and the joint-share
target at parting steps.
"""
import math
import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger()

# the decoder name rule of utils/hippo_lite.py, for reference (the probe rule reproduces it)
_GROUP_OLD_REF = {"q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv", "gate_proj": "gu", "up_proj": "gu",
                  "o_proj": "o", "down_proj": "down"}


def _kind(name):
    """Diagnostic label (not used by any rule): the kind of a linear area from its name (BERT / RoBERTa, Qwen / Llama)."""
    if name == "readout":
        return "readout"
    if "query" in name or "q_proj" in name:
        return "attn-qkv"
    if "attention.output" in name or "o_proj" in name:
        return "attn-out"
    if "intermediate" in name or "gate_proj" in name:
        return "ffn-in"
    if "output.dense" in name or "down_proj" in name:
        return "ffn-out"
    return "other"


def _is_norm(mod):
    """A normalisation module with a per-coordinate gain (RMSNorm, LayerNorm, ...): type-based, case-insensitive."""
    w = getattr(mod, "weight", None)
    return type(mod).__name__.lower().endswith("norm") and isinstance(w, torch.nn.Parameter) and w.dim() == 1


@torch.no_grad()
def detect_areas(model, readout_mods, probe):
    """One probe forward (probe() runs the model on a short input) with forward pre-hooks on every nn.Linear,
    normalisation module and nn.Embedding. Returns
      areas  {name: [linear modules]} -- linear maps (readout excluded) reading the same input tensor object, in
             named_modules order; name = the first member's name + '+' the other members' leaf names
      gains  {name: norm module} read in the probe (named_modules order)
      tables {name: embedding module} read in the probe (named_modules order)
    Every input tensor is kept alive until the grouping is done, so no id can be reused by a later tensor. The probe
    runs in eval mode (no dropout draw) and the RNG states are restored around it."""
    import random as _rd
    import numpy as _np
    ro = {id(m) for m in readout_mods}
    seen, keep, hs, outs = {}, [], [], {}
    for name, mod in model.named_modules():
        if (isinstance(mod, torch.nn.Linear) and id(mod) not in ro) or _is_norm(mod) \
                or isinstance(mod, torch.nn.Embedding):
            def pre(m_, inp, _n=name):
                x = inp[0] if isinstance(inp, tuple) and len(inp) else inp
                if torch.is_tensor(x) and _n not in seen:
                    keep.append(x)
                    seen[_n] = id(x)
            hs.append(mod.register_forward_pre_hook(pre))
        if _is_norm(mod) or id(mod) in ro:
            def post(m_, inp, out, _n=name):
                if id(m_) in ro:                                   # the readout's input (pre-hook below)
                    return None
                if torch.is_tensor(out) and _n not in outs:
                    keep.append(out)
                    outs[_n] = id(out)
            hs.append(mod.register_forward_hook(post))
        if id(mod) in ro:
            def pre_ro(m_, inp):
                x = inp[0] if isinstance(inp, tuple) and len(inp) else inp
                if torch.is_tensor(x) and "readout" not in seen:
                    keep.append(x)
                    seen["readout"] = id(x)
            hs.append(mod.register_forward_pre_hook(pre_ro))
    rs_ = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
           _rd.getstate(), _np.random.get_state())
    was = model.training
    model.eval()
    try:
        probe()
    finally:
        for h in hs:
            h.remove()
        if was:
            model.train()
        torch.set_rng_state(rs_[0])
        if rs_[1] is not None:
            torch.cuda.set_rng_state_all(rs_[1])
        _rd.setstate(rs_[2]); _np.random.set_state(rs_[3])
    areas, by_input, gains, tables = {}, {}, {}, {}
    for name, mod in model.named_modules():
        if name not in seen:
            continue
        if isinstance(mod, torch.nn.Linear) and id(mod) not in ro:
            key = seen[name]
            a = by_input.get(key)
            if a is None:
                a = by_input[key] = name
                areas[a] = []
            areas[a].append(mod)
        elif _is_norm(mod):
            gains[name] = mod
        elif isinstance(mod, torch.nn.Embedding):
            tables[name] = mod
    named, rename = {}, {}
    nm_ = {id(m): n for n, m in model.named_modules()}
    for a, mods in areas.items():
        rename[a] = a + "".join("+" + nm_[id(m)].rsplit(".", 1)[-1] for m in mods[1:])
        named[rename[a]] = mods
    # DOWNSTREAM of each gain: the area (or the readout) whose input is the norm's output tensor itself
    inv = {k: rename[v] for k, v in by_input.items()}         # input tensor id -> area
    if "readout" in seen:
        inv.setdefault(seen["readout"], "readout")
    down = {g: inv.get(outs.get(g)) for g in gains}
    del keep
    return named, gains, tables, down


def _gd_rank(ev, n):
    """Gavish-Donoho optimal hard threshold on the singular values sqrt(ev) of a second moment from n patterns."""
    sv = ev.clamp(min=0).sqrt().flip(0)
    D = float(ev.shape[0])
    bt = min(float(n), D) / max(float(n), D)
    om = 0.56 * bt ** 3 - 0.95 * bt ** 2 + 1.82 * bt + 1.43
    tau = om * float(sv[:int(min(n, D))].median())
    return int((sv > tau).sum())


def _pad64(U):
    """The basis with zero columns appended up to a multiple of 64: (x U) U^T is unchanged, and the GEMMs stay on the
    aligned kernels (an unaligned inner size runs ~10x slower)."""
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
            db = None
            if ctx.hasb:
                # BIAS: the weight on the constant input 1, which lies outside the held span: its held share is the
                # area's tail share c (T = c (I - U U^T) gives c e for any e orthogonal to U)
                db = dy2.float().sum(0)
                T_ = ctx.hx.T.get(ctx.area)
                if isinstance(T_, _LowT) and id(ctx.mod) not in ctx.hx._rhold:
                    s_ = (T_.sb if T_.av is not None else T_.c) * T_.m
                    if s_ != 1.0:
                        db.mul_(s_)
        return dx, dW, db, None, None, None


class _Prof:
    """Optional profiler (env HLE_PROF=1): device-timed wall time per component, active only in a step window of
    tasks 0 and 1 (it slows the steps but changes no number)."""
    def __init__(self):
        import os as _os
        self.on = bool(_os.environ.get("HLE_PROF"))
        self.live = False
        self.acc, self.cnt = {}, {}
        self.ev = []

    def wrap(self, key, fn):
        if not self.on:
            return fn

        def w(*a, **k):
            if not self.live:
                return fn(*a, **k)
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            r = fn(*a, **k)
            e1.record()
            self.ev.append((key, e0, e1))                     # (CUDA events: no host sync in the hot path)
            return r
        return w

    def mark(self, key, e0, e1):
        self.ev.append((key, e0, e1))

    def dump(self, tag, nsteps):
        if self.ev:
            torch.cuda.synchronize()                          # (once per window)
            for key, e0, e1 in self.ev:
                self.acc[key] = self.acc.get(key, 0.0) + e0.elapsed_time(e1) / 1000.0
                self.cnt[key] = self.cnt.get(key, 0) + 1
            self.ev = []
        if not self.acc:
            return
        logger.info("[PROFILE] %s over %d steps (ms per step, CUDA events): %s" % (
            tag, nsteps, " | ".join("%s %.2f (x%.1f/step)" % (k, 1000.0 * v / max(nsteps, 1), self.cnt[k] / max(nsteps, 1))
                                     for k, v in sorted(self.acc.items(), key=lambda kv: -kv[1]))))
        self.acc, self.cnt = {}, {}


PROF = _Prof()


def _re_layer(n):
    """parameter name with layer indices collapsed (for diagnostic logs)"""
    import re as _re
    return _re.sub(r"\.(\d+)\.", ".*.", n)


class _LowT:
    """Operator T = c (I - U U^T), U (d x r) the held span (bf16, padded to 64 columns), never materialised."""
    def __init__(self, Ub, c):
        self.Ub, self.c = Ub, float(c)
        self.m = 1.0
        self.av, self.b, self.sb = None, None, None   # minimum-interference form: T = b I + U diag(av - b) U^T
        self.dense = None                             # (owmx) the exact operator (I + mu C / f)^-1, d x d bf16
        self.shape, self.dtype = (Ub.shape[0], Ub.shape[0]), torch.bfloat16


class HippoIndexEnc:
    _CMOFF = 32                              # answer-step offsets per co-movement class slot

    def __init__(self, learner, model, probe, readout_mods=None, readout_hook=None):
        """probe(): one short forward of `model` (area detection). readout_mods: the linear maps that read the last
        hidden sequence (default: the output embedding); readout_hook: the module whose forward returns the readout's
        logits for every position (default: the output embedding itself)."""
        self.L = learner
        self.first = {}                      # area -> the module whose input is read
        if readout_mods is None:
            readout_mods = [model.get_output_embeddings()]
        head = readout_hook if readout_hook is not None else readout_mods[0]
        self.areas, gains_, tables_, self._gdown = detect_areas(model, readout_mods, probe)   # area -> [modules]
        # encoder-fitted options read from the learner (defaults = the decoder recipe)
        self._answ = bool(getattr(learner, "trunk_answer_weight", True))   # answer rows reweighted in trunk moments
        self._gshare_down = bool(getattr(learner, "gain_share_downstream", False))
        self.areas["readout"] = list(readout_mods)
        oe_ = model.get_output_embeddings()
        self._tied = oe_ is not None and model.get_input_embeddings().weight is oe_.weight
        # answer tokens are part of the input (decoder: the answer is read back) or not (encoder: the class is not a
        # token); the tables count only non-answer input positions
        self._anstok = bool(getattr(learner, "ans_in_input", True))
        # every hook is kept as (module, kind, fn) so it can be detached during evaluation, where all of them
        # return without effect (collect is None, _mcache is None, no gradient)
        self._hookspec, self._handles = [], []
        for area, mods in self.areas.items():
            self.first[area] = mods[0]
            self._hookspec.append((mods[0], "pre", self._make_hook(area)))
            if area != "readout":
                self._hookspec.append((mods[0], "fwd", self._make_cmp_hook(area)))
        self.diag = {}
        for name, mod in gains_.items():
            self.diag[name] = mod
            self._hookspec.append((mod, "pre", self._make_diag_hook(
                name, float(getattr(mod, "eps", getattr(mod, "variance_epsilon", 1e-6))),
                isinstance(mod, torch.nn.LayerNorm))))
        for area, mods in self.areas.items():
            for mod in mods:
                self._install(mod, area)
        self._xt = {}
        self._xbc = None                     # one-entry cache of the present area's bf16 input (forward)
        self._tmdm = None                    # step(): param -> operator / share vector, rebuilt only when T changes
        self.emb = model.get_input_embeddings()
        self.tabs = {}                       # per-row table area name -> embedding module
        for name, mod in tables_.items():
            tn = "tok" if mod is self.emb else "tab:" + name
            self.tabs[tn] = mod
            self._hookspec.append((mod, "fwd", self._make_emb_hook(tn)))
        if "tok" not in self.tabs:
            self.tabs["tok"] = self.emb
            self._hookspec.append((self.emb, "fwd", self._make_emb_hook("tok")))
        self._hookspec.append((head, "fwd", self._conflict_hook))
        # diagnostic: parameter -> (kind, area) for the step-leak measure
        self._pk = {}
        for a_, ms_ in self.areas.items():
            for m_ in ms_:
                self._pk[id(m_.weight)] = (_kind(a_) + "-w", a_)
                if m_.bias is not None:
                    self._pk[id(m_.bias)] = (_kind(a_) + "-b", a_)
        for g_, m_ in self.diag.items():
            k_ = "LN>" + _kind(self._gdown.get(g_) or "none")
            self._pk[id(m_.weight)] = (k_ + "-g", None)
            if isinstance(getattr(m_, "bias", None), torch.nn.Parameter):
                self._pk[id(m_.bias)] = (k_ + "-b", None)
        for t_, m_ in self.tabs.items():
            self._pk[id(m_.weight)] = ("table:" + t_.split(".")[-1], None)
        self._leak = {}
        # MASSIVE-COORDINATE OUTPUT HOLD: the residual stream's coordinates (width of the token table); the params that
        # write into them -- single-map areas whose output has the stream's width (attention out / FFN out), their
        # biases, the norms' gains / biases, the tables' columns -- and the massive set M found at sleep
        self._dstream = int(self.emb.weight.shape[1])
        self._sw = {}
        for a_, ms_ in self.areas.items():
            if a_ != "readout" and len(ms_) == 1 and ms_[0].out_features == self._dstream:
                self._sw[id(ms_[0].weight)] = "w"
                if ms_[0].bias is not None:
                    self._sw[id(ms_[0].bias)] = "b"
        for m_ in self.diag.values():
            if m_.weight.shape[0] == self._dstream:
                self._sw[id(m_.weight)] = "b"
                if isinstance(getattr(m_, "bias", None), torch.nn.Parameter):
                    self._sw[id(m_.bias)] = "b"
        for m_ in self.tabs.values():
            if m_.weight.shape[1] == self._dstream:
                self._sw[id(m_.weight)] = "t"
        self._M = None
        self._mhold = bool(getattr(learner, "massive_hold", False))
        # SLEEP IN THE WAKE MODE: with dropout kept (the encoders' native regime) the sleep reads run with dropout too,
        # so the stored statistics (span, tail, gains, tables) and the present ones are taken under the same forward
        self._sleep_train = bool(getattr(learner, "sleep_train_mode", False))
        self._cmp_all = bool(getattr(learner, "comparator_all_rows", False))
        self._sigcut = bool(getattr(learner, "massive_signal_cut", False))
        # OLD-ROW REGION HOLD (option): each old head row (class column k) and its bias are held together off the class's
        # REGION -- the Gavish-Donoho top directions of its augmented answer-state moment E[[h; 1][h; 1]^T], a
        # class-level statistic taken at the sleep -- in place of the readout span c (I - U U^T) and the comparator
        self._oregion = bool(getattr(learner, "oldrow_region", False))
        self._trunk_c1 = bool(getattr(learner, "trunk_c1", False))
        self._owm = bool(getattr(learner, "owm", True))   # MINIMUM-INTERFERENCE STEP (default); False: the tail share c
        # (option) EXACT minimum-interference step: the closed form (I + mu C / f)^-1 with C the full running moment of
        # every earlier input of the area (an aggregate statistic, d x d), not the GD-truncated U diag(lam) U^T plus an
        # isotropic tail tau (I - U U^T)
        self._owmx = bool(getattr(learner, "owm_exact", False))
        # (option) untied input tables: the decoder's per-row minimum-interference share f / (f + mu u_v) (gradient and,
        # with hle_tabstep, step) in place of the novelty share with owned rows held at 0
        self._tab_owm = bool(getattr(learner, "table_owm", False))
        # (option) state conflict against the row's own present class: a row conflicts with record j only if its
        # centred state is closer to j than to its own present class's running mean (centre: old records + present
        # class means); no radius, no degeneration for small K
        self._pdet = bool(getattr(learner, "present_detect", False))
        # (option) CLASS-LEVEL state conflict: a row is in conflict when its present class's running mean state lies
        # inside an earlier record's neighbourhood (the record's radius: its similarity to its nearest other earlier
        # record) -- every row of a counterpart class, not only the rows that happen to fall nearer the record
        self._pairc = bool(getattr(learner, "class_detect", False))
        # (option, with pdet) UNION: a row is in conflict when the row (pdet) OR its class (mutual nearest) is
        self._pairu = bool(getattr(learner, "class_detect_union", False))
        # (option) JOINT-PROPORTION TWIN CUT of the decision row's learning signal at a detected conflict
        self._jpcut = bool(getattr(learner, "twin_cut", False))
        self._jpcut_live = bool(getattr(learner, "twin_cut_live", False))   # axis: the live rival's readout difference
        self._jcS, self._jcR = [], []
        self._pcm = None                     # present classes' running decision-state sums [count (C), sum (C, d)]
        # (option) joint-share target at offset 0 when the model has a single decision row (encoders)
        self._js0 = bool(getattr(learner, "joint_share_offset0", False))
        # (option) weight decay folded into the update before the hold (u + wd p passes the same operator / share)
        self._wdfold = bool(getattr(learner, "wd_fold", False))
        self._wdp = {}
        # (option) the comparator's armed decision rows enter as their mismatch with the area's WHOLE held span
        # W = [U, patterns] (not only the class-mean patterns): a decision-row write interferes with every earlier
        # decision-row input, and those lie in W -- the armed rows then write only where no earlier input sits
        self._cmp_held = bool(getattr(learner, "cmp_held", False))
        self._cmp_owm = bool(getattr(learner, "cmp_owm", False))   # (option) give-back sized by the OWM share
        self._cmpc = {}
        self._cmpG, self._cmpBs, self._cmpOs = [], {}, {}   # batched comparator: groups, stacked bases / operators
        # (option) DECISION ROW AS AN ATTENTION SINK: in an attention-input area (q / k / v read the same input) the
        # decision row's input x_0 is read through the value / key maps by every position i with weight a_i0, so a
        # write along x_0 changes every position's attention output: its interference counts the decision row with
        # weight w_0 = 1 + sum_i a_i0^2 (head mean), measured from the area's own q / k maps; memory and wake alike
        self._sink = bool(getattr(learner, "sink_weight", False))
        # (option) TABLE ROWS IN THE STREAM'S UNITS: a lookup row's change dv enters the area the embedding norm feeds
        # as (gamma / sigma) dv at every position using it; its interference is compared with THAT area's noise floor
        # f_A: share f_A / (f_A + mu u_v mean(gamma^2) / sigma^2) (u_v the row's earlier use per position), in place
        # of the GD cut of the row-use distribution (degenerate for flat-use tables: positions, token_type)
        self._tab_ln = bool(getattr(learner, "table_stream", False))
        self._tablog = None
        self._sinkw = {}                     # diagnostic: area -> [sum w, n]
        self._cmpS = []
        self._full, self._Tx = {}, {}
        # (option) DOWNSTREAM GAIN: the OWM interference of a layer's areas measured at the decision state, a layer's
        # relative output change times its measured downstream gain g_l (mu -> mu g_l^2); g_l taken at the sleep
        self._lgain_on = bool(getattr(learner, "layer_gain", False))
        self._lgain = {}
        # MINIMUM-INTERFERENCE STEP for the per-coordinate parameters too (gains, table rows; default): share
        # f / (f + mu e_i) in place of the novelty share and the owned-coordinate mask
        self._owm_coord = bool(getattr(learner, "owm_coord", False))   # (ablation; off by default)
        self._gfm = {}
        self._cregC, self._cbas, self._rbas, self._rhold, self._rdef = {}, {}, None, {}, {}
        self.resume_hooks()

        self.collect = None                  # None | "now" (training, task >= 1) | "rec" (sleep) | "g2s" (2nd sleep read)
        self.now, self.rec, self.old = {}, {}, {}
        self.eig = {}                        # readout: (ev, V host, kp) | gains/tok: (ev, None, owned) | trunk: ("subspace",)
        self.T = {}                          # area -> operator (readout bf16 d x d, trunk _LowT)
        self.dD = {}                         # gain / token areas -> share vector
        self._mem = {}                       # trunk area -> {"U", "Ub", "lam", "tr", "n", "tail"}
        self.steps = 0
        self.next_build = 1
        self.n_eps = 0
        self.plast = 1.0                     # write gate of the present step (set by the conflict hook)
        self._tid = 0
        self._rowk, self._rowc = None, None
        self._sk = None                      # sleep: the batch's masks, row indices and record keys
        self.recs = {}                       # (cid, off) -> [n, sum_h]
        self.cids = {}
        self.next_cid = 0
        self._cidname, self._pcd = {}, None      # diagnostic: record class id -> name; per present class detection counts
        self._sc_M = None
        # CA1 comparator: per memory (class, answer step) every area's summed answer-step input
        self._pat, self._pkey, self._pc_n = {}, {}, None
        self._psk, self._pom, self._rgQ = {}, {}, {}
        self._cmpB, self._j2p, self._cmpA = {}, None, {}
        self._cxs, self._cxa, self._cmpX, self._cmpsel, self._cmpd = {}, {}, {}, None, {}
        self._ansk, self._ansix, self._ansfl = None, None, None
        # co-movement (present task only)
        self._cm, self._cmB, self._cm_rowfac, self._cm_sl = None, None, None, None
        # sleep: new keys and the second read's per-row energies along them
        self._kc, self._g2P, self._g2E, self._sk2, self._g2st, self._cest = {}, {}, {}, None, [], []
        self._scs = []
        self._grp = set()                    # sleep: the trunk areas whose moment the present read accumulates
        self._pha = {}                       # per task: the held span with the earlier memories' patterns (pattern hold)
        self._dg = []                        # diagnostic: per build: (area, c, passed energy share)
        self._fresh, self._Tf = set(), None  # FRESH READOUT: maps with no memory yet, and their operator
        self._sgw, self._sgr, self._sgref = {}, {}, {}   # diagnostic: decision-row log sigma: wake / sleep read / reference
        self._ncmp, self._njs, self._nrow = 0, 0, 0   # diagnostic: comparator-armed / joint-share rows / answer rows
        self.train_peak = None               # the finished task's training peak (bytes): the sleep stays below it
        nb_ = sum(1 for v in self.areas.values() for m_ in v if m_.bias is not None)
        nbg_ = sum(1 for m_ in self.diag.values() if isinstance(getattr(m_, "bias", None), torch.nn.Parameter))
        if PROF.on:
            for nm_ in ("_held_input", "apply", "step", "_build", "_cmp_arm", "_pdet_scores", "_rowproj_all", "_sink_w"):
                setattr(self, nm_, PROF.wrap("call:" + nm_, getattr(self, nm_)))
        logger.info("[HIPPO-ENC] one hippocampus indexing %d cortical areas (%d synapse groups, %d with a bias; readout "
                    "%d maps), %d gain areas (%d with a bias) and %d per-row tables %s"
                    % (len(self.areas), sum(len(v) for v in self.areas.values()), nb_, len(self.areas["readout"]),
                       len(self.diag), nbg_, len(self.tabs), sorted(self.tabs)))

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
        row on the same x), so _held_input's cache (keyed on the saved copy's address) is shared and x T is formed
        once per area. Cleared in pre_backward."""
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
        if id(mod) in self._rhold:
            return xb.reshape(-1, xb.shape[-1])          # (region-held old head: the plain gradient, held in apply())
        c_ = self._cmpX.get(id(mod)) if self._cmpsel is not None else None
        # the comparator's rows are the same tensor for every group of the area (_cmp_arm shares them): cached too
        T = self._Tf if (self._Tf is not None and id(mod) in self._fresh) else self.T.get(area)
        key = (area, xb.data_ptr(), tuple(xb.shape), c_ is not None, id(T))
        Xt = self._xt.get(key)
        if Xt is not None:
            return Xt
        X = xb.reshape(-1, xb.shape[-1])
        if isinstance(T, _LowT) and T.dense is None and X.dtype == torch.bfloat16:
            # the same product in two fused kernels (bf16 operands, fp32 accumulation, one rounding to bf16):
            # c-rule x T = s x - s (x U) U^T; minimum-interference x T = m b x + m (x U diag(av - b)) U^T
            if T.av is None:
                s_ = T.c * T.m
                Xt = torch.addmm(X, torch.mm(X, T.Ub), T.Ub.t(), beta=s_, alpha=-s_)
            else:
                y_ = torch.mm(X, T.Ub, out_dtype=torch.float32)
                y_.mul_(T.av - T.b)
                Xt = torch.addmm(X, y_.to(torch.bfloat16), T.Ub.t(), beta=float(T.b) * T.m, alpha=T.m)
            if c_ is not None:
                bs, ps = self._cmpsel
                Xt[bs * xb.shape[1] + ps] = c_[1].to(torch.bfloat16)
        else:
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
            fn = PROF.wrap("hook:" + fn.__qualname__.replace("HippoIndexEnc.", "").replace(".<locals>.hook", ""), fn)
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

    def _rows(self, ck, ans):
        if self._rowk is not ck:
            self._rowk = ck
            self._rowc = (ck[2].reshape(-1).nonzero(as_tuple=True)[0], ck[1].reshape(-1).nonzero(as_tuple=True)[0])
        return self._rowc[1] if ans else self._rowc[0]

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
        pm = (mt & (lb == -100)) if self._anstok else mt
        self._sk = {"shape": (B, Lq), "mt": mt, "ma": ma, "ix_all": ix_all, "ix_ans": ix_ans,
                    "n_all": int(ix_all.numel()), "n_ans": int(ix_ans.numel()), "bi": bi, "pi": pi, "rows": rows,
                    "rk": rk, "pm": pm, "n_pm": float(pm.sum())}

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
            if self.collect == "now":
                # wake: present energy, total and inside the held span U
                if torch._C._current_graph_task_id() != -1:
                    # skip checkpoint recomputes: they run AFTER pre_backward advanced self.steps, so
                    # _energy_needed() would answer for the next step and add a shifted set of batches
                    return None
                x = inp[0] if isinstance(inp, tuple) else inp
                if not torch.is_tensor(x) or x.dim() != 3 or area not in self._mem:
                    return None
                ck = self.L._mcache
                if ck is None or ck[0] != tuple(x.shape[:2]) or not self._energy_needed():
                    return None
                n_all = ck[4] if ans else ck[3]
                if n_all <= 0:
                    return None
                with torch.no_grad():
                    xd = x.detach()
                    if xd.dtype != torch.bfloat16:
                        xd = xd.to(torch.bfloat16)
                    mm_ = self._mem[area]
                    Ub = mm_["Ub"]
                    r_ = mm_["U"].shape[1]
                    if Ub is None:
                        Ub = mm_["Ua"]                               # (pattern hold: U is the first r columns)
                    X = xd.reshape(-1, xd.shape[-1]).index_select(0, self._rows(ck, ans))

                    def _e(Xr):
                        z_ = torch.mm(Xr, Ub, out_dtype=torch.float32).pow(2)
                        parts = [Xr.float().pow(2).sum().view(1), z_[:, :r_].sum().view(1)]
                        return torch.cat(parts)
                    C = _e(X)
                    if not ans and ck[4] > 0 and self._answ:
                        Xa = xd.reshape(-1, xd.shape[-1]).index_select(0, self._rows(ck, True))
                        C = C + _e(Xa) * (n_all / ck[4])
                    if not ans and self._sink and "attention.self.query" in area:
                        w_ = self._sink_w(area, x, ck[2])
                        if w_ is not None:
                            z0_ = xd[:, 0]                                        # (B, d) the decision rows
                            zz_ = torch.mm(z0_, Ub, out_dtype=torch.float32).pow(2)
                            wm_ = (w_ - 1.0).float()
                            C = C + torch.cat([(z0_.float().pow(2).sum(1) * wm_).sum().view(1),
                                               (zz_[:, :r_].sum(1) * wm_).sum().view(1)])
                    e = self.now.get(area)
                    if e is None:
                        self.now[area] = [C, float(n_all)]
                    else:
                        e[0] += C
                        e[1] += float(n_all)
                return None
            if self.collect is None:
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
                am_ = sk.get("am", self.L.cur_attn)
                if not ans and self._sink and "attention.self.query" in area and am_ is not None \
                        and tuple(am_.shape) == tuple(x.shape[:2]):
                    w_ = self._sink_w(area, x, am_)
                    if w_ is not None:
                        X0_ = x.detach()[:, 0].float() * (w_ - 1.0).clamp(min=0.0).sqrt().unsqueeze(1)
                        C.add_(X0_.t() @ X0_)
                if not ans and sk["n_ans"] > 0 and self._answ:
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

    def _make_diag_hook(self, name, eps, centred=False):
        """A gain's per-coordinate input energy: the normalised input the gain multiplies -- x / rms(x) for RMSNorm,
        (x - mean x) / std(x) for LayerNorm (centred)."""
        def hook(mod, inp):
            if self.collect is None or self.collect == "recC":
                return None
            x = inp[0] if isinstance(inp, tuple) else inp
            if not torch.is_tensor(x) or x.dim() < 3:
                return None
            if self.collect == "now":
                if torch._C._current_graph_task_id() != -1:
                    return None                                          # checkpoint recompute: see the area hook
                ck = self.L._mcache
                if ck is None or ck[0] != tuple(x.shape[:2]):
                    return None
                if self._sigcut and self._M is not None and x.requires_grad and x.shape[-1] == self._dstream:
                    # MASSIVE-COORDINATE SIGNAL CUT: the learning signal reaching a norm's input carries no component
                    # along the stream's massive coordinates. In post-LN that coordinate sets sigma for every token,
                    # so a signal along it asks to rescale the whole stream (the cheapest way to lower every earlier
                    # head's logits at once); the cut removes that drive at its source, in every path behind it.
                    M_ = self._M.to(x.device)

                    def _cut(g, _M=M_):
                        return g.index_fill(-1, _M, 0.0)
                    x.register_hook(_cut)
                if self.steps % 10 == 0 and x.dim() == 3 and ck[4] > 0:
                    # diagnostic: the stream scale at the decision rows: mean log sigma of the norm's input
                    with torch.no_grad():
                        xa_ = x.detach()[ck[1]].float()
                        if centred:
                            xa_ = xa_ - xa_.mean(-1, keepdim=True)
                        ls_ = 0.5 * torch.log(xa_.pow(2).mean(-1) + eps).mean()
                        e_ = self._sgw.get(name)
                        self._sgw[name] = [ls_, 1] if e_ is None else [e_[0] + ls_, e_[1] + 1]
                n_ = ck[3]
                if n_ <= 0 or not self._energy_needed():
                    return None
                m = ck[2]
                with torch.no_grad():
                    xf = x.detach().float()
                    if centred:
                        xf = xf - xf.mean(-1, keepdim=True)
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
            if self.collect == "rec" and x.dim() == 3 and sk["n_ans"] > 0:
                # diagnostic: the stream scale at the decision rows of the episode just learnt (finished model)
                with torch.no_grad():
                    xa_ = x.detach().reshape(-1, x.shape[-1]).index_select(0, sk["ix_ans"]).float()
                    if centred:
                        xa_ = xa_ - xa_.mean(-1, keepdim=True)
                    ls_ = 0.5 * torch.log(xa_.pow(2).mean(-1) + eps)
                    e_ = self._sgr.get(name)
                    self._sgr[name] = [ls_.sum(), float(ls_.numel())] if e_ is None else [e_[0] + ls_.sum(), e_[1] + float(ls_.numel())]
            with torch.no_grad():
                X = x.detach().reshape((-1,) + tuple(x.shape[2:])).index_select(0, sk["ix_all"])
                X = X.float().reshape(-1, x.shape[-1])
                if centred:
                    X = X - X.mean(-1, keepdim=True)
                X = X * torch.rsqrt(X.pow(2).mean(-1, keepdim=True) + eps)
                self._store("g:" + name, X.pow(2).sum(0), float(X.shape[0]))
            return None
        return hook

    def _make_emb_hook(self, tn):
        """A per-row table (token ids, absolute positions, token types): per row, the count of input positions that
        read it -- attended positions, minus the answer tokens when the answer is part of the input. The ids may
        cover the batch (B, L) or one row of positions broadcast over it (1, L)."""
        def hook(mod, inp, out):
            return self._emb_hook(mod, inp, out, tn)
        return hook

    def _row_weights(self, ids, pm, n_pm):
        """Weights of each id for a (B, L) position mask pm: the mask itself, or summed over the batch for (1, L) ids."""
        if tuple(ids.shape) == tuple(pm.shape):
            return pm.reshape(-1).float(), n_pm
        if ids.dim() == 2 and ids.shape[0] == 1 and ids.shape[1] == pm.shape[1]:
            return pm.float().sum(0).reshape(-1), n_pm
        return None, None

    def _emb_hook(self, mod, inp, out, tn="tok"):
        ids = inp[0] if isinstance(inp, tuple) else inp
        if not torch.is_tensor(ids) or ids.dim() != 2 or self.collect == "recC":
            return None
        if self.collect == "now":
            ck = self.L._mcache
            lb = self.L.cur_labels
            if ck is not None and ids.shape[1] == ck[0][1] and self._energy_needed():
                with torch.no_grad():
                    if self._anstok:
                        pm, n_ = ck[2] & (lb.to(ids.device) == -100), float(ck[3] - ck[4])
                    else:
                        pm, n_ = ck[2], float(ck[3])
                    w_, n_ = self._row_weights(ids, pm, n_)
                    if w_ is not None:
                        cnt = torch.bincount(ids.reshape(-1), weights=w_, minlength=mod.weight.shape[0])
                        self._store(tn, cnt, n_)
        elif self.collect == "rec":
            sk = self._sk
            if sk is not None and ids.shape[1] == sk["shape"][1]:
                with torch.no_grad():
                    w_, n_ = self._row_weights(ids, sk["pm"], sk["n_pm"])
                    if w_ is not None:
                        cnt = torch.bincount(ids.reshape(-1), weights=w_, minlength=mod.weight.shape[0])
                        self._store(tn, cnt, n_)
        D = self.dD.get(tn)
        # tied table only: an untied readout's held span lives in the last layer's space, not the input table's
        Tr = self.T.get("readout") if self.collect == "now" and self._tied and tn == "tok" else None
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
        """x @ T in fp32 from bf16 operands."""
        if isinstance(T, _LowT) and T.dense is not None:
            out = torch.mm(x.to(torch.bfloat16), T.dense, out_dtype=torch.float32)
            if T.m != 1.0:
                out.mul_(T.m)
            return out
        if isinstance(T, _LowT) and T.av is not None:
            y = torch.mm(x.to(torch.bfloat16), T.Ub, out_dtype=torch.float32)
            y.mul_(T.av - T.b)
            out = torch.mm(y.to(torch.bfloat16), T.Ub.t(), out_dtype=torch.float32)
            out.add_(x.to(torch.float32), alpha=T.b)
            if T.m != 1.0:
                out.mul_(T.m)
            return out
        if isinstance(T, _LowT):
            y = torch.mm(x.to(torch.bfloat16), T.Ub, out_dtype=torch.float32)
            m_ = torch.mm(y.to(torch.bfloat16), T.Ub.t(), out_dtype=torch.float32)
            # x - m written into m: x is upcast exactly inside the kernel, so no fp32 copy of x is made
            out = torch.sub(x, m_, out=m_)
            s_ = T.c * T.m
            if s_ != 1.0:                                   # x * 1.0 == x: the pass is skipped
                out.mul_(s_)
            return out
        return torch.mm(x.to(torch.bfloat16), T, out_dtype=torch.float32)

    @staticmethod
    def _tmm_(x, T):
        """x <- x @ T in place for an fp32 scratch x (the step's update): the same arithmetic as _tmm, without
        the result tensor and the copy back."""
        if not isinstance(T, _LowT) or x.dtype != torch.float32 or T.dense is not None:
            return x.copy_(HippoIndexEnc._tmm(x, T))
        if T.av is not None:
            y = torch.mm(x.to(torch.bfloat16), T.Ub, out_dtype=torch.float32)
            y.mul_(T.av - T.b)
            x.mul_(T.b)
            x.add_(torch.mm(y.to(torch.bfloat16), T.Ub.t(), out_dtype=torch.float32))
            if T.m != 1.0:
                x.mul_(T.m)
            return x
        y = torch.mm(x.to(torch.bfloat16), T.Ub, out_dtype=torch.float32)
        x.sub_(torch.mm(y.to(torch.bfloat16), T.Ub.t(), out_dtype=torch.float32))
        s_ = T.c * T.m
        if s_ != 1.0:
            x.mul_(s_)
        return x

    @staticmethod
    def _mm_rows(x, T, rows=32768):
        """x <- x @ T written back into x block by block of rows. Callers hold no_grad."""
        for i0 in range(0, x.shape[0], rows):
            HippoIndexEnc._tmm_(x[i0:i0 + rows], T)
        return x

    @staticmethod
    def _proj(x, T, rows=32768):
        """x @ T exactly as _mm_rows computes it: a tensor of at most `rows` rows is one block, projected in place in
        one call; taller tensors (the readout table) are projected in place block by block."""
        if x.shape[0] <= rows:
            return HippoIndexEnc._tmm_(x, T)       # in place: no result tensor (one fp32 param-sized buffer fewer)
        return HippoIndexEnc._mm_rows(x, T, rows)

    def step_audit(self, opt, Tm, Dm, Bm, Rm):
        """Diagnostic (logging only): which rule acts on each trainable parameter's POST-ADAM step, from the maps step()
        uses: operator (T), region row hold (ROW), per-coordinate share (D), bias share (B), per-row table share (R),
        or none; and which rule acts on its gradient only."""
        names = {}
        for g in opt.param_groups:
            for p in g["params"]:
                names[id(p)] = p
        nm = {id(p): n for n, p in self.L._unwrap(self.L.wrap_model).model.named_parameters()}
        cat = {}
        for i, p in names.items():
            if not p.requires_grad:
                continue
            if i in self._rhold:
                k = "ROW"
            elif i in Tm:
                k = "T-fresh" if (self._Tf is not None and Tm[i] is self._Tf) else "T"
            elif i in Dm:
                k = "D"
            elif i in Rm:
                k = "R"
            elif i in Bm:
                k = "B"
            else:
                k = "NONE"
            cat.setdefault(k, []).append(nm.get(i, "?"))
        logger.info("[STEP-RULE] task %d: post-Adam step rule per trainable parameter: %s" % (
            int(self._tid), " | ".join("%s %d" % (k, len(v)) for k, v in sorted(cat.items()))))
        for k, v in sorted(cat.items()):
            short = sorted({_re_layer(n) for n in v})
            logger.info("[STEP-RULE] task %d %s: %s" % (int(self._tid), k, ", ".join(short)))

    @torch.no_grad()
    def _rowproj_all(self, items):
        """Every old head at once: [gw; gb] of all old rows projected off their class regions with ONE batched
        product against the stacked bases."""
        if not items:
            return
        Gs = [torch.cat([gw.float(), (gb.float().unsqueeze(1) if gb is not None else gw.new_zeros(gw.shape[0], 1).float())], 1)
              for _, gw, gb in items]
        G_ = torch.cat(Gs, 0)
        idx_ = torch.cat([self._rhix[id(m_)] for m_, _, _ in items]) if len(items) > 1 else self._rhix[id(items[0][0])]
        B_ = self._rhall.index_select(0, idx_) if len(items) < len(self._rhix) else self._rhall
        G_ = G_ - torch.einsum("nr,ndr->nd", torch.einsum("nd,ndr->nr", G_, B_), B_)
        o_ = 0
        for _, gw, gb in items:
            n_ = gw.shape[0]
            gw.copy_(G_[o_:o_ + n_, :-1].to(gw.dtype))
            if gb is not None:
                gb.copy_(G_[o_:o_ + n_, -1].to(gb.dtype))
            o_ += n_

    @torch.no_grad()
    def _rowproj(self, m_, gw, gb):
        """in place: [gw_r, gb_r] of an old head's row r projected off its class region (augmented basis)"""
        B_ = self._rhold[id(m_)]
        G_ = torch.cat([gw.float(), (gb.float().unsqueeze(1) if gb is not None else gw.new_zeros(gw.shape[0], 1).float())], 1)
        G_ = G_ - torch.einsum("nr,ndr->nd", torch.einsum("nd,ndr->nr", G_, B_), B_)
        gw.copy_(G_[:, :-1].to(gw.dtype))
        if gb is not None:
            gb.copy_(G_[:, -1].to(gb.dtype))

    # ------------------------------------------------------------------ write through the index (the optimiser step)
    @torch.no_grad()
    def step(self, opt):
        """AdamW with Adam's step taken through the index; the comparator gives back, along the mismatch directions,
        what the index removed from Adam's step. The moments and the update u = m^/(sqrt(v^)+eps) come from torch's
        fused AdamW kernel (written into a zero buffer at lr -1); unwritten steps (lr 0) only move the moments."""
        if self._tmdm is None:
            # rebuilt only after _build / begin_task / sleep (T and dD change only there)
            Tm, Bm = {}, {}
            for area, mods in self.areas.items():
                T = self.T.get(area)
                if T is not None:
                    for mod in mods:
                        Tm[id(mod.weight)] = self._Tf if (self._Tf is not None and id(mod) in self._fresh) else T
                        if mod.bias is not None and isinstance(T, _LowT):
                            # BIAS: its Adam step at the area's tail share (minimum-interference: f / (f + mu))
                            Bm[id(mod.bias)] = (T.sb if T.av is not None else T.c) * T.m
            Dm = {}
            for name, mod in self.diag.items():
                D = self.dD.get("g:" + name)
                if D is not None:
                    Dm[id(mod.weight)] = D
                    if isinstance(getattr(mod, "bias", None), torch.nn.Parameter):
                        Dm[id(mod.bias)] = D                      # norm bias: the gain's per-coordinate share
            Rm = {}
            if bool(getattr(self.L, "table_step_share", False)):
                # UNTIED TABLES (encoder token / position / token-type tables): no operator acts on their step (a tied
                # decoder table steps through the readout operator), and Adam's elementwise normalisation undoes a
                # share applied to the gradient alone -- so, as for the gains, the per-row share also scales the step
                for tn, mod in self.tabs.items():
                    D = self.dD.get(tn)
                    if D is not None and id(mod.weight) not in Tm:
                        Rm[id(mod.weight)] = D
            if bool(getattr(self.L, "table_step_downstream", False)):
                # UNTIED TABLES THROUGH THE AREA THEY FEED: a table row is a vector written into the stream at the
                # positions where its id occurs; the stream (after the first norm it passes) is the input of the first
                # area, so the row's step goes through that area's operator -- as a tied decoder table's step goes
                # through the readout operator. (The row share stays on the gradient.)
                fg_ = next(iter(self._gdown.items()), (None, None))
                Td_ = self.T.get(fg_[1]) if fg_[1] is not None else None
                if isinstance(Td_, _LowT):
                    for tn, mod in self.tabs.items():
                        if id(mod.weight) not in Tm and mod.weight.shape[1] == Td_.shape[0]:
                            Tm[id(mod.weight)] = Td_
                            Rm.pop(id(mod.weight), None)
            self._tmdm = (Tm, Dm, Bm, Rm)
            if getattr(self, "_auditlog", None) != self._tid:
                self._auditlog = self._tid
                self.step_audit(opt, Tm, Dm, Bm, Rm)
        Tm, Dm, Bm, Rm = self._tmdm
        tdev_ = {}
        idle_, chunk_, nb_ = [], [], [0]

        def flush():
            # one fused kernel launch for the chunk's moments and updates, then each group through its operator
            if not chunk_:
                return
            # one zero buffer per chunk (one allocation and one memset)
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
                wd_ = self._wdp.get(id(p)) if self._wdfold else None
                if wd_:
                    u.add_(p.detach(), alpha=wd_)                 # the decay passes the hold with the step
                if id(p) in self._rhold:
                    self._rdef[id(p)] = (p, u.clone(), lr_)       # (region-held old head: weight and bias together)
                    continue
                T = Tm.get(id(p))
                cq_ = self._cmpd.get(id(p)) if self._cmpd else None
                uM_ = None
                if cq_ is not None and len(cq_) == 1 and u.dim() == 2 and cq_[0] in self._cmpA:
                    Qa_, Ma_ = self._cmpA[cq_[0]]
                    uM_ = u @ Ma_
                if self.steps % 10 == 0 and id(p) in self._pk:
                    # diagnostic: the Adam step before the index: what the index removes per kind
                    k0_ = self._pk[id(p)][0] + "#pre"
                    v0_ = u.float().pow(2).sum() * (lr_ * lr_)
                    e0_ = self._leak.get(k0_)
                    self._leak[k0_] = [v0_, v0_] if e0_ is None else [e0_[0] + v0_, e0_[1] + v0_]
                if T is not None and u.dim() == 2 and u.shape[1] == T.shape[0]:
                    u = self._proj(u, T)
                elif id(p) in Dm and u.dim() == 1:
                    u.mul_(Dm[id(p)].to(u.device))
                elif id(p) in Rm and u.dim() == 2 and u.shape[0] == Rm[id(p)].shape[0]:
                    u.mul_(Rm[id(p)].to(u.device, u.dtype).unsqueeze(1))
                elif id(p) in Bm and u.dim() == 1:
                    if Bm[id(p)] != 1.0:
                        u.mul_(Bm[id(p)])
                if self._M is not None and id(p) in self._sw:
                    sk_ = self._sw[id(p)]
                    M_ = self._M.to(u.device)
                    if self._mhold:
                        # the writes into the massive coordinates are withheld (held exactly)
                        if sk_ == "w" and u.dim() == 2:
                            u.index_fill_(0, M_, 0.0)
                            if uM_ is not None:
                                uM_.index_fill_(0, M_, 0.0)
                        elif sk_ == "b" and u.dim() == 1:
                            u.index_fill_(0, M_, 0.0)
                        elif sk_ == "t" and u.dim() == 2:
                            u.index_fill_(1, M_, 0.0)
                    if self.steps % 10 == 0 and id(p) in self._pk:
                        um_ = (u.index_select(0, M_) if sk_ in ("w", "b") else u.index_select(1, M_)).float()
                        k_ = self._pk[id(p)][0] + "@M"
                        e_ = self._leak.get(k_)
                        v_ = um_.pow(2).sum() * (lr_ * lr_)
                        self._leak[k_] = [v_, v_] if e_ is None else [e_[0] + v_, e_[1] + v_]
                if self.steps % 10 == 0 and id(p) in self._pk:
                    self._leak_add(p, u, lr_)
                p.add_(u, alpha=-lr_)
                del u
                if uM_ is not None:
                    if self.steps % 10 == 0 and id(p) in self._pk:
                        # diagnostic: the comparator's give-back |(u M) Q~^T| (Q~ orthonormal: = |u M|), per kind
                        kg_ = self._pk[id(p)][0] + "#cmp"
                        vg_ = uM_.float().pow(2).sum() * (lr_ * lr_)
                        eg_ = self._leak.get(kg_)
                        self._leak[kg_] = [vg_, vg_] if eg_ is None else [eg_[0] + vg_, eg_[1] + vg_]
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
                if decay_ and self._wdfold:
                    self._wdp[id(p)] = wd                        # (folded into u in flush, before the hold)
                elif decay_:
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
        if self._rdef:
            its_ = []
            for m_ in self.areas["readout"]:
                e_ = self._rdef.get(id(m_.weight))
                if e_ is None:
                    continue
                eb_ = self._rdef.get(id(m_.bias)) if m_.bias is not None else None
                its_.append((m_, e_, eb_))
            self._rowproj_all([(m_, e_[1], eb_[1] if eb_ is not None else None) for m_, e_, eb_ in its_])
            for m_, e_, eb_ in its_:
                e_[0].add_(e_[1], alpha=-e_[2])
                if eb_ is not None:
                    eb_[0].add_(eb_[1], alpha=-eb_[2])
            self._rdef = {}
        if idle_:
            # an unwritten step only moves the moments (multi-tensor, the same elementwise arithmetic)
            gs_ = [e[0].grad for e in idle_]
            torch._foreach_lerp_([e[1] for e in idle_], gs_, 1 - idle_[0][3])
            torch._foreach_mul_([e[2] for e in idle_], idle_[0][4])
            torch._foreach_addcmul_([e[2] for e in idle_], gs_, gs_, value=1 - idle_[0][4])
            for e in idle_:
                e[0].grad = None
        self._cmpd = {}
        # the comparator's rows and bases are consumed: free them now
        self._cmpX, self._cmpsel, self._cmpA, self._cmpG = {}, None, {}, []

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
            self._cmpX, self._cmpsel, self._cmpG = {}, None, []
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
                # the answer rows gathered and cast once (rk is in (bi, pi) order): row j equals x[b, p].float()
                H_ = x.detach()[sk["bi"], sk["pi"]].float()
                if self._oregion:
                    col_ = self.L.cur_labels.to(H_.device)[sk["bi"], sk["pi"] + 1].long()
                    Ha_ = torch.cat([H_.double(), H_.new_ones(H_.shape[0], 1).double()], 1)
                    for k_ in torch.unique(col_).tolist():
                        Hk_ = Ha_[col_ == k_]
                        e_ = self._cregC.get(k_)
                        C_ = Hk_.t() @ Hk_
                        self._cregC[k_] = [C_, float(Hk_.shape[0])] if e_ is None else [e_[0] + C_, e_[1] + float(Hk_.shape[0])]
                for j_, (b, p, k) in enumerate(sk["rk"]):
                    e = self.recs.get(k)
                    h = H_[j_]
                    if e is None:
                        self.recs[k] = [1.0, h.clone()]          # [count, sum of h]
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
            if self._pairu:
                hx_ = x.detach()[bi, pi].float()
                _, cin_, cjn_ = self._class_scores(hx_, gold, offs, update=True)
                Sp, own_ = self._pdet_scores(hx_, gold, update=False)
                same = offs.unsqueeze(1) == self._sc_off.to(x.device).unsqueeze(0)
                rin_ = (Sp > own_.unsqueeze(1)) & same
                csc_ = cin_.any(1)
                sc_ = rin_.any(1) | csc_
                jn = torch.where(csc_, cjn_, (Sp - own_.unsqueeze(1)).masked_fill(~same, -9.0).argmax(1))
            elif self._pairc:
                Sp, inside, jn = self._class_scores(x.detach()[bi, pi].float(), gold, offs, update=True)
                sc_ = inside.any(1)
            elif self._pdet:
                hx_ = x.detach()[bi, pi].float()
                Sp, own_ = self._pdet_scores(hx_, gold, update=True)
                same = offs.unsqueeze(1) == self._sc_off.to(x.device).unsqueeze(0)
                inside = (Sp > own_.unsqueeze(1)) & same
                sc_ = inside.any(1)
                jn = (Sp - own_.unsqueeze(1)).masked_fill(~same, -9.0).argmax(1)
            else:
                hs = F.normalize(x.detach()[bi, pi].float() - self._sc_c.to(x.device), dim=1)
                Sp = hs @ self._sc_M.to(x.device).t()
                same = offs.unsqueeze(1) == self._sc_off.to(x.device).unsqueeze(0)
                inside = (Sp > self._sc_nn.to(x.device).unsqueeze(0)) & same
                sc_ = inside.any(1)
                jn = (Sp - self._sc_nn.to(x.device).unsqueeze(0)).masked_fill(~same, -9.0).argmax(1)
            sc_nearest = self._rcnt.to(x.device)[jn]
            self._pcd_add(tg, bi, sc_, jn, x.device, gold)
            self._cm_collect(x, tg, bi, pi, offs)
            if self.steps % 10 == 0:
                self._scs.append(sc_.float().mean())                 # read on the host only when logged
            self._nrow += int(bi.numel())
            if len(self._scs) >= 20:
                logger.info("[HIPPO-ENC] last 200 steps (every 10th): answer steps in state conflict %.3f | (DIAG) "
                            "answer rows %d: comparator-armed %d, joint-share active %d"
                            % (sum(float(v_) for v_ in self._scs) / len(self._scs), self._nrow, self._ncmp,
                               int(float(self._njs))))
                self._scs = []
                self._ncmp, self._njs, self._nrow = 0, 0, 0
                if self._sgw and self._sgref:
                    nm_ = list(self.diag)
                    dv_ = [(i_, float(self._sgw[k][0]) / self._sgw[k][1] - self._sgref[k]) for i_, k in enumerate(nm_)
                           if k in self._sgw and k in self._sgref]
                    if dv_:
                        w_ = max(dv_, key=lambda t: abs(t[1]))
                        logger.info("[HLE-DIAG] decision-row log sigma, wake minus last sleep read: mean %+.3f, "
                                    "largest %+.3f at norm %d | per norm %s" % (
                                        sum(v for _, v in dv_) / len(dv_), w_[1], w_[0],
                                        " ".join("%+.2f" % v for _, v in dv_)))
                self._sgw = {}
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
            if self._jpcut and x.requires_grad and self._pcm is not None:
                # JOINT-PROPORTION TWIN CUT (the decoder's rule, applied to the single decision row): at a row in state
                # conflict with earlier record k, the learning signal dL/dh at the decision row loses its component
                # along the deciding axis u = mean state of k - running mean state of the present class, with weight
                # n_old / (n_old + n_new) (k's record count against the present class's rows per epoch: what k's own
                # samples would push back along u in joint training), only where gold already beats every other
                # class on this row (error gate; no class identity of k is read). Signal, not weights: the trunk's hold
                # is unchanged.
                g_ = gold.long().clamp(min=0)
                if self._jpcut_live:
                    # the LIVE rival's deciding direction: u = W_r - W_gold at the readout input, r the row's top
                    # non-gold class of the current logits (its identity unread, never compared with any record)
                    Wl_ = torch.cat([m_.weight.detach() for m_ in self.areas["readout"]], 0)[:z.shape[1]].float()
                    u_ = F.normalize(Wl_[riv] - Wl_[g_], dim=1)
                else:
                    cnt_, sm_ = self._pcm
                    Pm_ = sm_[g_] / cnt_[g_].clamp(min=1.0).unsqueeze(1)
                    u_ = F.normalize(self._recH.to(x.device)[jn] - Pm_, dim=1)
                w_ = (rp * (zg > zr).float() * sc_.float()).clamp(0.0, 1.0)
                self._jcS.append(torch.stack([w_.sum(), (w_ > 0).float().sum(), float(w_.numel()) + 0.0 * w_.sum()]))

                def _jcut(g, _b=bi, _p=pi, _u=u_, _w=w_, _self=self):
                    gg_ = g[_b, _p].float()
                    c_ = (gg_ * _u).sum(1)
                    rm_ = (c_ * _w).unsqueeze(1) * _u
                    _self._jcR.append(torch.stack([rm_.pow(2).sum(), gg_.pow(2).sum()]))
                    g2_ = g.clone()
                    g2_[_b, _p] = (gg_ - rm_).to(g.dtype)
                    return g2_
                x.register_hook(_jcut)
                if len(self._jcS) >= 200:
                    a_ = torch.stack(self._jcS).sum(0)
                    r_ = torch.stack(self._jcR).sum(0) if self._jcR else torch.zeros(2)
                    logger.info("[HIPPO-ENC] (twin cut) last 200 steps: rows cut %.3f, mean weight on cut rows %.3f, "
                                "signal energy removed at the decision row %.4f" % (
                                    float(a_[1] / a_[2].clamp(min=1.0)), float(a_[0] / a_[1].clamp(min=1.0)),
                                    float(r_[0] / max(float(r_[1]), 1e-30))))
                    self._jcS, self._jcR = [], []
            self.plast = float((1.0 - rp).mean().clamp(0.0, 1.0))
            if out.requires_grad and self._cm_rowfac is not None and self._cm_rowfac.shape[0] == bi.shape[0]:
                # JOINT SHARE TARGET: (1 - a) gold + a q at parting steps inside the answer, a = co-movement x share
                a_ = (self._cm_rowfac.float() * rp.float()).clamp(0.0, 1.0)
                if not (self._js0 and not self._anstok):
                    a_ = a_ * (offs > 0).float()               # (a single decision row IS the parting step)
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
                self._njs = self._njs + okj_.sum()                       # diagnostic: device tensor, read when logged

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
                if self._cmp_all:
                    # ENCODER: the class enters only at the decision row, so the present decision input's mismatch
                    # with every earlier memory's pattern is where a new class is novel: armed wherever gold does not
                    # yet win, not only on rows flagged in state conflict
                    ok = (pr >= 0) & (zg <= zr)
                sel_ = ok.nonzero(as_tuple=True)[0]                     # one host sync
                self._ncmp += int(sel_.numel())
                if sel_.numel():
                    self._cmp_arm(bi[sel_], pi[sel_], cxs_, mod, x, out, sel_)
        return None

    @torch.no_grad()
    def _pcd_add(self, tg, bi, sc_, jn, dev, gold=None):
        """Diagnostic (logging only): per present class and epoch: answer rows, rows in state conflict, and which record they fall near"""
        R_ = self._sc_M.shape[0]
        if self._pcd is None or self._pcd["R"] != R_:
            self._pcd = {"R": R_, "ix": {}, "rows": torch.zeros(8, 128, device=dev), "conf": torch.zeros(8, 128, device=dev),
                         "pair": torch.zeros(128, R_, device=dev), "gcol": torch.full((128,), -1, device=dev, dtype=torch.long)}
        P_ = self._pcd
        pix_ = torch.tensor([P_["ix"].setdefault(t_, len(P_["ix"])) % 128 for t_ in tg]).pin_memory() \
            .to(dev, non_blocking=True)[bi]
        ep_ = min(int(self.L.ep), 7)
        P_["rows"][ep_].index_add_(0, pix_, torch.ones_like(sc_, dtype=torch.float32))
        P_["conf"][ep_].index_add_(0, pix_, sc_.float())
        P_["pair"].index_put_((pix_, jn), sc_.float(), accumulate=True)
        if gold is not None:
            P_["gcol"].index_put_((pix_,), gold.long())

    def _pcd_log(self, task_id):
        P_ = self._pcd
        if P_ is None or not P_["ix"]:
            return
        keys = list(self.recs)
        rows, conf, pair = P_["rows"].cpu(), P_["conf"].cpu(), P_["pair"].cpu()
        ne_ = int((rows.sum(1) > 0).sum())
        out_ = []
        for nm_, i_ in sorted(P_["ix"].items(), key=lambda kv: -float(conf[:, kv[1] % 128].sum())):
            if i_ >= 128:
                continue
            sh_ = " ".join("%.2f" % (float(conf[e_, i_]) / max(float(rows[e_, i_]), 1.0)) for e_ in range(ne_))
            pr_ = pair[i_]
            if float(pr_.sum()) > 0:
                j_ = int(pr_.argmax())
                k_ = keys[j_][0] if j_ < len(keys) else -1
                top_ = "%s %.2f" % (self._cidname.get(k_, str(k_))[:22], float(pr_[j_]) / float(pr_.sum()))
            else:
                top_ = "-"
            # class level: the present class's running mean state against the records (centred cosine), and the
            # nearest record's own neighbourhood radius (its similarity to its nearest other earlier record)
            cl_ = ""
            g_ = int(P_["gcol"][i_])
            if g_ >= 0 and self._pcm is not None and float(self._pcm[0][g_]) > 0:
                Pm_ = self._pcm[1] / self._pcm[0].clamp(min=1.0).unsqueeze(1)
                dv_ = self._pcm[1].device
                # (the records' own geometry: centred by the records' mean, as their radius was measured)
                v_ = F.normalize(Pm_[g_:g_ + 1] - self._sc_c.to(dv_), dim=1)
                S_ = (v_ @ self._sc_M.to(dv_).t()).squeeze(0)
                jj_ = int(S_.argmax())
                kk_ = keys[jj_][0]
                cl_ = " | mean nearest %s cos %.2f (its radius %.2f%s)" % (
                    self._cidname.get(kk_, str(kk_))[:22], float(S_[jj_]), float(self._sc_nn[jj_]),
                    ", INSIDE" if float(S_[jj_]) > float(self._sc_nn[jj_]) else "")
            out_.append("%s [%s] near %s%s" % (nm_[:22], sh_, top_, cl_))
        logger.info("[DIAG-D] task %d state conflict per present class (share of its answer rows per epoch) and the "
                    "record most of its conflicts fall near: %s" % (int(task_id), " | ".join(out_)))
        self._pcd = None

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
            same = offs.unsqueeze(1) == self._sc_off.to(x.device).unsqueeze(0)
            if self._pairu and self._pcm is not None:
                h2_ = x.detach()[bi, pi].float()
                g2_ = sk2_["gold"].to(x.device)
                _, ins_, _ = self._class_scores(h2_, g2_, offs, update=False)
                Sr_, own_ = self._pdet_scores(h2_, g2_, update=False)
                Sp = torch.where(ins_, torch.full_like(Sr_, 2.0), Sr_)   # (class counterpart, else the row's own test)
                nn_ = torch.where(ins_, torch.zeros_like(Sr_), own_.unsqueeze(1).expand_as(Sr_))
            elif self._pairc and self._pcm is not None:
                _, ins_, _ = self._class_scores(x.detach()[bi, pi].float(), sk2_["gold"].to(x.device), offs, update=False)
                Sp = ins_.float() * 2.0 - 1.0                         # (inside exactly at the class's counterpart)
                nn_ = torch.zeros_like(Sp)
            elif self._pdet and self._pcm is not None:
                Sp, own_ = self._pdet_scores(x.detach()[bi, pi].float(), sk2_["gold"].to(x.device), update=False)
                nn_ = own_.unsqueeze(1)
            else:
                hs = torch.nn.functional.normalize(x.detach()[bi, pi].float() - self._sc_c.to(x.device), dim=1)
                Sp = hs @ self._sc_M.to(x.device).t()
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

    def _tab_stream(self):
        """(f_A, mu_A, mean gamma^2 / sigma^2) of the area the embedding norm feeds, or None"""
        g_ = next((n for n in self.diag if "embeddings" in n and n in self._gdown), None)
        if g_ is None:
            return None
        a_ = self._gdown.get(g_)
        m_ = self._mem.get(a_) if a_ is not None else None
        if m_ is None or "floor" not in m_ or g_ not in self._sgref:
            return None
        sig2_ = math.exp(2.0 * float(self._sgref[g_]))
        k_ = float(self.diag[g_].weight.detach().float().pow(2).mean()) / max(sig2_, 1e-30)
        return max(m_["floor"], 1e-30), m_["n"] / max(m_.get("nlast", m_["n"]), 1.0), k_

    @torch.no_grad()
    def _sink_w(self, area, x, am):
        """per sentence: w_0 = 1 + mean_h sum_i a_{i,0}^2 for an attention-input area, else None"""
        if not self._sink or "attention.self.query" not in area or len(self.areas[area]) < 2:
            return None
        mq, mk = self.areas[area][0], self.areas[area][1]
        H = int(getattr(self.L._unwrap(self.L.wrap_model).model.config, "num_attention_heads", 12))
        B, L, d = x.shape
        with torch.autocast("cuda", enabled=False):
            xf = x.detach().float()
            q = F.linear(xf, mq.weight.float(), mq.bias.float() if mq.bias is not None else None)
            k = F.linear(xf, mk.weight.float(), mk.bias.float() if mk.bias is not None else None)
            dh = q.shape[-1] // H
            q = q.view(B, L, H, dh).transpose(1, 2)
            k = k.view(B, L, H, dh).transpose(1, 2)
            lg = (q @ k.transpose(-1, -2)) / math.sqrt(dh)                # (B, H, L, L)
            amb = am.bool().to(x.device)
            lg = lg.masked_fill(~amb[:, None, None, :], float("-inf"))
            a0 = torch.softmax(lg, dim=-1)[..., 0]                         # (B, H, L): every query's weight on row 0
            a0 = a0 * amb[:, None, :].float()
            w = 1.0 + a0.pow(2).sum(-1).mean(1)                              # (B,)
        e = self._sinkw.get(area)
        self._sinkw[area] = [float(w.sum()), float(B)] if e is None else [e[0] + float(w.sum()), e[1] + float(B)]
        return w

    @torch.no_grad()
    def _class_scores(self, h, gold, offs, update):
        """(option class_detect) CLASS-LEVEL state conflict between a present class v (its running mean state, a
        class-level statistic) and an earlier record k, in the records' own geometry (centred by the records' mean,
        unit): v and k are MUTUAL nearest neighbours -- k is v's nearest record (same answer step), v is k's nearest
        present class, and v is nearer to k than k's nearest other earlier record (k's radius). At most one present
        class per record. Returns (similarities rows x records, inside rows x records, the counterpart per row)."""
        self._pdet_scores(h, gold, update=update)                 # (the running class sums, updated as pdet does)
        dev = h.device
        cnt, sm = self._pcm
        has = cnt > 0
        Pm = sm / cnt.clamp(min=1.0).unsqueeze(1)
        Sall = F.normalize(Pm - self._sc_c.to(dev), dim=1) @ self._sc_M.to(dev).t()     # (classes, records)
        Sall = Sall.masked_fill(~has.unsqueeze(1), -9.0)
        vbest = Sall.argmax(0)                                     # per record: its nearest present class
        g_ = gold.long().clamp(min=0)
        Sp = Sall[g_]                                              # (rows, records)
        same = offs.unsqueeze(1) == self._sc_off.to(dev).unsqueeze(0)
        jn = Sp.masked_fill(~same, -9.0).argmax(1)                 # v's nearest record (same step)
        R_ = Sp.shape[1]
        oh = F.one_hot(jn, R_).bool()
        mutual = vbest[jn] == g_
        within = Sp.gather(1, jn.unsqueeze(1)).squeeze(1) > self._sc_nn.to(dev)[jn]
        inside = oh & (mutual & within & has[g_]).unsqueeze(1)
        return Sp, inside, jn

    @torch.no_grad()
    def _pdet_scores(self, h, gold, update):
        """(option present_detect) per row: centred cosine to every earlier record, and to its own present class's running mean;
        centre = mean of the earlier records' means and the present classes' means (class-level statistics only)"""
        dev = h.device
        gold = gold.long().clamp(min=0)
        if self._pcm is None:
            # sized once from the dataset's class count: no host sync on gold per call
            C_ = int(self.L.CL_dataset.continual_config.get("NUM_CLASS", 0)) or 4096
            self._pcm = [torch.zeros(C_, device=dev), torch.zeros(C_, h.shape[1], device=dev)]
        if not self.L.CL_dataset.continual_config.get("NUM_CLASS") and int(gold.max()) >= self._pcm[0].shape[0]:
            C2_ = int(gold.max()) + 257
            n2_ = torch.zeros(C2_, device=dev); s2_ = torch.zeros(C2_, h.shape[1], device=dev)
            n2_[:self._pcm[0].shape[0]] = self._pcm[0]; s2_[:self._pcm[0].shape[0]] = self._pcm[1]
            self._pcm = [n2_, s2_]
        if update:
            self._pcm[0].index_add_(0, gold, torch.ones_like(gold, dtype=torch.float32))
            self._pcm[1].index_add_(0, gold, h)
        cnt, sm = self._pcm
        hasf = (cnt > 0).float()
        Pmean = sm / cnt.clamp(min=1.0).unsqueeze(1)
        H = self._recH.to(dev)
        # masked sums instead of boolean indexing: no host sync per step
        c = ((H.sum(0) + (Pmean * hasf.unsqueeze(1)).sum(0)) / (H.shape[0] + hasf.sum())).unsqueeze(0)
        M = F.normalize(H - c, dim=1)
        Pall = F.normalize(Pmean - c, dim=1) * hasf.unsqueeze(1)
        hs = F.normalize(h - c, dim=1)
        Sp = hs @ M.t()
        own = (hs * Pall[gold]).sum(1)
        own = torch.where(cnt[gold] > 0, own, torch.full_like(own, 2.0))   # (no own mean yet: not in conflict)
        return Sp, own

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
        fr_ = getattr(self.L, "fresh_readout", None)
        self._fresh = {id(m_) for m_ in fr_()} if callable(fr_) else set()
        self._Tf = None
        self._tid = task_id
        self.now, self.T, self.dD = {}, {}, {}
        self._tmdm = None
        self.steps, self.next_build = 0, 1
        self._pha = {}
        self._cmpBs, self._cmpOs = {}, {}                 # stacked comparator bases / operators: per task
        self.collect = "now" if self.eig else None
        self._Tx = {}
        self._pcm = None
        self._rhold = {}
        rh_list_ = []
        if self._oregion and self._cbas:
            # old heads (every readout map but the fresh one): per row, its class's region basis (padded)
            o_ = 0
            for m_ in self.areas["readout"]:
                n_ = m_.out_features
                if id(m_) not in self._fresh and all((o_ + j_) in self._cbas for j_ in range(n_)):
                    rm_ = max(self._cbas[o_ + j_].shape[1] for j_ in range(n_))
                    B_ = torch.zeros(n_, m_.in_features + 1, rm_, device=m_.weight.device)
                    for j_ in range(n_):
                        Bj_ = self._cbas[o_ + j_]
                        B_[j_, :, :Bj_.shape[1]] = Bj_.to(m_.weight.device)
                    self._rhold[id(m_)] = B_
                    rh_list_.append((m_, B_))
                    self._rhold[id(m_.weight)] = B_
                    if m_.bias is not None:
                        self._rhold[id(m_.bias)] = B_
                o_ += n_
        self._rhall, self._rhix = None, {}
        if rh_list_:
            # one stacked, zero-padded basis for every old row; per head its row indices into it
            rmx_ = max(B_.shape[2] for _, B_ in rh_list_)
            parts_, o_ = [], 0
            for m_, B_ in rh_list_:
                if B_.shape[2] < rmx_:
                    B_ = torch.cat([B_, B_.new_zeros(B_.shape[0], B_.shape[1], rmx_ - B_.shape[2])], 2)
                parts_.append(B_)
                self._rhix[id(m_)] = torch.arange(o_, o_ + B_.shape[0], device=B_.device)
                o_ += B_.shape[0]
            self._rhall = torch.cat(parts_, 0).contiguous()
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
                # the same rows as x[ba, pa], gathered by one index_select (cheaper on the host than 2-D indexing)
                xa_ = x.detach().reshape(-1, x.shape[-1]).index_select(0, self._ansfl)
                self._cxa[area] = xa_
            for m_ in self.areas[area]:                                # (the area's groups share this input)
                self._cxs[id(m_)] = (m_, area, xa_)
            return None
        return hook

    def _cmp_arm(self, b, p, cxs, head, xh, zh, sel):
        """Each conflicting answer step's input at every area and its mismatch with all memories' patterns; the
        learning signal at the area's output is caught in the backward pass.
        Batched: the areas of one width are stacked and projected in two batched products; the bases are stacked once
        per task (or per operator build under cmp_held)."""
        self._cmpsel = (b, p)
        self._cmpG = []
        # (every readout map reads the same last hidden sequence: each gets the readout's mismatch rows)
        ents = list(cxs.values()) + [(m_, "readout", None) for m_ in self.areas["readout"]]
        mods_, src_, order_ = {}, {}, []
        for mod, area, xin in ents:
            B_ = self._cmpB.get(area)
            if B_ is None:
                continue
            if area not in src_:
                if self._cmp_held:
                    T_ = self.T.get(area)
                    if isinstance(T_, _LowT):
                        B_ = T_.Ub                                 # (orthonormal [U, patterns] + zero columns)
                src_[area] = (xin, B_)
                order_.append(area)
                mods_[area] = []
            mods_[area].append(mod)
        grp_ = {}
        for a in order_:
            grp_.setdefault(src_[a][1].shape[0], []).append(a)
        with torch.no_grad():
            xr_ = None
            for d_, ars in grp_.items():
                key_ = tuple(ars)
                bs_ = tuple(src_[a][1] for a in ars)
                cb_ = self._cmpBs.get(key_)
                if cb_ is None or len(cb_[0]) != len(bs_) or any(x_ is not y_ for x_, y_ in zip(cb_[0], bs_)):
                    rm_ = max(B_.shape[1] for B_ in bs_)
                    Bp_ = torch.zeros(len(bs_), d_, rm_, device=bs_[0].device)
                    for i_, B_ in enumerate(bs_):
                        Bp_[i_, :, :B_.shape[1]] = B_.float()      # (zero columns change no product)
                    cb_ = self._cmpBs[key_] = (bs_, Bp_)
                Bp_ = cb_[1]
                xs_ = [src_[a][0] for a in ars]
                if all(x_ is not None for x_ in xs_):
                    X = torch.stack(xs_).index_select(1, sel).float()          # == xin[sel].float() per area
                else:
                    if xr_ is None:
                        xr_ = xh.detach()[b, p]
                    X = torch.stack([x_.index_select(0, sel) if x_ is not None else xr_ for x_ in xs_]).float()
                Xp = X - torch.bmm(torch.bmm(X, Bp_), Bp_.transpose(1, 2))
                del X                                                     # not needed after the projection
                self._cmpG.append([ars, Xp, None])
                for i_, a in enumerate(ars):
                    xpa_ = Xp[i_]
                    for mod in mods_[a]:
                        self._cmpX[id(mod)] = (None, xpa_, a)
            if self._cmpG:
                # the orthonormal mismatch bases depend only on the forward's rows: formed here, where the hook
                # already reads the armed count on the host, so the eigensolver's own sync costs no extra stall
                with torch.autocast("cuda", enabled=False):
                    Gs = torch.cat([torch.bmm(g_[1], g_[1].transpose(1, 2)) for g_ in self._cmpG])   # (areas, n, n)
                    ev_, V_ = torch.linalg.eigh(Gs)
                    Wt_ = V_ * ((ev_ > 1e-6 * ev_[:, -1:]).float() / ev_.clamp(min=1e-20).sqrt()).unsqueeze(1)
                    o_ = 0
                    for g_ in self._cmpG:
                        A_ = len(g_[0])
                        g_[2] = torch.bmm(g_[1].transpose(1, 2), Wt_[o_:o_ + A_])    # (A, d, n) orthonormal (or 0)
                        o_ += A_

    # ------------------------------------------------------------------ sleep: write the memory
    @torch.no_grad()
    def _write(self, area, C, n):
        """Delta-rule write of a trunk area's subspace memory (C None: deferred write after the second sleep read)."""
        kc_ = self._kc.pop(area, None)
        if kc_ is None:
            kc_ = self._keys(C, n)
        kc_ = {k_: (v_.to(kc_["dev"]) if torch.is_tensor(v_) else v_) for k_, v_ in kc_.items()}
        dev, K, lk, floor, tr_e, nd_ = kc_["dev"], kc_["K"], kc_["lk"], kc_["floor"], kc_["tr"], kc_["nd"]
        Ce = kc_.get("Ce") if C is not None else None             # the moment _keys already divided
        m = self._mem.get(area)
        if m is None or m["U"].shape[1] == 0:
            U, lam, tr, nn = K, lk, tr_e, float(n)
            beta = 1.0
        else:
            beta = float(n) / (m["n"] + float(n))
            Uo = m["U"].to(dev)
            lo = m["lam"].to(dev)
            B = kc_.get("B")                                             # the basis _keys made from the same
            if B is None:                                                #     [Uo, K]: bit for bit the same QR
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
                Kw_ = Kb * (1.0 - (1.0 - er_).clamp(min=0.0).sqrt()).unsqueeze(0)
                G_ = torch.eye(Kb.shape[0], device=dev) - Kw_ @ Kb.t()
                S = G_ @ S @ G_.t() + beta * ((Kb * lk) @ Kb.t())
            else:
                er_ = beta * torch.ones(Kb.shape[1], device=dev)
                Kw_ = Kb * (1.0 - (1.0 - er_).clamp(min=0.0).sqrt()).unsqueeze(0)
                G_ = torch.eye(Kb.shape[0], device=dev) - Kw_ @ Kb.t()
                S = G_ @ S @ G_.t() + beta * ((Kb * lk) @ Kb.t())
            S = 0.5 * (S + S.t())
            lw, W = torch.linalg.eigh(S)
            keep = lw > floor
            U, lam = B @ W[:, keep], lw[keep]
            tr, nn = (1.0 - beta) * m["tr"] + beta * tr_e, m["n"] + float(n)
        # TAIL: the episode's own per-position energy outside the new span U, running mean over episodes
        Uall = U
        if Ce is not None:
            tail_e = max(tr_e - float(((Ce @ Uall) * Uall).sum()), 0.0)
        else:
            Ua_ = B.t() @ Uall                                           # (the new span lies inside B)
            tail_e = max(tr_e - float(((kc_["Ceb"] @ Ua_) * Ua_).sum()), 0.0)
        tail = tail_e if m is None or "tail" not in m else (1.0 - beta) * m["tail"] + beta * tail_e
        self._mem[area] = {"U": U.cpu(), "Ub": _pad64(U.to(torch.bfloat16)), "lam": lam.cpu(), "tr": tr, "n": nn,
                           "tail": tail, "floor": float(floor), "nlast": float(n)}
        return U.shape[1], nd_

    @torch.no_grad()
    def _keys(self, C, n, area=None):
        """The episode's new keys: the Gavish-Donoho top of its per-position moment, their energies and its noise floor;
        for a deferred write also the merge basis B = [U_old, K] and the episode's moment inside it."""
        dev = C.device
        # C is the caller's own moment (popped from self.rec and dropped after this call): divided in place, so no
        # second d x d fp32 buffer is made; a non-deferred write reads Ce from the result
        Ce = C.div_(max(n, 1.0)) if C.dtype == torch.float32 else (C / max(n, 1.0)).float()
        ev, V = torch.linalg.eigh(Ce)
        ev = ev.clamp(min=0)
        k = _gd_rank(ev, n)
        sv = ev.sqrt().flip(0)
        D = float(ev.shape[0])
        bt = min(float(n), D) / max(float(n), D)
        tau = (0.56 * bt ** 3 - 0.95 * bt ** 2 + 1.82 * bt + 1.43) * float(sv[:int(min(n, D))].median())
        floor = tau * tau                                          # the episode's noise floor, in energy units
        kb = k
        out = {"dev": dev, "K": V[:, ev.shape[0] - kb:].contiguous(), "lk": ev[ev.shape[0] - kb:].clone(), "k": k,
               "floor": floor, "tr": float(ev.sum()), "nd": int(ev.shape[0])}
        m = self._mem.get(area) if area is not None else None
        if m is not None and m["U"].shape[1] > 0:
            Uo = m["U"].to(dev)
            B = torch.linalg.qr(torch.cat([Uo, out["K"]], 1)).Q
            out["Ceb"] = B.t() @ Ce @ B
            out["B"] = B.cpu()                     # _write reuses it (no second identical QR)
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
        model.train() if self._sleep_train else model.eval()   # (sleep reads in the wake forward's mode)
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
                self._sk2 = {"shape": (B, Lq), "ix_ans": ix_ans, "bi": bi, "pi": pi, "ma": ma, "gold": lb[bi, pi + 1]}
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
                        "n_ans": int(ix_ans.numel()), "am": am}
            model(input_ids=ids, attention_mask=am, use_cache=False, return_dict=True, output_hidden_states=False)
        self.collect = None
        self._sk = None

    @torch.no_grad()
    def sleep(self, task_id, loader, model):
        try:
            self._pcd_log(task_id)
        except Exception as ex_:                                       # (a diagnostic never stops a run)
            logger.info("[DIAG-D] skipped: %s" % ex_)
        """The finished model re-reads the episode just learnt (no learning, no graph; the loader iteration draws the
        global RNG) and the index is written."""
        import random as _rd
        import numpy as _np
        was = model.training
        model.train() if self._sleep_train else model.eval()   # (sleep reads in the wake forward's mode)
        self.L._mcache = None
        self.T = {}
        self._tmdm = None
        self.n_eps += 1
        self._leak_log(task_id)
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
                self._cidname[self.cids[t_]] = t_
            self._sleep_batch(ids, self.L.cur_attn, self.L.cur_labels, self.L.cur_tg)
            model(input_ids=ids, attention_mask=self.L.cur_attn, use_cache=False, return_dict=True,
                  output_hidden_states=False)
        self.collect = None
        self._sk = None
        if self._cregC:
            for k_, (C_, n_) in self._cregC.items():
                ev_, V_ = torch.linalg.eigh(C_ / n_)
                r_ = max(_gd_rank(ev_.clamp(min=0), n_), 1)
                self._cbas[k_] = V_[:, V_.shape[1] - r_:].float().contiguous()
            rn_ = [(self._cbas[k_].shape[1] / max(v_[1], 1.0), self._cbas[k_].shape[1], int(v_[1])) for k_, v_ in self._cregC.items()]
            mx_ = max(rn_)
            logger.info("[HIPPO-ENC] class regions after task %d: rank mean %.1f (min %d max %d) of %d | rank / n per class: "
                        "mean %.3f max %.3f (rank %d of n %d) -- a region near rank = n would be a sentence-level span" % (
                int(task_id), sum(self._cbas[k_].shape[1] for k_ in self._cregC) / len(self._cregC),
                min(self._cbas[k_].shape[1] for k_ in self._cregC), max(self._cbas[k_].shape[1] for k_ in self._cregC),
                next(iter(self._cregC.values()))[0].shape[0], sum(r[0] for r in rn_) / len(rn_), mx_[0], mx_[1], mx_[2]))
            self._cregC = {}
        self.next_cid += len(self.cids)
        self.cids = {}
        if was:
            model.train()
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
              if self._owmx and not (area == "tok" or area.startswith("g:") or area.startswith("tab:")):
                  Cc_ = C.detach().double().cpu()
                  e_ = self._full.get(area)
                  self._full[area] = [Cc_, float(n)] if e_ is None else [e_[0] + Cc_, e_[1] + float(n)]
                  del Cc_
              if area == "tok" or area.startswith("g:") or area.startswith("tab:"):
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
                  # (minimum-interference step, the decoder's rule) the per-coordinate noise floor is the GD cut of
                  # the coordinate energies (the smallest owned one); the earlier samples' weight is all of them over
                  # the episode just written
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
            logger.info("[HIPPO-ENC] after task %d: answer rows in conflict at the sleep read %.3f | deferred areas %d | "
                        "new keys' conflict share rho mean %.3f"
                        % (int(task_id), sum(self._g2st) / max(len(self._g2st), 1), len(g2d_),
                           sum(self._cest) / max(len(self._cest), 1)))
            self._kc, self._g2E = {}, {}
        self._cest = []
        if self._lgain_on:
            try:
                self._measure_gain(loader, model)
            except Exception as ex_:
                logger.info("[HIPPO-ENC] downstream-gain diagnostic skipped: %s" % ex_)
        self._massive_detect(task_id)
        if self._sgr:
            self._sgref = {k: float(v[0]) / max(v[1], 1.0) for k, v in self._sgr.items()}
            self._sgr = {}
        torch.cuda.empty_cache()
        if ranks:
            mb = sum(m["U"].numel() for m in self._mem.values()) * 6 / 2 ** 30
            tr_ = [(r, d) for (r, d) in zip(ranks, dims)]
            logger.info("[HIPPO-ENC] after task %d: %d areas written | rank / dim mean %.3f (rank %.1f of %.0f dims) | "
                        "readout owned %s | subspace memory %.2f GB (fp32 host + bf16 device)"
                        % (int(task_id), len(ranks), sum(r / d for r, d in tr_) / len(tr_),
                           sum(r for r, _ in tr_) / len(tr_), sum(d for _, d in tr_) / len(tr_),
                           ("%d of %d" % tuple(self._mem["readout"]["U"].shape[::-1])) if "readout" in self._mem else "-", mb))

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
                if self._tab_ln and (area == "tok" or area.startswith("tab:")) and self._tab_stream() is not None:
                    fA_, muA_, k_ = self._tab_stream()
                    D = fA_ / (fA_ + muA_ * k_ * ref_.float())
                    if self._tablog is None or (self._tid, area) not in self._tablog:
                        self._tablog = (self._tablog or set()) | {(self._tid, area)}
                        used_ = ref_ > 0
                        logger.info("[HIPPO-ENC] task %d table %s (stream units): f_A %.3g mu %.2f gamma^2/sigma^2 %.3g | "
                                    "share on used rows mean %.3f min %.4f max %.3f" % (
                                        int(self._tid), area, fA_, muA_, k_, float(D[used_].mean()) if bool(used_.any()) else -1,
                                        float(D[used_].min()) if bool(used_.any()) else -1, float(D[used_].max()) if bool(used_.any()) else -1))
                elif self._tab_owm and (area == "tok" or area.startswith("tab:")) and area in self._gfm:
                    f_, mu_ = self._gfm[area]
                    D = f_ / (f_ + mu_ * ref_.float())             # per-row minimum-interference share
                elif self._owm and self._owm_coord and area in self._gfm:
                    # MINIMUM-INTERFERENCE STEP for a per-coordinate parameter (gain g_i, table row v): its change moves
                    # every earlier output by dg_i x_i, so E_old (dg_i x_i)^2 = dg_i^2 e_i with e_i the coordinate's
                    # earlier energy; the same objective as the linear areas gives the share f / (f + mu e_i)
                    f_, mu_ = self._gfm[area]
                    D = f_ / (f_ + mu_ * ref_.float())
                self.dD[area] = D
                continue
            # trunk: T = c (I - U U^T), c = the tail's novelty share
            m = self._mem.get(area)
            if m is None or e is None:
                continue
            ev_now = e[0] / max(e[1], 1.0)
            tot, inu = float(ev_now[0]), float(ev_now[1])
            tail_now = max(tot - inu, 0.0)
            tail_past = m["tail"]
            c = max(0.0, min(1.0, 1.0 - tail_past / tail_now)) if tail_now > 0 else 1.0
            if self._trunk_c1 and area != "readout":
                c = 1.0                          # (option) trunk held by its span alone: the tail is fully plastic
            if area in self._cmpB and self._cmpB[area].shape[1] > 0:
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
                    m["Ua"], m["Ub"] = Ua_, None                # (one bf16 basis per area: U is its first r columns)
                self.T[area] = _LowT(Ua_, c)
            else:
                self.T[area] = _LowT(m["Ub"], c)
            if self._owm and "floor" in m:
                # MINIMUM-INTERFERENCE STEP (the decoder's rule, utils/hippo_lite.py): the step minimises <G, dW> +
                # (1/2eta)(|dW|^2 + mu/f tr(dW C dW^T)), C the earlier inputs' moment (the memory), mu = n_old / n_last,
                # f the area's GD noise floor: dW = -eta G (I + mu C / f)^-1. With C = U diag(lam) U^T + tau (I - U U^T),
                # tau = tail / (d - r): every held direction keeps f / (f + mu lam_i), the tail f / (f + mu tau); the
                # class-pattern columns stay fully held. A bias (the weight on the constant input 1 every earlier row
                # carries) keeps f / (f + mu).
                T_ = self.T[area]
                r_ = m["U"].shape[1]
                d_ = m["U"].shape[0]
                mu_ = m["n"] / max(m.get("nlast", m["n"]), 1.0)
                if self._lgain:
                    import re as _re
                    mm_ = _re.search(r"layer\.(\d+)\.", area)
                    if mm_ is not None and int(mm_.group(1)) in self._lgain:
                        mu_ = mu_ * self._lgain[int(mm_.group(1))] ** 2      # (option) interference at the decision
                f_ = max(m["floor"], 1e-30)
                lam_ = m["lam"].to(T_.Ub.device).float()
                tau_ = m["tail"] / max(d_ - r_, 1)
                av_ = torch.zeros(T_.Ub.shape[1], device=T_.Ub.device)
                av_[:r_] = f_ / (f_ + mu_ * lam_)
                T_.av, T_.b, T_.c = av_, float(f_ / (f_ + mu_ * tau_)), 1.0
                T_.sb = float(f_ / (f_ + mu_))
                c = T_.b
                if self._owmx and area in self._full:
                    Tx_ = self._Tx.get(area)
                    if Tx_ is None:
                        # EXACT closed form: (I + mu C / f)^-1 on the full earlier moment, then the class patterns held
                        Cs_, N_ = self._full[area]
                        # (on the device: host LAPACK is slow in a many-threaded process)
                        ev_, V_ = torch.linalg.eigh(Cs_.to(T_.Ub.device) / max(N_, 1.0))
                        sh_ = f_ / (f_ + mu_ * ev_.clamp(min=0.0))
                        Tx_ = (V_ * sh_.unsqueeze(0)) @ V_.t()
                        Pb_ = self._cmpB.get(area)
                        if Pb_ is not None and Pb_.shape[1] > 0:
                            Qp_ = torch.linalg.qr(Pb_.double().to(T_.Ub.device)).Q
                            Tx_ = Tx_ - Qp_ @ (Qp_.t() @ Tx_)
                            Tx_ = Tx_ - (Tx_ @ Qp_) @ Qp_.t()
                        Tx_ = (0.5 * (Tx_ + Tx_.t())).to(device=T_.Ub.device, dtype=torch.bfloat16).contiguous()
                        self._Tx[area] = Tx_
                        self._Txs = getattr(self, "_Txs", {})
                        self._Txs[area] = (float(sh_.mean()), float(sh_.min()), float(sh_.max()))
                    T_.dense = Tx_
                    c = self._Txs[area][0]
            # diagnostic: the share of the present input energy the operator passes: c^2 |x - U U^T x|^2 / |x|^2
            self._dg.append((area, c, (c * c) * tail_now / max(tot, 1e-30)))
        if self._gshare_down:
            # POST-LN GAINS: a norm's output IS the input of the area it feeds (BERT: the next block's q/k/v or the
            # FFN input; the last one the readout), so its gain g_j and bias b_j are weights on that area's input
            # coordinate j. They are held as that area's operator holds coordinate j: D_j = (T e_j)_j =
            # c (1 - |U_j|^2), U the held basis (span + pattern hold). Used in place of the per-coordinate novelty share,
            # whose statistics saturate after normalisation (every coordinate of a normalised input carries ~1).
            for g_, a_ in self._gdown.items():
                T_ = self.T.get(a_) if a_ is not None else None
                if isinstance(T_, _LowT) and ("g:" + g_) in self.dD:
                    if T_.av is not None:
                        # (minimum-interference operator: its diagonal, (T e_j)_j = b + sum_i (av_i - b) U_ji^2)
                        dj_ = (T_.b + (T_.Ub.float().pow(2) * (T_.av - T_.b)).sum(1)).clamp(0.0, 1.0) * T_.m
                        self.dD["g:" + g_] = dj_.to(self.dD["g:" + g_].device)
                    else:
                        rq_ = T_.Ub.float().pow(2).sum(1)
                        self.dD["g:" + g_] = ((1.0 - rq_).clamp(0.0, 1.0) * (T_.c * T_.m)).to(self.dD["g:" + g_].device)
        if self._fresh and self._Tf is None and self._cmpB.get("readout") is not None \
                and self._cmpB["readout"].shape[1] > 0:
            # FRESH READOUT (untied per-task heads): a head created for the present task carries no memory; its rows
            # are held only off the earlier memories' mean answer states (the places an earlier sentence sits, where a
            # new row rising would pull it into the new class), at full scale -- not by the readout's held variance
            # span U and its tail share c, which protect what the earlier rows store
            self._Tf = _LowT(_pad64(self._cmpB["readout"].to(torch.bfloat16)), 1.0)
            if bool(getattr(self.L, "fresh_free", False)):
                # (ablation) the fresh head unheld: an empty basis (x T = x)
                self._Tf = _LowT(torch.zeros_like(self._Tf.Ub), 1.0)
        self._build_diag()

    @torch.no_grad()
    def _leak_add(self, p, u, lr_):
        """Diagnostic (does not affect training): the applied step's size and its part on the earlier memories' pattern span (where the earlier
        answer rows sit, at the area's input): weights |u B|, B the area's pattern basis; biases, gains and tables
        act on every earlier row, so their whole step counts."""
        k_, a_ = self._pk[id(p)]
        if a_ == "readout" and k_.endswith("-w") and self._Tf is not None:
            for m_ in self.areas["readout"]:
                if m_.weight is p and id(m_) in self._fresh:
                    k_ = "readout-fresh-w"
        tot = u.float().pow(2).sum() * (lr_ * lr_)
        if k_.endswith("-w") and a_ is not None and self._cmpB.get(a_) is not None and u.dim() == 2 \
                and u.shape[1] == self._cmpB[a_].shape[0]:
            on = (u.float() @ self._cmpB[a_].float()).pow(2).sum() * (lr_ * lr_)
        else:
            on = tot
        e_ = self._leak.get(k_)
        if e_ is None:
            self._leak[k_] = [on, tot]
        else:
            e_[0] = e_[0] + on; e_[1] = e_[1] + tot

    @torch.no_grad()
    def _measure_gain(self, loader, model, n_batches=4, eps=1e-2):
        """DOWNSTREAM GAIN per encoder layer (eval mode, fp32, no graph, the present task's training sentences): each
        layer's output (all attended positions) is perturbed by a random direction of relative size eps; the gain is
        the relative change of the decision state (position 0, last layer) over eps. RNG states restored."""
        import random as _rd
        import numpy as _np
        enc = getattr(model, "enc", None)
        if enc is None or not hasattr(enc, "encoder"):
            return
        layers = list(enc.encoder.layer)
        rs_ = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), _rd.getstate(), _np.random.get_state())
        was = model.training
        model.eval()
        col0, self.collect = self.collect, None
        self.pause_hooks()
        gen = torch.Generator(device=next(model.parameters()).device).manual_seed(4321)
        acc = torch.zeros(len(layers), dtype=torch.float64)
        nb = 0
        try:
            for bi, batch in enumerate(loader):
                if bi >= n_batches:
                    break
                ids, am = batch["input_ids_with_ans"], batch["attention_mask_with_ans"]
                with torch.autocast("cuda", enabled=False):
                    h0 = enc(input_ids=ids, attention_mask=am).last_hidden_state[:, 0].float()
                    for l, lay in enumerate(layers):
                        def hk(mod, inp, out, _am=am):
                            o = out[0] if isinstance(out, tuple) else out
                            u = torch.randn(o.shape, generator=gen, device=o.device, dtype=o.dtype) * _am.unsqueeze(-1).to(o.dtype)
                            on = o.pow(2).sum(-1, keepdim=True).sqrt()
                            u = u / u.pow(2).sum(-1, keepdim=True).sqrt().clamp(min=1e-12) * on * eps
                            o2 = o + u
                            return (o2,) + tuple(out[1:]) if isinstance(out, tuple) else o2
                        hh = lay.register_forward_hook(hk)
                        try:
                            h1 = enc(input_ids=ids, attention_mask=am).last_hidden_state[:, 0].float()
                        finally:
                            hh.remove()
                        acc[l] += float(((h1 - h0).norm(dim=1) / h0.norm(dim=1).clamp(min=1e-12)).mean()) / eps
                nb += 1
        finally:
            self.collect = col0
            self.resume_hooks()
            if was:
                model.train()
            torch.set_rng_state(rs_[0]); torch.cuda.set_rng_state_all(rs_[1]); _rd.setstate(rs_[2]); _np.random.set_state(rs_[3])
        if nb:
            g = (acc / nb).tolist()
            # running mean over the tasks seen (each sleep measures the finished model)
            for l, v in enumerate(g):
                o = self._lgain.get(l)
                self._lgain[l] = v if o is None else 0.5 * (o + v)
            logger.info("[HIPPO-ENC] downstream gain per layer (relative change of the decision state / relative "
                        "change of the layer output): %s" % " ".join("L%d %.2f" % (l + 1, self._lgain[l]) for l in sorted(self._lgain)))

    @torch.no_grad()
    def _massive_detect(self, task_id, thr=0.1):
        """MASSIVE COORDINATES of the residual stream, from the gains' own stored statistics: any coordinate carrying
        more than `thr` of a norm's (normalised) input energy (i.e. above thr x d times the per-coordinate mean)."""
        M, sh = set(), []
        for g_, m_ in self.diag.items():
            e_ = self.old.get("g:" + g_)
            if e_ is None or m_.weight.shape[0] != self._dstream:
                continue
            ev = e_[0].float() / max(float(e_[1]), 1.0)
            r_ = ev / ev.sum().clamp(min=1e-30)
            j_ = int(r_.argmax())
            sh.append((j_, float(r_[j_])))
            M.update((r_ > thr).nonzero(as_tuple=True)[0].tolist())
        self._M = torch.tensor(sorted(M), dtype=torch.long) if M else None
        logger.info("[HIPPO-ENC] after task %d: massive stream coordinates %s (share > %.2f of a norm's input energy); "
                    "per norm top coordinate:share %s" % (int(task_id), sorted(M), thr,
                                                           " ".join("%d:%.2f" % x for x in sh)))

    def _leak_log(self, task_id):
        if self._sinkw:
            import re as _re
            ws_ = sorted(((int(_re.search(r"layer\.(\d+)\.", a).group(1)), v[0] / max(v[1], 1.0)) for a, v in self._sinkw.items()
                          if _re.search(r"layer\.(\d+)\.", a)))
            logger.info("[HIPPO-ENC] task %d: decision-row sink weight w_0 = 1 + sum_i a_i0^2 per layer: %s" % (
                int(task_id), " ".join("L%d %.2f" % (l + 1, w) for l, w in ws_)))
            self._sinkw = {}
        if getattr(self, "_cmpS", None):
            t_ = torch.stack(self._cmpS).sum(0)
            logger.info("[HLE-DIAG] task %d: comparator give-back share s_q (objective) mean %.4f over %d armed areas-steps" % (
                int(task_id), float(t_[0] / t_[1].clamp(min=1.0)), len(self._cmpS)))
            self._cmpS = []
        if not self._leak:
            return
        it = sorted(self._leak.items())
        logger.info("[HLE-DIAG] step leak task %d (every 10th step; sqrt of summed lr^2 |step on the earlier patterns|^2 "
                    "/ |step|^2): %s" % (int(task_id), " | ".join("%s %.2e/%.2e" % (k, float(v[0]) ** 0.5, float(v[1]) ** 0.5)
                                                                 for k, v in it)))
        self._leak = {}

    def _build_diag(self):
        """Diagnostic (logging only): tail share c and passed energy per kind of area, gain / table shares."""
        dg_, self._dg = self._dg, []
        if not dg_:
            return
        tr_ = [(c, p) for a, c, p in dg_ if a != "readout"]
        kc_ = {}
        for a, c, p in dg_:
            kc_.setdefault(_kind(a), []).append((c, p))
        kd_ = {}
        for g_, a_ in self._gdown.items():
            if ("g:" + g_) in self.dD:
                kd_.setdefault(_kind(a_ or "none"), []).append(float(self.dD["g:" + g_].mean()))
        if self.steps in (1, 64):
            logger.info("[HLE-DIAG] build step %d task %d per kind: c / passed %s | gains D by the area they feed: %s"
                        % (self.steps, self._tid,
                           " ".join("%s %.3f/%.4f" % (k, sum(c for c, _ in v) / len(v), sum(p for _, p in v) / len(v))
                                    for k, v in kc_.items()),
                           " ".join("LN>%s %.3f" % (k, sum(v) / len(v)) for k, v in kd_.items())))
        ro_ = [(c, p) for a, c, p in dg_ if a == "readout"]
        gD_ = [self.dD[a] for a in self.dD if a.startswith("g:")]
        gD = torch.cat(gD_) if gD_ else None
        gO_ = [self.eig[a][2].float() for a in self.dD if a.startswith("g:") and a in self.eig]
        gO = float(torch.cat(gO_).mean()) if gO_ else -1.0          # owned (GD top of the stored energies)
        tb_ = {a: self.dD[a] for a in self.dD if a == "tok" or a.startswith("tab:")}
        tbs = " ".join("%s %.3f" % (a.split(".")[-1], float(D[self.old[a][0] > 0].mean()) if (self.old.get(a) is not None
                       and bool((self.old[a][0] > 0).any())) else -1.0) for a, D in tb_.items())
        logger.info("[HLE-DIAG] build step %d task %d: trunk c mean %.3f min %.3f max %.3f, passed energy mean %.4f | "
                    "readout c %s passed %s | gains D mean %.3f (zero %.3f, of which owned %.3f) | tables D on used rows: %s"
                    % (self.steps, self._tid, sum(c for c, _ in tr_) / max(len(tr_), 1),
                       min((c for c, _ in tr_), default=-1), max((c for c, _ in tr_), default=-1),
                       sum(p for _, p in tr_) / max(len(tr_), 1),
                       "%.3f" % ro_[0][0] if ro_ else "-", "%.4f" % ro_[0][1] if ro_ else "-",
                       float(gD.mean()) if gD is not None else -1, float((gD == 0).float().mean()) if gD is not None else -1,
                       gO, tbs))

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
            b_ = getattr(mod, "bias", None)
            for g in (mod.weight.grad, b_.grad if isinstance(b_, torch.nn.Parameter) else None):
                # (gain and norm bias alike: the gain's per-coordinate share)
                if D is not None and g is not None:
                    if g.dtype == torch.float32:
                        gs_.append(g); Ds_.append(D.to(g.device))
                    else:
                        g.copy_((g.float() * D.to(g.device)).to(g.dtype))
        if gs_:
            torch._foreach_mul_(gs_, Ds_)                                 # fp32 g * D, written in place
        self._rowproj_all([(m_, m_.weight.grad, m_.bias.grad if m_.bias is not None else None)
                           for m_ in self.areas["readout"] if id(m_) in self._rhold and m_.weight.grad is not None])
        if not self.T or self._cmpsel is None or not self._cmpX:
            return
        xp_ = {}
        for area, mods in self.areas.items():
            for mod in mods:
                c_ = self._cmpX.get(id(mod))
                if c_ is None or mod.weight.grad is None:
                    continue
                if self._Tf is not None and id(mod) in self._fresh:
                    continue                    # (the pattern-only operator removes nothing along the mismatch)
                X_, Xp_, ar_ = c_
                xp_[ar_] = Xp_
                self._cmpd[id(mod.weight)] = (ar_,)
        self._cmpA = {}
        if xp_:
            self._cmp_batched(xp_)

    @torch.no_grad()
    def _cmp_ops(self, ars, dev):
        """The stacked operators of a group of areas of one width (zero-padded spans; cached per operator build):
        kind 'c' T = s (I - U U^T), kind 'owm' T = m (b I + U diag(av - b) U^T), 'none' T = 0; with the objective's
        give-back terms (U_r, lam, tau, mu, f) where cmp_owm applies (else s_q = 1)."""
        Ts_ = tuple(self.T.get(a) for a in ars)
        key_ = tuple(ars)
        co_ = self._cmpOs.get(key_)
        if co_ is not None and all(x_ is y_ for x_, y_ in zip(co_[0], Ts_)):
            return co_[1]
        kinds = set()
        for T_ in Ts_:
            if T_ is None:
                continue
            if not isinstance(T_, _LowT) or T_.dense is not None:
                kinds.add("other")
            else:
                kinds.add("owm" if T_.av is not None else "c")
        if "other" in kinds or len(kinds) > 1:
            self._cmpOs[key_] = (Ts_, None)
            return None
        kind = kinds.pop() if kinds else "none"
        A_, d_ = len(ars), None
        for T_ in Ts_:
            if T_ is not None:
                d_ = T_.Ub.shape[0]
        rm_ = max([T_.Ub.shape[1] for T_ in Ts_ if T_ is not None] or [1])
        d_ = d_ or 1
        U_ = torch.zeros(A_, d_, rm_, dtype=torch.bfloat16, device=dev)
        W_ = torch.zeros(A_, 1, rm_, device=dev)                          # (owm) av - b on the span columns
        sc_ = torch.zeros(A_, 1, 1, device=dev)                           # c m ('c') or m ('owm'); 0 where T is None
        b_ = torch.zeros(A_, 1, 1, device=dev)
        own_ = torch.zeros(A_, 1, device=dev)
        rr_, ownl_ = 1, [False] * A_
        for i_, (a, T_) in enumerate(zip(ars, Ts_)):
            if T_ is None:
                continue
            U_[i_, :, :T_.Ub.shape[1]] = T_.Ub
            if kind == "owm":
                W_[i_, 0, :T_.Ub.shape[1]] = (T_.av - T_.b) if torch.is_tensor(T_.av) else float(T_.av - T_.b)
                b_[i_] = float(T_.b)
                sc_[i_] = float(T_.m)
            else:
                sc_[i_] = float(T_.c * T_.m)
            if self._cmp_owm and a in self._mem and "floor" in self._mem[a]:
                own_[i_] = 1.0
                ownl_[i_] = True
                rr_ = max(rr_, self._mem[a]["U"].shape[1])
        Ur_ = torch.zeros(A_, d_, rr_, device=dev)
        lam_ = torch.zeros(A_, rr_, 1, device=dev)
        tau_ = torch.zeros(A_, 1, device=dev)
        mu_ = torch.zeros(A_, 1, device=dev)
        f_ = torch.ones(A_, 1, device=dev)
        for i_, (a, T_) in enumerate(zip(ars, Ts_)):
            if not ownl_[i_]:
                continue
            m_ = self._mem[a]
            r_ = m_["U"].shape[1]
            Ur_[i_, :, :r_] = T_.Ub[:, :r_].float()
            lam_[i_, :r_, 0] = m_["lam"].to(dev).float()
            tau_[i_] = m_["tail"] / max(m_["U"].shape[0] - r_, 1)
            mu_[i_] = m_["n"] / max(m_.get("nlast", m_["n"]), 1.0)
            f_[i_] = max(m_["floor"], 1e-30)
        ops_ = (kind, U_, W_, sc_, b_, own_, Ur_, lam_, tau_, mu_, f_, any(ownl_))
        self._cmpOs[key_] = (Ts_, ops_)
        return ops_

    @torch.no_grad()
    def _cmp_batched(self, xp_):
        """The comparator's bases for the step's give-back, every area of one width at once: Q~ the orthonormalised
        mismatch rows, M = Q~ diag(s) - T Q~."""
        with torch.autocast("cuda", enabled=False):
            for ars, Xp, Q in self._cmpG:
                keep_ = [i_ for i_, a in enumerate(ars) if a in xp_]
                if not keep_:
                    continue
                if len(keep_) != len(ars):
                    ars = [ars[i_] for i_ in keep_]
                    ix_ = self._cmpOs.get(("ix",) + tuple(keep_))
                    if ix_ is None:
                        ix_ = self._cmpOs[("ix",) + tuple(keep_)] = torch.tensor(keep_, device=Q.device)
                    Q = Q.index_select(0, ix_)
                ops_ = self._cmp_ops(ars, Q.device)
                if ops_ is None:
                    for i_, a in enumerate(ars):                          # (dense / mixed operators: per area)
                        Q_ = Q[i_]
                        T_ = self.T.get(a)
                        TQ_ = self._tmm(Q_.t(), T_).t() if T_ is not None else None
                        self._cmpA[a] = (Q_, Q_ - TQ_ if T_ is not None else Q_)
                    continue
                kind, U_, W_, sc_, b_, own_, Ur_, lam_, tau_, mu_, f_, anyown_ = ops_
                x_ = Q.transpose(1, 2)                                                    # (A, n, d) fp32
                if kind == "none":
                    for i_, a in enumerate(ars):
                        self._cmpA[a] = (Q[i_], Q[i_])
                    continue
                y_ = torch.bmm(x_.to(torch.bfloat16), U_, out_dtype=torch.float32)
                if kind == "owm":
                    y_.mul_(W_)
                    TQ = torch.bmm(y_.to(torch.bfloat16), U_.transpose(1, 2), out_dtype=torch.float32)
                    TQ.add_(x_ * b_).mul_(sc_)
                else:
                    TQ = torch.bmm(y_.to(torch.bfloat16), U_.transpose(1, 2), out_dtype=torch.float32)
                    TQ = torch.sub(x_, TQ, out=TQ).mul_(sc_)
                TQ = TQ.transpose(1, 2)                                                   # (A, d, n)
                if anyown_:
                    # (option) GIVE-BACK SIZED BY THE OBJECTIVE: s_q = f / (f + mu q^T C q),
                    # C = U diag(lam) U^T + tau (I - U U^T); areas without the objective keep s_q = 1
                    pq_ = torch.bmm(Ur_.transpose(1, 2), Q)                              # (A, r, n)
                    p2_ = pq_.pow(2)
                    qcq_ = (lam_ * p2_).sum(1) + tau_ * (1.0 - p2_.sum(1)).clamp(min=0.0)
                    s_ = torch.where(own_ > 0, f_ / (f_ + mu_ * qcq_), torch.ones_like(qcq_))  # (A, n)
                    M = Q * s_.unsqueeze(1) - TQ
                    v_ = (Q.pow(2).sum(1) > 0).float() * own_
                    self._cmpS.append(torch.stack([(s_ * v_).sum(), v_.sum()]))
                else:
                    M = Q - TQ
                for i_, a in enumerate(ars):
                    self._cmpA[a] = (Q[i_], M[i_])                    # (I - T) Q: what the index removes
