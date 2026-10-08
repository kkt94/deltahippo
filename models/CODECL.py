"""CODE-CL (Apolinario, Choudhary & Roy, ICCV 2025): conceptor-based gradient projection, on a decoder-only LLM.

Reference: the paper (Eqs. 1-6 and 8, Alg. 1) and the official code github.com/mapolinario94/CODE-CL
(models/conceptor_operations.py, cl_method/code_cl.py, cl_method/strategy.py, models/nn_models/layers.py,
models/nn_models/alexnet.py, script.sh). It runs under the same full fine-tuning protocol as DeltaHippo. The released
configuration uses the task-agnostic form with K = 0 (--codecl_K 0): no task-overlap analysis and no task-specific
mixing matrices, only the conceptor projection of the updates.

* Every nn.Linear of the LLM (q/k/v/o/gate/up/down of every block and lm_head, tied to the input table) keeps a
  conceptor C of its past input space (q/k/v and gate/up share one: same input tensor).
* After task t (except the last): conceptor of the task's inputs from one batch of b = 125 training sentences (every
  non-pad token is one feature row), C_post = R (R + alpha^-2 I)^-1, R = X^T X / rows; aperture adapted by x1.1 until
  sum ||C x|| / sum ||x|| >= 0.95; merged C <- C_post OR C (task 0: C <- C_post); singular values < 0.2 set to 0.
* Before task t >= 1 (task-overlap analysis, only when K > 0): pre-conceptor of task t's inputs on the current model; if
  Theta(C AND C_pre) / Theta(C) > 0.5 (and in_features > 50, K > 0), the top-K singular vectors U of C AND C_pre
  give the layer a learnable K x K matrix M_t: W_eff = W + W U M_t U^T - sg(W) U U^T (official layers.py forward).
  The evaluation here is class-incremental with no task identity, so the effective weight learnt for task t is
  written into W at the end of task t (W <- W_eff value) -- the only task-agnostic use of M_t.
* From task 1 on, the step on every projected weight is right-multiplied by (I - C) (Eq. 8; official
  update_gradient: grad <- grad - grad C). Under the protocol's AdamW the projection is applied to the AdamW step
  (decoupled decay included), so the written change is (I - C)-shaped as with the paper's SGD.
* Normalisation gains get no update from task 1 on (official update_gradient zeroes the batch-norm gradients).
* Optimiser state is new for every task (strategy.train builds a new optimiser per task).
"""
import logging
import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.cl_proj_common import ProjLearnerBase, str2bool

logger = logging.getLogger()


def get_CODECL_params(parser):
    parser.add_argument("--pdr_no_liger", type=str2bool, default=True, help="no fused linear-cross-entropy kernel")
    parser.add_argument("--codecl_aperture", type=float, default=6.0, help="conceptor aperture alpha")
    parser.add_argument("--codecl_mem_thr", type=float, default=0.95, help="target of the aperture-adaptation loop: sum ||C x|| / sum ||x||")
    parser.add_argument("--codecl_lower_sval", type=float, default=0.2, help="singular values of C below this -> 0")
    parser.add_argument("--codecl_K", type=int, default=80, help="number of free dimensions K per overlapping layer (0 = task-agnostic form, no mixing matrices)")
    parser.add_argument("--codecl_eps", type=float, default=0.5, help="task-overlap threshold on the capacity ratio")
    parser.add_argument("--codecl_basis_bs", type=int, default=125, help="sentences used to build a conceptor")
    parser.add_argument("--codecl_log_every", type=int, default=25, help="log the removed share every n steps")
    parser.add_argument("--codecl_foldcheck", type=int, default=0,
                        help="Diagnostic (evaluation only): on this many training sentences of the task, compare the "
                             "training-time forward (with M) before the fold with the plain forward after it; does "
                             "not affect training (0 = off)")


# ---------------------------------------------------------------- conceptor operations (models/conceptor_operations.py)
# Every matrix these operations touch is symmetric positive semi-definite, so the official code's torch.svd /
# torch.linalg.pinv are evaluated through a symmetric eigendecomposition (singular values = |eigenvalues|, same
# vectors), which is exact in arithmetic and much faster for the wide inputs of an LLM.
def sym_eig(A):
    """(U, S) of a symmetric matrix, S = |eigenvalues| sorted descending (= torch.svd of a symmetric PSD matrix)."""
    ev, U = torch.linalg.eigh(0.5 * (A + A.T))
    S = ev.abs()
    idx = torch.argsort(S, descending=True)
    return U[:, idx], S[idx]


def sym_pinv(A, tol):
    U, S = sym_eig(A)
    inv = torch.where(S > tol, 1.0 / S.clamp_min(tol), torch.zeros_like(S))
    return (U * inv) @ U.T


def aperture_adaptation(C, gamma):
    I = torch.eye(C.size(0), device=C.device, dtype=C.dtype)
    C = C @ torch.linalg.inv(C + (gamma ** -2) * (I - C))
    U, S = sym_eig(C)
    return (U * torch.clamp(S, min=1e-8, max=0.9999999)) @ U.T


def compute_conceptor(data, aperture):
    # data: (n, rows)
    I = torch.eye(data.size(0), device=data.device, dtype=data.dtype)
    R = data @ data.T / data.size(1)
    return R @ torch.linalg.inv(R + (aperture ** -2) * I)


def not_op(C):
    return torch.eye(C.size(0), device=C.device, dtype=C.dtype) - C


def and_op(C, B):
    dim = C.size(0)
    tol = 1e-6
    UC, SC = sym_eig(C)
    UB, SB = sym_eig(B)
    nC = int(torch.sum(SC > tol))
    nB = int(torch.sum(SB > tol))
    UC0 = UC[:, nC:]
    UB0 = UB[:, nB:]
    W, Sig = sym_eig(UC0 @ UC0.T + UB0 @ UB0.T)
    nS = int(torch.sum(Sig > tol))
    Wgk = W[:, nS:]
    I = torch.eye(dim, device=C.device, dtype=C.dtype)
    CandB = Wgk @ torch.linalg.inv(Wgk.T @ (sym_pinv(C, tol) + sym_pinv(B, tol) - I) @ Wgk) @ Wgk.T
    U, S = sym_eig(CandB)
    return (U * torch.clamp(S, min=1e-8, max=0.9999999)) @ U.T


def or_op(C, B):
    return not_op(and_op(not_op(C), not_op(B)))


def capacity(C):
    return sym_eig(C)[1].mean()


def task_conceptor(act, aperture, thr, max_it=200):
    """code_cl.update_basis (one layer): C = R (R + a^-2 I)^-1, then aperture_adaptation(C, 1.1) until
    sum_j ||C x_j|| / sum_j ||x_j|| >= thr. Every step keeps the eigenvectors of R, so it runs on the eigenvalues:
    s = l / (l + a^-2); adaptation s <- clamp(s / (s + 1.1^-2 (1 - s)), 1e-8, 1 - 1e-7)."""
    R = act @ act.T / act.size(1)
    U, lam = sym_eig(R)
    s = lam / (lam + aperture ** -2)
    Z = U.T @ act                                   # (n, rows)
    xn = torch.norm(act, dim=0).sum()
    it = 0
    while True:
        ratio = torch.norm(s[:, None] * Z, dim=0).sum() / xn
        if ratio < thr and it < max_it:
            s = torch.clamp(s / (s + (1.1 ** -2) * (1 - s)), min=1e-8, max=0.9999999)
            it += 1
        else:
            break
    return (U * s) @ U.T, it, float(ratio)


class CODECL(ProjLearnerBase):
    TAG = "CODE-CL"

    def __init__(self, params, CL_dataset, accelerator):
        self.conc = {}           # group -> conceptor of all past tasks (fp32, GPU; used by the projection)
        self.conc64 = {}         # group -> the same conceptor in fp64 on the host (used by the conceptor algebra)
        self.free = {}           # linear name -> (U (n x K), M parameter) for the present task
        self.state = {}
        self.ostep = 0
        self.rm_stats = defaultdict(list)
        self._g = torch.Generator(device="cpu")
        self._g.manual_seed(int(getattr(params, "seed", 0) or 0) + 1234)
        super().__init__(params, CL_dataset, accelerator)
        model = self.base_model()
        self.linears, self.group_of = {}, {}
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                self.linears[name] = mod
                g = name
                for a, b in (("k_proj", "q_proj"), ("v_proj", "q_proj"), ("up_proj", "gate_proj")):
                    if name.endswith("." + a):
                        g = name[: -len(a)] + b
                self.group_of[name] = g
        self.lin_param = {id(m.weight): n for n, m in self.linears.items()}
        self.norm_ids = {id(p) for n, p in model.named_parameters() if p.dim() == 1 and "norm" in n.lower()}
        others = [n for n, p in model.named_parameters() if id(p) not in self.lin_param and id(p) not in self.norm_ids]
        logger.info("[CODE-CL] %d linear maps (%d conceptor groups); %d norm gains (frozen from task 1); other: %s"
                    % (len(self.linears), len(set(self.group_of.values())), len(self.norm_ids), others))

    # ------------------------------------------------------------------ activations of one basis batch
    @torch.no_grad()
    def _activations(self, loader):
        """Inputs of every conceptor group over the first b sentences of a (shuffled) training loader."""
        model = self.base_model()
        groups = sorted(set(self.group_of.values()))
        feats = defaultdict(list)
        box = {}

        def make_hook(g):
            def hook(mod, inp, out):
                x = inp[0]
                feats[g].append(x.reshape(-1, x.shape[-1])[box["m"]].float().cpu())
            return hook
        hs = [self.linears[g].register_forward_hook(make_hook(g)) for g in groups]
        model.eval()
        need = int(self.params.codecl_basis_bs)
        got = 0
        try:
            for batch in loader:
                ids = batch["input_ids_with_ans"][: need - got]
                am = batch["attention_mask_with_ans"][: need - got]
                box["m"] = am.reshape(-1).bool()
                model(input_ids=ids, attention_mask=am, use_cache=False, return_dict=True)
                got += ids.shape[0]
                if got >= need:
                    break
        finally:
            for h in hs:
                h.remove()
            model.train()
        return {g: torch.cat(v, 0) for g, v in feats.items()}, got  # (rows, n) on the host

    def _task_conceptor(self, act, lower):
        return task_conceptor(act, float(self.params.codecl_aperture), float(self.params.codecl_mem_thr))

    @staticmethod
    def _lower(C, lower):
        if lower > 0:
            U, S = sym_eig(C)
            S = torch.where(S < lower, torch.zeros_like(S), S)
            C = (U * S) @ U.T
        return C

    # ------------------------------------------------------------------ task boundaries
    def begin_task(self, task_id):
        super().begin_task(task_id)
        self.state, self.ostep = {}, 0
        self.rm_stats = defaultdict(list)
        self.free = {}
        if int(task_id) > 0 and int(self.params.codecl_K) > 0:
            self._overlap(int(task_id))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _overlap(self, task_id):
        """strategy.task_similarity + layers.measure_tasks_similarity."""
        act, n = self._activations(self.train_loader_list[task_id])
        K = int(self.params.codecl_K)
        lines, nfree = [], 0
        for g in sorted(act):
            B, it, r = self._task_conceptor(act[g].cuda().double().T.contiguous(), 0.2)
            B = self._lower(B, 0.2)                        # task_similarity: lower_sval_bound=0.2
            C = self.conc64[g].cuda()
            Cand = and_op(C, B)
            ratio = float(capacity(Cand) / capacity(C))
            members = [nm for nm, gg in self.group_of.items() if gg == g]
            # the readout gets no free dimensions: in the official code the classifier heads are plain nn.Linear
            # (alexnet.py fc3 / resnet18.py linear), only CustomLinear/CustomConv2d layers carry M; with a tied
            # table, folding W_eff into lm_head would also rotate every input embedding
            is_head = self.linears[g] is self.base_model().get_output_embeddings()
            if ratio > float(self.params.codecl_eps) and C.size(0) > 50 and not is_head:
                U, S = sym_eig(Cand)
                U = U[:, :K].float().contiguous()
                for nm in members:
                    bound = 1.0 / math.sqrt(K)                # nn.Linear(K, K) default init
                    M = (torch.rand(K, K, generator=self._g) * 2 - 1) * bound
                    self.free[nm] = (U, nn.Parameter(M.to(U.device)))
                nfree += len(members)
            lines.append("%s:%.2f" % (g.replace("model.layers.", "L").replace("self_attn.", "").replace("mlp.", ""),
                                      ratio))
            del B, Cand, C
        del act
        for nm, (U, M) in self.free.items():
            mod = self.linears[nm]

            def make_fwd(mod, U, M):
                def fwd(x):
                    W = mod.weight
                    WU = W @ U
                    z = F.linear(x, U.T)                     # U^T x
                    y = F.linear(x, W) + F.linear(F.linear(z, M), WU) - F.linear(z, WU.detach())
                    return y if mod.bias is None else y + mod.bias
                return fwd
            mod.forward = make_fwd(mod, U, M)
        torch.cuda.empty_cache()
        logger.info("[CODE-CL] task %d overlap (%d sentences): %d/%d linear maps in case 1 (ratio > %.2f) with K=%d; "
                    "capacity ratios: %s" % (task_id, n, nfree, len(self.linears), float(self.params.codecl_eps), K,
                                             " ".join(lines)))

    def end_task(self, task_id):
        self.log_peak("task %d training" % int(task_id))
        self.state = {}
        nchk = int(getattr(self.params, "codecl_foldcheck", 0) or 0)
        chk = self._foldcheck(task_id, nchk) if (self.free and nchk > 0) else None
        if self.free:
            with torch.no_grad():                          # task-agnostic evaluation: W <- W_eff
                for nm, (U, M) in self.free.items():
                    mod = self.linears[nm]
                    W = mod.weight
                    WU = W @ U
                    W.add_(WU @ M @ U.T - WU @ U.T)
                    del mod.forward
            self.free = {}
        if chk is not None:
            self._foldcheck(task_id, nchk, before=chk)
        torch.cuda.empty_cache()
        if int(task_id) < int(self.CL_dataset.continual_config["NUM_TASK"]) - 1:
            self._update_conceptors(int(task_id))
            self.log_peak("task %d conceptor update" % int(task_id))
        super().end_task(task_id)

    def _update_conceptors(self, task_id):
        """strategy.update_basis -> code_cl.update_basis."""
        act, n = self._activations(self.train_loader_list[task_id])
        lower = float(self.params.codecl_lower_sval)
        lines = []
        for g in sorted(act):
            B, it, r = self._task_conceptor(act[g].cuda().double().T.contiguous(), lower)
            C = B if g not in self.conc64 else or_op(self.conc64[g].cuda(), B)
            C = self._lower(C, lower)
            self.conc64[g] = C.cpu()
            self.conc[g] = C.float()
            S = sym_eig(C)[1]
            lines.append("%s:cap=%.3f,dirs=%d/%d,it=%d" % (
                g.replace("model.layers.", "L").replace("self_attn.", "").replace("mlp.", ""), float(S.mean()),
                int((S > 1e-4).sum()), S.numel(), it))
            del B
        del act
        torch.cuda.empty_cache()
        logger.info("[CODE-CL] task %d conceptors (%d sentences, alpha=%g, gain target %.2f): %s"
                    % (task_id, n, float(self.params.codecl_aperture), float(self.params.codecl_mem_thr),
                       " ".join(lines)))

    @torch.no_grad()
    def _foldcheck(self, task_id, n, before=None):
        """Answer-token CE / teacher-forced token accuracy / sequence exact (all answer tokens) on the first n training
        sentences of the task (fixed order, no shuffle draw), in eval mode with the same bf16 autocast forward.
        Called once with the training-time forward (M active) before the fold and once with the plain model after.
        Diagnostic (evaluation only); does not affect training."""
        model = self.base_model()
        model.eval()
        ds = self.train_loader_list[task_id].dataset
        out_all, stats = [], [0.0, 0, 0, 0, 0]
        i = 0
        while i < min(n, len(ds)):
            items = [ds[j] for j in range(i, min(i + 8, n, len(ds)))]
            i += len(items)
            batch = self.train_loader_list[task_id].collate_fn(items)
            ids = batch["input_ids_with_ans"].cuda(); am = batch["attention_mask_with_ans"].cuda()
            lab = batch["labels_with_ans"].cuda()
            lg = model(input_ids=ids, attention_mask=am, use_cache=False, return_dict=True).logits.float()
            sl, tl = lg[:, :-1], lab[:, 1:]
            m = tl != -100
            ce = torch.nn.functional.cross_entropy(sl[m], tl[m], reduction="sum")
            ok = (sl.argmax(-1) == tl) | ~m
            stats[0] += float(ce); stats[1] += int(m.sum()); stats[2] += int((sl.argmax(-1) == tl)[m].sum())
            stats[3] += int(ok.all(-1).sum()); stats[4] += ids.shape[0]
            out_all.append(sl[m].cpu())
        model.train()
        lg_all = torch.cat(out_all, 0)
        tag = "before fold (training forward, M active)" if before is None else "after fold (plain model)"
        msg = "[CODE-CL] foldcheck task %d %s on %d train sentences: answer CE %.4f, token acc %.4f, sequence exact %.4f" % (
            int(task_id), tag, stats[4], stats[0] / max(stats[1], 1), stats[2] / max(stats[1], 1), stats[3] / max(stats[4], 1))
        if before is not None:
            d = (lg_all - before).abs()
            msg += " | logits vs before: max abs diff %.4e, mean abs diff %.4e, argmax agreement %.4f" % (
                float(d.max()), float(d.mean()), float((lg_all.argmax(-1) == before.argmax(-1)).float().mean()))
        logger.info(msg)
        return lg_all

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def _adamw_step(self, task_id, log_now):
        lr = float(self.params.lr)
        wd = float(self.params.weight_decay)
        b1, b2, eps = 0.9, 0.999, 1e-8
        self.ostep += 1
        bc1, bc2 = 1 - b1 ** self.ostep, 1 - b2 ** self.ostep
        params = [p for p in self.base_model().parameters()]
        params += [M for (U, M) in self.free.values()]
        seen = set()
        for p in params:
            if p.grad is None or id(p) in seen:
                continue
            seen.add(id(p))
            if int(task_id) > 0 and id(p) in self.norm_ids:
                continue                                  # BN analogue: zeroed gradients from task 1 on
            st = self.state.get(id(p))
            if st is None:
                st = self.state[id(p)] = (torch.zeros_like(p), torch.zeros_like(p))
            m, v = st
            g = p.grad
            m.mul_(b1).add_(g, alpha=1 - b1)
            v.mul_(b2).addcmul_(g, g, value=1 - b2)
            upd = (m / bc1) / ((v / bc2).sqrt().add_(eps))
            upd.add_(p, alpha=wd).mul_(-lr)
            nm = self.lin_param.get(id(p))
            if nm is not None and int(task_id) > 0:
                C = self.conc[self.group_of[nm]]
                rm = upd @ C
                if log_now:
                    self.rm_stats[nm].append((float(rm.pow(2).sum()), float(upd.pow(2).sum())))
                upd.sub_(rm)
            p.add_(upd)

    def observe_batch(self, task_id, ep, lm_input):
        out = self.lm_loss(lm_input)
        loss = out.loss
        model = self.base_model()
        model.zero_grad(set_to_none=True)
        for U, M in self.free.values():
            M.grad = None
        loss.backward()
        self.step += 1
        log_now = int(task_id) > 0 and self.step % int(self.params.codecl_log_every) == 0
        self._adamw_step(task_id, log_now)
        model.zero_grad(set_to_none=True)
        if self.step % self.params.info_per_steps == 0 and self.accelerator.is_main_process:
            logger.info("[CODE-CL] task=%d ep=%d step=%d: ce=%.4f" % (task_id, ep + 1, self.step, float(loss.detach())))
        if log_now:
            by = defaultdict(list)
            for n, v in self.rm_stats.items():
                by[n.split(".")[-1]].append(sum(a for a, b in v) / max(sum(b for a, b in v), 1e-30))
            ta = sum(a for v in self.rm_stats.values() for a, b in v)
            tb = sum(b for v in self.rm_stats.values() for a, b in v)
            logger.info("[CODE-CL] task=%d step=%d removed share of update (||dW C||^2/||dW||^2): all=%.4f | by type "
                        "mean[min,max]: %s" % (task_id, self.step, ta / max(tb, 1e-30), " ".join(
                            "%s=%.3f[%.3f,%.3f]" % (k, sum(v) / len(v), min(v), max(v)) for k, v in sorted(by.items()))))
            self.rm_stats = defaultdict(list)
