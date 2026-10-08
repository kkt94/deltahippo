"""Sculpting Subspaces / Adaptive SVD (Nayak et al., 2025, arXiv:2504.07097) on a decoder-only LLM.

Reference: the paper (Alg. 1 and its appendix hyperparameters) and the official code
github.com/Red-Hat-AI-Innovation-Team/orthogonal-subspace-learning (notebooks/finetune_svd.py: SVD split, gradient
projection, importance / adaptive rank code), with the factorised forward of the team's maintained implementation
(github.com/Red-Hat-AI-Innovation-Team/mini_trainer, src/mini_trainer/osft_utils.py: _factorized_linear). It runs
under the same full fine-tuning protocol as DeltaHippo.

* Target matrices: q/k/v/o/gate/up/down of every block. Everything else (tied input table / readout, norm gains)
  is trained without constraint (official code: "parameters not in svd_config stay trainable").
* Task 0: plain full fine-tuning (no previous task to measure importance on; the official T5 script starts the SVD
  sequence from a plainly fine-tuned first-task checkpoint).
* Before task t >= 1, the number k of frozen singular directions of each matrix (m = min(out, in)) is set by
  --sculpt_rank:
    - fixed: k = round((i-1)/n m) at task i (1-based) of n, i.e. a share of the spectrum that grows with the number of
      tasks seen (the released configuration);
    - adaptive: layer importance I = mean over tokens of cos(X[:m], Y[:m]) (as in the official code) on task t-1's
      training data, normalised to mean 1 over all target matrices; retained fraction r = mrr + I (trr - mrr)
      (paper defaults mrr = 0.1, trr = 0.8); k = round(r m) clamped to [1, m].
  Then SVD of the current weight W = U S V^T; the top k triplets (U_high, S_high, V_high) are frozen, the rest
  (U_low, S_low, V_low) are trainable parameters, and the layer computes
  x -> (x V_high^T * S_high) U_high^T + (x V_low^T * S_low) U_low^T.
* Every step, before AdamW: dU_low -= U_high U_high^T dU_low ; dV_low -= dV_low V_high^T V_high ; dS_low unchanged
  (finetune_svd.project_gradient_to_orthogonal_space, paper Alg. 1).
* At the end of every task the factors are merged back into the dense W (the next task's SVD is of the current W).
* A new AdamW per task (the parameters themselves are new at every task).
"""
import logging
import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.cl_proj_common import ProjLearnerBase, str2bool

logger = logging.getLogger()

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def get_SculptSVD_params(parser):
    parser.add_argument("--pdr_no_liger", type=str2bool, default=True, help="no fused linear-cross-entropy kernel")
    parser.add_argument("--sculpt_rank", type=str, default="adaptive",
                        help="adaptive (importance-based: mrr + I (trr - mrr)) | fixed (top (i-1)/n frozen at task i of n)")
    parser.add_argument("--sculpt_mrr", type=float, default=0.1, help="minimum retention ratio (adaptive rank)")
    parser.add_argument("--sculpt_trr", type=float, default=0.8, help="target retention ratio (adaptive rank)")
    parser.add_argument("--sculpt_log_every", type=int, default=25, help="log the removed share of the gradient every n steps")


class SculptSVD(ProjLearnerBase):
    TAG = "SCULPT"

    def __init__(self, params, CL_dataset, accelerator):
        self.fac = {}            # target name -> dict of factors
        self.rm_stats = defaultdict(list)
        super().__init__(params, CL_dataset, accelerator)
        model = self.base_model()
        self.targets = {n: m for n, m in model.named_modules()
                        if isinstance(m, nn.Linear) and n.split(".")[-1] in TARGETS}
        logger.info("[SCULPT] %d target matrices (%s); untargeted trainable params: %s"
                    % (len(self.targets), ",".join(TARGETS),
                       [n for n, p in model.named_parameters()
                        if not any(n == t + ".weight" for t in self.targets)]))

    # ------------------------------------------------------------------ importance (adaptive rank)
    @torch.no_grad()
    def _importance(self, loader):
        model = self.base_model()
        s = defaultdict(float)
        c = defaultdict(float)
        box = {}

        def make_hook(name):
            def hook(mod, inp, out):
                x, y = inp[0], out
                m = min(x.shape[-1], y.shape[-1])
                xs = x[..., :m].float().reshape(-1, m)[box["m"]]
                ys = y[..., :m].float().reshape(-1, m)[box["m"]]
                cs = F.cosine_similarity(xs, ys, dim=-1, eps=1e-8)
                s[name] = s[name] + cs.sum()
                c[name] += float(cs.numel())
            return hook

        hs = [m.register_forward_hook(make_hook(n)) for n, m in self.targets.items()]
        model.eval()
        try:
            for batch in loader:
                am = batch["attention_mask_with_ans"]
                box["m"] = am.reshape(-1).bool()
                model(input_ids=batch["input_ids_with_ans"], attention_mask=am, use_cache=False, return_dict=True)
        finally:
            for h in hs:
                h.remove()
            model.train()
        return {n: float(s[n]) / max(c[n], 1.0) for n in self.targets}

    def _ranks(self, task_id):
        ntask = int(self.CL_dataset.continual_config["NUM_TASK"])
        mode = str(self.params.sculpt_rank)
        ranks, frac = {}, {}
        if mode == "fixed":
            r = float(task_id) / ntask                    # task i (1-based) of n: top (i-1)/n frozen
            for n, mod in self.targets.items():
                m = min(mod.weight.shape)
                ranks[n] = max(1, min(m, int(round(r * m))))
                frac[n] = r
            logger.info("[SCULPT] task %d: fixed budget, top %.3f of the singular vectors frozen" % (task_id, r))
            return ranks
        imp = self._importance(self.train_loader_list[task_id - 1])
        mean_ = sum(imp.values()) / len(imp)
        mrr, trr = float(self.params.sculpt_mrr), float(self.params.sculpt_trr)
        for n, mod in self.targets.items():
            m = min(mod.weight.shape)
            In = imp[n] / (mean_ + 1e-8)
            r = mrr + In * (trr - mrr)
            ranks[n] = max(1, min(m, int(round(r * m))))
            frac[n] = ranks[n] / m
        by = defaultdict(list)
        for n in self.targets:
            by[n.split(".")[-1]].append((imp[n], frac[n]))
        logger.info("[SCULPT] task %d: importance mean %.4f; by type mean I [min,max] -> mean kept fraction [min,max]: %s"
                    % (task_id, mean_, " ".join("%s=%.4f[%.4f,%.4f]->%.3f[%.3f,%.3f]" % (
                        k, sum(a for a, b in v) / len(v), min(a for a, b in v), max(a for a, b in v),
                        sum(b for a, b in v) / len(v), min(b for a, b in v), max(b for a, b in v))
                        for k, v in sorted(by.items()))))
        logger.info("[SCULPT] task %d per-matrix kept fraction: %s" % (task_id, " ".join(
            "%s=%.2f" % (n.replace("model.layers.", "L").replace("self_attn.", "").replace("mlp.", ""), frac[n])
            for n in self.targets)))
        return ranks

    # ------------------------------------------------------------------ factorisation
    def _factorise(self, ranks):
        for n, mod in self.targets.items():
            W = mod.weight.data.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            k = ranks[n]
            f = {
                "Uh": U[:, :k].contiguous(), "Sh": S[:k].contiguous(), "Vh": Vt[:k, :].contiguous(),
                "Ul": nn.Parameter(U[:, k:].contiguous()), "Sl": nn.Parameter(S[k:].contiguous()),
                "Vl": nn.Parameter(Vt[k:, :].contiguous()), "k": k, "shape": tuple(W.shape),
            }
            del U, S, Vt, W
            self.fac[n] = f
            mod.weight.requires_grad_(False)
            mod.weight.data = torch.empty(0, device=mod.weight.device, dtype=mod.weight.dtype)

            def make_fwd(f):
                def fwd(x):
                    y = F.linear(F.linear(x, f["Vh"]) * f["Sh"], f["Uh"])
                    if f["Sl"].numel() > 0:
                        y = y + F.linear(F.linear(x, f["Vl"]) * f["Sl"], f["Ul"])
                    return y
                return fwd
            mod.forward = make_fwd(f)
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _merge(self):
        for n, f in self.fac.items():
            mod = self.targets[n]
            W = (f["Uh"] * f["Sh"].unsqueeze(0)) @ f["Vh"]
            if f["Sl"].numel() > 0:
                W = W + (f["Ul"] * f["Sl"].unsqueeze(0)) @ f["Vl"]
            mod.weight.data = W.contiguous()
            mod.weight.requires_grad_(True)
            del mod.forward                       # back to nn.Linear.forward
        self.fac = {}
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ task boundaries
    def begin_task(self, task_id):
        super().begin_task(task_id)
        model = self.base_model()
        if int(task_id) > 0:
            ranks = self._ranks(int(task_id))
            self._factorise(ranks)
            fparams = [f[k] for f in self.fac.values() for k in ("Ul", "Sl", "Vl")]
            rest = [p for p in model.parameters() if p.requires_grad]
            nf = sum(p.numel() for p in fparams)
            logger.info("[SCULPT] task %d: %d matrices factorised; trainable low-subspace factors %.1fM, "
                        "unconstrained params %.1fM" % (task_id, len(self.fac), nf / 1e6,
                                                        sum(p.numel() for p in rest) / 1e6))
            params = fparams + rest
        else:
            params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=float(self.params.lr),
                                           weight_decay=float(self.params.weight_decay))
        self.rm_stats = defaultdict(list)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def end_task(self, task_id):
        self.log_peak("task %d training" % int(task_id))
        self.optimizer = None
        if self.fac:
            self._merge()
        torch.cuda.empty_cache()
        super().end_task(task_id)

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def _project(self, log_now):
        for n, f in self.fac.items():
            Uh, Vh = f["Uh"], f["Vh"]
            dU, dV = f["Ul"].grad, f["Vl"].grad
            a = b = 0.0
            if dU is not None:
                pu = Uh @ (Uh.t() @ dU)
                if log_now:
                    a += float(pu.pow(2).sum()); b += float(dU.pow(2).sum())
                dU.sub_(pu)
            if dV is not None:
                pv = (dV @ Vh.t()) @ Vh
                if log_now:
                    a += float(pv.pow(2).sum()); b += float(dV.pow(2).sum())
                dV.sub_(pv)
            if log_now:
                self.rm_stats[n].append((a, b))

    def observe_batch(self, task_id, ep, lm_input):
        out = self.lm_loss(lm_input)
        loss = out.loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.step += 1
        log_now = bool(self.fac) and self.step % int(self.params.sculpt_log_every) == 0
        if self.fac:
            self._project(log_now)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.step % self.params.info_per_steps == 0 and self.accelerator.is_main_process:
            logger.info("[SCULPT] task=%d ep=%d step=%d: ce=%.4f" % (task_id, ep + 1, self.step, float(loss)))
        if log_now:
            by = defaultdict(list)
            for n, v in self.rm_stats.items():
                by[n.split(".")[-1]].append(sum(x for x, y in v) / max(sum(y for x, y in v), 1e-30))
            ta = sum(x for v in self.rm_stats.values() for x, y in v)
            tb = sum(y for v in self.rm_stats.values() for x, y in v)
            logger.info("[SCULPT] task=%d step=%d removed share of the U_low/V_low gradient: all=%.4f | by type "
                        "mean[min,max]: %s" % (task_id, self.step, ta / max(tb, 1e-30), " ".join(
                            "%s=%.3f[%.3f,%.3f]" % (k, sum(v) / len(v), min(v), max(v)) for k, v in sorted(by.items()))))
            self.rm_stats = defaultdict(list)
