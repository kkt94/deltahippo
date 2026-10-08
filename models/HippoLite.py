"""HippoLite: the DeltaHippo learner for decoder LLMs, one hippocampus over every synapse group (utils/hippo_lite.py).

Plain full fine-tuning of the whole model (answer-token cross-entropy) in standard mixed precision (fp32 weights and
AdamW state, bf16 autocast forward), relative step per trunk module, and one hippocampus (utils/hippo_lite.py) that
shapes where and how much is written. Task 0 trains with torch AdamW; from task 1 on the step is written whole with
probability = the conflict gate, through the index (minimum-interference operator per linear area). The hippocampus
sleeps after every task but the last.
Options: pdr_no_liger (read by utils/backbone.py, default True), hippo_tiedrow (default True), hippo_widthref (muP
width scaling of the matrices' step, 0 = off), hippo_orthobasis (fp64 orthonormal basis for the held span),
hippo_gradckpt (gradient checkpointing for large models), hippo_off (timing reference: plain fine-tuning, no sleep).
"""
import logging
import random

import numpy as np
import torch
import torch.nn as nn

from utils.backbone import get_backbone
from utils.evaluation import evaluate_sent_level_acc_with_generation
from utils.dataloader import get_dataloader
from utils.metric import ResultSummary
from utils.wrapmodel import WrapModel
from models.Base import BaseLearner
from utils.hippo_lite import HippoIndexLite

logger = logging.getLogger()


def get_HippoLite_params(parser):
    parser.add_argument("--pdr_no_liger", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=True,
                        help="the hippocampus reads the readout forward: no fused linear-cross-entropy kernel (Llama)")
    parser.add_argument("--hippo_tiedrow", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=True,
                        help="tied table: each row's step also weighed by its earlier use as an input, f / (f + mu u_v)")
    parser.add_argument("--hippo_off", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="(timing reference) no sleep: every task is plain fine-tuning on the same code path")
    parser.add_argument("--hippo_widthup", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="width rule without the cap at 1: narrower models step up by ref / width (ablation; off by default)")
    parser.add_argument("--hippo_widthref", type=float, default=0,
                        help="muP width scaling: matrices of a model wider than this step by ref / hidden (0 = off)")
    parser.add_argument("--hippo_gradckpt", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="memory mode: gradient checkpointing (energy hooks skip the recompute)")
    parser.add_argument("--hippo_orthobasis", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="held span + pattern-hold columns as one fp64 QR([U, P]) (numerical-rank tolerance)")


class HippoLite(BaseLearner):
    def __init__(self, params, CL_dataset, accelerator):
        self.hidx = None
        # the fields the hippocampus reads about the present batch
        self.cur_attn = None
        self.cur_labels = None
        self.cur_tg = None
        self._mcache = None
        self._clsSeen = {}
        self._spe = None
        self.ep = 0
        self._stepEp = 0
        # quantal write: its own generator (never the global RNGs), one draw per step from task 1 on
        self._prng = random.Random(12345)
        self._ptk = []
        super().__init__(params, CL_dataset, accelerator)

    # ------------------------------------------------------------------ build
    def build_metric(self):
        self.result_summary = ResultSummary(num_task=self.CL_dataset.continual_config["NUM_TASK"])

    def build_backbone(self):
        self.model, self.tokenizer = get_backbone(self.params, self.CL_dataset.continual_config["NUM_TASK"])
        self.model = self.model.to(torch.float32)
        if bool(getattr(self.params, "hippo_gradckpt", False)):
            # (the hippocampus's wake hooks return without effect inside the recomputation: they test the autograd
            # graph task, so energies and comparator rows are taken once per step)
            self.model.config.use_cache = False
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        _orig_fwd = self.model.forward

        def _amp_fwd(*a, **k):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return _orig_fwd(*a, **k)
        self.model.forward = _amp_fwd
        for p in self.model.parameters():
            p.requires_grad_(True)
        tied = self.model.get_input_embeddings().weight is self.model.get_output_embeddings().weight
        n = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("[HIPPO-LITE] full fine-tuning of every parameter: %.1fM trainable, tied table %s, dtype %s"
                    % (n / 1e6, tied, str(next(self.model.parameters()).dtype)))
        self.hidx = HippoIndexLite(self, self.model)
        self.hidx.tiedrow = bool(getattr(self.params, "hippo_tiedrow", True))
        self.hidx.orthobasis = bool(getattr(self.params, "hippo_orthobasis", False))
        if self.hidx.orthobasis:
            logger.info("[HIPPO-LITE] orthobasis: held span and pattern-hold columns as one fp64 QR([U, P])")

    def build_classifier(self):
        self.classifier = None

    def build_optimizer(self):
        # RELATIVE STEP: every trunk module gets lr x min(1, its weight rms / the model's mean trunk rms); the rest lr
        ps = [p for p in self.model.parameters() if p.requires_grad]
        lr0 = float(self.params.lr)
        base = getattr(self.model, "model", self.model)
        base = getattr(base, "language_model", base)
        allw = []
        for L in base.layers:
            for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
                allw.append(getattr(L.self_attn, nm).weight)
            for nm in ("gate_proj", "up_proj", "down_proj"):
                allw.append(getattr(L.mlp, nm).weight)
        rms = lambda w: float(w.detach().float().pow(2).mean().sqrt())
        mean_ = sum(rms(w) for w in allw) / max(len(allw), 1)
        scaled = {id(w_) for w_ in allw}
        # WIDTH (muP): an Adam step changes a matrix's output in proportion to its width, so the matrices of a model
        # wider than the reference width the lr was set on (hippo_widthref; 0 = off) step by ref / width; vector-like
        # parameters (input table, norm gains) keep lr. The untied readout is a matrix and is scaled as well.
        wref = float(getattr(self.params, "hippo_widthref", 0) or 0)
        dmod = float(getattr(self.model.config, "hidden_size", 0) or 0)
        wsc = (wref / dmod if bool(getattr(self.params, "hippo_widthup", False)) else min(1.0, wref / dmod)) \
            if wref > 0 and dmod > 0 else 1.0          # (widthup: narrower models step up by ref / width as well)
        head_ = self.model.get_output_embeddings().weight
        tied_ = head_ is self.model.get_input_embeddings().weight
        groups, rest, sc = [], [], []
        for p_ in ps:
            if id(p_) in scaled:
                s_ = min(1.0, rms(p_) / max(mean_, 1e-12))
                groups.append({"params": [p_], "lr": lr0 * s_ * wsc}); sc.append(s_)
            elif wsc != 1.0 and not tied_ and p_ is head_:
                groups.append({"params": [p_], "lr": lr0 * wsc})
            else:
                rest.append(p_)
        groups.insert(0, {"params": rest, "lr": lr0})
        logger.info("[HIPPO-LITE] relative step on %d trunk modules, scale mean %.3f (min %.3f); width scale %.3f "
                    "(ref %d / hidden %d) on the matrices" % (len(sc), sum(sc) / max(len(sc), 1), min(sc) if sc else 1.0,
                                                             wsc, int(wref), int(dmod)))
        self.optimizer = torch.optim.AdamW(groups, lr=lr0, weight_decay=float(self.params.weight_decay))

    def build_dataloader(self):
        self.train_loader_list, self.dev_loader_list, self.test_loader_list = \
            get_dataloader(self.params, self.CL_dataset, self.tokenizer)

    def build_buffer(self):
        self.buffer = None

    def accelerate_prepare(self):
        self.wrap_model = WrapModel(self.model, nn.ModuleList())
        (self.wrap_model, self.optimizer, *self.train_loader_list) = \
            self.accelerator.prepare(self.wrap_model, self.optimizer, *self.train_loader_list)
        if len(self.dev_loader_list) > 1:
            self.dev_loader_list = list(self.accelerator.prepare(*self.dev_loader_list))
            self.test_loader_list = list(self.accelerator.prepare(*self.test_loader_list))
        else:
            self.dev_loader_list = [self.accelerator.prepare(self.dev_loader_list[0])]
            self.test_loader_list = [self.accelerator.prepare(self.test_loader_list[0])]

    def _unwrap(self, m):
        return m.module if hasattr(m, "module") else m

    # ------------------------------------------------------------------ tasks
    def begin_task(self, task_id):
        super().begin_task(task_id)
        self._clsSeen = {}
        self.hidx.begin_task(task_id)

    def end_task(self, task_id):
        if torch.cuda.is_available():
            self.hidx.train_peak = torch.cuda.max_memory_allocated()
            logger.info("[HIPPO-LITE] task %d: peak allocated %.2f GB" % (int(task_id), self.hidx.train_peak / 1e9))
            torch.cuda.reset_peak_memory_stats()
        # sleep: iterates the (shuffled) training loader of the present task once more -- this draws the global RNG
        # (hippo_off: timing reference only -- no sleep, so every task trains as plain fine-tuning on the same code).
        # After the last task nothing would read the memory: no sleep (the evaluation below reads only the model).
        if not bool(getattr(self.params, "hippo_off", False)):
            if int(task_id) < int(self.CL_dataset.continual_config["NUM_TASK"]) - 1:
                self.hidx.sleep(task_id, self.train_loader_list[task_id], self._unwrap(self.wrap_model).model)
                if torch.cuda.is_available():
                    logger.info("[HIPPO-LITE] sleep after task %d: peak allocated %.2f GB"
                                % (int(task_id), torch.cuda.max_memory_allocated() / 1e9))
            else:
                self.hidx.rest()
                logger.info("[HIPPO-LITE] task %d is the last: no sleep" % int(task_id))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        super().end_task(task_id)

    def train_epochs(self, task_id):
        loader = self.train_loader_list[task_id]
        nep = int(self.params.training_epochs)
        self._spe = len(loader)
        model = self._unwrap(self.wrap_model).model
        model.train()
        for ep in range(nep):
            self.ep, self._stepEp = ep, 0
            if self.accelerator.is_main_process:
                logger.info("[HIPPO-LITE] Task %d | Epoch %d/%d" % (task_id + 1, ep + 1, nep))
            for batch in loader:
                self.observe_batch(task_id, ep, batch)

    def observe_batch(self, task_id, ep, lm_input):
        model = self._unwrap(self.wrap_model).model
        ids = lm_input["input_ids_with_ans"]
        attn = lm_input["attention_mask_with_ans"]
        labels = lm_input["labels_with_ans"]
        # the present batch: labels, classes, and the masks / counts the hippocampus's hooks reuse without host syncs
        self.cur_labels = labels
        self.cur_tg = [str(t).strip().lower() for t in lm_input["target"]]
        for c in self.cur_tg:
            self._clsSeen[c] = self._clsSeen.get(c, 0) + 1
        self._stepEp += 1
        m_ans = torch.zeros(labels.shape, dtype=torch.bool, device=labels.device)
        m_ans[:, :-1] = labels[:, 1:] != -100
        m_att = attn.bool()
        cnt_ = torch.stack([m_att.sum(), m_ans.sum()]).tolist()
        self._mcache = (tuple(labels.shape), m_ans, m_att, float(cnt_[0]), float(cnt_[1]))
        out = model(input_ids=ids, attention_mask=attn, labels=labels, use_cache=False, return_dict=True)
        loss = out.loss
        self.optimizer.zero_grad(set_to_none=True)
        self.hidx.pre_backward()
        self.accelerator.backward(loss)
        hx = self.hidx
        hx.apply()
        if hx.eig:
            # QUANTAL PLASTICITY: the step is written whole with probability equal to the conflict gate
            _lr0 = [g_["lr"] for g_ in self.optimizer.param_groups]
            _take = self._prng.random() < float(hx.plast)        # (the gate is read here, after the backward)
            self._ptk.append(float(_take))
            if len(self._ptk) >= 200:
                logger.info("[HIPPO-LITE] last 200 steps: steps written %.3f" % (sum(self._ptk) / len(self._ptk)))
                self._ptk = []
            _mul = 1.0 if _take else 0.0
            for g_ in self.optimizer.param_groups:
                g_["lr"] = g_["lr"] * _mul
            hx.step(self.optimizer)
            for g_, l_ in zip(self.optimizer.param_groups, _lr0):
                g_["lr"] = l_
            hx.plast = 1.0
        else:
            self.optimizer.step()
        # the gradient is consumed: free it now
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        if self.step % self.params.info_per_steps == 0 and self.accelerator.is_main_process:
            logger.info("[HIPPO-LITE] task=%d ep=%d step=%d: ce=%.4f" % (task_id, ep + 1, self.step, float(loss)))

    # ------------------------------------------------------------------ evaluation (generation, exact match)
    def evaluate_model(self, task_id):
        result_dict, log_dict = {}, {}
        cur = int(task_id)
        il_mode = self.params.il_mode
        acc_list, acc_next = self.evaluate_all_seen_task_tc(cur, "test", il_mode)
        result_dict["Test_Acc_List"] = acc_list
        for t in range(cur + 1):
            log_dict[f"Test_Acc_Task_{t}"] = acc_list[t]
        self.result_summary.update(cur, cur, acc_list[cur])
        log_dict["Test_Acc_Task_Seen"] = float(np.round(np.mean(acc_list[: cur + 1]), 3))
        log_dict["Test_Acc_Task_All"] = float(np.round(np.mean(acc_list), 3))
        if self.accelerator.is_main_process:
            logger.info(f"Mode = {il_mode}, Test Result = {log_dict}")
            logger.info(f"Result Summary Test After Task {cur} =\n{self.result_summary.print_format()}")
        return result_dict

    def evaluate_current_task(self, eval_task_id, cur_task_id, phase, il_mode):
        loaders = (self.train_loader_list if phase == "train"
                   else self.dev_loader_list if phase == "dev" else self.test_loader_list)
        model = self._unwrap(self.wrap_model).model
        model.eval()
        # the hippocampus hooks would only return without effect on every generation forward: they are detached while
        # evaluating (only when nothing is collected and no training batch is cached -- always so after the sleep)
        pause_ = self.hidx is not None and self.hidx.collect is None and self._mcache is None
        if pause_:
            self.hidx.pause_hooks()
        try:
            # one outer autocast region: its weight-cast cache lives for the whole evaluation instead of being rebuilt
            # by every forward (prefill + each decode step); the cached bf16 weights equal a per-forward cast
            # (costs one bf16 copy of the linears while evaluating)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                acc, _ = evaluate_sent_level_acc_with_generation(
                    model=model, eval_data_loader=loaders[eval_task_id], next_eval_data_loader=None,
                    tokenizer=self.tokenizer, accelerator=self.accelerator, params=self.params,
                    idx2label=self.CL_dataset.continual_config["idx2label"])
        finally:
            if pause_:
                self.hidx.resume_hooks()
        model.train()
        return acc, None
