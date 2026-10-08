"""HippoLiteEnc: HippoLite (models/HippoLite.py) for discriminative text encoders (BERT, RoBERTa).

The same learner and the same hippocampus rules (utils/hippo_enc.py, the model-agnostic counterpart of
utils/hippo_lite.py). What is encoder-specific is only how the model is read:
  * backbone: the bare encoder (AutoModel, no pooler) from ./hf_models/<name>; fp32 weights and AdamW state, bf16
    autocast forward (as HippoLite).
  * READOUT: the framework's per-task classifier heads (utils/classifier.get_classifier, Linear with bias). They read
    the FULL last hidden sequence (B, L, d) and give logits at every position (B, L, C), C = the classes of the tasks
    seen so far (CIL: concatenation of the heads 0..t). The loss is taken only at the extract position: the raw
    position-0 state ([CLS] / <s>), as the framework's sequential fine-tuning baseline reads it
    (utils/backbone.obtain_features, cls_token: no pooler).
  * LABELS in the generative convention: labels[:, 1] = the class index, -100 elsewhere, so position 0 is the one
    "answer step" (m_ans = labels[:, 1:] != -100) and gold = labels[b, p + 1]; the hippocampus's masks, conflict hook,
    records, comparator, joint-share target and held-input gradients run unchanged on (B, L, d) tensors.
  * the answer is not part of the input (the class is not a token): the per-row tables count every attended position.
  * evaluation: argmax over the concatenated heads at position 0 (the framework's classifier evaluation); hooks
    detached.
"""
import logging
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from utils.classifier import get_classifier
from utils.dataloader import get_dataloader
from utils.metric import ResultSummary
from utils.wrapmodel import WrapModel
from models.Base import BaseLearner
from utils.hippo_enc import HippoIndexEnc, PROF
from utils.hle_diag import EncDiag, LayerOwn

logger = logging.getLogger()

_EXTRACT = 0                                 # [CLS] / <s>: the answer step; its label sits at labels[:, 1]


def get_HippoLiteEnc_params(parser):
    parser.add_argument("--hle_widthref", type=float, default=0,
                        help="DeltaHippo width rule: maps with fan_in > ref step by ref / fan_in (0 = off)")
    parser.add_argument("--hle_diag", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=True,
                        help="diagnostics (evaluation only; do not affect training): margin split, per-layer drift, "
                             "acquisition (utils/hle_diag.py)")
    parser.add_argument("--hle_owm", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=True,
                        help="true: minimum-interference step in every area; false: areas held by their span with the "
                             "tail share c")
    parser.add_argument("--hle_owmg", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="minimum-interference share for the per-coordinate parameters (gains, table rows); false: the "
                             "novelty share with owned coordinates held (ablation; off by default)")
    parser.add_argument("--hle_tabowm", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="untied input tables: per-row share f / (f + mu u_v) instead of the novelty share")
    parser.add_argument("--hle_pdet", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="state conflict against the row's own present class mean (no radius)")
    parser.add_argument("--hle_pairc", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="class-level state conflict: the row's present-class mean inside an earlier record's radius "
                             "(ablation; off by default)")
    parser.add_argument("--hle_pairu", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="state conflict = row-level (pdet) OR class-level mutual-nearest (hle_pairc's test) "
                             "(ablation; off by default)")
    parser.add_argument("--hle_jpcut", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="joint-proportion twin cut of dL/dh at the decision row, at pdet conflicts "
                             "(ablation; off by default)")
    parser.add_argument("--hle_jpcut_live", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="(with hle_jpcut) cut axis W_r - W_gold, r the live top non-gold class "
                             "(ablation; off by default)")
    parser.add_argument("--hle_js0", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="joint-share target at offset 0 for a single decision row")
    parser.add_argument("--hle_wdfold", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="weight decay folded into the update before the hold")
    parser.add_argument("--hle_cmpheld", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="comparator's armed rows: mismatch with the whole held span [U, patterns] "
                             "(ablation; off by default)")
    parser.add_argument("--hle_cmpowm", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="comparator give-back sized by the minimum-interference share along each mismatch direction")
    parser.add_argument("--hle_sink", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="decision row weighted by 1 + sum_i a_i0^2 in attention-input areas (memory and wake)")
    parser.add_argument("--hle_tabln", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="table rows held in the units of the stream they feed (f_A of the area the embedding norm feeds)")
    parser.add_argument("--hle_owmx", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="exact minimum-interference step: (I + mu C / f)^-1 on the full running moment per area "
                             "(ablation; off by default)")
    parser.add_argument("--hle_gain", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="OWM interference of a layer's areas weighted by the layer's measured downstream gain^2 "
                             "(ablation; off by default)")
    parser.add_argument("--hle_c1", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="trunk areas held by their span alone (tail share c = 1) (ablation; off by default)")
    parser.add_argument("--hle_oldrow", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="old heads held per row off their class's region (GD basis of its [h; 1] moment), "
                             "in place of the readout span and the comparator")
    parser.add_argument("--hle_sigcut", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="no learning signal along the stream's massive coordinates at any norm input "
                             "(ablation; off by default)")
    parser.add_argument("--hle_tabdown", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="untied tables: the step goes through the operator of the first area the stream feeds "
                             "(ablation; off by default)")
    parser.add_argument("--hle_cmpall", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="comparator armed on every decision row where gold does not yet win (not only in conflict)")
    parser.add_argument("--hle_freshfree", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="the present task's head is not held at all (ablation; off by default)")
    parser.add_argument("--hle_sleepdrop", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="dropout kept: the sleep reads run in the same (train) mode as the wake forward")
    parser.add_argument("--hle_mhold", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="hold the residual stream's massive coordinates: no write into them (out/down rows, biases, "
                             "norm gains/biases, table columns) (ablation; off by default)")
    parser.add_argument("--hle_uniform", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="trunk moments over all attended positions alike (no answer-row reweighting)")
    parser.add_argument("--hle_gdown", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="norm gains / biases held as the area they feed holds their coordinate")
    parser.add_argument("--hle_tabstep", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="untied tables: the per-row share scales the Adam step as well as the gradient")
    parser.add_argument("--hle_fresh", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="the present task's head (no memory yet) is held only off the earlier memories' patterns")
    parser.add_argument("--hle_nodrop", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="encoder without dropout (wake statistics comparable with the sleep read) (ablation; off by default)")
    parser.add_argument("--hippo_gradckpt", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False,
                        help="gradient checkpointing to reduce activation memory (the energy hooks skip the recompute)")


class _CatHead(nn.Module):
    """The readout: heads 0..n-1 on every position of the last hidden sequence, concatenated (CIL logits)."""
    def __init__(self, heads):
        super().__init__()
        self.heads = heads
        self.n = 1

    def forward(self, h):
        return torch.cat([self.heads[i](h) for i in range(self.n)], dim=-1)


class EncoderClassifier(nn.Module):
    """Encoder + readout with the decoder model's call signature: forward(input_ids, attention_mask, labels) returns
    .logits (B, L, C) and, given labels, .loss = the shifted cross-entropy (labels[:, 1:] against logits[:, :-1]), i.e.
    the class at position 0."""
    def __init__(self, enc, heads):
        super().__init__()
        self.enc = enc
        self.readout = _CatHead(heads)
        self.config = enc.config

    def get_input_embeddings(self):
        return self.enc.get_input_embeddings()

    def get_output_embeddings(self):
        return None

    def gradient_checkpointing_enable(self, **kw):
        self.enc.gradient_checkpointing_enable(**kw)

    def forward(self, input_ids=None, attention_mask=None, labels=None, layers=False, **kw):
        o_ = self.enc(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=layers)
        h = o_.last_hidden_state
        if layers:
            self._lcls = torch.stack([t[:, _EXTRACT] for t in o_.hidden_states], 1)     # (B, layers + 1, d)
            # Diagnostic (evaluation only): content-token state, the mean over the attended positions other than
            # the decision position
            mk_ = attention_mask.clone().float()
            mk_[:, _EXTRACT] = 0.0
            mk_ = (mk_ / mk_.sum(1, keepdim=True).clamp(min=1.0)).unsqueeze(-1)
            self._lcon = torch.stack([(t.float() * mk_).sum(1) for t in o_.hidden_states], 1)
        z = self.readout(h)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(z[:, :-1].float().reshape(-1, z.shape[-1]),
                                   labels[:, 1:].reshape(-1).to(z.device), ignore_index=-100)
        return SimpleNamespace(loss=loss, logits=z, cls=h[:, _EXTRACT])


def _cls_batch(b):
    """A classification batch in the keys and label convention the hippocampus reads (input = the sentence without
    the answer; labels[:, 1] = the class index)."""
    ids = b["input_ids"]
    lb = torch.full_like(ids, -100)
    lb[:, _EXTRACT + 1] = b["label_idx_cil"].to(lb.device).long()
    out = dict(b)
    out["input_ids_with_ans"] = ids
    out["attention_mask_with_ans"] = b["attention_mask"]
    out["labels_with_ans"] = lb
    return out


class _ClsView:
    """A train loader seen through _cls_batch (iterates the loader itself: the same RNG draws)."""
    def __init__(self, loader):
        self.loader = loader

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for b in self.loader:
            yield _cls_batch(b)


def _resolve(name):
    base = os.path.basename(name.rstrip("/"))
    for p in (os.path.join("hf_models", base), name, "./" + base):
        if os.path.isdir(p):
            return p
    return name


class HippoLiteEnc(BaseLearner):
    def __init__(self, params, CL_dataset, accelerator):
        self.hidx = None
        self._ediag = EncDiag(self) if bool(getattr(params, "hle_diag", True)) else None
        self._lown = LayerOwn() if self._ediag is not None else None
        self.ans_in_input = False            # read by the hippocampus: the class is not an input token
        self.table_step_share = bool(getattr(params, "hle_tabstep", False))
        self.trunk_answer_weight = not bool(getattr(params, "hle_uniform", False))
        self.gain_share_downstream = bool(getattr(params, "hle_gdown", False))
        self.massive_hold = bool(getattr(params, "hle_mhold", False))
        self.sleep_train_mode = bool(getattr(params, "hle_sleepdrop", False))
        self.fresh_free = bool(getattr(params, "hle_freshfree", False))
        self.comparator_all_rows = bool(getattr(params, "hle_cmpall", False))
        self.table_step_downstream = bool(getattr(params, "hle_tabdown", False))
        self.massive_signal_cut = bool(getattr(params, "hle_sigcut", False))
        self.oldrow_region = bool(getattr(params, "hle_oldrow", False))
        self.trunk_c1 = bool(getattr(params, "hle_c1", False))
        self.owm = bool(getattr(params, "hle_owm", True))
        self.layer_gain = bool(getattr(params, "hle_gain", False))
        self.owm_exact = bool(getattr(params, "hle_owmx", False))
        self.table_owm = bool(getattr(params, "hle_tabowm", False))
        self.present_detect = bool(getattr(params, "hle_pdet", False))
        self.class_detect = bool(getattr(params, "hle_pairc", False))
        self.class_detect_union = bool(getattr(params, "hle_pairu", False))
        self.twin_cut = bool(getattr(params, "hle_jpcut", False))
        self.twin_cut_live = bool(getattr(params, "hle_jpcut_live", False))
        self.joint_share_offset0 = bool(getattr(params, "hle_js0", False))
        self.wd_fold = bool(getattr(params, "hle_wdfold", False))
        self.cmp_held = bool(getattr(params, "hle_cmpheld", False))
        self.cmp_owm = bool(getattr(params, "hle_cmpowm", False))
        self.sink_weight = bool(getattr(params, "hle_sink", False))
        self.table_stream = bool(getattr(params, "hle_tabln", False))
        self.owm_coord = bool(getattr(params, "hle_owmg", False))
        self.cur_attn = None
        self.cur_labels = None
        self.cur_tg = None
        self._mcache = None
        self._clsSeen = {}
        self._spe = None
        self.ep = 0
        self._stepEp = 0
        self._prng = random.Random(12345)
        self._ptk = []
        assert params.il_mode == "CIL", "HippoLiteEnc: CIL only"
        assert params.classifier == "Linear", "HippoLiteEnc: the readout is the framework's Linear heads"
        assert params.backbone_extract_token == "cls_token", "HippoLiteEnc: the answer step is position 0 ([CLS])"
        super().__init__(params, CL_dataset, accelerator)

    # ------------------------------------------------------------------ build
    def build_metric(self):
        self.result_summary = ResultSummary(num_task=self.CL_dataset.continual_config["NUM_TASK"])

    def build_backbone(self):
        path = _resolve(self.params.backbone)
        kw_ = {}
        if bool(getattr(self.params, "hle_nodrop", False)):
            # No dropout: the hippocampus compares the wake input statistics with the sleep read's (eval mode);
            # dropout noise would make every present input look novel outside the held span
            kw_ = {"hidden_dropout_prob": 0.0, "attention_probs_dropout_prob": 0.0}
        enc = AutoModel.from_pretrained(path, add_pooling_layer=False, dtype=torch.float32, **kw_)
        logger.info("[HIPPO-ENC] dropout: hidden %.2f, attention %.2f" % (enc.config.hidden_dropout_prob,
                                                                            enc.config.attention_probs_dropout_prob))
        self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side="right")
        heads = get_classifier(self.params, enc.config.hidden_size, self.CL_dataset.continual_config["CUR_NUM_CLASS"])
        self.model = EncoderClassifier(enc, heads).to("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(torch.float32)
        if bool(getattr(self.params, "hippo_gradckpt", False)):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        _orig_fwd = self.model.forward

        def _amp_fwd(*a, **k):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return _orig_fwd(*a, **k)
        self.model.forward = _amp_fwd
        for p in self.model.parameters():
            p.requires_grad_(True)
        n = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("[HIPPO-ENC] encoder %s from %s: full fine-tuning of every parameter: %.1fM trainable (%d heads, "
                    "classes %s), dtype %s" % (self.params.backbone, path, n / 1e6, len(heads),
                                               self.CL_dataset.continual_config["CUR_NUM_CLASS"],
                                               str(next(self.model.parameters()).dtype)))
        dev = next(self.model.parameters()).device
        pr = self.tokenizer(["a short probe sentence"], return_tensors="pt").to(dev)
        self.model.readout.n = 1

        def probe():
            self.model(input_ids=pr["input_ids"], attention_mask=pr["attention_mask"])
        self.hidx = HippoIndexEnc(self, self.model, probe, readout_mods=list(heads), readout_hook=self.model.readout)

    def build_classifier(self):
        self.classifier = None               # the heads are the readout inside self.model

    def build_optimizer(self):
        # RELATIVE STEP: every linear map of a non-readout area (and its bias) gets lr x min(1, its weight rms / the
        # mean rms of those maps); the readout heads the framework's classifier lr; the rest lr
        lr0, lrc = float(self.params.lr), float(self.params.classifier_lr)
        wd = float(self.params.weight_decay)
        mods = [m for a, ms in self.hidx.areas.items() if a != "readout" for m in ms]
        rms = lambda w: float(w.detach().float().pow(2).mean().sqrt())
        mean_ = sum(rms(m.weight) for m in mods) / max(len(mods), 1)
        sc_ = {}
        wref = float(getattr(self.params, "hle_widthref", 0) or 0)
        nw_ = 0
        for m in mods:
            s_ = min(1.0, rms(m.weight) / max(mean_, 1e-12))
            if wref > 0 and m.weight.shape[1] > wref:
                # width rule: a map whose input is wider than the reference steps by ref / fan_in
                s_ *= wref / m.weight.shape[1]; nw_ += 1
            sc_[id(m.weight)] = s_
            if m.bias is not None:
                sc_[id(m.bias)] = s_
        head_ = {id(p) for m in self.hidx.areas["readout"] for p in m.parameters()}
        groups, rest, heads, sc = [], [], [], []
        for p_ in self.model.parameters():
            if not p_.requires_grad:
                continue
            if id(p_) in sc_:
                groups.append({"params": [p_], "lr": lr0 * sc_[id(p_)]}); sc.append(sc_[id(p_)])
            elif id(p_) in head_:
                heads.append(p_)
            else:
                rest.append(p_)
        groups.insert(0, {"params": rest, "lr": lr0})
        groups.append({"params": heads, "lr": lrc})
        logger.info("[HIPPO-ENC] relative step on %d trunk parameters (%d linear maps + biases), scale mean %.3f "
                    "(min %.3f); %d readout parameters at lr %.1e; %d other parameters at lr %.1e; width ref %d: %d maps"
                    % (len(sc), len(mods), sum(sc) / max(len(sc), 1), min(sc) if sc else 1.0, len(heads), lrc,
                       len(rest), lr0, int(wref), nw_))
        self.optimizer = torch.optim.AdamW(groups, lr=lr0, weight_decay=wd)

    def build_dataloader(self):
        self.train_loader_list, self.dev_loader_list, self.test_loader_list = \
            get_dataloader(self.params, self.CL_dataset, self.tokenizer)

    def build_buffer(self):
        self.buffer = None

    def accelerate_prepare(self):
        self.wrap_model = WrapModel(self.model, nn.ModuleList())
        (self.wrap_model, self.optimizer, *self.train_loader_list) = \
            self.accelerator.prepare(self.wrap_model, self.optimizer, *self.train_loader_list)
        self.train_loader_list = [_ClsView(l_) for l_ in self.train_loader_list]
        if len(self.dev_loader_list) > 1:
            self.dev_loader_list = list(self.accelerator.prepare(*self.dev_loader_list))
            self.test_loader_list = list(self.accelerator.prepare(*self.test_loader_list))
        else:
            self.dev_loader_list = [self.accelerator.prepare(self.dev_loader_list[0])]
            self.test_loader_list = [self.accelerator.prepare(self.test_loader_list[0])]

    def _unwrap(self, m):
        return m.module if hasattr(m, "module") else m

    def fresh_readout(self):
        """Read by the hippocampus at begin_task: the readout maps with no memory yet (the present task's head)."""
        if not bool(getattr(self.params, "hle_fresh", False)):
            return []
        m = self._unwrap(self.wrap_model).model if hasattr(self, "wrap_model") else self.model
        return [m.readout.heads[m.readout.n - 1]]

    # ------------------------------------------------------------------ tasks
    def begin_task(self, task_id):
        super().begin_task(task_id)
        self._clsSeen = {}
        self._unwrap(self.wrap_model).model.readout.n = int(task_id) + 1
        self.hidx.begin_task(task_id)

    def end_task(self, task_id):
        if PROF.on:
            PROF.live = False
            PROF.dump("task %d (window end)" % int(task_id), max(1, min(200, (self._spe or 0) * int(self.params.training_epochs) - 50)))
        sd_, sl_ = getattr(self, "_spd", None), getattr(self, "_spd_last", None)
        if sd_ is not None and sl_ is not None and sd_[2] == task_id and sl_[2] == task_id and sl_[1] > sd_[1]:
            logger.info("[SPEED] task %d: %.4f s/step over steps %d..%d (unsynced wall time)" % (
                int(task_id), (sl_[0] - sd_[0]) / (sl_[1] - sd_[1]), sd_[1], sl_[1]))
        if torch.cuda.is_available():
            self.hidx.train_peak = torch.cuda.max_memory_allocated()
            logger.info("[HIPPO-ENC] task %d: peak allocated %.2f GB" % (int(task_id), self.hidx.train_peak / 1e9))
            torch.cuda.reset_peak_memory_stats()
        if self._ediag is not None:
            try:
                self._ediag.acquisition(task_id)
            except Exception as ex_:                               # diagnostics must never stop a run
                logger.info("[DIAG-A] skipped: %s" % ex_)
        self.hidx.sleep(task_id, self.train_loader_list[task_id], self._unwrap(self.wrap_model).model)
        if torch.cuda.is_available():
            logger.info("[HIPPO-ENC] sleep after task %d: peak allocated %.2f GB" % (int(task_id), torch.cuda.max_memory_allocated() / 1e9))
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
                logger.info("[HIPPO-ENC] Task %d | Epoch %d/%d" % (task_id + 1, ep + 1, nep))
            for batch in loader:
                self.observe_batch(task_id, ep, batch)

    def observe_batch(self, task_id, ep, lm_input):
        model = self._unwrap(self.wrap_model).model
        ids = lm_input["input_ids_with_ans"]
        attn = lm_input["attention_mask_with_ans"]
        labels = lm_input["labels_with_ans"]
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
        # speed log: unsynced wall time per step from step 300 of each task to its end
        import time as _tm
        sp0_ = self._stepEp + ep * (self._spe or 0)
        if sp0_ == 300:
            self._spd = (_tm.perf_counter(), 300, task_id)
        elif getattr(self, "_spd", None) is not None and self._spd[2] == task_id and sp0_ > 300:
            self._spd_last = (_tm.perf_counter(), sp0_, task_id)
        if PROF.on:
            # profiling: steps 50..249 of every task are timed per component with CUDA events (no host sync)
            sp_ = self._stepEp + ep * (self._spe or 0)
            PROF.live = 50 <= sp_ < 250
            if sp_ == 250:
                PROF.dump("task %d" % task_id, 200)
            if PROF.live:
                ev_ = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                ev_[0].record()
        out = model(input_ids=ids, attention_mask=attn, labels=labels)
        loss = out.loss
        if PROF.live:
            ev_[1].record(); PROF.mark("L:forward(+hooks)", ev_[0], ev_[1])
        self.optimizer.zero_grad(set_to_none=True)
        self.hidx.pre_backward()
        self.accelerator.backward(loss)
        if PROF.live:
            ev_[2].record(); PROF.mark("L:pre_backward+backward", ev_[1], ev_[2])
        hx = self.hidx
        hx.apply()
        if hx.eig:
            # QUANTAL PLASTICITY: the step is written whole with probability equal to the conflict gate
            _lr0 = [g_["lr"] for g_ in self.optimizer.param_groups]
            _take = self._prng.random() < hx.plast
            self._ptk.append(float(_take))
            if len(self._ptk) >= 200:
                logger.info("[HIPPO-ENC] last 200 steps: steps written %.3f" % (sum(self._ptk) / len(self._ptk)))
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
        self.optimizer.zero_grad(set_to_none=True)
        if PROF.live:
            ev_[3].record(); PROF.mark("L:apply+step+zero", ev_[2], ev_[3]); PROF.mark("L:TOTAL", ev_[0], ev_[3])
        self.step += 1
        if self.step % self.params.info_per_steps == 0 and self.accelerator.is_main_process:
            logger.info("[HIPPO-ENC] task=%d ep=%d step=%d: ce=%.4f" % (task_id, ep + 1, self.step, float(loss)))

    # ------------------------------------------------------------------ evaluation (classifier heads, argmax)
    def evaluate_model(self, task_id):
        result_dict, log_dict = {}, {}
        cur = int(task_id)
        il_mode = self.params.il_mode
        acc_list, acc_next = self.evaluate_all_seen_task_tc(cur, "test", il_mode)
        if self._ediag is not None:
            try:
                self._ediag.summary(cur)
            except Exception as ex_:
                logger.info("[DIAG-M] summary skipped: %s" % ex_)
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
        n0_ = model.readout.n
        model.readout.n = int(cur_task_id) + 1                  # CIL: every head seen so far
        pause_ = self.hidx is not None and self.hidx.collect is None and self._mcache is None
        if pause_:
            self.hidx.pause_hooks()
        hit, zs, ys, hs, cs_ = [], [], [], [], []
        lnh_, lncap_ = [], {}
        lown_ = self._lown if (self._ediag is not None and phase == "test" and eval_task_id == 0) else None
        if lown_ is not None:
            lown_.hooks(model)
        if eval_task_id == 0 and self.hidx is not None:
            # Diagnostic (evaluation only): every norm's input / output at the decision position, task 0's test
            # sentences only; does not affect training
            for i_, (g_, m_) in enumerate(self.hidx.diag.items()):
                def _cap(mod, inp, out, _i=i_):
                    lncap_.setdefault(_i, []).append((inp[0][:, _EXTRACT].float().cpu(), out[:, _EXTRACT].float().cpu()))
                lnh_.append(m_.register_forward_hook(_cap))
            # Diagnostic: FFN at the decision position: input x and pre-activation of the FFN input map, activations a
            # entering the FFN output map
            for a_, ms_ in self.hidx.areas.items():
                if len(ms_) != 1 or a_ == "readout":
                    continue
                m_ = ms_[0]
                if "intermediate" in a_ or "gate_proj" in a_:
                    def _ci(mod, inp, out, _a=a_):
                        lncap_.setdefault(("in", _a), []).append((inp[0][:, _EXTRACT].float().cpu(), out[:, _EXTRACT].float().cpu()))
                    lnh_.append(m_.register_forward_hook(_ci))
                elif m_.out_features == self.hidx._dstream and m_.in_features != self.hidx._dstream:
                    def _co(mod, inp, _a=a_):
                        lncap_.setdefault(("out", _a), []).append((inp[0][:, _EXTRACT].float().cpu(), None))
                    lnh_.append(m_.register_forward_pre_hook(_co))
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for b in loaders[eval_task_id]:
                    o_ = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], layers=True)
                    z = o_.logits[:, _EXTRACT]
                    hs.append(model._lcls.float().cpu())
                    cs_.append(model._lcon.float().cpu())
                    pred = z.argmax(-1)
                    pred, y = self.accelerator.gather_for_metrics((pred, b["label_idx_cil"].to(pred.device)))
                    hit.append((pred == y).float().cpu())
                    zs.append(z.float().cpu()); ys.append(y.cpu())
        finally:
            for h_ in lnh_:
                h_.remove()
            if lown_ is not None:
                lown_.remove()
            if pause_:
                self.hidx.resume_hooks()
            model.readout.n = n0_
        model.train()
        acc = float(np.round(torch.cat(hit).mean().item() * 100, 3)) if hit else 0.0
        if zs:
            self._eval_diag(eval_task_id, cur_task_id, torch.cat(zs), torch.cat(ys), acc, torch.cat(hs))
            if self._ediag is not None and phase == "test":
                try:
                    self._ediag.margin(eval_task_id, cur_task_id, torch.cat(hs), torch.cat(ys))
                except Exception as ex_:
                    logger.info("[DIAG-M] skipped: %s" % ex_)
        if lown_ is not None:
            try:
                lown_.report(model, cur_task_id)
            except Exception as ex_:                               # diagnostics must never stop a run
                logger.info("[DIAG-L] skipped: %s" % ex_)
        if cs_:
            self._content_diag(eval_task_id, cur_task_id, torch.cat(cs_), torch.cat(ys))
        if lncap_ and ys:
            self._ln_diag(cur_task_id, lncap_, torch.cat(ys))
        return acc, None

    @torch.no_grad()
    def _ln_diag(self, cur, cap, Y):
        """Diagnostic (evaluation only, logging): post-LN rescaling test on task 0's test sentences at the decision position: per norm,
        the massive input coordinate j (largest share of |x - mu|^2), the relative change of x_j and of sigma since the
        previous evaluation, the measured change of the norm's output |dy|, and the change predicted by the sigma
        rescaling alone, -dsigma/sigma (y - b): its size relative to |dy| and its cosine with dy (class means)."""
        Y = Y.long()
        ks = sorted(set(Y.tolist()))
        st = {}
        mods = list(self.hidx.diag.values())
        ffn = {k: v for k, v in cap.items() if isinstance(k, tuple)}
        cap = {k: v for k, v in cap.items() if not isinstance(k, tuple)}
        self._ffn_diag(cur, ffn, Y, ks)
        for i_, lst in cap.items():
            X = torch.cat([a for a, _ in lst]); O = torch.cat([b for _, b in lst])
            mu = X.mean(1, keepdim=True)
            sg = (X - mu).pow(2).mean(1).sqrt()
            st[i_] = (torch.stack([X[Y == k].mean(0) for k in ks]), torch.stack([sg[Y == k].mean() for k in ks]),
                      torch.stack([O[Y == k].mean(0) for k in ks]))
        prev = getattr(self, "_lnstate", None)
        self._lnstate = (ks, st)
        if prev is None or prev[0] != ks:
            return
        out = []
        for i_, (Xm, sg, Om) in st.items():
            Xp, sgp, Op = prev[1][i_]
            xc = Xp - Xp.mean(1, keepdim=True)
            e_ = xc.pow(2).mean(0)
            j = int(e_.argmax())
            shj = float(e_[j] / e_.sum())
            dxj = float(((Xm[:, j] - Xp[:, j]).abs() / Xp[:, j].abs().clamp(min=1e-6)).mean())
            ds = (sg - sgp) / sgp
            b_ = mods[i_].bias.detach().float().cpu() if getattr(mods[i_], "bias", None) is not None else 0.0
            pred = -ds.unsqueeze(1) * (Op - b_)
            dy = Om - Op
            cos = float(F.cosine_similarity(pred, dy, dim=1).mean())
            out.append("%d:j%d sh%.2f dxj%+.3f dsig%+.3f |dy|%.2f pred/|dy| %.2f cos %.2f" % (
                i_, j, shj, dxj, float(ds.mean()), float(dy.norm(dim=1).mean()),
                float(pred.norm(dim=1).mean() / dy.norm(dim=1).mean().clamp(min=1e-12)), cos))
        for k0 in range(0, len(out), 9):
            logger.info("[HLE-DIAG] LN rescaling T%d task 0 (norm idx: massive coord, its share, rel dx_j, rel dsigma, "
                        "|dy|, sigma-only prediction / |dy|, cos): %s" % (cur, " | ".join(out[k0:k0 + 9])))

    def _eval_diag(self, e, cur, Z, Y, acc, H=None):
        """Diagnostic (evaluation only, logging): where the test sentences of task e land among the heads after task cur, and each
        head's logit level on them."""
        nc = self.CL_dataset.continual_config["CUR_NUM_CLASS"]
        bd = np.cumsum([0] + list(nc[:cur + 1]))
        hid = torch.bucketize(Z.argmax(1), torch.tensor(bd[1:]), right=True)
        win = [float((hid == h).float().mean()) for h in range(cur + 1)]
        hmax = [float(Z[:, bd[h]:bd[h + 1]].max(1).values.mean()) for h in range(cur + 1)]
        gold = float(Z.gather(1, Y.long().unsqueeze(1)).mean())
        logger.info("[HLE-DIAG] after T%d eval task %d (acc %.2f): win share per head %s | mean head-max logit %s | "
                    "gold logit %.2f" % (cur, e, acc, "[" + " ".join("%.2f" % w for w in win) + "]",
                                         "[" + " ".join("%.2f" % v for v in hmax) + "]", gold))
        if H is not None:
            self._drift_diag(e, cur, H, Y)
        if e == cur:
            m = self._unwrap(self.wrap_model).model
            hs = m.readout.heads
            logger.info("[HLE-DIAG] after T%d heads: weight row-norm %s | bias mean %s" % (
                cur, "[" + " ".join("%.3f" % float(hs[h].weight.norm(dim=1).mean()) for h in range(cur + 1)) + "]",
                "[" + " ".join("%.3f" % float(hs[h].bias.mean()) for h in range(cur + 1)) + "]"))
            lb_ = [mm.bias for a, ms in self.hidx.areas.items() if a != "readout" for mm in ms if mm.bias is not None]
            gn_ = [g for g in self.hidx.diag.values()]
            gb_ = [g.bias.detach() for g in gn_ if isinstance(getattr(g, "bias", None), torch.nn.Parameter)]
            logger.info("[HLE-DIAG] after T%d trunk: linear bias rms %.4f | norm gain mean %.4f, norm bias rms %.4f" % (
                cur, float(torch.cat([b_.detach().flatten() for b_ in lb_]).pow(2).mean().sqrt()) if lb_ else -1,
                float(torch.cat([g.weight.detach() for g in gn_]).mean()) if gn_ else -1,
                float(torch.cat(gb_).pow(2).mean().sqrt()) if gb_ else -1))

    @torch.no_grad()
    def _drift_diag(self, e, cur, H, Y):
        """Diagnostic (evaluation only, logging; does not affect training). Per class of test task e: its mean [CLS] state and gold
        logit at this evaluation, kept until the next one. The change of the mean gold logit since the previous
        evaluation splits exactly (logits are linear in h) into READOUT (this readout on the previous mean state minus
        the previous mean gold logit) + TRUNK (this readout's gold row on the change of the mean state); the newest
        head's top logit on task e's states likewise into readout (its row on the previous states) and trunk. Also the
        common-mode share of the state drift: |mean_k d_k|^2 / mean_k |d_k|^2."""
        m = self._unwrap(self.wrap_model).model
        hd = m.readout.heads
        nc = self.CL_dataset.continual_config["CUR_NUM_CLASS"]
        bd = np.cumsum([0] + list(nc))
        W = torch.cat([hd[h].weight.detach().float().cpu() for h in range(cur + 1)])
        b = torch.cat([hd[h].bias.detach().float().cpu() for h in range(cur + 1)])
        Y = Y.long()
        ks = sorted(set(Y.tolist()))
        HL = torch.stack([H[Y == k].mean(0) for k in ks])          # (classes, layers + 1, d)
        Hm = HL[:, -1]
        g_now = torch.stack([(Hm[i] @ W[k] + b[k]) for i, k in enumerate(ks)])
        st = getattr(self, "_dstate", None)
        if st is None:
            st = self._dstate = {}
        prev = st.get(e)
        st[e] = (ks, Hm, g_now, HL)
        if prev is None or prev[0] != ks:
            return
        _, Hp, gp, HLp = prev
        dL = HL - HLp                                               # per class, per layer
        # per layer: the coordinate holding most of the state's energy, its share of |h|^2 and of the drift's |d|^2
        mx_ = []
        for l in range(HL.shape[1]):
            e_ = HLp[:, l].pow(2).mean(0)
            j_ = int(e_.argmax())
            mx_.append("%d:%.2f/%.2f" % (j_, float(e_[j_] / e_.sum().clamp(min=1e-30)),
                                         float(dL[:, l, j_].pow(2).mean() / dL[:, l].pow(2).sum(1).mean().clamp(min=1e-30))))
        lay = " ".join("%.2f/%.2f" % (float(dL[:, l].mean(0).norm() / HL[:, l].norm(dim=1).mean()),
                                      float(dL[:, l].norm(dim=1).mean() / HL[:, l].norm(dim=1).mean()))
                       for l in range(HL.shape[1]))
        ro = torch.stack([(Hp[i] @ W[k] + b[k]) for i, k in enumerate(ks)]) - gp
        tr = torch.stack([((Hm[i] - Hp[i]) @ W[k]) for i, k in enumerate(ks)])
        d = Hm - Hp
        cm = float(d.mean(0).pow(2).sum() / d.pow(2).sum(1).mean().clamp(min=1e-30))
        msg = ("[HLE-DIAG] drift T%d task %d: gold logit change %+.2f = readout %+.2f + trunk %+.2f | state drift |d| %.3f "
               "(|h| %.2f), common-mode share %.2f" % (cur, e, float((g_now - gp).mean()), float(ro.mean()),
                                                       float(tr.mean()), float(d.norm(dim=1).mean()),
                                                       float(Hm.norm(dim=1).mean()), cm))
        if e < cur:
            Wn, bn = W[bd[cur]:bd[cur + 1]], b[bd[cur]:bd[cur + 1]]
            nw_now = (Hm @ Wn.t() + bn).max(1).values.mean()
            nw_prev = (Hp @ Wn.t() + bn).max(1).values.mean()
            msg += " | newest head top logit on these states %.2f (on the previous states %.2f: trunk part %+.2f)" % (
                float(nw_now), float(nw_prev), float(nw_now - nw_prev))
        logger.info(msg)
        logger.info("[HLE-DIAG] drift T%d task %d per layer (emb, L1..): common-mode / total [CLS] drift relative to |h|: %s"
                    % (cur, e, lay))
        logger.info("[HLE-DIAG] drift T%d task %d per layer: top coordinate : its share of |h|^2 / of |d|^2: %s"
                    % (cur, e, " ".join(mx_)))

    @torch.no_grad()
    def _ffn_diag(self, cur, ffn, Y, ks):
        """Diagnostic (evaluation only): which FFN neurons write the massive stream coordinate at the decision position, and why their write
        changed since the previous evaluation: per layer, the top-8 neurons k by |W_out[j,k] a_k|, the change of their
        summed write, and the change of their pre-activations split exactly (linear in the class-mean input) into a
        WEIGHT part (W_in,now x_prev + b_now - pre_prev) and an INPUT part (pre_now - W_in,now x_prev - b_now)."""
        if self.hidx._M is None:
            return
        j = int(self.hidx._M[-1]) if 588 not in self.hidx._M.tolist() else 588
        # the largest-share massive coordinate, or coordinate 588 when it is among them (RoBERTa-base)
        mean = lambda T: torch.stack([T[Y == k].mean(0) for k in ks])
        ins = {a: (mean(torch.cat([x for x, _ in v])), mean(torch.cat([o for _, o in v])))
               for (kd, a), v in ffn.items() if kd == "in"}
        outs = {a: mean(torch.cat([x for x, _ in v])) for (kd, a), v in ffn.items() if kd == "out"}
        prev = getattr(self, "_ffnstate", None)
        self._ffnstate = (ks, ins, outs)
        if prev is None or prev[0] != ks:
            return
        rows = []
        ain = list(ins); aout = list(outs)
        for li, (ai, ao) in enumerate(zip(ain, aout)):
            mi = self.hidx.areas[ai][0]; mo = self.hidx.areas[ao][0]
            Wo = mo.weight.detach().float().cpu()[j]                       # (ffn,)
            a_now, a_prev = outs[ao], prev[2][ao]
            wr_prev = (a_prev * Wo).mean(0)
            K = wr_prev.abs().topk(8).indices
            w_now = float((a_now[:, K] * Wo[K]).sum(1).mean()); w_prev = float((a_prev[:, K] * Wo[K]).sum(1).mean())
            w_all_now = float((a_now * Wo).sum(1).mean()); w_all_prev = float((a_prev * Wo).sum(1).mean())
            Wi = mi.weight.detach().float().cpu()[K]; bi = mi.bias.detach().float().cpu()[K] if mi.bias is not None else 0.0
            x_now, p_now = ins[ai]; x_prev, p_prev = prev[1][ai]
            mid = x_prev @ Wi.t() + bi
            wpart = (mid - p_prev[:, K]).norm(dim=1).mean(); ipart = (p_now[:, K] - mid).norm(dim=1).mean()
            rows.append("L%d write %.2f->%.2f (top8 %.2f->%.2f) dpre weight %.3f input %.3f |pre| %.2f" % (
                li, w_all_prev, w_all_now, w_prev, w_now, float(wpart), float(ipart), float(p_prev[:, K].norm(dim=1).mean())))
        logger.info("[HLE-DIAG] FFN write into stream coordinate %d at the decision position, T%d task 0: %s"
                    % (j, cur, " | ".join(rows)))

    @torch.no_grad()
    def _content_diag(self, e, cur, C, Y):
        """Diagnostic (evaluation only): the content tokens' mean state per layer, drift since the previous
        evaluation relative to its norm (class means), i.e. whether old sentences' content positions move before their
        decision position does."""
        Y = Y.long()
        ks = sorted(set(Y.tolist()))
        Cm = torch.stack([C[Y == k].mean(0) for k in ks])
        st = getattr(self, "_cstate", None)
        if st is None:
            st = self._cstate = {}
        prev = st.get(e)
        st[e] = (ks, Cm)
        if prev is None or prev[0] != ks or e > 0:
            return
        d = (Cm - prev[1]).norm(dim=2).mean(0) / prev[1].norm(dim=2).mean(0).clamp(min=1e-12)
        logger.info("[HLE-DIAG] content-token drift T%d task %d per layer (emb, L1..), relative: %s"
                    % (cur, e, " ".join("%.3f" % float(v) for v in d)))
