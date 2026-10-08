"""Shared protocol for the projection baselines (Sculpting Subspaces, CODE-CL).

The scaffolding mirrors models/HippoLite.py (the DeltaHippo learner) without the hippocampus, so that the
baselines run on exactly the same protocol: same backbone loading (utils/backbone.py), fp32 master weights with a
bf16-autocast forward, same data pipeline / prompts / label verbalisation (utils/dataloader.py), answer-token
cross-entropy from the HF model, and the same generation-based exact-match evaluation
(utils/evaluation.evaluate_sent_level_acc_with_generation, greedy, backbone_max_new_token) on every seen task after
every task. Optimisers are built by the methods themselves (both reset their optimiser at every task boundary).
"""
import logging

import numpy as np
import torch
import torch.nn as nn

from utils.backbone import get_backbone
from utils.evaluation import evaluate_sent_level_acc_with_generation
from utils.dataloader import get_dataloader
from utils.metric import ResultSummary
from utils.wrapmodel import WrapModel
from models.Base import BaseLearner

logger = logging.getLogger()


def str2bool(x):
    return str(x).lower() in ("1", "true", "yes")


class ProjLearnerBase(BaseLearner):
    TAG = "BASE"

    # ------------------------------------------------------------------ build
    def build_metric(self):
        self.result_summary = ResultSummary(num_task=self.CL_dataset.continual_config["NUM_TASK"])

    def build_backbone(self):
        self.model, self.tokenizer = get_backbone(self.params, self.CL_dataset.continual_config["NUM_TASK"])
        # standard mixed precision as in DeltaHippo: fp32 weights / optimiser state, bf16 autocast forward
        self.model = self.model.to(torch.float32)
        _orig_fwd = self.model.forward

        def _amp_fwd(*a, **k):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return _orig_fwd(*a, **k)
        self.model.forward = _amp_fwd
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.tied = self.model.get_input_embeddings().weight is self.model.get_output_embeddings().weight
        n = sum(p.numel() for p in self.model.parameters())
        logger.info("[%s] full fine-tuning: %.1fM parameters, tied table %s, dtype %s"
                    % (self.TAG, n / 1e6, self.tied, str(next(self.model.parameters()).dtype)))

    def build_classifier(self):
        self.classifier = None

    def build_optimizer(self):
        # built per task by the method (both methods reset the optimiser at every task boundary)
        self.optimizer = None

    def build_dataloader(self):
        self.train_loader_list, self.dev_loader_list, self.test_loader_list = \
            get_dataloader(self.params, self.CL_dataset, self.tokenizer)

    def build_buffer(self):
        self.buffer = None

    def accelerate_prepare(self):
        self.wrap_model = WrapModel(self.model, nn.ModuleList())
        (self.wrap_model, *self.train_loader_list) = \
            self.accelerator.prepare(self.wrap_model, *self.train_loader_list)
        if len(self.dev_loader_list) > 1:
            self.dev_loader_list = list(self.accelerator.prepare(*self.dev_loader_list))
            self.test_loader_list = list(self.accelerator.prepare(*self.test_loader_list))
        else:
            self.dev_loader_list = [self.accelerator.prepare(self.dev_loader_list[0])]
            self.test_loader_list = [self.accelerator.prepare(self.test_loader_list[0])]

    def _unwrap(self, m):
        return m.module if hasattr(m, "module") else m

    def base_model(self):
        return self._unwrap(self.wrap_model).model

    def decoder_layers(self):
        m = self.base_model()
        b = getattr(m, "model", m)
        b = getattr(b, "language_model", b)
        return b.layers

    # ------------------------------------------------------------------ training loop
    def train_epochs(self, task_id):
        loader = self.train_loader_list[task_id]
        nep = int(self.params.training_epochs)
        model = self.base_model()
        model.train()
        for ep in range(nep):
            if self.accelerator.is_main_process:
                logger.info("[%s] Task %d | Epoch %d/%d" % (self.TAG, task_id + 1, ep + 1, nep))
            for batch in loader:
                self.observe_batch(task_id, ep, batch)

    def lm_loss(self, lm_input):
        model = self.base_model()
        out = model(input_ids=lm_input["input_ids_with_ans"], attention_mask=lm_input["attention_mask_with_ans"],
                    labels=lm_input["labels_with_ans"], use_cache=False, return_dict=True)
        return out

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
        model = self.base_model()
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            acc, _ = evaluate_sent_level_acc_with_generation(
                model=model, eval_data_loader=loaders[eval_task_id], next_eval_data_loader=None,
                tokenizer=self.tokenizer, accelerator=self.accelerator, params=self.params,
                idx2label=self.CL_dataset.continual_config["idx2label"])
        model.train()
        return acc, None

    def log_peak(self, what):
        if torch.cuda.is_available():
            logger.info("[%s] %s: peak allocated %.2f GB" % (self.TAG, what, torch.cuda.max_memory_allocated() / 1e9))
            torch.cuda.reset_peak_memory_stats()
