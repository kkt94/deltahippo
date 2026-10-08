# DeltaHippo

Code for **"Hippocampal Consolidation without Replay for Continual Learning of Fully Fine-tuned Networks"**.

DeltaHippo keeps every parameter of a network trainable across a sequence of classification tasks without replaying
earlier data, without a teacher model and without any memory at inference time. A training-only "hippocampus" indexes
the activity geometry of every synapse group (the set of linear maps that read the same input). At the end of each
task (sleep) it writes the subspace and class-level records of that task by gated-delta write and erase. During later
tasks (wake) it detects conflicts with earlier records and shapes both the gradient that enters the optimizer and the
update the optimizer produces. At test time the plain model is evaluated with every hook detached.

The repository covers the four model families of the paper:

| Family | Models | Entry point | Method code |
|---|---|---|---|
| Decoder LLMs | Qwen3-0.6B / 4B / 8B, Llama-3.2-1B / 3B, Llama-3.1-8B | `main_CL.py` | `models/HippoLite.py`, `utils/hippo_lite.py` |
| Text encoders | BERT-base (uncased), RoBERTa-base | `main_CL.py` | `models/HippoLiteEnc.py`, `utils/hippo_enc.py` |
| Vision | ViT-B/16 (ImageNet-21k), ResNet-50 | `tools/vision_hippo.py` | `utils/hippo_vis.py` |
| Baselines (LLMs) | Sculpting Subspaces, CODE-CL | `main_CL.py` | `models/SculptSVD.py`, `models/CODECL.py`, `utils/cl_proj_common.py` |

## Repository layout

```
main_CL.py              continual-learning driver for text (decoder LLMs, encoders, baselines)
models/                 learners: HippoLite (LLMs), HippoLiteEnc (encoders), SculptSVD, CODECL, Base
utils/                  framework (data, prompts, backbones, evaluation, metrics) and the method modules
                        hippo_lite.py (LLMs), hippo_enc.py (encoders), hippo_vis.py (vision)
tools/                  vision runner (vision_hippo.py), its diagnostics (vision_diag.py), ImageNet-R cache builder
configs/llm/            DeltaHippo, decoder LLMs (BENCH.yaml, BENCH_wide.yaml, BENCH_8b.yaml)
configs/encoder/        DeltaHippo, BERT / RoBERTa
configs/baselines/      Sculpting Subspaces (sculpt_*.yaml) and CODE-CL (codecl_*.yaml)
dataset/                class-incremental split definitions (continual_config.json); the data itself is not shipped
scripts/                run scripts and data preparation notes
```

The text framework (`main_CL.py`, `models/Base.py`, `utils/`) is derived from the codebase of Zheng, Qiu & Ma (2024),
<https://github.com/zzz47zzz/codebase-for-incremental-learning-with-llm>.

## Installation

Python 3.12 and a CUDA GPU. The reported runs used PyTorch 2.9.1 (CUDA 12.8) and transformers 4.57.6:

```
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install flash-attn==2.8.3 --no-build-isolation      # optional, see below
```

When `flash-attn` is importable, the Llama and Qwen3-4B/8B backbones are loaded with `flash_attention_2`, otherwise
with PyTorch SDPA. The reported decoder runs used flash-attn.

GPU memory (peak, training / sleep): about 23 GB for Llama-3.2-1B and 140-145 GB for the 8B models (gradient
checkpointing on). All runs fit a single GPU; no multi-GPU setup is needed.

## Data

No data are distributed. See [`scripts/prepare_data.md`](scripts/prepare_data.md) for the sources and the expected
layout (`dataset/<name>/continual_config.json` + `continual_data.json`; `data_vision/` for CIFAR-100 and ImageNet-R).
The text splits follow Zheng, Qiu & Ma (2024): TACRED 8 tasks x 5 relations, FewRel 8 x 10, Banking77 7 x 11,
CLINC150 15 x 10. Vision: CIFAR-100 10 x 10, ImageNet-R 10 x 20.

## Model weights

Weights are not shipped. Download them from the Hugging Face Hub (the Llama models require accepting the Meta license
on the Hub first):

| `--backbone` / `MODEL` | Hugging Face id | Expected location |
|---|---|---|
| `Qwen3-0.6B` | `Qwen/Qwen3-0.6B` | `./Qwen3-0.6B` |
| `Qwen3-4B` | `Qwen/Qwen3-4B` | `./Qwen3-4B` |
| `Qwen3-8B` | `Qwen/Qwen3-8B` | `./Qwen3-8B` |
| `Llama-3.2-1B` | `meta-llama/Llama-3.2-1B` | `./Llama-3.2-1B` |
| `Llama-3.2-3B` | `meta-llama/Llama-3.2-3B` | `./Llama-3.2-3B` |
| `Llama-3.1-8B` | `meta-llama/Llama-3.1-8B` | `./Llama-3.1-8B` |
| `bert-base-uncased` | `google-bert/bert-base-uncased` | `./hf_models/bert-base-uncased` (optional) |
| `roberta-base` | `FacebookAI/roberta-base` | `./hf_models/roberta-base` (optional) |
| ViT-B/16 (`--model vit`) | `google/vit-base-patch16-224-in21k` | `./hf_models/vit-base-in21k` (required) |
| ResNet-50 (`--model resnet50`) | torchvision `ResNet50_Weights.IMAGENET1K_V2` | torch hub cache (downloaded automatically) |

The decoder backbones are loaded from the directory named by `--backbone`, relative to the repository root, e.g.

```
huggingface-cli download Qwen/Qwen3-0.6B --local-dir Qwen3-0.6B
huggingface-cli download meta-llama/Llama-3.2-1B --local-dir Llama-3.2-1B
huggingface-cli download google/vit-base-patch16-224-in21k --local-dir hf_models/vit-base-in21k
```

The encoders are read from `./hf_models/<name>` when that directory exists, otherwise from the Hub by name.

## Running

All scripts run from any directory, write the log to `experiments/logs/<run>.log` and print the last result line.

### Decoder LLMs (Table 1)

```
bash scripts/run_llm.sh BENCH MODEL [SEED]      # BENCH: tacred | fewrel | banking77 | clinc150
bash scripts/run_llm.sh tacred Llama-3.2-1B 42
```

which runs

```
python main_CL.py --exp_prefix NAME --cfg configs/llm/<config>.yaml --backbone MODEL \
  --classifier None --training_epochs 3 --info_per_steps 25 --seed SEED
```

with the configuration chosen by the architecture rules below:

| Models | Configuration |
|---|---|
| Qwen3-0.6B, Llama-3.2-1B, Llama-3.2-3B | `configs/llm/BENCH.yaml` |
| Qwen3-4B | `configs/llm/BENCH_wide.yaml` |
| Qwen3-8B, Llama-3.1-8B | `configs/llm/BENCH_8b.yaml` |

All decoder runs use the same recipe: full fine-tuning of every parameter, AdamW (weight decay 5e-4), learning rate
2e-5, batch 8, 3 epochs per task, fp32 weights and optimizer state with a bf16 autocast forward, greedy generation
(at most 10 new tokens) and exact match at test time.

### Text encoders (Extended Data Table 1)

```
bash scripts/run_encoder.sh BENCH MODEL [SEED]  # MODEL: bert-base-uncased | roberta-base
bash scripts/run_encoder.sh tacred bert-base-uncased 42
```

(`main_CL.py --cfg configs/encoder/BENCH.yaml --backbone MODEL --classifier Linear --training_epochs 3
--info_per_steps 25 --seed SEED`). Learning rate 2e-5 (heads 1e-3), batch 8, 3 epochs per task, one linear head per
task on [CLS].

### Vision (Extended Data Table 2)

```
bash scripts/run_vision.sh MODEL DATASET [SEED] # MODEL: vit | resnet50   DATASET: cifar100 | imr
bash scripts/run_vision.sh vit cifar100 1993
```

The flags of the reported runs are:

```
# ViT-B/16 (backbone lr 3e-5)
python tools/vision_hippo.py --model vit --dataset {cifar100,imr} --method ours --seed SEED \
  --fresh_head --oldrow --oldrow_span --fresh_aug --wdfold --pdet --cmpowm --lastcls --tabln --sink --lr_bb 3e-5 \
  --out experiments/vision/NAME
# ResNet-50 (backbone lr: default 1e-4)
python tools/vision_hippo.py --model resnet50 --dataset {cifar100,imr} --method ours --seed SEED \
  --fresh_head --oldrow --oldrow_span --fresh_aug --wdfold --pdet --cmpowm --no_owm --cmpcons \
  --out experiments/vision/NAME
```

10 tasks, batch 64, 3 epochs per task, heads lr 1e-3. `--method seq` and `--method seqstar` run the SEQ and SEQ*
references with the same protocol.

### Baselines (Table 1, Qwen3-0.6B and Llama-3.2-1B)

```
bash scripts/run_baselines.sh METHOD BENCH MODEL [SEED]   # METHOD: sculpt | codecl
bash scripts/run_baselines.sh codecl tacred Qwen3-0.6B 42
```

`sculpt` is Sculpting Subspaces (Nayak et al., 2025) with the fixed rank budget ((i-1)/n of the singular directions
of each matrix protected at task i of n); `codecl` is CODE-CL (Apolinario, Choudhary & Roy, 2025) in its
task-agnostic form K = 0. Both use the same optimizer, learning rate, batch and epochs as DeltaHippo.

## Architecture rules

The method is the same for every decoder model; three rules depend only on the architecture:

- **Relative step.** Every trunk matrix (q/k/v/o, gate/up/down) steps with `lr x min(1, rms(W) / mean rms)`, the mean
  taken over all trunk matrices of the model (`models/HippoLite.py`, `build_optimizer`). The vision and encoder
  learners apply the same rule to every linear / convolutional map.
- **Width rule** (`hippo_widthref: 2560`, in the 8B configurations only). The matrices of Qwen3-8B and Llama-3.1-8B
  step by `2560 / hidden_size` = 0.625; the models up to 4B, including Llama-3.2-3B, use no width scaling. Vector-like
  parameters (input table, norm gains) keep the base learning rate.
- **fp64 orthonormal basis** (`hippo_orthobasis: true`, the `_wide` and `_8b` configurations). For models whose widest
  synapse-group input exceeds 9,000 (the MLP down-projection input of Qwen3-4B, 9,728; Qwen3-8B, 12,288;
  Llama-3.1-8B, 14,336) the held span and the class-pattern columns are kept as one fp64 QR basis with a
  numerical-rank tolerance.
- The 8B configurations additionally enable gradient checkpointing (`hippo_gradckpt: true`), which changes memory
  use only.

## Output

Text runs log, after every task, a line

```
Mode = CIL, Test Result = {'Test_Acc_Task_0': ..., ..., 'Test_Acc_Task_Seen': ..., 'Test_Acc_Task_All': ...}
```

followed by the accuracy matrix (`Result Summary Test After Task t`; row t = accuracy on every task after training
task t, -1 for tasks not yet seen). `Test_Acc_Task_Seen` is the mean test accuracy over the tasks seen so far; after
the last task it is the **final average accuracy (AA)** reported in the paper. Vision runs log one `ROW Tt: ... | AA`
line per task, a final `FINAL AA` line and write `result.json` to the `--out` directory.

Final AA (%) reported in the paper for DeltaHippo:

| Model | TACRED | FewRel | Banking77 | CLINC150 |
|---|---|---|---|---|
| Qwen3-0.6B | 73.22 | 71.95 | 85.13 | 89.07 |
| Llama-3.2-1B | 76.77 | 78.29 | 83.67 | 92.27 |
| Qwen3-4B | 71.61 | 76.71 | 85.33 | 94.98 |
| Llama-3.2-3B | 73.47 | 80.63 | 87.27 | 92.18 |
| Qwen3-8B | 80.21 | 78.87 | 90.78 | 95.27 |
| Llama-3.1-8B | 77.75 | 81.11 | 87.86 | 95.16 |
| BERT-base | 68.71 ± 1.90 | 64.38 ± 2.48 | 86.41 ± 0.43 | 90.18 ± 0.65 |
| RoBERTa-base | 68.64 ± 0.96 | 67.90 ± 3.60 | 87.67 ± 0.44 | 91.02 ± 0.47 |

| Model | CIFAR-100 | ImageNet-R |
|---|---|---|
| ViT-B/16 | 88.12 ± 0.15 | 64.34 ± 0.45 |
| ResNet-50 | 68.65 ± 0.98 | 61.16 ± 0.61 |

Results can vary slightly with GPU type and library versions. Setting `PDR_DETERMINISTIC=1` requests deterministic
PyTorch kernels (off by default).

## Citation

```
@article{deltahippo,
  title  = {Hippocampal Consolidation without Replay for Continual Learning of Fully Fine-tuned Networks},
  author = {Kim, Keuntae and Choi, Yong Suk},
  year   = {2026}
}
```

## License

The DeltaHippo code (`models/HippoLite.py`, `models/HippoLiteEnc.py`, `utils/hippo_lite.py`, `utils/hippo_enc.py`,
`utils/hle_diag.py`, `utils/hippo_vis.py`, `tools/`, the clustering utilities in `utils/`) and our implementations of the
baselines (`models/SculptSVD.py`, `models/CODECL.py`, `utils/cl_proj_common.py`) are released under the MIT License
(see `LICENSE`).

Third-party code:
- The text continual-learning framework (`main_CL.py`, `models/Base.py`, `models/__init__.py` and the remaining files in
  `utils/`) is adapted, with modifications, from the codebase of Zheng, Qiu & Ma (2024),
  <https://github.com/zzz47zzz/codebase-for-incremental-learning-with-llm>; these files remain the work of their original
  authors and are included for reproducibility.
- The baseline implementations follow the official code of Sculpting Subspaces
  (<https://github.com/Red-Hat-AI-Innovation-Team/orthogonal-subspace-learning>, MIT; OSFT in
  <https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer>, Apache-2.0) and CODE-CL
  (<https://github.com/mapolinario94/CODE-CL>, MIT).

Dataset and model licences are those of their respective providers.
