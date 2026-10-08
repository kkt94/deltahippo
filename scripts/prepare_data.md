# Preparing the data

No dataset is distributed with this repository. TACRED is licensed by the Linguistic Data Consortium
(LDC2018T24); the other benchmarks are public but are not redistributed here.

All paths below are relative to the repository root (the directory that contains `main_CL.py`).

## Text benchmarks (decoder LLMs, text encoders, baselines)

The class-incremental splits follow the protocol and the preprocessing of Zheng, Qiu & Ma (2024),
*Learn or Recall? Revisiting Incremental Learning with Pre-trained Language Models* (ACL 2024), whose public
codebase is <https://github.com/zzz47zzz/codebase-for-incremental-learning-with-llm>.

### Expected layout

```
dataset/
  tacred_task8/       continual_config.json  continual_data.json
  fewrel_task8/       continual_config.json  continual_data.json
  banking77_task7/    continual_config.json  continual_data.json
  clinc150_task15/    continual_config.json  continual_data.json
```

`continual_config.json` (shipped) holds the number of tasks and classes, the class names and the class order
(`CUR_CLASS`, `label2idx`, `idx2label`, ...). It contains no example text.

`continual_data.json` (not shipped) maps each task id (`"0"`, `"1"`, ...) to its splits:

```
{
  "0": {
    "train": {"input": [...], "target": [...], "label_idx_cil": [...], "label_idx_til": [...]},
    "dev":   {"input": [...], "target": [...], "label_idx_cil": [...], "label_idx_til": [...]},
    "test":  {"input": [...], "target": [...], "label_idx_cil": [...], "label_idx_til": [...]}
  },
  "1": { ... },
  ...
}
```

`input` is the raw sentence (for TACRED/FewRel with entity markers `[E11] ... [E12]`, `[E21] ... [E22]`), `target`
the natural-language class name, `label_idx_cil` the class index over all tasks and `label_idx_til` the index inside
the task.

### Building the files

1. Clone the codebase of Zheng et al. (2024) and follow its README ("Step 2: prepare the dataset"): obtain the raw
   datasets (for TACRED, from the LDC; the FewRel/TACRED inputs are read from `dataset/fewrel/FewRel-2021.pkl` and
   `dataset/tacred/TACRED-2021.pkl`, Banking77 from `dataset/banking77/{train.csv,test.csv,categories.json}`, CLINC150 from
   `dataset/clinc150/{data_full,label_dict}.json`).
2. Run its preprocessing with seed 1:

   ```
   python utils/dataformat_preprocess.py --dataset tacred    --seed 1
   python utils/dataformat_preprocess.py --dataset fewrel    --seed 1
   python utils/dataformat_preprocess.py --dataset banking77 --seed 1
   python utils/dataformat_preprocess.py --dataset clinc150  --seed 1
   ```

   This writes `dataset/{tacred_task8,fewrel_task8,banking77_task7,clinc150_task15}/` with the two json files.
3. Copy the four `continual_data.json` files into the matching folders under `dataset/` here. The generated
   `continual_config.json` must be identical to the shipped one (same class order); check with

   ```
   cmp <their>/dataset/tacred_task8/continual_config.json dataset/tacred_task8/continual_config.json
   ```

Split sizes (train / dev / test): TACRED 7,391 / 1,259 / 1,259 (40 relations, 8 tasks x 5);
FewRel 33,600 / 11,200 / 11,200 (80 relations, 8 x 10); Banking77 10,003 / 3,080 / 3,080 (77 intents, 7 x 11);
CLINC150 15,000 / 3,000 / 4,500 (150 in-scope intents, 15 x 10).

## Vision benchmarks

```
data_vision/
  cifar-100-python/            train, test, meta   (the "python version" of CIFAR-100)
  imagenet-r/<wnid>/*.jpg      200 class folders
  cache/                       written by tools/vision_data_prep.py
```

1. CIFAR-100: download `cifar-100-python.tar.gz` from <https://www.cs.toronto.edu/~kriz/cifar.html> and extract it
   into `data_vision/` (this creates `data_vision/cifar-100-python/`).
2. ImageNet-R: download `imagenet-r.tar` from <https://github.com/hendrycks/imagenet-r> and extract it into
   `data_vision/` (this creates `data_vision/imagenet-r/`). Then build the uint8 cache (Resize 256, CenterCrop 224,
   per-class fixed-seed 80/20 split):

   ```
   python tools/vision_data_prep.py
   ```

   This writes `data_vision/cache/imr_{train,test}_{x,y}.npy` and `imr_wnids.txt`.

The class order of both vision benchmarks is `numpy.random.RandomState(1993).permutation` (the PyCIL convention) and
is computed in `tools/vision_hippo.py`.
