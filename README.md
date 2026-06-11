# UDBM-ICML26

Official training and evaluation code for **Uncertainty-Aware Diffusion Bridge Model (UDBM)**.

UDBM uses:

- **Stage 1**: an auxiliary restoration network for pixel-wise uncertainty prediction.
- **Stage 2**: an uncertainty-aware diffusion bridge with a denoising/restoration network.
- **Single-step inference** by default for the released evaluation script.

## Repository Structure

```text
UDBM-ICML26/
|-- train_s1.py                 # train Stage 1 uncertainty estimator
|-- train_s2.py                 # train Stage 2 UDBM bridge
|-- test_s2.py                  # evaluate Stage 2 checkpoints
|-- train_s1.sh                 # simple Stage 1 training command
|-- train_s2.sh                 # simple Stage 2 training command
|-- train.sh                    # simple two-stage training command
|-- test.sh                     # simple standard benchmark evaluation
|-- test_real.sh                # simple real-world evaluation
|-- test_cd11.sh                # simple CD11 evaluation
|-- requirements.txt
|-- scripts/                    # configurable shell wrappers
|-- data/                       # universal paired dataset loader
|-- metrics/                    # PSNR, SSIM, NIQE utilities
|-- pretrained/                 # copied final checkpoints
|-- src_s1/
|   |-- model_udbm_s1_s.py
|   |-- model_udbm_s1_m.py
|   `-- model_udbm_s1_l.py
`-- src/
    |-- local_arch.py
    |-- model_udbm_s2_s.py
    |-- model_udbm_s2_m.py
    `-- model_udbm_s2_l.py
```

The repository is intentionally trimmed to the code needed for Stage 1 training, Stage 2 training, and Stage 2 evaluation.

## Model Zoo

| Variant | Stage-1 uncertainty estimator | Stage-2 restoration bridge | Stage-1 file | Stage-2 file |
|---|---|---|---|---|
| UDBM-S | `width=16`, `blocks=[1,1,1,10]` | `width=16`, `blocks=[1,1,1,28]` | `src_s1/model_udbm_s1_s.py` | `src/model_udbm_s2_s.py` |
| UDBM-M | `width=16`, `blocks=[1,1,1,28]` | `width=24`, `blocks=[1,1,1,28]` | `src_s1/model_udbm_s1_m.py` | `src/model_udbm_s2_m.py` |
| UDBM-L | `width=32`, `blocks=[1,1,1,16]` | `width=48`, `blocks=[1,1,1,28]` | `src_s1/model_udbm_s1_l.py` | `src/model_udbm_s2_l.py` |

`L` is the default variant.

## Installation

The original experiments used Python 3.7 and PyTorch 1.12. Install the PyTorch/CUDA build that matches your machine, then install the remaining dependencies:

```bash
cd UDBM-ICML26
pip install -r requirements.txt
```

If `sk-video` fails during dependency installation, install it separately:

```bash
pip install sk-video
```

## Dataset Layout

The default data root is:

```text
./datasets/all_in_one
```

Pass another root with `--dataroot`. The code uses relative paths only.

### Training Data

Prepare the training folders as follows:

```text
<dataroot>/
|-- LOL/train/
|   |-- low
|   `-- high
|-- syn_rain/train/
|   |-- input
|   `-- target
|-- Deblur/train/
|   |-- input
|   `-- target
|-- Snow100K/train/Snow100K-L/
|   |-- synthetic
|   `-- gt
`-- RESIDE/OTS_ALPHA/
    |-- haze/OTS
    `-- clear/clear_images
```

### Standard Test Data

The testing loader follows the original all-in-one restoration task folders, including:

```text
<dataroot>/
|-- syn_rain/test/
|   |-- Test100
|   |-- Test1200
|   |-- Test2800
|   |-- Rain100H
|   `-- Rain100L
|-- Snow100K/test/
|   |-- Snow100K-S
|   `-- Snow100K-L
|-- Deblur/test/GoPro/
`-- RESIDE/SOTS/outdoor/
```

### Real-World Test Data

Real-world branches are also placed under `--dataroot`:

```text
<dataroot>/
|-- real_dark/real_dark/
|   |-- MEF                         # real_dark_mef
|   |-- DICE                        # real_dark_dice
|   `-- NPE                         # real_dark_npe
|-- real_rain/real_rain/
|   `-- Practical                   # real_rain
|-- Snow100K/
|   `-- realistic                   # real_snow
`-- Deblur/test/
    |-- HIDE/
    |   |-- input                   # real_hide
    |   `-- target
    |-- RealBlur_J/
    |   |-- input                   # real_j
    |   `-- target
    `-- RealBlur_R/
        |-- input                   # real_r
        `-- target
```

`real_dark`, `real_rain`, and `real_snow` are unpaired in this loader, so PSNR/SSIM are not meaningful for them. Use the restored images and no-reference metrics for those subsets.

### CD11 Test Data

The CD11 branch is mapped to:

```text
<dataroot>/
`-- cd11/
    |-- clear                       # shared target folder
    |-- haze                        # h
    |-- haze_rain                   # hr
    |-- haze_snow                   # hs
    |-- low                         # l
    |-- low_haze                    # lh
    |-- low_haze_rain               # lhr
    |-- low_haze_snow               # lhs
    |-- low_rain                    # lr
    |-- low_snow                    # ls
    |-- rain                        # r
    `-- snow                        # s
```

Keep file ordering and names consistent across each degraded folder and `cd11/clear`.

## Pretrained Weights

Final weights copied into this repository are stored under `pretrained/`:

```text
pretrained/
|-- udbm_s/
|   |-- stage1.pt
|   `-- model-600.pt
|-- udbm_m/
|   |-- stage1.pt
|   `-- model-600.pt
`-- udbm_l/
    |-- stage1.pt
    `-- model-600.pt
```

`stage1.pt` is the uncertainty estimator checkpoint. `model-600.pt` is the Stage 2 UDBM checkpoint.

## Shell Commands

Activate your environment first:

```bash
conda activate fcl
```

The root-level `.sh` files are simple command wrappers. They can be run directly with `sh`.

```bash
# Stage 1 training.
sh train_s1.sh

# Stage 2 training. Uses ./ckpt_universal/udbm_l_s1/model-600.pt by default.
sh train_s2.sh

# Two-stage training. This runs Stage 1 first, then Stage 2.
sh train.sh

# Evaluate released pretrained weights on the standard tasks.
sh test.sh

# Evaluate real-world branches.
sh test_real.sh

# Evaluate CD11.
sh test_cd11.sh
```

These scripts keep the same style as the raw command:

```bash
CUDA_VISIBLE_DEVICES=0 sh test_real.sh
```

which is equivalent to running:

```bash
CUDA_VISIBLE_DEVICES=0 python test_s2.py \
  --variant L \
  --tasks real_dark real_blur real_rain real_snow
```

Common variables can be overridden before `sh`:

```bash
CUDA_VISIBLE_DEVICES=0,1 VARIANT=M DATAROOT=./datasets/all_in_one sh train_s1.sh
CUDA_VISIBLE_DEVICES=0 VARIANT=S sh test.sh
CUDA_VISIBLE_DEVICES=0 VARIANT=L RESULT_DIR=./result_real sh test_real.sh
CUDA_VISIBLE_DEVICES=0 VARIANT=L TASKS="rain snow" sh test.sh
```

Additional Python arguments can be appended after the script command:

```bash
sh train_s2.sh --train_num_steps 300000 --save_and_sample_every 500
sh test.sh --sampling_timesteps 1
```

Available root-level scripts:

| Script | Purpose |
|---|---|
| `train_s1.sh` | Train Stage 1 uncertainty estimator |
| `train_s2.sh` | Train Stage 2 UDBM bridge |
| `train.sh` | Train Stage 1 and then Stage 2 |
| `test.sh` | Evaluate pretrained weights on standard tasks |
| `test_real.sh` | Evaluate pretrained weights on real-world tasks |
| `test_cd11.sh` | Evaluate pretrained weights on CD11 |
| `test_checkpoint.sh` | Evaluate checkpoints from `ckpt_universal` |

The `scripts/` directory contains more configurable wrappers that use `conda run -n fcl` by default. Use them when you want the script to select the conda environment automatically.

| Script | Purpose |
|---|---|
| `scripts/train_s1.sh` | Train Stage 1 uncertainty estimator |
| `scripts/train_s2.sh` | Train Stage 2 UDBM bridge |
| `scripts/train_two_stage.sh` | Run Stage 1 and then Stage 2 |
| `scripts/test_pretrained.sh` | Evaluate copied pretrained weights |
| `scripts/test_checkpoint.sh` | Evaluate checkpoints from `ckpt_universal` |

Common variables:

| Variable | Default | Description |
|---|---|---|
| `CONDA_ENV` | `fcl` | Conda environment used through `conda run -n` |
| `VARIANT` | `L` | `S`, `M`, or `L` |
| `GPUS` | `0` | CUDA device list, for example `0` or `0,1` |
| `NUM_PROCESSES` | number of GPUs in `GPUS` | Number of accelerate processes for training |
| `DATAROOT` | `./datasets/all_in_one` | Dataset root |
| `TASKS` | `light_only rain blur fog snow` | Evaluation task list |
| `RESULT_DIR` | `./result` | Output folder for restored images |
| `MILESTONE` | `600` | Stage-2 checkpoint milestone |

Examples:

```bash
# Train UDBM-L Stage 1 on two GPUs.
GPUS=0,1 VARIANT=L DATAROOT=./datasets/all_in_one scripts/train_s1.sh

# Train UDBM-L Stage 2 with the default Stage-1 checkpoint.
GPUS=0,1 VARIANT=L scripts/train_s2.sh

# Train both stages.
GPUS=0,1 VARIANT=L scripts/train_two_stage.sh

# Evaluate released pretrained UDBM-L weights.
GPUS=0 VARIANT=L scripts/test_pretrained.sh

# Evaluate only selected tasks.
GPUS=0 VARIANT=L TASKS="rain snow cd11" scripts/test_pretrained.sh
```

Additional Python arguments can be appended after the script command:

```bash
GPUS=0,1 VARIANT=M scripts/train_s2.sh \
  --train_num_steps 300000 \
  --save_and_sample_every 500
```

Set `CONDA_ENV=` to use the currently activated shell environment instead of `conda run -n fcl`.

## Training

### Stage 1

Train the uncertainty estimator first.

Recommended wrapper:

```bash
CUDA_VISIBLE_DEVICES=0,1 VARIANT=L DATAROOT=./datasets/all_in_one sh train_s1.sh
```

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python train_s1.py \
  --variant L \
  --dataroot ./datasets/all_in_one
```

Multi-GPU with `accelerate`:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --multi_gpu \
  --num_processes 2 \
  train_s1.py \
  --variant L \
  --dataroot ./datasets/all_in_one
```

By default, Stage 1 checkpoints are written to:

```text
./ckpt_universal/udbm_l_s1/
```

With the default `--save_and_sample_every 1000`, the 600k-step checkpoint is:

```text
./ckpt_universal/udbm_l_s1/model-600.pt
```

Resume Stage 1:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --multi_gpu \
  --num_processes 2 \
  train_s1.py \
  --variant L \
  --resume_milestone 300
```

### Stage 2

Train Stage 2 after Stage 1 finishes. Pass the Stage 1 checkpoint through `--ckpt_path_s1`.

Recommended wrapper:

```bash
CUDA_VISIBLE_DEVICES=0,1 VARIANT=L sh train_s2.sh
```

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python train_s2.py \
  --variant L \
  --dataroot ./datasets/all_in_one \
  --ckpt_path_s1 ./ckpt_universal/udbm_l_s1/model-600.pt
```

Multi-GPU with `accelerate`:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --multi_gpu \
  --num_processes 2 \
  train_s2.py \
  --variant L \
  --dataroot ./datasets/all_in_one \
  --ckpt_path_s1 ./ckpt_universal/udbm_l_s1/model-600.pt
```

If `--ckpt_path_s1` is omitted, the script uses:

```text
./ckpt_universal/udbm_<variant>_s1/model-600.pt
```

Stage 2 checkpoints are written to:

```text
./ckpt_universal/udbm_l_s2/
```

Resume Stage 2:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --multi_gpu \
  --num_processes 2 \
  train_s2.py \
  --variant L \
  --ckpt_path_s1 ./ckpt_universal/udbm_l_s1/model-600.pt \
  --resume_milestone 300
```

## Batch Composition

The training batch is controlled inside the trainer by five task-specific DataLoaders. `--train_batch_size` is kept only for compatibility with the original trainer signature.

Task order:

```text
fog, light_only, rain, snow, blur
```

Default task ratio:

```text
fog : light_only : rain : snow : blur = 8 : 2 : 4 : 4 : 2
```

| `gradient_accumulate_every` | Task batch sizes | Per-step sum | Effective sum |
|---:|---|---:|---:|
| `1` | `16,4,8,8,4` | `40` | `40` |
| `2` | `8,2,4,4,2` | `20` | `40` |

Override the task batch sizes with:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 \
  train_s2.py \
  --variant L \
  --gradient_accumulate_every 2 \
  --task_batch_sizes 8,2,4,4,2 \
  --ckpt_path_s1 ./ckpt_universal/udbm_l_s1/model-600.pt
```

## Evaluation

Evaluate a checkpoint trained under `ckpt_universal`:

```bash
CUDA_VISIBLE_DEVICES=0 VARIANT=L sh test_checkpoint.sh
```

Equivalent Python command:

```bash
CUDA_VISIBLE_DEVICES=0 python test_s2.py \
  --variant L \
  --dataroot ./datasets/all_in_one \
  --ckpt_path_s1 ./ckpt_universal/udbm_l_s1/model-600.pt \
  --results_folder ./ckpt_universal/udbm_l_s2 \
  --milestone 600 \
  --result_dir ./result
```

Evaluate the released UDBM-L weights:

```bash
CUDA_VISIBLE_DEVICES=0 VARIANT=L sh test.sh
```

Equivalent Python command:

```bash
CUDA_VISIBLE_DEVICES=0 python test_s2.py \
  --variant L \
  --dataroot ./datasets/all_in_one \
  --ckpt_path_s1 ./pretrained/udbm_l/stage1.pt \
  --results_folder ./pretrained/udbm_l \
  --milestone 600 \
  --result_dir ./result
```

For UDBM-S or UDBM-M, replace `L` and `udbm_l` with the matching variant.

Run selected tasks:

```bash
CUDA_VISIBLE_DEVICES=0 python test_s2.py \
  --variant L \
  --tasks rain snow
```

Run real-world and CD11 branches:

```bash
CUDA_VISIBLE_DEVICES=0 sh test_real.sh
CUDA_VISIBLE_DEVICES=0 sh test_cd11.sh
```

Task expansion:

| Input task | Expanded test sets |
|---|---|
| `rain` | `rain1`, `rain2`, `rain3`, `rain4`, `rain5` |
| `snow` | `snow1`, `snow2` |
| `real_dark` | `real_dark_mef`, `real_dark_dice`, `real_dark_npe` |
| `real_blur` | `real_hide`, `real_j`, `real_r` |
| `cd11` | `l`, `h`, `r`, `s`, `lh`, `lr`, `ls`, `hr`, `hs`, `lhr`, `lhs` |

Restored images are saved under:

```text
<result_dir>/fsf_final_small_s1/<task>/
```

The script reports PSNR, SSIM, and NIQE where ground truth is available.

## Quick Reproduction

Use the released UDBM-L checkpoints:

```bash
CUDA_VISIBLE_DEVICES=0 VARIANT=L sh test.sh
```

Train UDBM-L from scratch:

```bash
CUDA_VISIBLE_DEVICES=0,1 VARIANT=L DATAROOT=./datasets/all_in_one sh train.sh
```

## Implementation Notes

- Stage 2 loads Stage 1 during trainer initialization. The Stage 1 checkpoint must contain a top-level `model` key, which is produced by `train_s1.py`.
- Use `accelerate launch` for multi-GPU training. The `--gpu` argument is kept for simple single-process debugging.
- `metrics/niqe.py` loads `metrics/niqe_pris_params.npz` from this repository.

## Citation

```bibtex
@inproceedings{udbm2026,
  title     = {Unifying Heterogeneous Degradations: Uncertainty-Aware Diffusion Bridge Model for All-in-One Image Restoration},
  author    = {Anonymous},
  booktitle = {International Conference on Machine Learning},
  year      = {2026}
}
```
