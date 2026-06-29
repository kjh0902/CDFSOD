# GroundingDINO Baseline for NEU-DET

This repository is trimmed for GroundingDINO baseline training on the NEU-DET
dataset. Domain-RAG retrieval, background generation, outpainting, and inpainting
code has been removed.

## Expected Dataset Layout

Place the dataset at the repository root:

```text
datasets/
  NEU-DET/
    annotations/
      train.json
      test.json
    train/
    test/
  split.py
  kshot_split.py
```

The default config reads:

- train images from `datasets/NEU-DET/train/`
- test images from `datasets/NEU-DET/test/`
- train annotations from `datasets/NEU-DET/annotations/train.json`
- test annotations from `datasets/NEU-DET/annotations/test.json`

## Environment

Recommended on the remote server:

```bash
conda create -n cdfsod python=3.10 -y
conda activate cdfsod
pip install -U pip
pip install -r requirements.txt
```

The root `requirements.txt` uses PyTorch CUDA 12.8 wheels, which are suitable
for the RTX 5090 with the installed CUDA 13-capable NVIDIA driver.

## Prepare Splits

If you have one COCO JSON and already split images into `train/` and `test/`,
run:

```bash
python datasets/split.py --input datasets/NEU-DET/annotations/data.json
```

To make few-shot annotation files from `train.json`:

```bash
python datasets/kshot_split.py --shots 1 5 10
```

Then train with the full train split:

```bash
python mmdetection/tools/train.py \
  mmdetection/configs/mm_grounding_dino/CDFSOD/GroundingDINO-few-shot-SwinB.py \
  --amp \
  --device cuda:0
```

For a 1-shot run:

```bash
python mmdetection/tools/train.py \
  mmdetection/configs/mm_grounding_dino/CDFSOD/GroundingDINO-few-shot-SwinB.py \
  --amp \
  --device cuda:0 \
  --cfg-options train_dataloader.dataset.ann_file=annotations/1_shot.json
```
