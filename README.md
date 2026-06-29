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

Recommended on the remote server. If the old `cdfsod` environment already has a
broken `mmcv._ext`, remove it and start from a clean environment.

```bash
conda deactivate || true
conda env remove -n cdfsod -y || true
conda create -n cdfsod python=3.10 -y
conda activate cdfsod

conda install -c nvidia cuda-toolkit=12.8 -y
conda install -c conda-forge "gcc_linux-64=13.*" "gxx_linux-64=13.*" -y

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

python -m pip install -U pip
python -m pip install "setuptools==80.9.0" "wheel==0.45.1"
pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt

pip uninstall -y mmcv mmcv-full || true
rm -rf "$CONDA_PREFIX/lib/python3.10/site-packages/mmcv"
rm -rf "$CONDA_PREFIX/lib/python3.10/site-packages/mmcv-"*.dist-info

MMCV_WITH_OPS=1 FORCE_CUDA=1 MAX_JOBS=8 pip install -v \
  --no-build-isolation \
  --no-cache-dir \
  --no-binary mmcv \
  "mmcv>=2.1.0,<2.2.0"

pip install -e ./mmdetection --no-build-isolation
```

Keep the `CUDA_HOME`, `CC`, and `CXX` exports in the same shell while building
MMCV. After installation, verify the CUDA ops, not just `mmcv.__version__`:

```bash
python - <<'PY'
import torch
import mmcv
from mmcv.ops import roi_align

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
print("mmcv:", mmcv.__version__)
print("mmcv.ops roi_align:", roi_align)
PY
```

If `from mmcv.ops import roi_align` fails, do not train yet. Remove `mmcv` again
and rebuild it in the same active conda environment after confirming:

```bash
which nvcc
nvcc --version
ls -l "$CC" "$CXX"
"$CXX" --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

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
