# CDFSOD GroundingDINO Baseline

This repository is a compact GroundingDINO baseline for CDFSOD few-shot
detection. The extra Domain-RAG, generation, inpainting, and broad MMDetection
experiment files are intentionally kept out of the main workflow.

## Server Paths

Clone this repository on the remote server here:

```bash
cd /home/aislab5090/CDFSOD/junhyung/grounding_dino_idea
git clone https://github.com/kjh0902/CDFSOD.git
cd CDFSOD
```

Datasets are expected here:

```text
/home/aislab5090/CDFSOD/junhyung/datasets/
  clipart1k/
  NEU-DET/
  UODD/
```

Each dataset should follow this COCO-style layout:

```text
DATASET_NAME/
  annotations/
    train.json
    test.json
    1_shot.json
    5_shot.json
    10_shot.json
  train/
  test/
```

For example:

```text
/home/aislab5090/CDFSOD/junhyung/datasets/
  NEU-DET/
    annotations/
      train.json
      test.json
      1_shot.json
    train/
    test/
```

## Environment

Use a clean environment. Keep the compiler/CUDA exports in the same shell while
building MMCV.

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

Verify the install:

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

## Train

```bash
bash mmdetection/run_all_training.sh NEU-DET 1
```

The script arguments are:

```bash
bash mmdetection/run_all_training.sh DATASET SHOT GPU_COUNT
```

Examples:

```bash
bash mmdetection/run_all_training.sh NEU-DET 1 1
bash mmdetection/run_all_training.sh clipart1k 5 4
bash mmdetection/run_all_training.sh UODD 10 4
```

The default data root is:

```bash
/home/aislab5090/CDFSOD/junhyung/datasets
```

Override it only if needed:

```bash
CDFSOD_DATA_ROOT=/other/datasets bash mmdetection/run_all_training.sh NEU-DET 1 1
```
