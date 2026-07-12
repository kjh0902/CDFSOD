# CDFSOD GroundingDINO 실험 코드

이 저장소는 CDFSOD few-shot detection 실험을 위한 Grounding DINO 기반 코드입니다.
현재 기본 실험은 support set의 object caption으로 class-level text prototype을 만들어 사용하는 방식입니다.

## 1. 서버 경로

원격 서버에서는 아래 위치에 clone해서 사용합니다.

```bash
cd /home/aislab5090/CDFSOD/junhyung/grounding_dino_idea
git clone https://github.com/kjh0902/CDFSOD.git
cd CDFSOD
```

데이터셋 기본 경로는 아래와 같습니다.

```text
/home/aislab5090/CDFSOD/junhyung/datasets/
  NEU-DET/
  clipart1k/
  UODD/
```

각 데이터셋은 COCO 형식으로 아래 구조를 사용합니다.

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

## 2. 환경 설치

```bash
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

설치 확인:

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

## 3. Caption 파일 생성

class prototype 방식은 support set의 각 GT bbox crop caption을 먼저 생성해야 합니다.
BLIP에는 전체 이미지가 아니라 각 GT bounding box crop이 입력됩니다.

NEU-DET 1-shot 예시:

```bash
python mmdetection/tools/generate_instance_captions.py \
  --dataset-root /home/aislab5090/CDFSOD/junhyung/datasets/NEU-DET \
  --ann-file annotations/1_shot.json \
  --img-prefix train \
  --output annotations/1_shot_captions.json
```

다른 shot도 같은 규칙입니다.

```bash
python mmdetection/tools/generate_instance_captions.py \
  --dataset-root /home/aislab5090/CDFSOD/junhyung/datasets/NEU-DET \
  --ann-file annotations/5_shot.json \
  --img-prefix train \
  --output annotations/5_shot_captions.json
```

## 4. 기본 실험: class-level text prototype

현재 기본값은 class prototype 방식입니다.

각 support object caption은 아래 prompt가 됩니다.

```text
{class_name}, {instance_caption}, {domain_attribute}.
```

학습 중 매 iteration마다:

1. 전체 support prompt bank를 현재 BERT로 encoding
2. 각 prompt의 `[CLS]` feature 추출
3. 같은 class의 K개 feature 평균
4. class별 text prototype 생성
5. detection loss가 BERT까지 역전파

실행:

```bash
bash mmdetection/run_all_training.sh NEU-DET 1 1
bash mmdetection/run_all_training.sh NEU-DET 5 1
bash mmdetection/run_all_training.sh clipart1k 1 1
bash mmdetection/run_all_training.sh UODD 1 1
```

멀티 GPU:

```bash
bash mmdetection/run_all_training.sh NEU-DET 1 4
```

## 5. Baseline: class name만 사용

prototype을 끄면 기존처럼 class name prompt만 사용합니다.

```bash
CDFSOD_USE_CLASS_PROTOTYPES=0 bash mmdetection/run_all_training.sh NEU-DET 1 1
```

## 6. Debug 모드

첫 iteration에서 prototype 관련 로그를 보고 싶으면 아래처럼 실행합니다.

```bash
CDFSOD_DEBUG_TEXT_PROTOTYPE=1 bash mmdetection/run_all_training.sh NEU-DET 1 1
```

출력되는 항목:

- 클래스별 support prompt 개수
- enriched prompt 예시
- BERT 입력 shape
- prompt별 CLS feature shape
- 최종 prototype shape
- `prototype.requires_grad`
- BERT parameter의 `requires_grad`
- backward 이후 BERT gradient norm
- class 간 attention 차단 여부
- 한 class caption 변경 시 다른 class prototype이 변하지 않는지 확인

## 7. 자주 바꾸는 설정

데이터셋 루트 변경:

```bash
CDFSOD_DATA_ROOT=/other/datasets bash mmdetection/run_all_training.sh NEU-DET 1 1
```

caption 파일 직접 지정:

```bash
CDFSOD_CAPTION_FILE=annotations/1_shot_captions.json \
  bash mmdetection/run_all_training.sh NEU-DET 1 1
```

domain attribute 직접 지정:

```bash
CDFSOD_DOMAIN_ATTRIBUTE="industrial steel surface defect image" \
  bash mmdetection/run_all_training.sh NEU-DET 1 1
```

현재 기본 domain attribute:

```text
NEU-DET: industrial steel surface defect image
clipart1k: clipart style object image
UODD: underwater object detection image
```

## 8. Validation best checkpoint 저장

checkpoint hook은 validation `coco/bbox_mAP` 기준으로 best checkpoint를 저장합니다.

```python
checkpoint=dict(
    interval=1,
    save_best='coco/bbox_mAP',
    rule='greater'
)
```

학습 결과 폴더 예시:

```text
mmdetection/work_dirs/NEU-DET_1shot_class_prototype/
mmdetection/work_dirs/NEU-DET_1shot_class_name/
```

## 9. Best checkpoint test

class prototype 방식:

```bash
cd mmdetection
python tools/test.py \
  configs/mm_grounding_dino/CDFSOD/GroundingDINO-few-shot-SwinB.py \
  work_dirs/NEU-DET_1shot_class_prototype/best_coco_bbox_mAP_epoch_*.pth
```

baseline 방식:

```bash
cd mmdetection
CDFSOD_USE_CLASS_PROTOTYPES=0 python tools/test.py \
  configs/mm_grounding_dino/CDFSOD/GroundingDINO-few-shot-SwinB.py \
  work_dirs/NEU-DET_1shot_class_name/best_coco_bbox_mAP_epoch_*.pth
```

## 10. 실험 흐름 요약

class prototype 실험:

```bash
git pull
conda activate cdfsod

python mmdetection/tools/generate_instance_captions.py \
  --dataset-root /home/aislab5090/CDFSOD/junhyung/datasets/NEU-DET \
  --ann-file annotations/1_shot.json \
  --img-prefix train \
  --output annotations/1_shot_captions.json

bash mmdetection/run_all_training.sh NEU-DET 1 1
```

class name baseline:

```bash
git pull
conda activate cdfsod
CDFSOD_USE_CLASS_PROTOTYPES=0 bash mmdetection/run_all_training.sh NEU-DET 1 1
```
