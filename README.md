# FT-FSOD: CD-FSOD 실험 저장소

이 저장소는 FT-FSOD의 **CD-FSOD 6개 target dataset 재현만** 지원한다.
`codex/acl-pre-enhancer-unpooled-raw-mean-etf`는
`acl-nearest-etf-loss`의 `2c0892a2f35da0813182720049b0866f553f394e`에서 시작한다.
ACL Progressive Fine-Tuning, augmentation, optimizer, scheduler, detection loss,
validation 및 checkpoint 설정을 유지하며, standard serial Grounding DINO decoder의
마지막 layer로 추론한다. HED는 사용하지 않는다. 기존 `_HED.py` 파일명과 등록된
모델 이름은 config/checkpoint 호환성을 위해 유지한다.

## Offline Qwen class descriptions

Qwen은 support GT bbox crop들을 class별로 함께 보고 공통 시각 설명 한 개를 생성한다.
검증/test 이미지는 사용하지 않는다. JSON만 detector에 전달하며 Qwen 모델은 detector,
optimizer, checkpoint에 포함되지 않는다. 구현은 `codex/qwen3-vl-visual-descriptions`의
`26a352b`에서 crop 생성기와 BERT prototype 경로를 이식했다.

먼저 학습 환경과 별도로 preprocessing 환경을 준비한다. 다음은 저장소의 CUDA 12.8
PyTorch 버전과 Qwen3-VL을 지원하는 Transformers 버전을 사용하는 예시다.

```bash
conda create -n qwen-offline python=3.10 -y
conda activate qwen-offline
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install transformers==4.57.1 pillow==11.3.0

export CDFSOD_PATH=/path/to/datasets
DATASET=NEU-DET
for SHOT in 1 5 10; do
  python tools/generate_instance_captions.py \
    --dataset-root "${CDFSOD_PATH}/${DATASET}" \
    --ann-file "annotations/${SHOT}_shot.json" \
    --img-prefix train \
    --output "annotations/${SHOT}_shot_captions.json" \
    --model-name Qwen/Qwen3-VL-8B-Instruct \
    --device cuda --batch-size 1
done
```

`--batch-size`는 class 수다. class의 모든 crop을 한 요청에 넣으므로 필요한 메모리는
crop 수와 해상도에 따라 달라진다. 이 예시는 실제 Qwen/GPU 실행 검증을 포함하지 않는다.
`DATASET`은 `ArTaxOr`, `DIOR`, `FISH`, `NEU-DET`, `UODD`, `clipart1k` 중 선택한다.
각 shot의 support annotation을 별도로 처리해야 한다.

JSON은 아래 형식이며 생성기는 `ann_ids`, `image_ids`, `bboxes`, `file_names`도 보존한다.
설정의 모든 class에 정확히 하나의 비어 있지 않은 설명이 필요하다. 누락, 중복, 알 수 없는
class, 잘못된 bbox는 오류로 처리한다. JSON category ID가 아닌 `category_name`으로 매칭하며
class-name token과 ETF prototype 순서는 config의 `class_names`를 따른다.

```json
{"captions": [{"category_id": 1, "category_name": "crazing", "caption": "Thin branching lines across a rough surface."}]}
```

위 JSON은 생성기의 단일 entry 예시다. 학습에는 해당 데이터셋의 모든 class entry가 필요하다.
Base detector가 읽는 파일은 아래처럼 **class name → description** 객체다.
이미 이 형식의 offline Qwen 파일이 있으면 그대로 사용한다.

```json
{"crazing": "Thin branching lines across a rough surface."}
```

위 생성기를 사용했다면 metadata 파일을 보존하고 detector용 파일로 변환한다.
아래 예시는 앞의 생성 명령과 같은 `CDFSOD_PATH`, `DATASET`, `SHOT`를 사용한다.

```bash
for SHOT in 1 5 10; do
  CAPTION_FILE="${CDFSOD_PATH}/${DATASET}/annotations/${SHOT}_shot_captions.json"
  mv "$CAPTION_FILE" "${CAPTION_FILE%.json}.metadata.json"
  python - "${CAPTION_FILE%.json}.metadata.json" "$CAPTION_FILE" <<'PY'
import json
import sys
from pathlib import Path
entries = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))['captions']
captions = {entry['category_name']: entry['caption'] for entry in entries}
assert len(captions) == len(entries), 'Duplicate class descriptions'
Path(sys.argv[2]).write_text(json.dumps(captions, ensure_ascii=False, indent=2), encoding='utf-8')
PY
done
```

18개 finetune config는 각자 `annotations/{shot}_shot_captions.json`을 지정한다.

```bash
conda activate ft-fsod
python tools/train.py configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_NEU-DET_1shot.py
# 다른 JSON 사용 시 (평가 시에도 동일한 override 사용)
python tools/train.py configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_NEU-DET_5shot.py \
  --cfg-options model.support_caption_file=/absolute/path/5_shot_captions.json
```

BERT는 정리된 class name과 설명을 `class_name: description.`으로 class마다 독립적으로
인코딩한다. 문장 전체 attention을 사용한 뒤 class-name subword의 마지막 hidden state만
선택하고, 각 token에 `text_feat_map`을 적용한다. Description, special, padding token은
projection/Feature Enhancer에 전달하지 않는다. 설명은 BERT 문맥을 통해 class token에 영향을 준다.

선택된 token은 class 순서와 class 내부 subword 순서대로 연결한다. Text self-attention은
같은 class의 subword 사이에만 허용하고, position ID는 원래 BERT prompt 내 위치를 유지한다.
모든 enhancer layer, query selection, decoder/head는 평균 전 token-level representation을 사용한다.
학습 positive map과 추론 label-to-token map은 각 class의 모든 subword를 가리킨다.

학습·평가 feature cache는 사용하지 않으며 텍스트/tokenization만 재사용한다. 긴 설명은
기존 BERT token 제한에 따라 잘리고 class name이 잘리면 오류가 발생한다. 선택된 전체
class-name subword 수가 head의 `max_text_len`을 넘으면 오류가 발생한다.

기존 class-name 텍스트 경로로 비교하려면
`--cfg-options model.use_class_name_token_prototypes=False`를 사용한다.
이 옵션과 `build_prototype_*` 메서드 이름은 호환성을 위해 유지하지만, 활성화된 detection
경로에서는 pooling하지 않는다. DN query, detection loss, Progressive Fine-Tuning hook과
serial decoder는 base commit 그대로다.

가벼운 CPU sanity check (PyTorch와 Pillow 필요, MMDetection 확장/모델 다운로드 불필요):

```bash
python -m unittest discover -s tests -p test_qwen_offline_sanity.py -v
```

실제 detector 메서드에 작은 BERT 대역을 연결해 subword 선택/gradient/shape/positive map을 검사하고,
mock Qwen으로 crop·JSON 경로를 검사한다. 기준 ACL commit과 핵심 메서드 및 hook/decoder/head,
18개 config의 텍스트 옵션 외 설정이 동일한지도 확인한다. 전체 detector/GPU 학습 검증은 별도다.

## Nearest Simplex ETF auxiliary loss

`grounding_dino_HED.py`에서 class-name token을 BERT 직후 선택한다.
Reference A (`4118d18`)의 unpooled token mapping과 Reference B (`1ae964e`)의
raw mean ETF helper를 사용하되, 모든 enhancer layer에는 class-name token만 전달한다.

```text
offline Qwen → "class name: description" → BERT
  → class-name subword 선택 → text_feat_map → [B, N_token, 256]
    ├─ ETF: class별 raw mean → [B, C, 256] → Nearest Simplex ETF
    └─ Detection: token 그대로 → Feature Enhancer → query selection
                  → serial decoder → 기존 detection head/loss
```

Class별 mean은 ETF branch에서만 사용하며 token별 L2 정규화를 먼저 하지 않는다.
ETF는 enhancer 호출 전에 계산하며, detection tensor를 교체하거나 수정하지 않는다.
각 이미지의 전체 C개 class prototype에 적용하며 GT에 등장한 class만 고르지 않는다.
Batch에 확장된 같은 support token을 사용해도 loss는 batch 평균이므로 batch 크기로 증폭되지 않는다.

### 정확한 nearest-ETF 해

공식 [ETF_distance.py](https://github.com/evanmarkou/Guiding-Neural-Collapse/blob/main/nc/ETF_distance.py)의
목적함수와 [ddn_modules.py](https://github.com/evanmarkou/Guiding-Neural-Collapse/blob/main/nc/models/ddn_modules.py)의
proximal 항을 제외한 목적함수는 `||Y - P H / sqrt(C-1)||_F²`이며,
`H = I - 11ᵀ/C`, `PᵀP = I`이다. 이 제약 아래 target norm이 항상 1이므로
교차항을 최대화하는 orthogonal Procrustes 문제로 정확히 풀 수 있다.

샘플별 prototype `X`에 대해 class 평균을 빼고 전체 행렬을 정규화한다.
`Z = (X - mean_classes(X)) / max(||X - mean_classes(X)||_F, 1e-6)`.
개별 class vector는 L2 정규화하지 않는다.

Helmert basis `Q`를 `QᵀQ = I`, `QQᵀ = H`가 되도록 구성하고,
`ZᵀQ = UΣVᵀ`를 thin SVD로 분해하면 nearest target은
`T = Q V Uᵀ / sqrt(C-1)`이다. 따라서 `TTᵀ = H/(C-1)`이고 `||T||_F = 1`이다.
이 식은 `D >= C`에서 공식 Stiefel 표현과 같은 target 집합을 가지며,
redundant null direction을 제거하여 simplex의 최소 차원 `D = C-1`도 지원한다.
고정된 canonical ETF 방향을 target으로 사용하지 않는다.

```python
loss_nearest_etf = nearest_etf_loss_weight * (
    (normalized_prototypes - detached_nearest_target).square()
    .sum(dim=(-2, -1)).mean()
)
```

원소별 평균이 아닌 샘플별 squared Frobenius distance의 batch 평균이다.
SVD와 target 구성 전체는 `torch.no_grad()` 안에서 실행한다. Gradient는
중심화·정규화와 raw mean을 거쳐 BERT 및 `text_feat_map`으로 전달된다.
ETF 자체는 Feature Enhancer, visual backbone, decoder/head에 gradient를 전달하지 않으며,
기존 detection loss가 해당 경로를 학습한다. Target solve를 통한 gradient는 없다. Pymanopt, proximal 항, DDN,
implicit differentiation, 이전 step의 target cache는 도입하지 않는다.

FP16/BF16 입력은 autocast를 끄고 FP32로 계산하며 FP64 입력은 유지한다.
`C < 2` 또는 `D < C-1`은 오류다. Rank 부족 시 SVD가 유효한 최적해 하나를
선택한다. 모든 prototype이 같으면 정규화 분모를 epsilon으로 clamp하고
`Z=0`, loss=1인 유한한 확장으로 처리한다. `C=2`의 서로 다른 두 prototype은
중심화·정규화 후 이미 simplex이므로 보조 loss가 0이다.

### 실험 설정과 검증

Detector 기본값과 18개 finetune config는 모두 `nearest_etf_loss_weight=1.0`이다. 가중치가 0이거나 prototype 모드가 꺼져 있으면
ETF 계산과 loss key를 생략한다. 추론은 가중치와 무관하게 ETF를 계산하지 않는다.
학습 파라미터나 checkpoint state는 추가되지 않는다. 기존 PFT의 parameter group과
`lr=0` 동결 및 전환 규칙을 그대로 사용한다.

```bash
# 가중치 조정
python tools/train.py configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_NEU-DET_1shot.py \
  --cfg-options model.nearest_etf_loss_weight=0.05
# ETF만 비활성화하고 unpooled class-name detection 경로 유지
python tools/train.py configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_NEU-DET_1shot.py \
  --cfg-options model.nearest_etf_loss_weight=0.0

# CPU 수치/회귀/단일-rank Gloo DDP 검사 (PyTorch, Pillow)
python -m unittest discover -s tests -p 'test_*.py' -v
```

테스트는 ETF Gram 조건, 해석적 최솟값, 회전된 ETF, translation/scale/rotation
불변성, batch 평균, detached target, gradient, 저정밀·퇴화 입력을 검사한다.
실제 detector 메서드와 enhancer loop/fusion에 가벼운 BERT/attention 대역을 연결해
projection 입력의 subword 선택, enhancer 이전 raw mean, ETF 단독 gradient의 분리,
ETF 활성화 전후 detection 출력 보존을 확인한다. 실제 PFT hook의 LR 동결/전환,
serial decoder/final-layer 추론, checkpoint 사용/미사용 단일-rank CPU Gloo DDP도 검사한다.
CUDA autocast 검사는 CUDA가 있을 때만 실행되며, 이 경량 검증은 전체 MMDetection
GPU 학습이나 정확도 실험을 대신하지 않는다.

## 지원 환경

| 항목 | 고정값 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 물리 GPU 0 한 장 |
| Python | 3.10 |
| PyTorch | 2.7.1 + CUDA 12.8 wheel |
| torchvision | 0.22.1 + CUDA 12.8 wheel |
| CUDA build toolkit | 12.8 (conda 환경 내부) |
| MMEngine | 0.10.7 |
| MMCV | 2.2.0 소스 빌드, `sm_120` |
| FairScale | 0.4.13 |

`nvidia-smi`의 `CUDA Version: 13.2`는 드라이버가 지원하는 최대 버전이다. 이 저장소는
RTX 5090 코드가 포함된 공식 PyTorch `cu128` wheel과 CUDA 12.8로 빌드한 MMCV를 사용한다.

## 1. 설치

```bash
git clone https://github.com/kjh0902/CDFSOD.git
cd CDFSOD

conda env create -f environment.yml
conda activate ft-fsod
bash scripts/install_rtx5090.sh
```

설치 스크립트는 PyTorch의 `sm_120` 지원, GPU 0 CUDA matmul/backward, MMCV CUDA NMS,
FairScale activation checkpointing import를 검사한다. 다시 검사하려면:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/verify_environment.py
```

## 2. 데이터와 사전학습 checkpoint

기본 dataset root는 서버의 다음 경로로 설정되어 있다.

```text
/home/aislab5090/CDFSOD/junhyung/datasets
```

필요한 구조는 다음과 같다.

```text
/home/aislab5090/CDFSOD/junhyung/datasets/
├── ArTaxOr/
├── clipart1k/
├── DIOR/
├── FISH/
├── NEU-DET/
└── UODD/
    ├── annotations/{1_shot,5_shot,10_shot,test}.json
    ├── train/
    └── test/
```

다른 위치를 사용해야 할 때만 환경 변수로 덮어쓴다.

```bash
export CDFSOD_PATH=/absolute/path/to/datasets
```

Swin-B 사전학습 checkpoint를 준비한다.

```bash
mkdir -p checkpoints
wget -O checkpoints/grounding_dino_swin-b_pretrain_all-f9818a7c.pth \
  https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-b_pretrain_all/grounding_dino_swin-b_pretrain_all-f9818a7c.pth
```

경로가 다르면 다음 변수를 사용한다.

```bash
export MMGDINOB_PATH=/absolute/path/to/grounding_dino_swin-b_pretrain_all-f9818a7c.pth
```

BERT와 NLTK resource도 최초 실험 전에 cache한다.

```bash
python -c "from transformers import AutoTokenizer, BertModel; AutoTokenizer.from_pretrained('bert-base-uncased'); BertModel.from_pretrained('bert-base-uncased')"
python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng
```

## 3. dataset/shot별 학습 및 평가

`run_cdfsod.sh`는 지정한 config 하나를 학습한 뒤, 기존 validation metric으로 생성된
`best_coco_bbox_mAP_iter_*.pth`를 찾아 즉시 평가한다.

```bash
bash run_cdfsod.sh --dataset NEU-DET --shot 1 --gpu 0
bash run_cdfsod.sh --dataset NEU-DET --shot 5 --gpu 0
bash run_cdfsod.sh --dataset NEU-DET --shot 10 --gpu 0
```

지원 dataset과 이름은 다음과 같다.

```text
ArTaxOr
Clipart1k
DIOR
FISH        # DeepFish도 alias로 사용 가능
NEU-DET
UODD
```

예시:

```bash
bash run_cdfsod.sh --dataset ArTaxOr --shot 1 --gpu 0
bash run_cdfsod.sh --dataset Clipart1k --shot 5 --gpu 0
bash run_cdfsod.sh --dataset DIOR --shot 10 --gpu 0
bash run_cdfsod.sh --dataset DeepFish --shot 1 --gpu 0
bash run_cdfsod.sh --dataset UODD --shot 5 --gpu 0
```

중단된 동일 실험을 이어서 실행할 때는 `--resume`을 추가한다.

```bash
bash run_cdfsod.sh --dataset NEU-DET --shot 1 --gpu 0 --resume
```

동일 서버에서 다른 distributed 작업이 이미 `29500` 포트를 사용한다면 빈 포트를 지정한다.

```bash
bash run_cdfsod.sh --dataset DIOR --shot 5 --gpu 0 --port 29501
```

실행할 config와 결과 경로만 확인하려면 `--dry-run`을 추가한다.

## 4. 결과 구조와 집계

학습 checkpoint, 로그, 평가 결과는 dataset과 shot별로 분리된다.

```text
exp_cdfsod_results/
├── ArTaxOr/{1shot,5shot,10shot}/
├── Clipart1k/{1shot,5shot,10shot}/
├── DIOR/{1shot,5shot,10shot}/
├── FISH/{1shot,5shot,10shot}/
├── NEU-DET/{1shot,5shot,10shot}/
└── UODD/{1shot,5shot,10shot}/
```

각 실험 폴더에는 기존 naming convention의 best checkpoint와 `results.pkl`이 저장된다.
완료된 실험의 mAP를 모아 보려면:

```bash
python analyze_results_cdfsod.py
```

## 문제 해결

- `please install fairscale`: `python -m pip install -r requirements.txt`를 실행한다.
- `Weights only load failed` 또는 `HistoryBuffer was not an allowed global`: 최신
  `tools/train.py`와 `tools/test.py`를 사용한다. 이 호환 처리는 출처를 신뢰하는 checkpoint에만
  사용해야 한다.
- `No module named mmcv._ext`: `scripts/install_rtx5090.sh`로 MMCV CUDA ops를 다시 빌드한다.
- `no kernel image is available`: CUDA 12.4 이하 wheel이 섞인 환경일 수 있다. conda 환경을
  새로 만들고 설치 스크립트를 다시 실행한다.
- MMCV 빌드 OOM: `MAX_JOBS=1 bash scripts/install_rtx5090.sh`로 재실행한다.
- `Address already in use`: `--port`에 사용 중이지 않은 값을 지정한다.
- CUDA OOM: config를 변경하기 전에 `nvidia-smi`로 GPU 0의 다른 process를 확인한다.
- 마지막 validation 직후 `munmap_chunk(): invalid pointer` / SIGABRT:
  [종료 처리 수정과 재현·진단 절차](UODD_SHUTDOWN.md)를 참고한다.
  최신 train/test 진입점은 분산 process group을 명시적으로 해제한다.
  해당 네이티브 오류의 원인 확정에는 실행 서버의 재현 결과 또는 backtrace가 필요하다.

## 원본 및 인용

- 원본 코드: <https://github.com/Intellindust-AI-Lab/FT-FSOD>
- 논문: *A Closer Look at Cross-Domain Few-Shot Object Detection: Fine-Tuning Matters and Parallel Decoder Helps* (CVPR 2026)
- 라이선스: Apache-2.0 (`LICENSE`)

```bibtex
@inproceedings{yu2026acloser,
  title={A Closer Look at Cross-Domain Few-Shot Object Detection: Fine-Tuning Matters and Parallel Decoder Helps},
  author={Yu, Xuanlong and Sha, Youyang and Liu, Longfei and Shen, Xi and Yang, Di},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
