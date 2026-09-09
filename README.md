# FT-FSOD: CD-FSOD 실험 저장소

이 저장소는 FT-FSOD의 **CD-FSOD 6개 target dataset 재현만** 지원한다. 논문의 HED,
Progressive Fine-Tuning, augmentation, optimizer, scheduler, validation metric 및 checkpoint
설정은 원본 그대로 유지한다. `grounding_dino_acl_qwen` 브랜치는 offline Qwen 설명을
BERT class-name prototype으로 바꾸는 텍스트 입력 경로를 추가한다.

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
prototype 순서는 config의 `class_names`를 따른다.

```json
{"captions": [{"category_id": 1, "category_name": "crazing", "caption": "Thin branching lines across a rough surface."}]}
```

위 JSON은 단일 entry 예시다. 학습에는 해당 데이터셋의 모든 class entry가 필요하다.
18개 finetune config는 각자 `annotations/{shot}_shot_captions.json`을 지정한다.

```bash
conda activate ft-fsod
python tools/train.py configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_NEU-DET_1shot.py
# 다른 JSON 사용 시 (평가 시에도 동일한 override 사용)
python tools/train.py configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_NEU-DET_5shot.py \
  --cfg-options model.support_caption_file=/absolute/path/5_shot_captions.json
```

BERT는 정리된 class name과 설명을 `class_name: description.`으로 class마다 독립적으로
인코딩한다. 문장 전체 attention을 사용하고 class-name subword의 마지막 hidden state만
평균한 뒤 `text_feat_map`을 적용한다. 설명 토큰 자체는 detection token으로 전달하지 않는다.
BERT/projection gradient와 기존 ACL 학습 단계별 LR 정책을 유지한다. 학습·평가 feature
cache는 사용하지 않으며 텍스트/tokenization만 재사용한다. 긴 설명은 기존 BERT token 제한에
따라 잘리고 class name이 잘리면 오류가 발생한다.

기존 class-name 텍스트 경로로 비교하려면
`--cfg-options model.use_class_name_token_prototypes=False`를 사용한다.
HED decoder, DN query, detection loss, Progressive Fine-Tuning hook은 기준 ACL 코드 그대로다.

가벼운 CPU sanity check (PyTorch와 Pillow 필요, MMDetection 확장/모델 다운로드 불필요):

```bash
python -m unittest discover -s tests -p test_qwen_offline_sanity.py -v
```

실제 detector 메서드에 작은 BERT 대역을 연결해 pooling/gradient/shape/positive map을 검사하고,
mock Qwen으로 crop·JSON 경로를 검사한다. 기준 ACL commit과 핵심 메서드 및 hook/decoder/head,
18개 config의 텍스트 옵션 외 설정이 동일한지도 확인한다. 전체 detector/GPU 학습 검증은 별도다.

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
