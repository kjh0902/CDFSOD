# CDFSOD: Full LoRA + FE ETF

`acl-hed-etf-mu-orthogonality`의 Grounding DINO Swin-B / serial ACL 구조를
기반으로 한 `full-lora-fe-etf` 실험 브랜치다. 기존 사전학습 파라미터를 전부
동결하고 LoRA adapter만 첫 epoch부터 100 epoch 동안 학습한다.
Feature Enhancer 최종 `memory_text`에 기존 ETF loss를 가중치 **5.0**으로 적용한다.
BERT ETF와 BERT/FE μ-Orthogonality 가중치는 모두 0이다.

6개 CD-FSOD dataset의 1/5/10-shot annotation, 데이터 root, augmentation,
train/test pipeline, detection loss, query selection, denoising, evaluation 및
seed=42 / deterministic 설정은 시작 브랜치와 동일하다.

## 지원 환경

| 항목 | 고정값 |
|---|---|
| GPU | NVIDIA RTX 3090 24GB × 2 중 물리 GPU 0 또는 1 한 장 |
| OS / Driver | Ubuntu/Linux / 560.35.03 |
| Python | 3.10 |
| PyTorch | 2.7.1 + CUDA 12.6 wheel |
| torchvision | 0.22.1 + CUDA 12.6 wheel |
| CUDA build toolkit | 12.6 (conda 환경 내부) |
| MMEngine | 0.10.7 |
| MMCV | 2.2.0 소스 빌드, `sm_86` |
| FairScale | 0.4.13 |

`nvidia-smi`의 `CUDA Version: 12.6`은 드라이버가 지원하는 최대 버전이며, 로컬 nvcc 설치를
의미하지 않는다. [공식 PyTorch 2.7.1 설치표](https://pytorch.org/get-started/previous-versions/)의
`cu126` wheel과 같은 CUDA 12.6 toolkit으로 MMCV를 빌드한다. 제공된 드라이버는
[CUDA 12.6 GA의 Linux 최소 버전 560.28.03](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-toolkit-release-notes/index.html)을 충족한다.

## 1. 설치

```bash
git clone --branch full-lora-fe-etf https://github.com/kjh0902/CDFSOD.git
cd CDFSOD

conda env create -f environment.yml
conda activate ft-fsod
bash scripts/install_rtx3090.sh
```

설치 스크립트는 PyTorch의 `sm_86` 지원, 선택한 GPU의 CUDA matmul/backward, MMCV CUDA NMS,
FairScale activation checkpointing import를 검사한다. 설치 검사는 기본 GPU 0을 사용하며,
`GPU_ID=1 bash scripts/install_rtx3090.sh`로 GPU 1을 선택할 수 있다. 다시 검사하려면:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python tools/verify_environment.py
```

## 2. 데이터와 사전학습 checkpoint

기본 dataset root는 서버의 다음 경로로 설정되어 있다.

```text
/home/aislab/Desktop/junhyung/cdfsod/datasets
```

필요한 구조는 다음과 같다.

```text
/home/aislab/Desktop/junhyung/cdfsod/datasets/
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
export CDFSOD_PATH=/home/aislab/Desktop/junhyung/cdfsod/datasets
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
bash run_cdfsod.sh --dataset NEU-DET --shot 1 --gpu 0 \
  --work-dir work_dirs/experiment_A_NEU-DET_1shot --resume
```

동일 서버에서 다른 distributed 작업이 이미 `29500` 포트를 사용한다면 빈 포트를 지정한다.

```bash
bash run_cdfsod.sh --dataset DIOR --shot 5 --gpu 0 --port 29501
```

실행할 config와 결과 경로만 확인하려면 `--dry-run`을 추가한다.

### Full LoRA + FE ETF 고정 설정

위 명령으로 별도의 weight 옵션 없이 실험 설정이 적용된다. 기존 checkout에서는:

```bash
git fetch origin
git switch full-lora-fe-etf
git pull --ff-only origin full-lora-fe-etf
bash run_cdfsod.sh --dataset NEU-DET --shot 1 --gpu 0
bash run_cdfsod.sh --dataset UODD --shot 5 --gpu 1
```

각 명령은 학습 후 가장 높은 `coco/bbox_mAP` checkpoint를 평가한다.
두 명령은 각 GPU에서 별도로 실행할 수 있다. 모든 18개 dataset/shot config가
`configs_cdfsod/grounding_dino_swin-b_full_lora_fe_etf.py`를 상속한다.

| 항목 | 설정 |
|---|---|
| Architecture / pretrained | 기존 Grounding DINO Swin-B checkpoint |
| Rank / alpha / scaling | 32 / 64 / 2 |
| LoRA dropout | 0.05, adapter 입력에만 적용 |
| Optimizer / weight decay | AdamW / 0.01 |
| Batch size / epochs | 4 / 100 |
| Gradient clipping | max_norm=0.1 |
| Seed | 42, 기존 deterministic 옵션 유지 |
| Loss | 기존 detection loss + 5.0 × FE ETF |
| Strategy | Single-stage, 모든 adapter가 첫 epoch부터 학습 |

| Module | Learning rate |
|---|---:|
| Swin-B backbone | 1e-4 |
| BERT (`language_model`) | 1e-4 |
| Feature Enhancer (`encoder`) | 2e-4 |
| Decoder | 2e-4 |
| Detection Head | 2e-4 |
| Text Projection (`text_feat_map`) | 2e-4 |
| Neck | 2e-4 |

모든 Linear에 LoRA를 적용한다. Swin의 QKV, attention/FFN, BERT의 attention/FFN,
FE의 visual/text/fusion projection, Decoder와 bbox regression branch를 포함한다.
PyTorch MultiheadAttention의 결합형 QKV와 output projection도 adapter를 거치며,
각 Q/K/V projection에 rank 32를 적용한다. Neck은 Conv 기반이므로 모든 Conv2d에
공간 convolution A와 1×1 convolution B의 LoRA를 적용한다. Swin patch embedding의
Conv2d도 포함한다. Encoder proposal용 `memory_trans_fc`는 Detection Head LR을 사용한다.

원래 weight, bias, LayerNorm/GroupNorm, embedding, contrastive head bias/scale,
query/level embedding은 모두 동결한다. 파라미터가 없는 contrastive classification
연산에는 레이어를 추가하지 않는다. Adapter는 A=Kaiming / B=0으로 시작하므로
초기 모델 출력이 유지된다. 기존 checkpoint key를 유지하며, 저장된 학습 checkpoint는
원본 weight와 adapter를 함께 포함해 기존 test/resume 명령으로 불러올 수 있다.
새로운 adapter에 대한 missing-key 로그는 최초 사전학습 checkpoint 로딩 시 예상된다.
이전 Full FT 실험의 work directory를 `--resume`으로 재사용하지 않는다.

`BBoxHeadFirstHook6`와 stage 전환은 사용하지 않는다. `FullLoRAHook`는 학습 시작과
매 epoch마다 원본 동결 상태, adapter optimizer 구성과 LR 비율을 검사한다.
ReduceOnPlateau는 `coco/bbox_mAP`, greater, factor=0.5, patience=8,
threshold=1e-4 (rel), cooldown=1, min_value=1e-6을 사용한다. 공통 absolute floor에서
1:2 비율이 사라지는 것을 막기 위해 느린 그룹의 최저 LR은 1e-6, 빠른 그룹은 2e-6으로
적용한다. 100 epoch 동안 반복 감쇠 및 optimizer/scheduler 재시작 시 비율을 검증했다.

FE ETF의 수식, class-name WordPiece token의 raw mean pooling, class centering,
전체 Frobenius normalization, SVD target 계산 및 target detach는 기존 구현 그대로다.
GT에 없는 클래스와 empty-GT 이미지도 기존대로 포함한다. BERT ETF,
BERT Orthogonality, FE Orthogonality는 모두 계산하지 않는다. FISH의 단일 클래스는
기존 ETF 정의에 따라 보조 loss가 0이다. Inference에서는 ETF를 계산하지 않는다.

기존 loss weight CLI 옵션은 호환성을 위해 남아 있으며 이 실험에서는 생략한다.
Native MMEngine CLI도 같은 config를 사용할 수 있다:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_NEU-DET_1shot.py \
  --work-dir work_dirs/full_lora_fe_etf_NEU-DET_1shot
```

CPU 검증은 아래 명령으로 실행한다. LoRA 수치 및 gradient, 원본 weight 동결,
checkpoint/optimizer 복원, 실제 MMEngine scheduler의 100 epoch LR 감쇠와 재시작,
18개 config의 데이터/모델 보존, FE ETF와 기존 ACL regression을 검사한다.
Bash 검사에는 Bash가 필요하다. 실제 CUDA 학습과 데이터셋 성능 평가는 별도다.

```bash
python -m unittest discover -s tests -v
```

구현 시 로컬 검증 결과는 67개 통과 / CUDA 전용 1개 skip이다. 별도 CPU smoke에서
실제 Swin-B (2/2/18/2), BERT 12층, FE/Decoder/Head 전체를 구성하고 초기화했다.
학습 가능한 LoRA는 18,061,696개 scalar / 786개 tensor이며, 기존 1,076개 state-dict
key는 유지된다. 작은 합성 이미지와 query 8개, 무작위 BERT weight, MMCV의 CPU
deformable-attention 경로를 사용해 detection + FE ETF forward/backward 및 AdamW
step을 실행했고 모든 adapter tensor에 gradient가 전달됐다. 실제 pretrained
checkpoint 파일을 사용한 CUDA 학습 및 데이터셋 mAP는 이 검증에 포함하지 않았다.

## 4. 결과 구조와 집계

학습 checkpoint, 로그, 평가 결과는 `--work-dir`에 저장한다. 옵션을 생략하면 dataset/shot
아래에 timestamp와 process ID를 포함한 새 run directory를 생성해 실험 간 덮어쓰기를 피한다.
`--resume`은 기존 실험을 가리키는 `--work-dir`과 함께 사용한다.

```text
exp_cdfsod_results/
├── ArTaxOr/{1shot,5shot,10shot}/
├── Clipart1k/{1shot,5shot,10shot}/
├── DIOR/{1shot,5shot,10shot}/
├── FISH/{1shot,5shot,10shot}/
├── NEU-DET/{1shot,5shot,10shot}/
└── UODD/{1shot,5shot,10shot}/
```

위 dataset/shot 폴더 아래의 개별 run directory 또는 지정한 `--work-dir`에는
기존 naming convention의 best checkpoint와 `results.pkl`이 저장된다.
완료된 실험의 mAP를 모아 보려면:

```bash
python analyze_results_cdfsod.py
```

## 문제 해결

- `please install fairscale`: `python -m pip install -r requirements.txt`를 실행한다.
- `Weights only load failed` 또는 `HistoryBuffer was not an allowed global`: 최신
  `tools/train.py`와 `tools/test.py`를 사용한다. 이 호환 처리는 출처를 신뢰하는 checkpoint에만
  사용해야 한다.
- `No module named mmcv._ext`: `scripts/install_rtx3090.sh`로 MMCV CUDA ops를 다시 빌드한다.
- `no kernel image is available`: `sm_86`을 포함한 cu126 PyTorch와 현재 toolkit으로
  빌드한 MMCV인지 확인한 뒤 설치 스크립트를 다시 실행한다.
- MMCV 빌드 OOM: `MAX_JOBS=1 bash scripts/install_rtx3090.sh`로 재실행한다.
- `Address already in use`: `--port`에 사용 중이지 않은 값을 지정한다.
- CUDA OOM: config를 변경하기 전에 `nvidia-smi`로 선택한 물리 GPU의 다른 process를 확인한다.

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
