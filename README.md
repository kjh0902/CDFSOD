# FT-FSOD: CD-FSOD 실험 저장소

이 저장소는 FT-FSOD의 **CD-FSOD 6개 target dataset 재현만** 지원한다. 논문의 HED,
Progressive Fine-Tuning, augmentation, optimizer, scheduler, validation metric 및 checkpoint
설정은 원본 그대로 유지하며, BERT → `text_feat_map` 직후 또는 최종 Feature Enhancer의
`memory_text`에 **Raw Mean Simplex ETF + μ-Orthogonality 보조 loss**를 적용할 수 있다. HED parallel decoder와 기존 detection 경로는 그대로 사용한다.

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
git clone --branch acl-hed-etf-mu-orthogonality https://github.com/kjh0902/CDFSOD.git
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
/home/aislab/Desktop/fewshot/data/CD-FSODdata
```

필요한 구조는 다음과 같다.

```text
/home/aislab/Desktop/fewshot/data/CD-FSODdata/
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
export CDFSOD_PATH=/home/aislab/Desktop/fewshot/data/CD-FSODdata
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

### BERT / FE ETF + μ-Orthogonality 실험

두 위치 모두 이미지별로 **전체 dataset class**의 class-name WordPiece token을
Raw Mean pooling한다. Token이나 개별 prototype을 정규화하지 않는다. BERT 위치는
`text_feat_map` 직후, FE 위치는 최종 enhancer의 `memory_text`이다. 같은 위치의 두
loss는 prototype을 한 번 계산해 같은 tensor를 공유하며, GT에 없는 클래스와 empty-GT
이미지도 포함한다. Prompt에는 모든 클래스가 있어야 하며 잘리거나 누락된 span은 오류로 처리한다.

ETF는 기존 `nearest_etf_loss.py`를 그대로 사용한다. Class centering → 전체 행렬
Frobenius normalization → 매번 SVD로 계산한 Nearest Simplex ETF와 제곱 거리를 구한다.
Target만 detach한다. μ-Orthogonality는 `μ = mean_c p_c`, `r_c = p_c - μ`에 대해
`mean_{i,c} ((μ_i · r_{i,c}) / (||μ_i|| ||r_{i,c}|| + 1e-6))²`를 FP32로 계산한다.
μ와 residual 모두 detach하지 않는다. FISH의 단일 클래스는 residual이 0이므로 두 loss가 0이다.

| Config option | 기본 weight | 학습 로그 |
|---|---:|---|
| `model.bert_etf_loss_weight` | 1.0 | `loss_bert_etf` |
| `model.bert_orth_loss_weight` | 0.0 | `loss_bert_orth` |
| `model.fe_etf_loss_weight` | 0.0 | `loss_fe_etf` |
| `model.fe_orth_loss_weight` | 0.0 | `loss_fe_orth` |

각 weight는 독립적으로 조절하며, 0이면 해당 loss를 계산하거나 로그에 추가하지 않는다.
네 weight가 모두 0이면 보조 loss용 클래스 매핑과 pooling도 생략한다. 기본값은 시작
브랜치의 BERT ETF 단독 실험을 유지한다. Inference에서는 보조 loss를 계산하지 않는다.

아래 weight 1.0은 실행 예시이며 각 loss의 weight는 실험에 맞게 설정한다. 두 명령은
순서대로 실행하고, 각 실행은 한 GPU만 사용한다.

```bash
# Experiment A: BERT ETF + BERT μ-Orthogonality
bash run_cdfsod.sh --dataset NEU-DET --shot 1 --gpu 0 \
  --work-dir work_dirs/experiment_A_NEU-DET_1shot \
  --bert-etf-loss-weight 1.0 --bert-orth-loss-weight 1.0 \
  --fe-etf-loss-weight 0 --fe-orth-loss-weight 0

# Experiment B: FE ETF + FE μ-Orthogonality
bash run_cdfsod.sh --dataset NEU-DET --shot 1 --gpu 1 \
  --work-dir work_dirs/experiment_B_NEU-DET_1shot \
  --bert-etf-loss-weight 0 --bert-orth-loss-weight 0 \
  --fe-etf-loss-weight 1.0 --fe-orth-loss-weight 1.0

# Native MMEngine CLI에서도 동일하게 설정 가능
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_NEU-DET_1shot.py \
  --work-dir work_dirs/experiment_B_NEU-DET_1shot \
  --cfg-options model.bert_etf_loss_weight=0 model.bert_orth_loss_weight=0 \
                model.fe_etf_loss_weight=1.0 model.fe_orth_loss_weight=1.0
```

`--etf-loss-weight`와 `--raw-mean-etf-loss-weight`는 `--bert-etf-loss-weight`의 alias로 유지한다.
기존 config의 `model.raw_mean_etf_loss_weight`는 `model.bert_etf_loss_weight`로 변경한다.
`--gpu 1`이면 `CUDA_VISIBLE_DEVICES=1`로 물리 GPU 1만 노출하고, 내부 worker는
local rank 0 / `cuda:0`을 사용한다. Train과 test 모두 한 노드·한 process로 실행한다.
Batch size, epoch, optimizer, scheduler, Progressive Fine-Tuning hook, Stage 1/2
freeze 정책, Parallel Decoder / DN Query, matching, query selection과 inference는 유지한다.
FE는 Stage 1에서 기존 hook에 따라 lr=0이며, upstream BERT/backbone으로의 gradient는 유지한다.

CPU PyTorch만으로 수치, gradient 및 HED 회귀 검사를 실행할 수 있다. CLI 검사는 Bash를
사용하며 학습·평가 launcher의 인자와 환경을 확인한다. 실제 GPU 학습 검사는 별도로 수행한다.

```bash
python -m unittest discover -s tests -v
```

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
