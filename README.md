# FT-FSOD: CD-FSOD 실험 저장소

이 저장소는 FT-FSOD의 **CD-FSOD 6개 target dataset 재현만** 지원한다.
Progressive Fine-Tuning, augmentation, optimizer, scheduler, validation metric 및 checkpoint
설정은 시작 브랜치 그대로 유지하며, 최종 Feature Enhancer의 `memory_text`에
**Batch-shared μ-Orthogonal Token ETF Loss**를 적용한다. BERT 보조 loss는 사용하지 않는다.
Decoder와 Detection Head는 `codex/acl-raw-mean-etf-no-hed`의 순수 ACL 구조를 사용한다.
6개 Decoder 레이어가 순차적으로 query/reference를 갱신하고, 학습에서는 DN Query를
한 번만 생성한다. Head는 마지막 Decoder 레이어로 예측하며, 클래스 매핑·박스 후처리와
평가 경로는 유지한다. 기존 config와 실행 스크립트 호환성을 위해 등록된 모델·Head 이름과
`rand_dnquery_rate` 인자는 유지하지만, HED Parallel Decoder와 레이어별 DN Query는 사용하지 않는다.

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
git clone --branch acl-batch-shared-mu-orth-token-etf https://github.com/kjh0902/CDFSOD.git
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

### Batch-shared μ-Orthogonal Token ETF

최종 FE의 `memory_text`에서 **전체 dataset class**의 class-name WordPiece token을
사용한다. GT에 없는 클래스와 empty-GT 이미지도 포함한다. 기존 전체 클래스 매핑을
재사용하며, 누락·잘림·빈 span·padding을 가리키는 Token은 오류로 처리한다.
이미지별 클래스 평균을 구한 다음 Batch 전체에서 같은 클래스의 평균을 구한다.

$$
p_{i,c}=\frac{1}{K_{i,c}}\sum_k t_{i,c,k},\quad
\bar p_c=\frac1B\sum_i p_{i,c},\quad
\mu=\frac1C\sum_c\bar p_c
$$
$$
s=\sqrt{\sum_c\|\bar p_c-\mu\|^2},\qquad
Z_c=\frac{\bar p_c-\mu}{\max(s,10^{-6})}
$$

`nearest_etf_loss.py`와 같은 Unit-Frobenius Simplex ETF를 사용한다.
Helmert Basis `Q`와 μ의 직교 여공간 Basis `W`에 대해 Constrained Procrustes를 푼다.

$$
W^\top Z^\top Q=U\Sigma V^\top,\qquad
M^*=\frac{Q(WUV^\top)^\top}{\sqrt{C-1}}
$$

이 Target은 `M* μ = 0`, `M* M*ᵀ = (I - 11ᵀ/C)/(C-1)`을 만족하며
`||Z-M||_F²`를 최소화한다. SVD를 직교 여공간 안에서 수행하므로 Prototype의
Rank가 부족한 경우에도 직교 조건을 유지한다. μ가 정확히 0이면 `W=I`이며,
미소한 비영 μ는 크기를 재조정하여 방향을 안정적으로 계산한다.
비영 μ에서는 `D≥C`, 영 μ에서는 `D≥C−1`이 필요하다.

최종 목적함수는 개별 Token을 Batch 공통 ETF Vertex로 직접 정렬한다.

$$
L_{\mathrm{UnifiedETF}}=\frac1{BC}\sum_{i,c}\frac1{K_{i,c}}\sum_k
\left\|\frac{t_{i,c,k}-\mu}{\max(s,10^{-6})}-\operatorname{sg}(m_c^*)\right\|_2^2
$$

차원별 제곱합 뒤 클래스 내 Token 평균, 이미지·클래스 평균을 순서대로 계산한다.
따라서 클래스 이름의 Token 수와 Batch Size에 따라 가중치가 바뀌지 않으며,
Token/클래스 벡터의 개별 정규화나 차원 `D`로의 평균은 하지 않는다.
Target만 detach하고 Token, μ, s는 미분한다. Autocast를 끄고 FP32로 계산하며,
매 Batch Target을 다시 구한다. EMA나 저장된 Target은 사용하지 않는다.
FISH처럼 `C=1`이면 SVD 없이 미분 가능한 0을 반환한다.

| Config option | 기본 weight | 학습 로그 |
|---|---:|---|
| `model.unified_etf_loss_weight` | 10.0 | `loss_unified_etf` |

Weight는 유한한 비음수이며, 0이면 보조 Loss·전체 클래스 매핑·SVD를 모두 생략한다.
Inference에서는 보조 Loss를 계산하지 않는다. 기존 BERT/FE ETF·Orth weight와
`--etf-loss-weight`, `--raw-mean-etf-loss-weight` alias는 폐기했으며, 이전 옵션을
사용하면 새 옵션으로 변경하라는 오류를 표시한다.

```bash
# 기본 weight=10.0
bash run_cdfsod.sh --dataset NEU-DET --shot 1 --gpu 0

# CLI에서 weight 변경
bash run_cdfsod.sh --dataset NEU-DET --shot 1 --gpu 1 \
  --work-dir work_dirs/unified_etf_NEU-DET_1shot \
  --unified-etf-loss-weight 5.0

# Native MMEngine CLI
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_NEU-DET_1shot.py \
  --work-dir work_dirs/unified_etf_NEU-DET_1shot \
  --cfg-options model.unified_etf_loss_weight=5.0
```

Train과 test는 기존대로 한 노드·한 GPU·한 process로 실행한다. Batch는 해당
forward의 로컬 Minibatch이며, GPU 사이 통계를 모으는 통신은 추가하지 않는다.
기본 Batch Size 4와 Max Epochs(FISH 16, Clipart1k 50, 나머지 100), optimizer, scheduler,
Progressive FT, Detection Loss, augmentation, dataset, evaluation을 유지한다.
FE는 Stage 1에서 기존 hook에 따라 lr=0이며 upstream Gradient는 유지한다.

CPU PyTorch로 수학·Token 가중치·Gradient·ACL 회귀 검사를 실행할 수 있다.
CLI 검사는 Bash로 학습·평가 launcher의 인자와 환경을 확인한다.
현재 로컬 검증 환경은 CPU PyTorch이며 MMCV 확장이 없어 실제 GPU 학습은 검증 범위에
포함되지 않는다. 기존 수학 Helper는 참고와 회귀 검증용으로 보존한다.

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
