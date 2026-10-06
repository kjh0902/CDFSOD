# FT-FSOD: Background-anchored Nearest Deformed ETF

이 브랜치는 `grounding_dino_acl_serial_decoder` (`4118d18`)에서 시작한다.
ACL Progressive Fine-Tuning과 standard serial Grounding DINO decoder를 유지하고,
최종 Feature Enhancer(FE) output에 background-anchored nearest deformed ETF 보조
loss를 적용한다. 기존 optimizer, scheduler, augmentation, validation 및 checkpoint
설정은 유지한다. 입력은 foreground class name과 `background`만 사용하며 설명 JSON이나
별도 설명 생성 모델은 필요하지 않다.

## Background anchor와 foreground detection

기존 `class_a. class_b. ...` prompt 뒤에 `background. `를 추가하고,
BERT → `text_feat_map` → 모든 FE layer를 통과시킨다. 학습과 추론 모두 같은 경로를
사용한다. 최종 `memory_text`에서 class-name subword만 raw mean하여 foreground
`p_c`와 background `p_bg`를 얻는다. 구두점, 특수 토큰, padding은 pooling에서 제외한다.
Token/class별 L2 normalization은 적용하지 않는다.

FE 직후 추가된 background 구간과 구두점을 제거하고 원래 foreground prompt의 token
순서, `[SEP]` 위치와 padding mask를 복원한다. Background는 query selection, decoder
cross-attention/classification, positive map 및 prediction label에 포함되지 않는다.
Detection class 수와 label mapping은 기존 foreground C개 그대로다.

Image별 loss는 다음과 같다.

```text
Q[c] = p_c - p_bg
Q_hat = Q / max(||Q||_F, 1e-6)
s[c] = 1 + alpha * present[c]
A = diag(s) E
A_hat = A / ||A||_F
U, singular_values, Vh = svd(Q_hat.T @ A_hat, full_matrices=False)
R = U @ Vh
T = A_hat @ R.T
L = mean_b sum_c,d (Q_hat[b,c,d] - T[b,c,d])**2
```

`E`는 foreground C개에 대한 Helmert basis 기반 canonical Simplex ETF다.
Background를 ETF vertex에 포함하지 않는다. `present`는 augmentation 이후 image의
GT labels로 판단하는 binary presence이며 instance count를 사용하지 않는다.
`Q`와 deformation 이후 `A`는 class-mean centering하지 않고, matrix 전체에 대해서만
Frobenius normalization한다. 모두 present 또는 모두 absent이면 공통 row scaling은
normalization으로 상쇄된다.

SVD와 target solve만 `no_grad`로 처리한다. `p_bg`는 detach하지 않으며 보조 loss의
feature gradient는 BERT, projection과 FE까지 흐른다. ACL이 FE를 LR=0으로 유지하는
단계에서도 graph는 유지한다. Geometry는 autocast를 끄고 FP32로 계산하고 FP64 입력은
보존한다. C≥2일 때 D≥C−1이 필요하며 rank-deficient 입력에도 유효한 SVD 해를 사용한다.
FISH처럼 C=1이면 background 경로는 유지하고 보조 loss만 미분 가능한 0으로 반환한다.

## Hyperparameter와 실행

18개 finetune config 모두 다음 옵션을 사용한다. Alpha와 weight는 dataset, class 수,
image의 present class 수 및 학습 단계와 무관한 constant scalar다.

```python
use_background_anchor=True
bg_anchored_etf_alpha=0.5
bg_anchored_etf_loss_weight=0.1
```

Alpha=0.5는 present class의 상대 크기를 50% 증가시키고, weight=0.1은 detection loss에
더하는 보조 항의 가중치다. 학습 로그의 이름은 `loss_bg_anchored_nearest_deformed_etf`다.
유한한 음이 아닌 scalar만 허용한다. 기존 CLI로 재정의할 수 있다.

```bash
python tools/train.py configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_NEU-DET_5shot.py \
  --cfg-options model.bg_anchored_etf_alpha=0.25 model.bg_anchored_etf_loss_weight=0.05
```

Weight=0은 background의 FE 참여를 유지하면서 SVD/보조 loss 계산만 생략한다.
`model.use_background_anchor=False`는 background 추가와 보조 loss를 모두 끄고 기존
class-name detection 경로를 사용한다. 모델 옵션의 기본 flag는 False이며, finetune
config에서 True로 활성화한다. 추론에는 GT labels나 SVD 계산이 필요하지 않다.

CPU 검증에는 PyTorch만 필요하며 MMDetection 확장이나 사전학습 모델 다운로드는
필요하지 않다. 테스트는 실제 detector 메서드와 FE layer loop를 작은 attention 대역에
연결해 gradient, background 분리 및 detection 인터페이스를 검증한다.

```bash
python -m unittest discover -s tests -v
```

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
