# FT-FSOD: CD-FSOD 실험 저장소

이 저장소는 FT-FSOD의 **CD-FSOD 6개 target dataset 재현만** 지원한다. 논문의 HED,
Progressive Fine-Tuning, augmentation, optimizer, scheduler, validation metric 및 checkpoint
설정은 원본 그대로 유지한다. RTX 5090 단일 GPU 환경과 dataset/shot별 실행 인터페이스를
제공하며, 이 브랜치는 아래의 GT-aware negative-only class separation loss를 추가한다.

## GT-aware negative-only class separation loss

기존 ACL/HED detection 구조와 detection loss를 유지한다. `forward_encoder()`가
전체 Feature Enhancer를 통과한 뒤 반환하는 최종 token-level `memory_text`에서
class-name token만 읽어 auxiliary loss를 계산한다.

Class g, j의 similarity는 모든 token pair의 **raw dot product 평균**이다.
Class별 token 수가 달라도 동일하게 동작하며, 다음 등가식으로 계산한다.

```text
mu_g = mean_a(t_g,a)
s(g,j) = mean_a,b(t_g,a dot t_j,b) = mu_g dot mu_j
L_sep(image) = mean_{g in unique GT} log(1 + sum_{j != g} exp(s(g,j) / temperature))
L_sep = mean_image L_sep(image)
L_total = L_detection + lambda_sep * L_sep
```

현재 이미지의 unique GT class만 anchor로 사용하며, dataset의 나머지 모든 class를
negative로 사용한다. 같은 이미지에 등장하는 다른 GT class도 negative다. GT object가
중복되어도 anchor 가중치는 증가하지 않는다. GT가 없는 이미지는 미분 가능한 0을
반환하고 전체 batch 평균에 포함한다. 단일-class 입력도 negative가 없으므로 0이다.

Token/class normalization, second-order representation, centering 및 target solve는
사용하지 않는다. `logsumexp([0, negative logits...])`로 수식을 안정적으로 계산하며,
auxiliary 연산은 autocast를 끈 FP32로 수행하고 FP64 입력은 유지한다. Gradient는
anchor와 negative의 class-name token을 통해 Feature Enhancer로 전달된다.

전체 class prompt는 기존 `CocoDataset(return_classes=True)`의 `metainfo.classes`에서
온다. GT label로 span을 선택하기 전에 전체 class mapping을 보관한다. 명시적
`tokens_positive`도 모든 class의 span을 class 순서로 제공해야 한다 (dict는 0..C-1 key).
기존 mapping의 class key는 1..C이고 GT label은 0..C-1이다. Class 누락, 빈 token,
prompt truncation, padding/범위 밖 token, 잘못된 GT label은 오류로 처리한다.

원래 token-level `memory_text`는 변경 없이 query selection, cross-modality decoder,
contrastive classification으로 전달된다. GT positive map, classification target,
HED, inference 경로 및 checkpoint parameter key는 유지된다.

18개 few-shot config의 `model`에는 다음 옵션이 기본 적용되어 있다.

```python
lambda_sep=0.1,
temperature=1.0,
```

`lambda_sep`는 유한한 비음수, `temperature`는 유한한 양수여야 한다. Loss key는
`loss_class_separation`이며 로그 값에는 `lambda_sep`가 이미 반영된다. 모델 생성자의
기본값은 `lambda_sep=0.0`, `temperature=1.0`이다. `lambda_sep=0.0`이면 auxiliary
mapping 생성과 loss 계산을 생략하며, inference에서도 실행하지 않는다. 예를 들어:

```bash
python tools/train.py CONFIG --cfg-options model.lambda_sep=0.1 model.temperature=1.0
```

공통 pretraining config에는 이 옵션을 추가하지 않았다. Raw dot product이므로 loss
크기는 feature magnitude와 temperature에 의존하며, 위 설정은 실험 시작값이다.

CPU PyTorch만으로 수학·gradient·ACL 회귀 테스트를 실행할 수 있다.

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

회귀 테스트는 실제 prompt mapping, detector 메서드, encoder loop, HED head forward를
사용하고 무거운 dependency는 작은 test double로 대체한다. 기준 ACL commit은
`8926970ebff1a549088b0a4c87c272e1a70fe0dd`이다. 동일 난수 상태에서 base 및 loss
활성화/비활성화의 detection 출력·positive map·HED 입력을 비교한다. CUDA 테스트는
CUDA가 없으면 skip한다. 이 테스트는 실제 MMCV/CUDA 학습이나 mAP 평가를 대체하지 않는다.

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
