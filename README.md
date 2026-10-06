# FT-FSOD: CD-FSOD 실험 저장소

이 저장소는 FT-FSOD의 **CD-FSOD 6개 target dataset 재현만** 지원한다. 논문의 HED,
Progressive Fine-Tuning, augmentation, optimizer, scheduler, validation metric 및 checkpoint
설정은 원본 그대로 유지하며, BERT → `text_feat_map` 직후에 **Raw Mean Simplex ETF
보조 loss**를 추가했다. HED parallel decoder와 기존 detection 경로는 그대로 사용한다.

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

### BERT Raw Mean Simplex ETF loss

전체 dataset class의 class-name token을 `text_feat_map` 직후 클래스별로 평균한다.
Token과 개별 class prototype에 L2 정규화를 적용하지 않는다. 각 이미지의 전체 class
prototype 행렬을 class 축으로 centering하고, 행렬 전체의 Frobenius norm으로 정규화한 뒤
SVD로 구한 nearest Simplex ETF와의 제곱 Frobenius 거리를 계산한다. SVD target은
detach하며, loss는 이미지별로 계산해 batch 평균을 낸다.

GT에 없는 클래스와 empty-GT 이미지도 전체 class loss에 포함된다. BERT와
`text_feat_map`에는 gradient가 흐르며, 기존 Progressive Fine-Tuning hook의 freeze/unfreeze
설정을 따른다. Feature enhancer, HED decoder 및 detection head 입력은 변경하지 않는다.
학습 loss dict에 `loss_raw_mean_etf`가 추가되며, inference에서는 계산하지 않는다.

기본 weight는 **1.0**이다. 각 finetune config의 `model.raw_mean_etf_loss_weight`를 수정하거나
CLI에서 덮어쓸 수 있다. `0`은 ETF mapping과 loss 계산을 모두 비활성화한다.

```bash
bash run_cdfsod.sh --dataset NEU-DET --shot 1 --gpu 0 --etf-loss-weight 0.1

python tools/train.py \
  configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_NEU-DET_1shot.py \
  --cfg-options model.raw_mean_etf_loss_weight=0.1
```

ETF를 켠 학습 prompt는 `bbox_head.num_classes`개의 모든 dataset class를 포함해야 한다.
일부 클래스만 제공하거나 class token이 잘리는 prompt는 오류로 처리한다. 명시적인
`tokens_positive`도 모든 클래스의 span을 제공해야 한다. 필요한 feature 차원은 `D >= C - 1`이다.
FISH처럼 클래스가 하나뿐인 dataset은 클래스 간 ETF 구조를 정의할 수 없으므로,
ETF 항을 gradient가 0인 zero loss로 처리한다. Weight 기본값은 동일하게 1.0이다.

CPU PyTorch만으로 수치, gradient 및 HED 회귀 검사를 실행할 수 있다. 전체 GPU 학습 검사는
별도로 학습 환경에서 실행해야 한다.

```bash
python -m unittest discover -s tests -p 'test_*etf*.py' -v
```

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
