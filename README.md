# FT-FSOD: CD-FSOD 실험 저장소

이 저장소는 FT-FSOD의 **CD-FSOD 6개 target dataset 재현만** 지원한다. 논문의 HED,
Progressive Fine-Tuning, augmentation, optimizer, scheduler, validation metric 및 checkpoint
설정은 원본 그대로 유지한다. RTX 5090 단일 GPU 환경과 dataset/shot별 실행 인터페이스를
제공하며, 이 브랜치는 Centered ETF와 Encoder/Decoder Common Quality auxiliary loss를 추가한다.

## Centered ETF + Encoder/Decoder Common IoU Quality

기준 브랜치는 `codex/acl-centered-etf-decoder-mu-focal`(530e2e8)이다.
기존 serial ACL decoder, Hungarian one-to-one detection loss, encoder query
selection, token classification/confidence, DN supervision 및 inference를 유지하고,
Decoder Common Binary Focal을 두 개의 class-agnostic Quality Focal Loss로 대체한다.

최종 Feature Enhancer의 `memory_text`에서 각 dataset class의 class-name token을
raw mean하여 `p_c`를 만든다. 클래스별 가중치는 동일하며 token normalization은 없다.
`mu = mean_c(p_c)`, `d_c = p_c - mu`를 사용한다. 기존 centered ETF는 sample별
전체 residual matrix를 Frobenius normalize하고, detached SVD로 구한 per-image
nearest Simplex ETF와의 squared distance를 batch mean한다. 이 helper와 legacy
`second_order_etf_loss` 이름은 변경하지 않았다.

두 quality branch는 같은 최종 `mu`를 사용한다.

- Encoder: `gen_encoder_output_proposals`의 `output_memory`와 해당 encoder
  regression branch가 예측한 box를 사용한다. 기존 top-k selection 이전의 모든
  유효 후보가 대상이다. Padding, nonfinite base proposal/prediction 및 퇴화 box는
  제외한다. 기존 detection encoder loss는 원래 top-k 후보만 그대로 사용한다.
- Decoder: 마지막 layer의 일반 matching query와 그 layer가 예측한 box를
  사용한다. DN query와 중간 decoder layer는 quality loss에 포함하지 않는다.
- Score는 각각 `feature @ mu / sqrt(D) + bias`다. Encoder/Decoder bias는 별도의
  parameter이며 초기값은 둘 다 `-log(99)`다. Detection classifier bias와도 별개다.
  Auxiliary score는 selection, Hungarian matching cost 또는 최종 confidence에 쓰지 않는다.

Assignment는 각 branch의 box로 독립적으로 계산하며 class logits나 Hungarian
매칭을 참조하지 않는다. GT별 IoU top-5 후보 중 IoU > 0인 후보의 합집합을 positive로
삼는다. 여러 GT와 겹치는 positive는 전체 GT 중 최대 IoU를 target으로 사용한다.
같은 후보를 중복 계산하지 않는다. 미선택 후보의 최대 IoU가 0.5 이상이면 ignore,
그보다 낮으면 target 0인 negative다. GT가 없으면 모든 유효 후보가 negative다.
후보가 5개보다 적으면 있는 후보만 사용하며, overlap이 전혀 없으면 positive를
강제로 만들지 않는다. 높은 IoU의 중복 후보를 무조건 background로 밀지 않도록
ignore를 두되, 최종 one-to-one detection 감독은 그대로 유지한다.

기존 `QualityFocalLoss`의 soft tensor target 경로를 사용한다:
`QFL = BCEWithLogits(score, IoU) * abs(IoU - sigmoid(score))**2`.
Encoder/Decoder 각각의 unique positive 수를 distributed mean하고 최소 1로 clamp하여
해당 branch의 loss sum을 나눈다. Detection classification의 avg_factor를 재사용하지
않으며, 양성이 없어도 유효 negative를 학습한다. Ignore/invalid 후보는 제외한다.
IoU/assignment는 `no_grad`로 계산하여 quality loss의 box-regression target 경로를
차단한다. Score는 AMP에서도 FP32로 계산하며 feature, raw class means 및 해당 bias에
역전파된다. 공유 encoder/decoder parameter는 feature 경로를 통해 업데이트될 수 있다.

18개 few-shot config의 기본값:

```python
model = dict(
    second_order_etf_loss_weight=1.0,
    bbox_head=dict(
        enc_mu_quality_loss_weight=0.1,
        dec_mu_quality_loss_weight=0.1,
        common_quality_topk=5,
        common_quality_ignore_iou_thr=0.5,
        common_quality_beta=2.0))
```

Weight를 0으로 설정하면 해당 branch만 비활성화된다. 생성자의 quality weight 기본값은
둘 다 0.0이므로 공통 pretraining config는 그대로 사용할 수 있다. 예를 들어 encoder만
비활성화하려면 train 명령에
`--cfg-options model.bbox_head.enc_mu_quality_loss_weight=0.0`을 추가한다.
옛 `mu_focal_loss_weight` 옵션은 제거되었으므로 별도 custom config에도 위 옵션을 사용한다.
ETF가 꺼져도 quality가 하나라도 켜져 있으면 all-class mapping과 `mu`를 계산한다.

로그에는 이미 weight가 적용된 `loss_second_order_etf`, `enc_loss_mu_quality`,
`dec_loss_mu_quality`가 출력된다. Stage 1/2 hook, optimizer 및 scheduler 설정은
변경하지 않았다. 기존과 동일한 입력/가중치에서 detection loss와 inference 경로를
보존한다는 의미이며, 추가 loss로 학습한 가중치의 최종 detection 결과까지 같다는
의미는 아니다.

CPU regression tests:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Production assignment, QFL, head 및 detector methods와 CPU PyTorch를 사용한다.
MMCV base 초기화, Hungarian solver 및 box loss 일부는 test double로 대체한다.
IoU top-k/ignore, GT 충돌, independent normalization/bias, target detach, feature/mu
역전파, DN 제외, empty GT/invalid 후보, AMP, 18개 config 및 기존 detection/DN loss와
출력 보존을 검증한다. 전체 CUDA 학습이나 mAP 평가는 별도 실험이 필요하다.

설계 참고: [Generalized Focal Loss](https://arxiv.org/abs/2006.04388)의 continuous quality
감독과 [DETRs with Hybrid Matching](https://arxiv.org/abs/2207.13080)의 학습용
one-to-many 보조 감독을 참고했다. Top-5/ignore 0.5와 위 normalization은 본 실험의
명시적인 기본값이며, 해당 논문의 전체 알고리즘을 재현하거나 최적값을 주장하지 않는다.

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
