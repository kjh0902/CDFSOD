# FT-FSOD: Stagewise Raw Mean Simplex ETF

이 실험 브랜치는 `codex/acl-raw-mean-etf-no-hed`의 serial decoder와 scaled dot
product classifier를 유지하면서, ACL progressive fine-tuning stage에 따라
**같은 Raw Mean ETF loss의 입력 위치만 전환**한다. HED를 복원하지 않는다.

| 학습 구간 | ETF 입력 | ETF loss |
|---|---|---|
| Stage 1 | BERT → `text_feat_map` 이후, FE 이전 token | BERT Raw Mean ETF만 적용 |
| Stage 2 | 전체 Feature Enhancer의 최종 `memory_text` | FE Raw Mean ETF만 적용 |

전환은 기존과 동일하게 language-model LR이 초기값의 절반 이하가 된 다음 epoch
시작에 일어난다. FE unfreeze와 ETF 위치 변경은 하나의 hook에서 함께 수행한다.
두 위치의 ETF를 동시에 더하지 않으며 ramp/cross-fade를 사용하지 않는다.

## Raw Mean 정의와 gradient

`mmdet/models/losses/raw_mean_etf_loss.py`의 공통 계산을 두 stage에서 사용한다.

```text
전체 dataset class prompt → 기존 tokens_positive / get_positive_map
→ 해당 위치의 class-name token을 raw mean하여 [B,C,D]
→ class 방향 centering
→ sample별 전체 [C,D] matrix Frobenius normalization
→ nearest Simplex ETF (Helmert basis + reduced Procrustes SVD)
→ squared Frobenius distance → batch mean
```

Token별 또는 class vector별 normalization은 없다. SVD target만 `no_grad`이고
mean/centering/matrix normalization에는 gradient가 흐른다. FP32에서 계산하며
FP64 입력은 보존한다. 이미지에 GT가 없는 class도 포함한다. `C>=2`, `D>=C-1`이며
누락/중복/범위 밖 token, padding, prompt truncation은 오류로 처리한다.

Stage 1에서 ETF 자체의 gradient는 BERT와 projection에만 흐른다. Stage 2에서는
FE 및 그에 연결된 text/visual 입력 경로에도 흐른다. Detection loss는 별도로
기존 경로를 학습한다. Detection용 token, GT positive map, inference는 변경하지 않는다.

## Freeze 범위와 설정

**Stage 1의 frozen/trainable module 범위는 원래 `BBoxHeadFirstHook6`와 같다.**
`encoder.*`와 기존 Other group을 LR=0으로 두고, backbone, language_model,
text_feat_map, neck, decoder, bbox_head, dn_query_generator는 기존대로 학습한다.
`requires_grad=False`로 바꾸지 않으므로 clipping과 optimizer moment 동작도 보존한다.

18개 few-shot config는 다음 설정을 사용한다.

```python
model = dict(raw_mean_etf_loss_weight=1.0, stagewise_raw_mean_etf=True)
custom_hooks = [dict(
    type='StagewiseRawMeanETFHook',
    adjust_scheduler_patience=True,
    patience_frozen=3,
    patience_unfrozen=6,
    stage2_fe_lr_mult=0.5,
)]
```

| 항목 | 기존 로그 설정 | 새 설정 |
|---|---:|---:|
| ETF λ (두 stage 공통) | 1.0 | 1.0 |
| AdamW 기본 LR / weight decay | 1e-4 / 0.05 | 유지 |
| Backbone / BERT LR multiplier | 0.2 / 0.2 | 유지 |
| Stage 2 FE LR | 전환 시 head LR | 전환 시 head LR × 0.5 |
| Stage 1 / Stage 2 plateau patience | 3 / 8 | 3 / 6 |
| Plateau factor / cooldown / min LR | 0.5 / 1 / 1e-6 | 유지 |
| Gradient clipping max norm | 0.1 | 유지 |

FE 배율은 Stage 2 진입 시 `encoder.*` group에 한 번만 적용한다. 다른 Other
group은 기존대로 head LR을 따르고, 이후에는 기존 scheduler가 각 LR을 줄인다.
기본값에서 최초 전환 직후 head LR=5e-5, FE LR=2.5e-5, BERT/backbone LR=1e-5다.
기존 파일의 ETF weight 기본값은 0.1이었으나 제공된 실행 로그의 실제 값은 1.0이다.
새 config는 비교를 위해 **실험 로그의 λ=1.0**에 맞췄다.

## 로그에 근거한 소폭 조정

제공된 NEU-DET/UODD/Clipart1k × 1/5/10-shot × 두 위치의 총 18개 학습 실행을
분석했다. 새 GPU 학습이나 hyperparameter sweep은 실행하지 않았다. 아래 값은
검증된 최적값이 아니라 다음 실험의 공통 기본값이다.

- NEU-DET 5-shot의 FE best mAP는 freeze 구간 23.6에서 unfreeze 구간 22.4로
  낮아졌다. 두 위치의 전환 시점은 모두 152 iter였다. FE LR을 head의 절반으로
  낮추어 전환 후 업데이트를 완화한다. 이는 과도한 LR이 원인이었다는 인과 증명은 아니다.
- Clipart1k 5-shot FE는 best 이후 ETF가 0.3299→0.0963으로 감소했지만 mAP는
  62.1→61.3으로 하락했다. Stage 2 patience를 8→6으로 줄여 plateau 이후 LR 감소를
  조금 앞당긴다. 학습 epoch 수와 best-checkpoint 선택 규칙은 유지한다.
- λ=1에서 ETF/detection loss 비율의 실행별 중앙값은 BERT 약 0.02~0.98%,
  FE 약 0.10~3.78%다. 전체 grad norm은 약 55.6~1600.9로 clipping보다 크지만,
  이는 ETF 전용 gradient가 아니다. 이 정보만으로 λ나 clipping을 크게 바꾸지 않는다.
- 저장된 validation 항목은 mAP/AP50/AP75/size AP이며 **validation loss는 없다**.
  ETF/detection gradient 방향이나 module별 비율도 없으므로 추정값을 관측값처럼
  해석하지 않는다. 데이터셋별 tuning이나 평가 split 변경은 하지 않는다.

로그의 `loss_raw_mean_etf`는 weight가 반영된 유일한 ETF 학습 loss다.
`etf_raw`, `etf_to_detection_ratio`(weighted ETF / 전체 detection loss), `etf_stage`는
분리된 detached 진단값이며 총학습 loss에 중복 합산되지 않는다. λ=0이면 ETF 계산을
생략하고 stage만 기록한다. Hook은 초기화·전환·재개 및 매 epoch의 위치, λ,
epoch/iteration과 module별 실제 LR을 기록한다.

## 실행, override와 resume

기존 `run_cdfsod.sh` 인터페이스를 사용한다. GPU 학습은 준비된 Linux 환경에서 실행한다.

```bash
bash run_cdfsod.sh --dataset NEU-DET --shot 5 --dry-run
bash run_cdfsod.sh --dataset NEU-DET --shot 5 --gpu 0
bash run_cdfsod.sh --dataset NEU-DET --shot 5 --gpu 0 --resume
```

새 실행의 출력 디렉터리는 이전 실험과 분리한다. 직접 train.py를 호출하면
기존 CLI의 `--cfg-options`로 설정을 변경할 수 있다. 예를 들어 **stagewise ETF를
유지하면서 LR/patience만 기존 정책으로 복원**하려면:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_NEU-DET_5shot.py \
  --work-dir exp_stagewise_etf/NEU-DET/5shot \
  --cfg-options custom_hooks.0.stage2_fe_lr_mult=1.0 custom_hooks.0.patience_unfrozen=8
```

`model.raw_mean_etf_loss_weight=0.0`은 ETF를 끄며, 다른 양수로 바꾸면 두 stage에
같은 λ를 적용한다. `stagewise_raw_mean_etf=True`에는 전용 hook이 필수다.
기존 `second_order_etf_loss_weight`는 고정 FE Raw Mean 호환 옵션으로 유지한다.
새 weight와 동시에 지정하거나 stagewise 모드와 함께 사용하면 오류를 낸다.

Checkpoint metadata에 stage, 초기 LR 기준, parameter group 식별 정보와 전환
시점을 저장한다. `--resume`는 optimizer/scheduler 상태를 복원하며 Stage 2를
Stage 1로 되돌리거나 FE LR 배율을 다시 적용하지 않는다. LR 감소 직후 저장된
Stage 1 checkpoint는 다음 epoch 시작에 한 번 전환한다. 일반 `load_from`은
가중치 초기화로 취급하여 Stage 1부터 시작한다. Stage metadata가 없는 기존
checkpoint는 stagewise `--resume`를 지원하지 않으며 `resume=False`와
`load_from`으로 새 실행을 시작해야 한다. Model state_dict parameter key는 동일하다.

## 검증

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

CPU PyTorch와 MMEngine 0.10.7로 수학, source 선택, ETF 단독 gradient, Stage 1
freeze 동등성, AdamW step, 실제 plateau scheduler, stage 경계 resume 및 실제
`Runner.save_checkpoint/resume`를 검증한다. Detector/encoder 경로 테스트는
production 메서드와 가벼운 모듈을 사용하며 MMCV/CUDA 전체 학습을 대체하지 않는다.
기존 second-order fixture는 raw mean 조건에 맞게 수정했다. CUDA가 없는 환경의
CUDA 전용 테스트는 skip한다. 새 GPU 학습 성능은 아직 검증하지 않았다.

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
