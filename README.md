# FT-FSOD: CD-FSOD 실험 저장소

이 저장소는 FT-FSOD의 **CD-FSOD 6개 target dataset 실험**을 지원한다. 이 브랜치는
ACL Progressive Fine-Tuning을 유지하면서 HED를 standard serial Grounding DINO decoder로
교체하고, BERT + Linear 직후와 Feature Enhancer 이후에 Raw Mean Prototype + Nearest ETF
auxiliary loss를 동시에 적용한다. 시작 branch의 최신 commit
`30090c3d48ba0fe7aabb04546a6cf72dbbca1118`과 distributed process-group cleanup을 보존한다.
Augmentation, optimizer, scheduler, validation metric 및 checkpoint 설정은 유지한다.
RTX 5090 단일 GPU 환경과 dataset/shot별 실행 인터페이스를 제공한다.

## BERT + FE Raw Mean Prototype + Nearest ETF auxiliary losses

`grounding_dino_acl`을 기반으로 하며 ACL progressive fine-tuning을 유지한다.
LLM/Qwen description, support caption, detection용 class prototype은 사용하지 않는다.
`loss()`에서 BERT 출력이 `text_feat_map` (Linear)을 통과한 직후의
`text_dict['embedded']`로 기존 BERT ETF를 계산한다. 참고 branch
`acl-stagewise-raw-mean-etf`의 FE 계산과 동일하게 최종
`head_inputs_dict['memory_text']`로 FE ETF를 추가한다. 두 loss는 모든 stage에서
동시에 적용되며, weight는 독립적이다.

```text
L = L_det + lambda_B * L_ETF_BERT + lambda_F * L_ETF_FE
Stage 1: BERT ETF ON, FE ETF ON
Stage 2: BERT ETF ON, FE ETF ON
```

stage에 따른 ETF 위치 전환이나 전용 hook을 도입하지 않는다. Detection에는 기존처럼
최종 enhancer `memory_text`를 전달한다.

계산 순서는 다음과 같다 (`eps=1e-6`).

```text
전체 dataset class prompt → 기존 tokens_positive / get_positive_map
→ BERT → text_feat_map (Linear) → BERT ETF용 projected text features
                             → Feature Enhancer → FE ETF용 최종 text features
→ 동일한 전체 class mapping으로 각 위치에서 class-name token 선택
→ class별 raw token 평균 p_c = sum(t_i) / token 수 [D]
→ 이미지별 prototype stack [B,C,D] (batch 평균 없음)
→ class dimension centering
→ sample별 전체 [C,D] matrix Frobenius normalization (norm.clamp_min(eps))
→ sample별 nearest Simplex ETF target (Helmert basis + reduced Procrustes SVD)
→ squared Frobenius distance → batch mean
```

token 하나인 class도 동일하게 처리한다. token별 L2 normalization 없이 raw feature를
평균하며, 평균 후 각 `p_c`도 L2 normalize하지 않는다. `[B,C,D]`를 기존
`nearest_etf_loss()`에 그대로 전달한다. 해당 함수의 class centering, 전체 matrix
Frobenius normalization, nearest simplex ETF 계산 및 scalar loss의 batch 평균은
변경하지 않는다. 기존 SVD target solve만 `no_grad`이고, 두 loss의 feature 및 prototype
계산에는 `detach()`나 `no_grad()`를 적용하지 않는다. BERT ETF는 Linear와 BERT에,
FE ETF는 Feature Enhancer를 거쳐 Linear, BERT, image backbone까지 gradient를 전달한다.
Stage 1에서 FE parameter의 learning rate가 0이어도 computational graph는 유지되며,
FE input/output을 detach하지 않는다. Detection loss의 기존 gradient 경로는 유지한다. Auxiliary
계산은 FP32로 수행하며 FP64 입력은 보존한다.

전체 class prompt는 기존 `CocoDataset(return_classes=True)`의 `metainfo.classes`에서
온다. GT label로 span을 선택하기 전 전체 class mapping을 보관하므로 NEU-DET은 GT가
일부이거나 비어 있어도 항상 6개 class를 사용한다. 두 ETF는 한 번 생성한 같은 mapping을
공유하며 description 전체 평균이나 GT class subset으로 바꾸지 않는다.
명시적 `tokens_positive` 역시 모든
class의 span을 class 순서로 제공해야 한다 (dict는 0..C-1 key).
class 수 불일치, 누락된 token, prompt truncation, padding/범위 밖 token은 오류로
처리한다. `C >= 2`, `D >= C-1`이 필요하다.

원래 token-level `memory_text`는 변경 없이 query selection, cross-modality decoder,
contrastive classification으로 전달된다. GT positive map, classification target 및
checkpoint parameter key는 유지된다.

## Standard serial decoder와 progressive fine-tuning

`grounding_dino_acl_serial_decoder`의 serial 전환 커밋 `4118d18`에서 decoder 관련
변경만 반영했다. Qwen/support-caption 기능은 도입하지 않는다. Detector의
`_init_layers()`는 표준 `GroundingDinoTransformerDecoder`를 생성한다.

```text
Query → Decoder layer 1 → 2 → 3 → 4 → 5 → 6
```

각 layer는 직전 layer의 query와 갱신된 reference points를 입력받는다. 학습 시 DN
query를 한 번만 생성하고 모든 layer에서 사용하며, 기존 layer별 학습 loss는 유지한다.
추론은 마지막 layer의 class score와 bbox를 사용한다. HED의 parallel forward와
레이어별 추가 DN query 생성은 제거했다.

기존 detector/head 등록 이름과 decoder import alias는 config 호환을 위해 유지한다.
`rand_dnquery_rate`는 호환용 인자로만 받아 사용하지 않는다. `BBoxHeadFirstHook6`,
ReduceOnPlateau scheduler, Stage 1/Stage 2 전환 및 optimizer는 변경하지 않는다.
18개 실험 config에서는 ETF weight 설정만 분리한다.

## ETF 설정과 검증

18개 few-shot config의 `model`에는 다음 옵션이 기본 적용되어 있다.

```python
bert_raw_mean_etf_loss_weight=0.1
fe_raw_mean_etf_loss_weight=0.1
```

기존 BERT weight `0.1`을 유지하고 FE 기본 weight도 `0.1`로 설정했다.
loss key는 `loss_bert_raw_mean_etf`와 `loss_fe_raw_mean_etf`이며, 로그 값에는 각각의
weight가 이미 반영된다. MMEngine은 두 key를 detection loss와 함께 합산한다.
옵션을 생략한 모델 생성자의 기본값은 각각 `0.0`이다. 어느 weight든 `0.0`으로 설정하면
해당 ETF 계산을 생략하며, 둘 다 0일 때만 auxiliary mapping 생성도 생략한다.
공통 pretraining config에는 이 옵션을 추가하지 않았다.

예를 들어 두 weight를 각각 `1.0`으로 지정하려면:

```bash
python tools/train.py configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_NEU-DET_1shot.py \
  --cfg-options model.bert_raw_mean_etf_loss_weight=1.0 model.fe_raw_mean_etf_loss_weight=1.0
```

기존 `model.raw_mean_etf_loss_weight` override는 `model.bert_raw_mean_etf_loss_weight`로
변경해야 한다. 모델 parameter와 checkpoint key는 유지한다.

CPU PyTorch만으로 수학·gradient·ACL 회귀 테스트를 실행할 수 있다.

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

회귀 테스트는 실제 prompt mapping, detector 메서드, encoder/serial decoder loop와
detection head 메서드를 사용하고 무거운 dependency는 작은 test double로 대체한다.
joint ETF 변경 기준 commit은 `30090c3d48ba0fe7aabb04546a6cf72dbbca1118`이다.
두 ETF의 입력 위치와 공유 class mapping, 독립 weight, joint gradient 합산,
기존 hook의 Stage 1/2에서 FE gradient 및 parameter update 여부,
6개 decoder layer의 순차 실행, 단일 DN 생성 및 마지막 layer 추론을
검증한다. 동일 난수 상태의 serial 모델에서 ETF 활성화/비활성화에 따른 detection
출력·positive map·입력·gradient 일치도 검증한다. CUDA 테스트는
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
