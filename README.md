# FT-FSOD 재현 환경 (RTX 5090, 단일 GPU)

이 저장소는 [FT-FSOD 원본 코드](https://github.com/Intellindust-AI-Lab/FT-FSOD)를
RTX 5090 한 장(`GPU 0`)에서 재현하기 위한 실행용 저장소다. 논문의 모델 구조와 학습
method는 변경하지 않았으며, CUDA/PyTorch/MMCV 호환성, GPU 지정, 설치 및 실행 절차만
정리했다.

## 지원 환경

| 항목 | 고정값 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 (compute capability 12.0) |
| 사용 GPU | 물리 GPU 0 한 장 |
| Python | 3.10 |
| PyTorch | 2.7.1 + CUDA 12.8 wheel |
| torchvision | 0.22.1 + CUDA 12.8 wheel |
| CUDA build toolkit | 12.8 (conda 환경 내부) |
| MMEngine | 0.10.7 |
| MMCV | 2.2.0 소스 빌드, commit `a8073c74bf83d62ec36a103f835faa4837fb6585` |
| FairScale | 0.4.13 |

`nvidia-smi`의 `CUDA Version: 13.2`는 드라이버가 지원하는 최대 CUDA 버전이다. 이
프로젝트는 RTX 5090용 코드가 포함된 공식 `cu128` PyTorch wheel과 동일한 CUDA 12.8
툴킷으로 MMCV를 빌드한다. 시스템 CUDA 13.2에 맞춰 패키지를 임의로 바꾸지 않는다.

## 1. 설치

Miniconda/Miniforge가 설치된 Linux 서버에서 실행한다. MMCV 컴파일을 위해 약 15 GB의
여유 공간과 C/C++ 빌드 도구가 필요하다.

```bash
git clone https://github.com/kjh0902/CDFSOD.git
cd CDFSOD

conda env create -f environment.yml
conda activate ft-fsod

# PyTorch cu128 설치, MMCV sm_120 빌드, MMEngine patch, CUDA smoke test
bash scripts/install_rtx5090.sh
```

설치 스크립트는 다음까지 검증하고 실패 시 즉시 종료한다.

- PyTorch가 RTX 5090과 `sm_120`을 인식하는지
- GPU 0에서 CUDA matmul/backward가 동작하는지
- 직접 빌드한 MMCV CUDA NMS가 GPU 0에서 동작하는지
- FairScale activation checkpointing을 import할 수 있는지
- Python/PyTorch/torchvision/MMEngine/MMCV 버전이 고정값과 일치하는지

설치 후 다시 확인하려면 다음을 실행한다.

```bash
conda activate ft-fsod
CUDA_VISIBLE_DEVICES=0 python tools/verify_environment.py
```

## 2. 데이터와 사전학습 모델

CD-FSOD 데이터는 아래 구조로 둔다. 각 데이터셋의 annotation은 COCO 형식이다.

```text
data/cdfsod/
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

데이터는 [CD-FSOD benchmark](https://github.com/lovelyqian/CDFSOD-benchmark)에서
준비한다. Swin-B 사전학습 checkpoint를 내려받는다.

아래 URL과 같이 출처를 신뢰할 수 있는 checkpoint만 사용한다. PyTorch 2.6 이상에서는
MMEngine checkpoint의 학습 이력 메타데이터를 읽기 위해 제한되지 않은 pickle loader가
필요하며, 이 저장소의 `tools/train.py`와 `tools/test.py`가 해당 호환 설정을 자동 적용한다.

```bash
mkdir -p checkpoints
wget -O checkpoints/grounding_dino_swin-b_pretrain_all-f9818a7c.pth \
  https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-b_pretrain_all/grounding_dino_swin-b_pretrain_all-f9818a7c.pth

# BERT와 NLTK resource를 미리 cache
python -c "from transformers import AutoTokenizer, BertModel; AutoTokenizer.from_pretrained('bert-base-uncased'); BertModel.from_pretrained('bert-base-uncased')"
python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng
```

기본 경로는 `src_path.py`에 정의되어 있다. 서버의 데이터 위치가 다르면 코드를 수정하지
않고 환경 변수로 지정할 수 있다.

```bash
export CDFSOD_PATH=/absolute/path/to/cdfsod
export MMGDINOB_PATH=/absolute/path/to/grounding_dino_swin-b_pretrain_all-f9818a7c.pth
```

ODinW-13 또는 RF100-VL을 실행할 때는 같은 방식으로 `ODINW_PATH`,
`RF100_VL_FSOD_PATH`, `MMGDINOL_PATH`를 지정한다.

## 3. CD-FSOD 학습 및 평가

아래 명령은 1/5/10-shot 설정과 여섯 target dataset을 순서대로 학습하고, 각 실험의 best
checkpoint를 평가한다. 모든 launcher는 GPU 0 한 장만 노출한다.

```bash
conda activate ft-fsod
GPU_ID=0 PORT=29500 bash run_mmgdinob_traineval_cdfsod.sh
```

결과와 checkpoint는 `exp_cdfosd_results/`에 저장된다. 단일 config만 실행하려면:

```bash
bash tools/dist_train.sh \
  configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_ArTaxOr_1shot.py \
  1 29500 0 \
  --work-dir exp_cdfosd_results/swinB_all_ArTaxOr_1shot
```

학습을 중단했다가 이어서 실행할 때는 같은 명령 끝에 `--resume`을 추가한다.

## 4. 저장된 checkpoint 평가

```bash
bash tools/dist_test.sh \
  configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_ArTaxOr_1shot.py \
  /path/to/best_coco_bbox_mAP_iter_xxx.pth \
  1 29501 0 \
  --work-dir exp_cdfosd_results/eval_ArTaxOr_1shot \
  --out exp_cdfosd_results/eval_ArTaxOr_1shot/results.pkl
```

원 논문의 공개 fine-tuned checkpoint는
[Hugging Face](https://huggingface.co/Xuanlong/FT-FSOD-CD-FSOD)에서 받을 수 있다.

## 5. CD-Mixed 및 다른 benchmark

```bash
# CDFSOD_PATH와 CDMIXED_PATH가 필요하다.
bash create_cdmixed_set.sh
GPU_ID=0 PORT=29501 bash run_mmgdinob_eval_cdmixed.sh

# Swin-L checkpoint와 해당 데이터 경로가 필요하다.
GPU_ID=0 PORT=29502 bash run_mmgdinol_traineval_odwin.sh
GPU_ID=0 PORT=29503 bash run_mmgdinol_traineval_rf100vl.sh
```

결과 집계:

```bash
python analyze_results_cdfsod.py exp_cdfosd_results
python analyze_results_odinw.py exp_odinwfsod_results
python analyze_results_rf100.py exp_rf100vlfsod_results
```

## 문제 해결

- `no kernel image is available`: `torch==2.6` 이하 또는 CUDA 12.4 wheel을 설치한
  환경이다. 환경을 삭제하고 `environment.yml`부터 다시 만든다.
- `No module named mmcv._ext`: `mmcv-lite` 또는 다른 PyTorch ABI용 MMCV가 설치된
  상태다. `scripts/install_rtx5090.sh`로 CUDA ops를 다시 빌드한다.
- `please install fairscale`: 이전 requirements에 activation checkpointing 의존성이
  누락된 경우다. 최신 코드를 받은 뒤 `python -m pip install -r requirements.txt`를 실행한다.
- `Weights only load failed` 또는 `HistoryBuffer was not an allowed global`: PyTorch 2.6+
  호환 처리가 포함되지 않은 이전 코드다. 최신 `tools/train.py`/`tools/test.py`를 사용한다.
  다른 진입점을 직접 사용해야 하고 checkpoint 출처를 신뢰한다면
  `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`을 지정한다.
- MMCV 빌드 OOM: `MAX_JOBS=1 bash scripts/install_rtx5090.sh`로 재실행한다.
- `Address already in use`: 실행 명령의 `PORT`를 사용하지 않는 값으로 바꾼다.
- CUDA OOM: 논문 config의 batch size나 모델 구조를 임의 변경하기 전에 다른 GPU
  process가 없는지 `nvidia-smi`로 확인한다.

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
