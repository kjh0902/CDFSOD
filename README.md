# ACL Baseline Diagnostic Logging

This branch runs the ACL **sequential decoder without HED**. The four BERT/FE
ETF and orthogonality weights are all **0.0**. The original progressive
fine-tuning hook, augmentation, optimizer, scheduler and batch sizes are retained.
FISH runs for 16 epochs, Clipart1k for 50, and all other targets for 100.
Validation runs every epoch. **No model/optimizer checkpoints are saved**, and
there is no post-training best-checkpoint reload or test invocation. Initial
pretrained weights and activation checkpointing are retained.


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
git clone --branch acl-diagnostic-logging https://github.com/kjh0902/CDFSOD.git
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

## 3. Run the diagnostic baseline

Run from the repository root in the supported Linux GPU environment:

```bash
bash run_cdfsod.sh --dataset NEU-DET --shot 1 --gpu 0
bash run_cdfsod.sh --dataset FISH --shot 5 --gpu 1 --port 29501
bash run_cdfsod.sh --dataset DIOR --shot 10 --gpu 0 --dry-run
```

Datasets: ArTaxOr, Clipart1k, DIOR, FISH (DeepFish alias), NEU-DET, UODD.
Shots: 1, 5, 10. Each invocation uses one physical GPU and a unique work directory.
`--work-dir PATH` selects the output directory. `--resume` is rejected because
there are no checkpoints. Existing weight flags/aliases accept only zero.
The diagnostic hook also checks these invariants for direct `tools/train.py` use.

The logger follows the existing LoggerHook interval and short-epoch rules.
Stage numbers come from BBoxHeadFirstHook6, not an assumed epoch threshold.
Training records include current losses, window means, every optimizer-group LR,
and the original optimizer's pre-clipping gradient norm. LR=0 is the existing
progressive freeze mechanism; logging does not change it.

## 4. Compact diagnostics (schema v2)

The baseline training/validation and checkpoint-free launcher are unchanged.
The default storage policy is:

| Data | Frequency / scope |
|---|---|
| Loss, all optimizer-group LRs, existing gradient norm | Existing iteration cadence, plus an all-iteration mean every epoch |
| COCO mAP, AP50/AP75, per-class AP/AP50/AP75 | Every validation epoch |
| GT snapshot, prompts, category/label mapping | Once per run / unique prompt |
| Final detections and exact query/global-class provenance | Every image, every epoch; float32 boxes/scores |
| Encoder candidates, decoder layers, BERT/FE features | Epoch 1, every 5 epochs, first validation after a stage transition, final epoch |
| Final decoder token logits and final FE token features | Only on the same <=32 fixed probe images, at internal snapshot epochs |

Probe selection covers available classes and an empty-GT image when possible,
then fills by stable image-ID hashes. It never draws from training/validation RNGs.
Stage events are recorded before training the first epoch of the new stage.
Their internal snapshot is the normal validation after that epoch; no extra
forward/evaluation is added. The immediately preceding epoch may have only final
detections and scalars, rather than internal tensors.

```text
diagnostics/
  manifest.json             # schema, config, commit, policy, fixed probe IDs, completion status
  dataset.json              # original GT, crowd/ignore, image sizes, category-label mapping
  train.jsonl               # unchanged existing iteration logging cadence
  train_epochs.jsonl        # epoch means over ALL train iterations, LR, gradient norm mean
  stage_events.jsonl        # only when the progressive stage changes
  metrics.jsonl             # every epoch's COCO metrics, snapshot reasons and actual tensor bytes
  prompts/<hash>.json        # shared prompt/mapping metadata
  epoch_001/
    index.jsonl             # every image/chunk, capture_level, exact tensor references
    batch_000000.npz         # compressed batch shards; light epochs contain final detections only
    text/<hash>.npz          # snapshot BERT features, deduplicated by actual stored bytes
    text/<hash>.npz.json
    complete.json           # finalized only after all images and metrics were stored
```

### Retained internal information

- Encoder candidates are the union of the actual selected queries, up to 128
  highest-selection-score unselected candidates, and up to 5 highest-IoU valid
  candidates per non-crowd/non-ignored GT. `encoder_candidate_indices` preserves
  the original flattened encoder IDs; `selected_indices` keeps the actual native
  query selection order. Candidate arrays now have M rows, not N full grid rows.
- For ALL valid candidates, the logger also computes GT-wise maximum IoU,
  selected-candidate maximum IoU, candidate counts at IoU >= .50/.75, and the
  best correct-class score above those IoUs. These small summaries retain evidence
  that a good proposal existed but was not selected. Max-token-logit quantiles
  summarize the full candidate score distribution. Boxes and IoU summaries use
  the existing unclipped normalized-cxcywh-to-original-pixels convention.
- All six decoder layers retain every query's class scores and normalized boxes.
  A class score is the same mean(sigmoid(mapped token logits)) as inference;
  there is no native objectness head. `decoder_boxes[-1]` and
  `decoder_class_scores[-1]` are the complete final pre-postprocessing queries.
- BERT token features before/after text_feat_map remain available on snapshot
  epochs. All six FE layers retain raw class-name-token means as
  `fe_class_prototypes[layer,class,channel]`. These support class geometry,
  inter-layer comparisons, BERT-to-FE and temporal drift analyses without keeping
  every token vector at every enhancer layer.
- Probe images additionally retain `decoder_final_token_logits[query,token]`
  and `fe_final_text[token,channel]`. Full six-layer token logits and full six-layer
  FE token states are removed. No visual/query hidden states or attention weights
  are stored.
- Final `det_boxes` (original-pixel xyxy) and `det_scores` stay float32.
  Actual query and global class indices are stored using the smallest lossless
  uint dtype. The reader reconstructs local class and flat top-k indices exactly
  by integer arithmetic, never by score/box matching or re-running top-k.
- Base anchors, rectangular padding and proposal-valid masks are reconstructed
  from spatial shapes, valid heights/widths and the versioned proposal recipe.
  Nonrectangular padding falls back to the exact bit-packed mask. Retained
  candidate boxes/scores and other large internal floats stay float16.
- Compression remains standard pickle-free NPZ DEFLATE. There is no additional
  int8/lossy score quantization. Missing internal tensors are deliberate and
  explicitly marked `capture_level=detections`, not fabricated by the reader.

### Offline reading and limitations

The Python/Numpy-only reader supports both existing v1 logs and new v2 logs:

```bash
python tools/analysis_tools/inspect_acl_diagnostics.py /path/to/diagnostics --verify --analyze
```

`DiagnosticReader.observations()`, `load(record)`, `proposals(arrays)` and
`selected_candidate_rows(arrays)` expose all stored data. Final detection
confusion, missed GT, duplicate/localization/background examples can be examined
EVERY epoch from detections and GT. `--score-threshold .3` controls only this
example analysis, never the saved predictions or original COCO metrics.
Internal query-selection coverage, query confusion and FE class-feature drift
are reported only for snapshot epochs, with explicit internal observation counts.
Greedy diagnostic matching is not a replacement for COCO evaluation.

Unretained rejected candidates cannot be individually inspected later. Raw
within-class token behavior is restricted to the fixed probes' final layers;
intermediate FE token states and intermediate decoder token logits cannot be
recovered. Intermediate decoder class scores/boxes and FE class means remain
available at snapshots. Float16 offline IoUs/scores may differ slightly from
native precision; native final detection arrays and provenance are unchanged.
Class AP/AP50/AP75 continue to come from the original evaluator's precision
array, including threshold lookup by value and null/reason for undefined metrics.

## 5. Storage measurement and capacity protection

A reproducible CPU benchmark loads the original capture/writer directly from
commit `09efa0c`, feeds v1 and v2 IDENTICAL synthetic tensors and measures actual
compressed files, including metadata/scalars:

```bash
python tools/analysis_tools/benchmark_acl_diagnostics.py /new/output/directory --epochs 9 --images 8 --max-epochs 100
```

Parameters: 20,000 encoder candidates, 900 queries, 26 tokens, 6 classes,
256 feature channels, six layers, 8 images, first 9 epochs of a 100-epoch schedule,
and an illustrative stage transition at epoch 7. Internal snapshots are 1/5/7.
All 8 synthetic images are token probes, a conservative probe fraction compared
with <=32 probes in a full validation dataset.

| Measured files | Previous v1 | Compact v2 | Reduction |
|---|---:|---:|---:|
| First 9 synthetic epochs | 59,154,835 bytes | 4,635,695 bytes | 92.16% |
| 100 synthetic epochs (projection, 22 snapshots) | 657,241,809 bytes | 35,904,412 bytes | 94.54% |

These are actual compressed-file measurements, not a NEU-DET GPU training run.
Applying the measured prefix ratio to the reported 1.8 GB / 9 epochs suggests
about **0.14 GB / 9 epochs**. Actual data, proposal count, image geometry, token
length and stage timing affect the result. `comparison.json` also projects a full
run from measured light/snapshot epoch sizes and explicitly labels that estimate.
Normalizing that projection to the reported NEU-DET rate gives approximately
**20 GB -> 1.1 GB for 100 epochs**; this is a size estimate, not a measured
NEU-DET run or a capacity guarantee.
The benchmark asserts exact equality of native detection boxes, scores and
query provenance between captures. Existing logs are not modified or deleted.

Preflight budgets light epochs and internal snapshots separately, including one
possible progressive-stage snapshot. It recalibrates their compression rates
separately during validation. Defaults still require **10 GiB reserve** plus
**1.2 times projected remaining data**. Capacity is checked before each file
write and at batch/epoch boundaries; no optimistic compression ratio is assumed
before the first measurement.

```python
dict(type='ACLDiagnosticHook', out_dir='/large-disk/new-run/diagnostics',
     internal_interval=5, token_probe_images=32,
     reserve_gib=10.0, capacity_margin=1.2)
```

The destination must be new. A storage error stops training immediately, on
both light and snapshot epochs. It never triggers extra thinning or deletion.
Files are atomically finalized before index references are appended; only fully
stored epochs/runs receive completion markers. Compression and analysis
reductions add runtime but do not modify model inputs/outputs, gradients,
optimizer updates, progressive fine-tuning, validation or RNG state.

## 6. Local regression tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Tests use real production detector/head/encoder methods and small CPU fixtures;
CUDA/MMCV extensions and pretrained weights are not needed. They check config
preservation, numerical/RNG invariance, AP50/AP75 export, tied-score provenance,
proposal reconstruction, serialization, offline analysis and injected storage
failures. These tests do not claim full GPU/dataset training verification.

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
