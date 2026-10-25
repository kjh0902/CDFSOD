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

## 4. Diagnostic files and offline reading

All data is in `<work-dir>/diagnostics/`. In distributed direct-runner use,
pass the relevant `diagnostics/rank_XXXX/` directory to the reader instead.

```text
diagnostics/
  manifest.json             # schema, exact config, commit, tensor conventions, status
  dataset.json              # original GT, crowd/ignore, image sizes, category-label map
  train.jsonl               # epoch/step/stage, every loss, LR groups, gradient norm
  metrics.jsonl             # standard COCO metrics plus structured per-class AP/AP50/AP75
  prompts/<hash>.json        # static prompts, token strings and class mappings shared across epochs
  epoch_001/
    index.jsonl             # run/epoch/image/rank/batch/chunk -> tensors
    batch_000000.npz         # bounded batch shards, compressed and pickle-free
    text/<content-hash>.npz  # actual equal BERT features deduplicated within an epoch
    text/<content-hash>.npz.json
    complete.json           # written only after all images and metrics are committed
```

Read/verify/analyze without torch, MMEngine, model weights, tokenizer or images:

```bash
python tools/analysis_tools/inspect_acl_diagnostics.py /path/to/work-dir/diagnostics --verify --analyze
```

The reader can also be loaded directly with `importlib.util.spec_from_file_location`
from `mmdet/utils/acl_diagnostic_io.py`; use `DiagnosticReader(directory)`,
`observations()`, `load(record)` and `proposals(arrays)`.

### Tensor conventions

- Each observation has image metadata, prompt chunk, class offset, one-based
  model token mapping, and a reference to the original GT snapshot. Dataset
  `class_index` and all query/candidate/detection indices are zero-based;
  `category_id` preserves the original COCO category ID.
- `bert_before_map`, `bert_after_map`, actual input token IDs, token strings,
  position IDs, valid-token and text attention masks are shared only when their
  stored contents and prompt metadata are identical in the same epoch.
- `fe_text[layer,token,channel]` is stored for all six enhancer layers and every
  image. Its final layer is `fe_text[-1]`.
- `decoder_token_logits[layer,query,token]`, `decoder_class_scores[layer,query,class]`
  and `decoder_boxes[layer,query,4]` cover all six serial layers. Only actual
  token columns are stored; omitted padded logit columns are -inf. Class score
  is the model's mean of sigmoid(token logits), not a mean-logit or softmax.
- `decoder_boxes[-1]` is the complete pre-postprocessing query prediction;
  it is not duplicated. Decoder/encoder boxes use normalized cxcywh with the
  same resized-image convention as the original prediction head.
- Final `det_boxes` are xyxy in original pixels and `det_scores` are float32.
  `det_query_index`, `det_class_index_local`, `det_class_index` and
  `det_flat_topk_index` are copied from actual postprocessing. Global class
  index = local index + prompt-chunk class offset. For mapped class prompts,
  flat index = query index * local class count + local class index.
  Ties and repeated query IDs with different classes are preserved exactly.
- Encoder arrays are `encoder_boxes[N,4]`, `encoder_class_scores[N,C]`,
  `selection_max_token_logit[N]` (float32), `selection_argmax_token[N]`,
  and `selected_indices[Q]`. Candidate order is unchanged, including invalid
  candidates. Masks are bit-packed with little bit order. The reader restores
  base anchors from `spatial_shapes`, `valid_hw` and `padding_mask` using
  recipe version 1 (0.05 * 2**level sizes, 0.01/0.99 validity bounds).
  Selected candidate rows are referenced by index rather than duplicated.
- Large float arrays use float16 + NPZ compression. No full encoder NxT token
  logit array, visual/query hidden state or attention weight matrix is saved.
  Masks are structural masks, not attention weights. There is no native
  objectness head; max class score is explicitly only an objectness proxy.
- Class AP/AP50/AP75 come from the same COCO precision array used by the existing
  evaluator, at area=all and its final maxDets setting. AP50/AP75 threshold
  positions are looked up by value. Additional scalar keys are
  `coco/bbox_class_<category_id>_AP`, `_AP50`, `_AP75`. Standard mAP/scheduler
  inputs are unchanged. Undefined metrics are JSON null with a reason.

`--analyze` demonstrates saved-data-only candidate/selected/final-query GT IoU
coverage, per-query class confusion, objectness proxies and BERT-to-FE token
cosines by epoch. Query confusion is best-IoU matching, not a replacement for
COCO's one-to-one detection evaluation. The reader exposes each epoch's BERT
features for temporal drift comparisons. The NPZ quantization means offline
recomputed scores/IoUs can differ slightly from full precision; saved final
scores, selected indices and detection origins retain their original values.

## 5. Disk capacity and failure behavior

The logger performs a conservative uncompressed capacity estimate before
training using model dimensions, validation resize bounds, dataset size and
all epochs. It recalibrates with actual compressed bytes during validation.
Capacity is rechecked at batch/epoch boundaries and before each file write.
The defaults reserve **10 GiB** plus **1.2 times** projected remaining data.
Preflight may intentionally reject a disk that could fit unusually compressible
data; it does not assume an optimistic compression ratio.

Configure `ACLDiagnosticHook` in the dataset/shot config when needed:

```python
dict(type='ACLDiagnosticHook', out_dir='/large-disk/acl-run/diagnostics',
     reserve_gib=10.0, capacity_margin=1.2)
```

The destination must not already exist. Storage errors or insufficient capacity
stop the run immediately; data is never silently sampled, dropped or pruned.
NPZ/JSON files are atomically renamed before index rows refer to them. A failed
run keeps a running manifest and any `.partial` file for diagnosis; it cannot
be mistaken for a completed epoch/run. Compression and I/O add runtime but do
not modify model inputs, outputs, gradients, parameter updates or RNG state.

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
