# Serial ACL + GT region–text contrastive loss

기준: `grounding_dino_acl`의 `8926970ebff1a549088b0a4c87c272e1a70fe0dd`.
브랜치: `acl-serial-visual-text-contrastive`.

## 모델과 손실

기본 `GroundingDINO` / `GroundingDINOHead`를 사용한다. Decoder는 앞 layer의
query와 갱신된 reference를 다음 layer에 전달하는 원본 serial 구현이며,
DN query는 batch당 한 번 생성하고 추론에는 마지막 decoder layer를 사용한다.
HED 전용 detector/head/layer 파일, 등록 이름, 사용하지 않던 parallel decoder
클래스들은 삭제했다. 기본 serial encoder/decoder layer와 detection head는
기준 commit의 구현을 유지한다.

`L_total = L_detection + lambda_region_text * L_region_text`

- Visual: 최종 FE `memory`에 proposal masking → `memory_trans_fc` →
  `memory_trans_norm`을 적용한 **동일한 `output_memory` tensor**를 사용한다.
  이 tensor가 language-guided query selection의 classification branch에도 입력된다.
- `spatial_shapes`와 `level_start_index`로 모든 spatial level을 복원한다.
  각 GT box의 크기 `s = sqrt(width * height)`로 FPN level을 선택한다.
  `k = floor(4 + log2(s / 224))`를 사용 가능한 P3~P6으로 제한한다.
  stride `(8,16,32,64)`에서 크기 구간은 각각 `<224`, `[224,448)`,
  `[448,896)`, `>=896` pixel이다. 구현은 경계 안정성을 위해 log2 입력에
  작은 epsilon을 더한다. 선택된 **한 level에서만 3×3 RoIAlign**
  (`aligned=True`, sampling ratio 2)과 공간 평균을 수행하며 level 평균은 없다.
  결과는 원래 GT 순서를 유지한 **GT 객체마다 [256] 벡터 하나**다.
  같은 class의 여러 객체를 합치지 않는다.
- Box는 현재 `batch_data_samples.gt_instances.bboxes`의 augmented/resized xyxy
  pixel 좌표를 그대로 사용한다. `scale_factor`나 원본 크기로 재변환하지 않는다.
  Swin-B/ChannelMapper의 실제 stride `(8,16,32,64)`를 사용하므로 padding 또는
  feature 크기의 올림 때문에 `img_shape` 비율로 box를 늘이는 문제를 피한다.
- Text: 최종 FE `memory_text`에서 class-name span의 token만 raw mean한다.
  기존 ACL mapping을 dataset의 **모든 class**에 대해 만들고 GT label을 적용한다.
  class 누락, 비어 있는 span, 잘린 prompt 또는 padding token은 오류로 처리한다.
- `logits[i,c] = visual[i] @ text[c]`. 모든 dataset class에 대해 cross-entropy를
  계산한다. 이미지에 없는 class도 negative다. L2 normalization, temperature,
  projection head, class balancing은 없다.
- 모든 객체의 loss를 합산하여 객체 수로 나눈다. DDP에서는 전 rank의 GT 수를
  분모에 반영하고 DDP gradient averaging을 보정한다. 이미지별/class별 평균이 아니다.
- RoIAlign, raw mean 및 dot product는 FP32로 계산해 AMP overflow를 줄인다.
  dtype 변환은 gradient를 유지하며 양쪽 feature 경로에 detach를 적용하지 않는다.
- Empty GT batch는 양쪽 graph에 연결된 0을 반환한다. **FISH는 단일 class이므로
  softmax CE가 항상 0**이다. 이것은 요청한 objective의 결과이며 임의의 negative를
  추가하지 않는다.

추가 모델, 다른 experimental auxiliary loss, prototype transformation은 없다.
학습 parameter와 checkpoint key는 추가하지 않는다.

## 설정과 실행

18개 `configs_cdfsod/final_configs_bs4/*.py`에 다음 값을 적용했다.

```python
model = dict(
    type='GroundingDINO',
    lambda_region_text=0.01,
    region_text_roi_size=3,
    region_text_featmap_strides=(8, 16, 32, 64),
    bbox_head=dict(type='GroundingDINOHead', num_classes=num_classes))
```

초기값 0.01은 raw dot-product 보조 손실의 작은 시작 가중치이며 튜닝된 값이 아니다.
로그의 `loss_region_text`는 **가중치가 적용된** 항이다. Backbone/neck의 stride를
변경하면 `region_text_featmap_strides`도 맞춰야 한다. Stride는 양수이며
level마다 두 배가 되는 순서여야 한다. 기준 scale은 stride 16에서 224 pixel이다.

```bash
# Region-text experiment
python tools/train.py configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_NEU-DET_1shot.py --work-dir work_dirs/serial_region_text

# Serial + Progressive FT baseline
python tools/train.py configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_NEU-DET_1shot.py --work-dir work_dirs/serial_baseline --cfg-options model.lambda_region_text=0
```

0일 때는 all-class mapping, RoIAlign, 보조 loss 및 추가 head payload를 생성하지 않는다.
Inference에는 가중치와 무관하게 보조 경로가 없다.

`BBoxHeadFirstHook6`는 Stage 1에서 **모든 parameter group의 설정된 LR을 유지**한다.
기존에 LR=0으로 동결하던 `other` group(Feature Enhancer/encoder,
`memory_trans_*` 등)도 Stage 1부터 학습하며, `requires_grad=False`를 설정하지 않는다.
Language-model LR이 원래 값의 절반 이하로 내려가면 Stage 2로 전환하고
`other` group에 현재 head LR을 적용하는 기존 제어 흐름은 유지한다.
Validation plateau scheduler와 patience(이 설정에서는 Stage 1: 3, Stage 2: 8),
optimizer, augmentation, validation 및 checkpoint 설정도 유지한다.
`patience_frozen`/`patience_unfrozen` 이름은 config 호환성을 위해 유지한다.
Epoch 기반 대안 `StageWiseFreezeHook`에도 동일하게 Stage 1 동결 제거를 적용했다.

## 검증

```bash
python -m unittest discover -s tests -v
```

로컬 PyTorch 2.14.0 CPU / torchvision 0.29.0 CPU에서 **21개 테스트 통과**.
MMEngine 0.10.7로 18개 config의 상속/파싱도 확인했다.

수치 테스트는 실제 torchvision RoIAlign forward/backward를 사용해 좌표, FPN 경계와
clamping, 비정방형 box, 객체당 한 level 선택, GT 순서, 미사용 level의 호출 생략,
선택된 level/image로만 흐르는 gradient, 객체 단위 평균, absent-class negative,
class-name raw mean, 양쪽 gradient,
AMP, empty GT, 단일 class, 잘린 prompt와 DDP 분모를 검사한다.

통합 회귀는 기준 commit의 독립된 원본 serial detector 메서드와 현재 구현을
동일한 weight/RNG/input으로 실행한다. 실제 proposal projection, serial decoder
반복 로직, detection-head forward를 사용하되 encoder/attention/tokenizer와 loss
reduction은 소형 fixture로 대체한다. 공유 prompt/서로 다른 prompt/명시적 span,
empty/nonempty GT에서 `lambda=0`의 head 입력·출력, fixture loss 및 RNG가 bitwise
일치한다. 활성화된 보조 경로의 정확한 tensor 연결과 gradient, 실제 Progressive FT
hook의 Stage 1 FE/projection 실제 SGD 갱신, 모든 그룹의 trainability,
Stage 2 전환 및 patience 유지도 검사한다. 원본 detection head의 소스와 config의
나머지 설정이 바뀌지 않았다는 검사도 포함한다.

**로컬에는 MMCV/CUDA 실행 환경이 없어 실제 전체 모델의 detection loss 회귀나
학습/mAP는 실행하지 않았다.** 전체 모델용 검증은 정상 학습 환경에서 아래처럼
실행할 수 있다. 데이터셋 대신 synthetic image/GT를 사용하지만 model, pretrained
checkpoint 및 실제 detection loss는 config 그대로 사용한다. optimizer step은 없다.

```bash
python tools/analysis_tools/check_region_text_baseline.py configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_NEU-DET_1shot.py --checkpoint checkpoints/grounding_dino_swin-b_pretrain_all-f9818a7c.pth --device cuda:0
```

이 스크립트는 같은 모델의 weight, buffer 및 RNG를 복원하며 기준 commit의 serial
메서드와 현재 `lambda=0` 메서드로 실제 detection 출력과 모든 detection loss key를
bitwise 비교한다. 비교 루틴 자체는 CPU fixture 테스트에 포함되어 있다.
기준 commit을 읽기 때문에 Git history가 있는 checkout에서 실행해야 한다.
