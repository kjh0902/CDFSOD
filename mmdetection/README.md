# CDFSOD Grounding DINO 사용법

이 디렉터리의 CDFSOD 설정은 일반 Grounding DINO 구조를 사용합니다.
데이터셋의 class name 목록이 일반 text prompt로 BERT에 전달되고, support 이미지는
few-shot 학습 샘플로만 사용됩니다.

```bash
bash run_all_training.sh NEU-DET 1 1
```

인자는 순서대로 `DATASET`, `SHOT`, `GPU_COUNT`입니다. 상세 데이터 구조와 실험
설정은 저장소 루트의 `README.md`를 참고하세요.
