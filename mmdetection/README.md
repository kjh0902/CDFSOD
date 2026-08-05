# CDFSOD Grounding DINO 사용법

이 디렉터리의 CDFSOD 설정은 textualized visual token을 사용하는 Grounding DINO
구조입니다. 학습 중에는 매 iteration마다 고정 resize된 전체 support set의 모든 GT
object token을 다시 생성합니다. 평가에서는 최종 checkpoint로 같은 token을 한 번
생성해 캐시하고 모든 test image의 BERT 입력에 재사용합니다.

```bash
bash run_all_training.sh NEU-DET 1 1
```

인자는 순서대로 `DATASET`, `SHOT`, `GPU_COUNT`입니다. 상세 데이터 구조와 실험
설정은 저장소 루트의 `README.md`를 참고하세요.
