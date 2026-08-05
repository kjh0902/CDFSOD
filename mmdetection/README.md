# CDFSOD Grounding DINO 사용법

이 디렉터리의 CDFSOD 설정은 textualized visual token을 사용하는 Grounding DINO
구조입니다. 평가를 시작할 때 support set을 고정 resize로 한 번 처리하고, 클래스별
모든 GT object에서 생성한 token을 캐시하여 모든 test image의 BERT 입력에
재사용합니다.

```bash
bash run_all_training.sh NEU-DET 1 1
```

인자는 순서대로 `DATASET`, `SHOT`, `GPU_COUNT`입니다. 상세 데이터 구조와 실험
설정은 저장소 루트의 `README.md`를 참고하세요.
