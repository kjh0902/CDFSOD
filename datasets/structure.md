# Datasets Folder Structure

This baseline expects NEU-DET in COCO format:

```text
datasets/
  NEU-DET/
    annotations/
      train.json
      test.json
    train/
    test/
  split.py
  kshot_split.py
```

`split.py` creates `train.json` and `test.json`.
`kshot_split.py` creates files such as `1_shot.json`, `5_shot.json`, and
`10_shot.json` from `train.json`.
