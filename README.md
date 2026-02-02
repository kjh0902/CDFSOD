# [NeurIPS 2025] Domain-RAG: Retrieval-Guided Compositional Image Generation for Cross-Domain Few-Shot Object Detection

[📚 Paper (NeurIPS 2025)](https://arxiv.org/abs/2506.05872) | [📂 Project Page](https://yuli-cs.net/papers/domain-rag) | [📦 Dataset Scripts](#dataset-preparation) | [🚀 Quick Start](#quick-start) | [🎥 Video](#video) | [✉️ Contact](#contact)

---

**Domain-RAG** is a novel retrieval-augmented generative framework designed for **Cross-Domain Few-Shot Object Detection (CD-FSOD)**. We leverage large-scale vision-language models (GroundingDINO), a curated COCO-style retrieval corpus, and Flux-based background generation to synthesize diverse, domain-aware training data that enhances FSOD generalization under domain shift.

<p align="center">
  <img src="assets/framework.svg" alt="DomainRAG Pipeline" width="700"/>
</p>

---

## ✨ Highlights

- 🔍 **Retrieval-Augmented Generation**: retrieve semantically similar source images for novel-class prompts.
- 🎨 **Flux-Redux Integration**: compose diverse backgrounds with target foregrounds for domain-aligned generation.
- 📦 **Support for Multiple Target Domains**: ArTAXOr, Clipart1k, DIOR, DeepFish, UODD, NEU-DET, and more.
- 🧪 **Strong Benchmarks**: surpasses GroundingDINO baseline in few-shot setting across CD-FSOD & RS-FSOD & CAMO-FS.

---

## 🔧 Installation

```bash
git clone https://github.com/LiYu0524/Domain-RAG.git
cd Domain-RAG
conda create -n domainrag python=3.10
conda activate domainrag
pip install -r requirements.txt
```
## Quick start 

You can refer to `./domainrag.sh`


## GroundingDINO Training

After completing the total generation stage of Domain-RAG, we provide a simple and reproducible pipeline for few-shot object detection training based on GroundingDINO.

### Step 1: Install the Environment

Please first refer to `README_mmlab.md` for detailed instructions on setting up the basic environment for mmGroundingDINO, including installing dependencies, configuring MMDetection, and preparing the required tools.

We also provide an `environment.yaml` file for convenience. However, we strongly recommend following the official installation instructions for GroundingDINO to ensure compatibility and avoid potential dependency conflicts.

### Step 2: Prepare Few-Shot Configurations

Before starting training, please make sure that all required few-shot configuration files have been generated and placed in the `configs/grounding_dino` directory. These files should follow the naming convention below:

For example:
- `CDFSOD_detection_few-shot_ArTaxOr_1shot.py`

If any required configuration file is missing, the training script will automatically terminate with an error.

### Step 3: Start Training

Training is automated using the script `auto_modify_swin_t_config.py`. This script will back up the original Swin-T configuration file, modify the `_base_` field to reference the corresponding few-shot configuration, set the maximum number of training epochs (default: 30), create a dedicated working directory for each dataset and shot setting, and finally launch the training process.

To start training, simply run:

```bash
python auto_modify_swin_t_config.py
```

## Video

Walkthrough video(Chinese version): [Watch here](https://www.bilibili.com/video/BV1YznKzkEEK/?spm_id_from=333.337.search-card.all.click&vd_source=23bede4ceb3dc1ea2ffc645933850555)

## Contact

For questions and collaboration, please contact:

- Yu Li : `<liyu24@m.fudan.edu.cn>`
- Xingyu Qiu :  `<xyqiu24@m.fudan.edu.cn>`
- Yuqian Fu :  `<yuqian.fu@insait.ai>`


## Citation

If you find **Domain-RAG** useful in your research, please cite:

```bibtex
@article{li2025domain,
  title={Domain-RAG: Retrieval-Guided Compositional Image Generation for Cross-Domain Few-Shot Object Detection},
  author={Li, Yu and Qiu, Xingyu and Fu, Yuqian and Chen, Jie and Qian, Tianwen and Zheng, Xu and Paudel, Danda Pani and Fu, Yanwei and Huang, Xuanjing and Van Gool, Luc and others},
  journal={arXiv preprint arXiv:2506.05872},
  year={2025}
}
```

If you find **CD-Vito** useful in your research, please cite:
```bibtex
@inproceedings{fu2024cross,
  title={Cross-domain few-shot object detection via enhanced open-set object detector},
  author={Fu, Yuqian and Wang, Yu and Pan, Yixuan and Huai, Lian and Qiu, Xingyu and Shangguan, Zeyu and Liu, Tong and Fu, Yanwei and Van Gool, Luc and Jiang, Xingqun},
  booktitle={European Conference on Computer Vision},
  pages={247--264},
  year={2024},
  organization={Springer}
}
```
