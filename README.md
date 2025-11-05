<div align="center">
<img width="380" height="163" alt="image" src="https://github.com/user-attachments/assets/1de5f268-7f4f-4de4-abd6-73617ccddfa9" />
</div>
<h1 align="center">Interactive Thinking with Images</h1>

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/XXXX.XXXXX)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/datasets/We-Math/V-Thinker)
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

</div>

<h5 align="center">If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<div align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Orbitron&size=20&duration=3000&pause=1000&color=005DE3&center=true&vCenter=true&width=800&lines=Welcome+to+V-Thinker;Interactive+Thinking+with+Images;Powered+by+BUPT+x+Tencent+WeChat" alt="Typing Animation" />
</div>

---

## 📣 Latest News

- **[Nov 6, 2026]**: 🚀 V-Thinker codebase and datasets released!
- **[Nov 6, 2026]**: 📄 Our paper is now available on arXiv.
- **[Nov 6, 2026]**: 🎯 Introduced **VTBench**, an expert-verified benchmark for vision-centric interactive reasoning.

---

## 📂 Datasets

| Dataset | Description | Size | Download |
|---------|-------------|------|----------|
| **V-Interaction-400K** | Interactive reasoning with 25+ domains | 400K | [🤗 HuggingFace](https://huggingface.co/datasets/We-Math/V-Interaction-400K) |
| **V-Perception-40K** | Point-level perception alignment | 40K | [🤗 HuggingFace](https://huggingface.co/datasets/We-Math/V-Perception-40K) |
| **VTBench** | Expert-verified interactive benchmark | 1.5K | [🤗 HuggingFace](https://huggingface.co/datasets/We-Math/VTBench) |


---

## 💡 Overview

> *"The soul never thinks without an image." — Aristotle*

**V-Thinker** is a general-purpose multimodal reasoning assistant that enables **Interactive Thinking with Images** through end-to-end reinforcement learning. Unlike traditional vision-language models, V-Thinker actively **interacts** with visual content—editing, annotating, and transforming images to simplify complex problems.

<img width="682" height="299" alt="image" src="https://github.com/user-attachments/assets/ef4ddafe-a802-4216-b9d5-045d4b62c36f" />

---

## ✨ Key Features

### 🔄 Data Evolution Flywheel

Automated synthesis of high-quality interactive reasoning data across three dimensions:

- **Diversity**: Knowledge-driven synthesis from 25+ domains → **22,319 nodes** across 7 layers
- **Quality**: Coordinated checker-repairer mechanism for multi-modal consistency
- **Difficulty**: Progressive expansion via parallel & sequential strategies

**Outputs**: 
- 📊 **V-Interaction-400K**: Large-scale interactive reasoning dataset
- 🎯 **V-Perception-40K**: Point-level perception alignment dataset

---

### 📚 Visual Progressive Training Curriculum

Two-stage framework progressively building perception and interactive reasoning:

**Stage 1: Perception Alignment** → Fine-grained visual grounding with point-level supervision

**Stage 2: Interactive Reasoning** → Cold-start SFT + RL in sandboxed code executor

---

## 🎬 Interactive Image Examples

<img width="679" height="280" alt="image" src="https://github.com/user-attachments/assets/76d76528-1cad-4928-a716-7ad04bfe9f08" />

---

## 📊 VTBench Benchmark

Expert-verified benchmark with **1,500 QA pairs** across three hierarchical dimensions:

<img width="686" height="298" alt="image" src="https://github.com/user-attachments/assets/c853148c-5916-4614-9824-2c9096a75138" />

| Metric | Specification |
|--------|---------------|
| **Samples** | 1,500 expert-verified pairs (500 per task type) |
| **Sources** | 9 open-source benchmarks |
| **Domains** | Logical Reasoning, Geometry, Algebra, Statistics |


## 🚀 Quick Start

### Installation

```bash
conda create -n vthinker python=3.10
conda activate vthinker
pip install -r requirements.txt
```

### Training

```bash
# Perception Alignment
python src/train_perception.py --config_path ./config/base_config.yaml

# Interactive Reasoning (SFT + RL)
python src/train_interactive_sft.py --config_path ./config/base_config.yaml
python src/train_interactive_rl.py --config_path ./config/base_config.yaml
```

### Inference

```bash
# Run on VTBench
python src/run_vthinker.py --benchmark vtbench --eval

# Run on general benchmarks
python src/run_vthinker.py --benchmark mathvision --eval
```

---

## 🏆 Performance Results

### VTBench Results

| Model | Perception | Instruction-Guided | Interactive Reasoning |
|-------|------------|-------------------|----------------------|
| GPT-4o | 2.3 | 3.7 | 38.3 |
| InternVL3-78B | 10.8 | 16.0 | 43.4 |
| Qwen2.5-VL-7B | 9.6 | 8.8 | 32.2 |
| **V-Thinker-7B** | **18.0** (+8.4) | **34.6** (+25.8) | **41.8** (+9.6) |

### General Reasoning Benchmarks

| Model | MathVision | We-Math | VisuLogic |
|-------|------------|---------|-----------|
| Qwen2.5-VL-7B | 23.0 | 61.7 | 26.0 |
| **V-Thinker-7B** | **29.3** (+6.3) | **62.8** (+1.1) | **26.6** (+0.6) |

---

## 🔬 Case Studies
<img width="641" height="637" alt="截屏2025-11-06 00 54 13" src="https://github.com/user-attachments/assets/7f10bc54-ca02-4d63-9960-17af739f6fd3" />

<img width="578" height="131" alt="image" src="https://github.com/user-attachments/assets/de73c181-d413-4617-b775-08bdd4e6e624" />

<img width="585" height="711" alt="image" src="https://github.com/user-attachments/assets/35133170-ce70-41c0-891c-b82091aa6329" />


## 🔬 Evovled Knowledge System
<img width="589" height="374" alt="image" src="https://github.com/user-attachments/assets/07169a81-55ea-4841-83a0-ec00617dc3ad" />



---

## 📄 Citation

```bibtex
@article{vthinker2026,
  title={V-Thinker: Interactive Thinking with Images},
  author={Qiao, Runqi and Tan, Qiuna and Yang, Minghan and others},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

---

## 📞 Contact

**Email**: qrq@bupt.edu.cn, qiunatan@bupt.edu.cn  
**Issues**: [GitHub Issues](https://github.com/We-Math/V-Thinker/issues)

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

[![Star History Chart](https://api.star-history.com/svg?repos=We-Math/V-Thinker&type=Date)](https://www.star-history.com/#We-Math/V-Thinker&Date)

<div align="center"><b>⭐ Star us on GitHub to stay updated! ⭐</b></div>
