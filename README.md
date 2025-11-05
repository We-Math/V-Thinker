<div align="center">
<img width="380" height="163" alt="image" src="https://github.com/user-attachments/assets/1de5f268-7f4f-4de4-abd6-73617ccddfa9" />
</div>

# <h1 align="center">Interactive Thinking with Images</h1>

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

- **[January 2026]**: 🚀 V-Thinker codebase and datasets released!
- **[January 2026]**: 📄 Our paper is now available on arXiv.
- **[January 2026]**: 🎯 Introduced **VTBench**, an expert-verified benchmark for vision-centric interactive reasoning.

---

## 💡 Overview

> *"The soul never thinks without an image." — Aristotle*

**V-Thinker** is a general-purpose multimodal reasoning assistant that enables **Interactive Thinking with Images** through end-to-end reinforcement learning. Unlike traditional vision-language models, V-Thinker actively **interacts** with visual content—editing, annotating, and transforming images to simplify complex problems.

### 🎯 Three Paradigms of Vision-Centric Reasoning

<div align="center">
  <img src="./figures/fig1.png" width="80%" />
</div>

1. **Direct Thinking**: Traditional reasoning without visual interaction
2. **Thinking with Images**: Using images to assist reasoning (e.g., o3)
3. **Interactive Thinking with Images** ⭐ (Ours): Actively modifying images during reasoning

---

## ✨ Key Features

### 🔄 Data Evolution Flywheel

Automated synthesis of high-quality interactive reasoning data across three dimensions:

<div align="center">
  <img src="./figures/fig4.png" width="95%" />
</div>

- **Diversity**: Knowledge-driven synthesis from 25+ domains → **22,319 nodes** across 7 layers
- **Quality**: Coordinated checker-repairer mechanism for multi-modal consistency
- **Difficulty**: Progressive expansion via parallel & sequential strategies

**Outputs**: 
- 📊 **V-Interaction-400K**: Large-scale interactive reasoning dataset
- 🎯 **V-Perception-40K**: Point-level perception alignment dataset

---

### 📚 Visual Progressive Training Curriculum

Two-stage framework progressively building perception and interactive reasoning:

<div align="center">
  <img src="./figures/fig6.png" width="85%" />
</div>

**Stage 1: Perception Alignment** → Fine-grained visual grounding with point-level supervision

**Stage 2: Interactive Reasoning** → Cold-start SFT + RL in sandboxed code executor

---

## 🎬 Interactive Reasoning Examples

<table>
  <tr>
    <td width="50%">
      <img src="./figures/example_geometry.png" width="100%"/>
      <p align="center"><b>Geometry with Auxiliary Lines</b></p>
    </td>
    <td width="50%">
      <img src="./figures/example_counting.png" width="100%"/>
      <p align="center"><b>Visual Counting & Labeling</b></p>
    </td>
  </tr>
  <tr>
    <td colspan="2">
      <img src="./figures/fig11-cot.png" width="100%"/>
      <p align="center"><b>Complete Interactive Reasoning Trajectory: Think → Edit → Verify</b></p>
    </td>
  </tr>
</table>

---

## 📊 VTBench Benchmark

Expert-verified benchmark with **1,500 QA pairs** across three hierarchical dimensions:

<div align="center">
  <img src="./figures/fig7.png" width="95%" />
</div>

```
Perception → Instruction-Guided Interaction → Interactive Reasoning
```

| Metric | Specification |
|--------|---------------|
| **Samples** | 1,500 expert-verified pairs (500 per task type) |
| **Sources** | 9 open-source benchmarks |
| **Domains** | Logical Reasoning, Geometry, Algebra, Statistics |

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

## 📂 Datasets

| Dataset | Description | Size | Download |
|---------|-------------|------|----------|
| **V-Interaction-400K** | Interactive reasoning with 25+ domains | 400K | [🤗 HuggingFace](https://huggingface.co/datasets/We-Math/V-Interaction-400K) |
| **V-Perception-40K** | Point-level perception alignment | 40K | [🤗 HuggingFace](https://huggingface.co/datasets/We-Math/V-Perception-40K) |
| **VTBench** | Expert-verified interactive benchmark | 1.5K | [🤗 HuggingFace](https://huggingface.co/datasets/We-Math/VTBench) |

---

## 🔬 Key Insights

<table>
  <tr>
    <td width="50%">
      <img src="./figures/tree_fan_final.png" width="100%"/>
      <p align="center"><b>Knowledge System Evolution</b><br/>22,319 nodes across 25 domains</p>
    </td>
    <td width="50%">
      <img src="./figures/zhexian.png" width="100%"/>
      <p align="center"><b>Scaling Analysis</b><br/>~50× expansion after 5 iterations</p>
    </td>
  </tr>
</table>

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

## 🤝 Related Work

> [**DeepAgent**](https://github.com/RUC-NLPIR/DeepAgent) - General reasoning agent with scalable toolsets [![Stars](https://img.shields.io/github/stars/RUC-NLPIR/DeepAgent.svg?style=social)](https://github.com/RUC-NLPIR/DeepAgent)

> [**We-Math**](https://github.com/We-Math/We-Math) - Large-scale visual math benchmark [![Stars](https://img.shields.io/github/stars/We-Math/We-Math.svg?style=social)](https://github.com/We-Math/We-Math)

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
