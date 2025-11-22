<h1 align="center">Learning Protein-Ligand Binding in Hyperbolic Space</h1>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2508.15480-b31b1b.svg)](https://arxiv.org/abs/2508.15480)
[![AAAI 2026](https://img.shields.io/badge/AAAI-2026-blue.svg)](https://aaai.org/conference/aaai/aaai-26/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

<p align="center">
  <img src="figures/hypseek.jpg" width="700"/>
</p>

## 🚀 Quick Start

### Installization

Clone the repository:
```bash
git clone https://github.com/jianhuiwemi/HypSeek.git
cd HypSeek
```
Install OpenBabel:
```bash
apt-get install -y openbabel
```
Install Python packages:
```bash
pip install numpy scikit-bio==0.6.2 rdkit biopandas
```
Install Uni-Core:
```bash
git clone https://github.com/dptech-corp/Uni-Core.git
cd Uni-Core
python setup.py install
cd ..
```
Install ProDy:
```bash
git clone https://github.com/prody/ProDy.git
cd ProDy
python setup.py build_ext --inplace --force
pip install -Ue .
cd ..
```
We also provide a full Conda environment file (`environment.yml`) for users who encounter dependency issues.
### Data Preparation

## 📦 Dataset & Checkpoints

The HypSeek project provides complete training data, test datasets, and all pretrained / finetuned model checkpoints.

### 📂 Datasets

The following datasets are included in the project root directory:

- `data.zip` — Training dataset  
- `test_datasets.zip` — Testing / benchmark datasets  

After downloading, extract them under the project root:

```
HypSeek/
 ├── data/
 └── test_datasets/
```

---

### 🧩 Model Checkpoints

We release both pre-trained weights and downstream-task finetuned models.

#### 1. `pretrain.zip` — Pre-trained Checkpoints

Contains the pretrained model weights used before downstream finetuning. Extract under:

```
HypSeek/pretrain/
```

#### 2. Finetuned Checkpoints (`Checkpoints/` directory)

| File | Purpose |
|------|---------|
| `checkpoint_avg_41-50_rk.pt` | **Affinity Ranking** |
| `checkpoint_avg_41-50_vs.pt` | **Virtual Screening** |

## 🔧 Usage

### Training

AANet uses a **two-phase training strategy**:

#### Phase 1: Alignment (Representation Learning)
```bash
bash fpocket_neg_10A_siglip.sh
```

#### Phase 2: Aggregation (Pocket Selection)
```bash
bash finetune_chembl_fpocket_neg_10A_siglip_icrossatt_mollinear_wocollision_attn_kl.sh
```

> **Note**: Modify the **conda environment** and paths in the scripts, or run in the Unicore Docker environment.

### Evaluation

Run evaluation on different benchmarks:

```bash
# DUD-E benchmark
bash test_finetune_chembl_fpocket_neg_10A_siglip_icrossatt_mollinear_wocollision_attn_kl.sh <device_id> DUDE

# LIT-PCBA benchmark
bash test_finetune_chembl_fpocket_neg_10A_siglip_icrossatt_mollinear_wocollision_attn_kl.sh <device_id> PCBA
```

Results will be saved in the `./test` directory.

## 📈 Performance

AANet achieves state-of-the-art performance on multiple virtual screening benchmarks by effectively handling structural uncertainty in protein pockets.

## 📝 Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{zhu_aanet_2025,
    title = {{AANet}: {Virtual} {Screening} under {Structural} {Uncertainty} via {Alignment} and {Aggregation}},
    booktitle = {Proceedings of the Thirty-Ninth Annual Conference on Neural Information Processing Systems (NeurIPS 2025)},
    url = {https://openreview.net/forum?id=TUh4GDposM},
    author = {Zhu, Wenyu and Wang, Jianhui and Gao, Bowen and Jia, Yinjun and Tan, Haichuan and Zhang, Ya-Qin and Ma, Wei-Ying and Lan, Yanyan},
    year = {2025},
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Copyright (c) 2025 Institute for AI Industry Research (AIR), Tsinghua University**

Portions of this code are adapted from projects developed by DP Technology, licensed under the MIT License.

## 💐 Acknowledgments

This work builds upon [Uni-Mol](https://github.com/dptech-corp/Uni-Mol) and [Unicore](https://github.com/dptech-corp/Uni-Core). We thank the authors for their open-source contributions.

## 📧 Contact

For questions or collaborations, please contact:
- Wenyu Zhu: [GitHub](https://github.com/Wiley-Z)
- Yanyan Lan: [Homepage](https://air.tsinghua.edu.cn/en/info/1046/1194.htm)
