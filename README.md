<h1 align="center">🫀 ECG-Soup: Harnessing Multi-Layer Synergy for ECG Foundation Models</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2509.00102"><img src="https://img.shields.io/badge/arXiv-2509.00102-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/Xuanphu108/ecg_ssl"><img src="https://img.shields.io/badge/Code-GitHub-181717.svg?logo=github" alt="GitHub"></a>
  <img src="https://img.shields.io/badge/PhysioNet%20Challenge%202025-🥇%201st%20Place-ffd700.svg" alt="PhysioNet Challenge 2025 - 1st Place">
</p>

<p align="center">
  <b>📄 Paper:</b> <a href="https://arxiv.org/abs/2509.00102">ECG-Soup: Harnessing Multi-Layer Synergy for ECG Foundation Models</a> (arXiv:2509.00102)
</p>

This is the official code repository accompanying my work on **ECG-Soup: Harnessing Multi-Layer Synergy for ECG Foundation Models**.  
The project provides implementations for several self-supervised and supervised training paradigms on multi-source ECG datasets.

---

## 📦 Installation & Setup

### 1. Environment Setup
```bash
# Create conda environment
conda env create -f environment.yml

# Activate the environment
conda activate ecg_ssl

# Install additional dependencies (if needed)
pip install pandas==1.4.1
```

### 2. Data Preparation

This project uses five publicly available 12-lead ECG datasets. Download each dataset from its official source and arrange it under `./datasets/ecg_data/` using the **exact folder names** expected by `source/data_preprocessing.py`.

#### 2.1 Download the raw datasets

| Dataset | Source | Target folder (`./datasets/ecg_data/`) |
|---------|--------|----------------------------------------|
| **PTB-XL** | [physionet.org/content/ptb-xl/1.0.1](https://physionet.org/content/ptb-xl/1.0.1/) | `ptb_xl/` |
| **CinC 2020** | [physionet.org/content/challenge-2020/1.0.2](https://physionet.org/content/challenge-2020/1.0.2/) | `cinc2020/` |
| **Zheng (Chapman–Shaoxing)** | [figshare.com/collections/ChapmanECG/4560497](https://figshare.com/collections/ChapmanECG/4560497/2) | `zheng2020/` |
| **Ribeiro (CODE-test)** | [zenodo.org/records/3765780](https://zenodo.org/records/3765780) | `ribeiro2020_test/` |
| **CODE-15%** | [zenodo.org/records/4916206](https://zenodo.org/records/4916206) | `code_15/` |

```bash
# PTB-XL (~3 GB)
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.1/

# CinC 2020 — PhysioNet/CinC Challenge 2020 (~7.5 GB); the ECG data lives in the training/ folder.
# NOTE: the /sources/ folder on PhysioNet contains teams' submitted code, NOT the ECG data.
wget -r -N -c -np https://physionet.org/files/challenge-2020/1.0.2/

# Ribeiro CODE-test (~218 MB)
wget https://zenodo.org/records/3765780/files/data.zip

# CODE-15% (~46 GB): exams.csv + exams_part0..17.zip
wget https://zenodo.org/records/4916206/files/exams.csv
for i in $(seq 0 17); do wget "https://zenodo.org/records/4916206/files/exams_part${i}.zip"; done
```

#### 2.2 Expected raw directory layout

```
./datasets/ecg_data/
  ├─ ptb_xl/                 
  │   ├─ ptbxl_database.csv
  │   ├─ scp_statements.csv
  │   └─ records100/         
  ├─ cinc2020/               
  │   ├─ dx_mapping_scored.csv
  │   ├─ dx_mapping_unscored.csv
  │   ├─ Dx_map.csv          
  │   ├─ ICBEB2018/         
  │   ├─ ICBEB2018_2/        
  │   ├─ INCART/             
  │   ├─ PTB/                
  │   ├─ PTB-XL/             
  │   └─ Georgia/            
  ├─ zheng2020/              
  │   ├─ Diagnostics.xlsx
  │   └─ ECGData/            
  ├─ ribeiro2020_test/       
  │   ├─ ecg_tracings.hdf5
  │   ├─ attributes.csv
  │   └─ annotations/        
  └─ code_15/                
      ├─ exams.csv
      └─ hdf5/               
```

> **CinC folder renaming.** PhysioNet ships the Challenge 2020 sources with different folder names. Rename them to match `prepare_data_cinc`:
> `cpsc_2018 → ICBEB2018`, `cpsc_2018_extra → ICBEB2018_2`, `st_petersburg_incart → INCART`, `ptb → PTB`, `ptb-xl → PTB-XL`, `georgia → Georgia`.
>
> **CODE-15.** Unzip every `exams_part{i}.zip` into a single `code_15/hdf5/` folder and keep `exams.csv` at the `code_15/` root.

#### 2.3 Preprocess

Run the preprocessing script to resample every dataset to **100 Hz** and export a unified memory-mapped format:

```bash
cd source
python data_preprocessing.py
```

The processed datasets are written to `./datasets/ecg_data_processed/`:

```
./datasets/ecg_data_processed/
  ├─ ptb_xl_fs100/
  ├─ cinc_fs100/
  ├─ zheng_fs100/
  ├─ ribeiro_fs100/
  └─ code_15_fs100/
```

---

## 🧪 Training Pipelines

This repository includes multiple training approaches:

### A0. 🧠 **Supervised Training (from scratch)**

```bash
sh scripts/supervised_from_scratch.sh
```

---

### A1. 🩺 **Masked Vision Transformer**

1. **Pretraining**  
   ```bash
   sh scripts/pretraining_masked_transformer.sh
   ```
2. **Linear probing**  
   ```bash
   sh scripts/linear_probing_masked_transformer.sh
   ```
3. **Fine-tuning**  
   ```bash
   sh scripts/finetuning_masked_transformer.sh
   ```

---

### A2. 🔁 **Contrastive Predictive Coding (CPC)**

1. **Pretraining**  
   ```bash
   sh scripts/pretraining_cpc.sh
   ```
2. **Linear probing**  
   ```bash
   sh scripts/linear_probing_cpc.sh
   ```
3. **Fine-tuning**
   ```bash
   sh scripts/finetuning_cpc.sh
   ```
4. **Supervised training from scratch**  
   ```bash
   sh scripts/cpc_from_scratch.sh
   ```

---

### A3. 🌟 **SimCLR**

1. **Pretraining (XResNet1D-50 backbone)**  
   ```bash
   sh scripts/pretraining_simclr.sh
   ```
2. **Linear probing**  
   ```bash
   sh scripts/linear_probing_simclr.sh
   ```
3. **Fine-tuning**  
   ```bash
   sh scripts/finetuning_simclr.sh
   ```

> **NOTE:** Fine-tuning is performed using the checkpoint obtained from the linear probing stage.
---

## 📊 Datasets

| Dataset    | Leads | Sampling rate | Description                      |
|------------|-------|---------------|-----------------------------------|
| PTB-XL     | 12    | 100 Hz        | Physionet clinical ECG dataset   |
| CINC       | 12    | 100 Hz        | CinC Challenge dataset           |
| Zheng      | 12    | 100 Hz        | ECG dataset from China          |
| Ribeiro    | 12    | 100 Hz        | CODE-15 test     |
| CODE-15    | 12    | 100 Hz        | CODE-15 dataset from Brazil          |

✅ All datasets are **resampled to 100 Hz** and preprocessed into a **unified structure** for downstream tasks.

---

## 🧠 Methods Implemented

- **Supervised training** 
- **Masked Vision Transformer**
- **CPC (Contrastive Predictive Coding)**
- **SimCLR**
- Linear probing and fine-tuning strategies
- Multi-dataset pretraining support

---

## 🧾 Acknowledgements & References

This work builds upon and is inspired by the following projects:

- **[ecg-selfsupervised](https://github.com/tmehari/ecg-selfsupervised)**  
  by *Temesgen Mehari & Nils Strodthoff* — licensed under **GPL-v3**.

- **[ST-MEM: Spatio-Temporal Masked Electrocardiogram Modeling](https://github.com/bakqui/ST-MEM)**  
  by *VUNO Inc.* (Na, Yeongyeon; Park, Minje; Tae, Yunwon; Joo, Sunghoon) — © VUNO Inc. All rights reserved.  
  Use/modification/distribution only with explicit permission from VUNO.

---

## ✨ Citation

If you find this work or code helpful in your research, please cite our paper:

> **ECG-Soup: Harnessing Multi-Layer Synergy for ECG Foundation Models**  
> Phu X. Nguyen, Huy Phan, Hieu Pham, Christos Chatzichristos, Bert Vandenberk, Maarten De Vos.  
> *arXiv preprint arXiv:2509.00102*, 2025. [[Paper]](https://arxiv.org/abs/2509.00102)

```bibtex
@article{nguyen2025ecgsoup,
  title   = {ECG-Soup: Harnessing Multi-Layer Synergy for ECG Foundation Models},
  author  = {Nguyen, Phu X. and Phan, Huy and Pham, Hieu and Chatzichristos, Christos and Vandenberk, Bert and De Vos, Maarten},
  journal = {arXiv preprint arXiv:2509.00102},
  year    = {2025},
  url     = {https://arxiv.org/abs/2509.00102}
}
```

---

## 📬 Contact

If you have questions or would like to collaborate, feel free to reach out:

- **Author:** Phu X. Nguyen  
- **Email:** [phu.nguyen@kuleuven.be](mailto:phu.nguyen@kuleuven.be) · [nxphu.1994@gmail.com](mailto:nxphu.1994@gmail.com)  
- **Affiliation:** Stadius Center, ESAT, KU Leuven, Belgium

---