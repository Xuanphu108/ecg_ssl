# 🫀 ECG-Soup: Harnessing Multi-Layer Synergy for ECG Foundation Models
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
Follow the instructions in `data_preprocessing.py` to download and preprocess the ECG datasets.  
The expected directory structure after preprocessing:

```
./datasets/
  ├─ ecg_data_processed/
    ├─ cinc/
    ├─ zheng/
    ├─ ribeiro/
    ├─ code_15/
    └─ ptb_xl/
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