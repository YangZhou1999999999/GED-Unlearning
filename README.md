# 🚀 GED-Unlearning

**Collapse-Resistant GAN Unlearning**

Official implementation of:

> *Retain What Matters, Forget What Hurts: Collapse-Resistant GAN Unlearning*

---

## 📌 Overview

GANs can memorize sensitive data, raising privacy concerns.  
Machine Unlearning aims to remove specific identities from trained models.

However, existing methods suffer from:

> ❗ **Retain Collapse** — retained samples drift toward the forget target, degrading generation quality.

---

## 💡 Our Method

We propose **GED-Unlearning**, which prevents retain collapse via:

- 🔹 **Gradient Filtering (GF)**  
  Avoids conflicting updates between *retain* and *forget*

- 🔹 **Identity Disentanglement (IDM)**  
  Separates identities in latent space using:
  - Learnable mask
  - Mutual information minimization
  - Orthogonality constraints

---

## ⚙️ Installation

```bash
conda create -n ged python=3.9
conda activate ged

pip install torch torchvision
pip install lpips facenet-pytorch pandas tqdm click
```


##🚀 Running Unlearning
```bash
python unlearn.py --exp experiment \
  --inversion goae \
  --inversion_image_path ./data/CelebAHQ/512 \
  --target extra \
  --target_d 30.0 \
  --local \
  --adj \
  --glob \
  --loss_local_mse_lambda 5e-2 \
  --loss_adj_mse_lambda 5e-2 \
  --loss_local_lpips_lambda 2.0 \
  --loss_local_id_lambda 0.2 \
  --loss_adj_lpips_lambda 1.5 \
  --loss_adj_id_lambda 1.5 \
  --loss_global_cos_sim_lambda 300 \
  --loss_global_mse_sim_lambda 200 \
  --loss_global_orth_lambda 0.01 \
  --loss_global_lambda 4.0 \
  --loss_glob_mse_lambda 0.002 \
  --loss_glob_mask_lambda 10 \
  --orthogonal True \
  --globa_extra True \
  --loss_global_batch 2 \
  --mask True \
  --loss_sparsity 0.5 \
  --use_filter True \
  --filter_start 0 \
  --filter_end 400 \
  --filter_layer False \
  --seed 0
```


##📊 Evaluation
```bash
1️⃣ ID Scores
python evaluate_id.py --exp experiment
2️⃣ FID Scores
python evaluate_fid.py --exp experiment
---

##📈 Results
✅ Effective identity removal
✅ Better image quality than baselines
✅ ~44% FID improvement
