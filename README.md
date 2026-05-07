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
---

## 🚀 Running Unlearning
```bash
python unlearn.py --exp experiment \
  --inversion goae \
  --inversion_image_path ./data/CelebAHQ/512 \
  --target extra \
  --target_d 30.0 \
  --local \
  --adj \
  --glob \
  --orthogonal True \
  --globa_extra True \
  --mask True \
  --use_filter True \
  --filter_layer False \
```
---

## 📊 Evaluation
```bash
1️⃣ ID Scores
python evaluate_id.py --exp experiment
2️⃣ FID Scores
python evaluate_fid.py --exp experiment
```
---

## 📈 Results
- ✅ Better image quality than baselines
- ✅ ~44% FID improvement
