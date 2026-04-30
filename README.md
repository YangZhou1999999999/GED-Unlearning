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
