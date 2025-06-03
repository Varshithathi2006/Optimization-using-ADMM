# Optimization-using-ADMM

# 🔢 Applications of ADMM in Graphical Lasso Optimization and Image Super-Resolution

This project explores the use of the **Alternating Direction Method of Multipliers (ADMM)** for two advanced computational applications:
1. **Graphical Lasso Optimization** – For high-dimensional statistical modeling.
2. **Super-Resolution Imaging** – For enhancing low-resolution images using TV regularization.

> Course: 22AIE122 – Mathematics For Computing-2  
> Department: Artificial Intelligence  
> Institution: Amrita Vishwa Vidyapeetham

---

## 👩‍💻 Team Members

- **Varshitha Thilak Kumar** – CB.SC.U4AIE23258  
- **Siri Sanjana S** – CB.SC.U4AIE23249  
- **Shreya Arun** – CB.SC.U4AIE23253  
- **Anagha Menon** – CB.SC.U4AIE23212  
---

## 📌 Project Objectives

- Apply ADMM to solve the **Graphical Lasso** problem with better convergence and scalability.
- Utilize ADMM in **Image Super-Resolution** to reconstruct high-quality images from low-resolution inputs.
- Compare ADMM-based methods with traditional approaches (like bicubic interpolation).

---

## 🔍 Applications

### 1. 📈 Graphical Lasso Optimization with ADMM

- Estimates **sparse precision matrices** for high-dimensional data.
- Useful in financial modeling to detect inter-stock dependencies.
- ADMM splits the problem into sub-tasks and iteratively enforces:
  - Data fidelity
  - Sparsity via L1 penalty
  - Matrix positive definiteness

### 2. 🖼️ Super-Resolution Imaging

- Applies **Total Variation (TV) regularization** with ADMM.
- Converts low-resolution images to high-resolution output.
- Implemented in both **Python** and **MATLAB** for cross-validation.
- Preserves edges and fine details while suppressing noise.

---

## ⚙️ Technologies & Tools

- **Python (NumPy, Pandas, Matplotlib, skimage)**
- **MATLAB**
- Jupyter Notebook / MATLAB Live Script for iterative visualization

---

## 📊 Results Summary

- **Graphical Lasso**:
  - ADMM provided better sparsity enforcement and convergence compared to coordinate descent.
  - Achieved strong similarity with true precision matrix in synthetic stock data.

- **Super-Resolution**:
  - ADMM with TV produced visually sharper outputs than basic bicubic interpolation.
  - Effectively removed noise while maintaining critical features.

---

## 🧠 Key Concepts

- **ADMM**: Decomposes complex optimization problems into easier sub-problems with guaranteed convergence.
- **L1 Regularization**: Promotes sparsity in matrices.
- **Total Variation Denoising**: Reduces noise without blurring edges.
- **Soft-thresholding**: Used for enforcing sparsity during ADMM iterations.

---

## 📁 File Structure

admm-project/
├── graphical_lasso_admm.py
├── super_resolution_admm.py
├── super_resolution_admm.m
├── test_images/
│ └── test.png
├── plots/
├── README.md
└── mfc_report.pdf


---

## 📖 References

- Grechkin et al., Pathway Graphical Lasso, AAAI Conference, 2015.
- Zhu, Y., Augmented ADMM for Generalized Lasso, JCGS, 2017.
- Zhao, J. et al., Image super-resolution via adaptive sparse representation, KBS, 2017.

---

