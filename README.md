# Kaplan-Scaling-Replications-AtharvaPandey

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Google Colab](https://img.shields.io/badge/Colab-F9AB00.svg?style=for-the-badge&logo=googlecolab&logoColor=white)
![LaTeX](https://img.shields.io/badge/LaTeX-008080.svg?style=for-the-badge&logo=LaTeX&logoColor=white)

A rigorous replication study of Kaplan et al. (2020) "Scaling Laws for Neural Language Models" conducted for the CSOC'26 NLP Capstone.

## 📝 Project Overview
This repository contains a full, from-scratch implementation of a decoder-only autoregressive Transformer. The project validates core neural scaling theory by empirical analysis of how cross-entropy loss ($L$) scales with respect to model parameters ($N$) and training data volume ($D$).

## 📈 Empirical Findings
My experiments on the **Tiny Shakespeare** corpus yielded the following power-law exponents:

| Metric | Exponent / Ratio |
| :--- | :--- |
| **Parameter Exponent ($\alpha_N$)** | 0.0774 |
| **Data Exponent ($\alpha_D$)** | 0.1329 |
| **Scaling Ratio ($\gamma$)** | **0.5825** |

## 📂 Repository Structure
```text
├── report/
│   └── main.pdf           # Detailed research paper
├── src/
│   ├── model.py           # Core Transformer architecture
│   ├── train_scaling.py   # Training sweep configuration
│   └── plot_scaling.py    # Regression & visualization logic
└── notebooks/
    └── replication.ipynb  # Interactive development environment
