# Antenatal-High-Risk-Pregnancy-Classification

Code and derived summary tables for the study:

**Discrepancy Between Machine Learning Performance and Clinical Validity in Antenatal Risk Classification: A Guideline-Based Stepwise Analysis**

Kaan Arik, Metin Yildiz, Burak Aggul

---

## Overview

This repository contains the analysis code and derived summary tables supporting the conclusions of the manuscript. The study audits a 998-record antenatal care (ANC) register from Bangladesh to examine how the health-worker-assigned "high-risk pregnancy" label relates to recorded clinical findings and standard danger signs.

Key findings:

- The risk label follows a near-deterministic **period-3 row-order cycle** (99.90% lag-3 match, *p* < 10⁻¹⁵).
- **HBsAg positivity** shows complete separation: all 109 positive entries were labeled high-risk.
- Tree ensembles reached **AUROC ≈ 0.99** under patient-name-grouped cross-validation, but this discrimination was driven by **weight, TT dose, and normotensive blood pressure**—variables whose row-order periods are multiples of 3—rather than by obstetric findings.
- Agreement between the register label and a nine-item clinical danger-sign checklist was near chance (**κ = 0.04** with VDRL; **0.10** without).

The results show that high discrimination on this register reflects record structure and coding artefacts, not clinical risk.

---

## Repository Contents

| File / Folder | Description |
|---|---|
| `data/` | De-identified analytical dataset (names replaced by SHA-256 hashes at ingestion). |
| `code/` | Python scripts for data parsing, integrity audits, periodicity analysis, Firth regression, and predictive modeling. |
| `tables/` | Derived summary tables reported in the manuscript (Tables 2–7). |
| `figures/` | Figure-generation scripts and output figures (Figures 1–9). |
| `requirements.txt` | Python dependencies (NumPy, SciPy, etc.). |

> **Note:** Raw names were removed from all analytical datasets and outputs. The dataset analyzed in the study is publicly available at IEEE DataPort: https://doi.org/10.21227/ddfa-mf77

---

## Methods Summary

- **Data integrity audit:** missingness, constant variables, duplicates, name repetition, and within-name age consistency.
- **Row-order periodicity:** exact lag-*k* match rate for *k* = 1–49, Pearson autocorrelation, and the Wald–Wolfowitz runs test.
- **Statistical analysis:** Mann–Whitney U, Welch's *t*, χ²/Fisher exact tests, Cramér's V, and variance inflation factors.
- **Multivariable model:** Firth bias-reduced penalized logistic regression (complete separation on HBsAg).
- **Predictive modeling:** L2 logistic regression, CART, EBM-style GA2M, Random Forest, and XGBoost-style GBDT under 5 × 5 repeated GroupKFold nested cross-validation.
- **Explainability and uncertainty:** Monte Carlo Shapley values, partial dependence plots, split conformal prediction (90% coverage), and decision curve analysis.
- **Clinical concordance:** Cohen's κ between the register label and a nine-item WHO-based danger-sign checklist.
- **Reporting:** STROBE and TRIPOD+AI checklists (Appendix A of the manuscript).

---

## Reproducibility

- Fixed random seed: **42**
- All models implemented in self-contained NumPy/SciPy code (no external ML libraries for the main models).
- Sanity controls: shuffled-label Random Forest (AUROC ≈ 0.53) and feature-exclusion sensitivity analyses.

To reproduce the main results:

```bash
pip install -r requirements.txt
python code/run_all.py
