# Volve Field Time-to-Depth Prediction: Machine Learning vs. Geophysical Baseline

This repository implements a production-grade, modular Python pipeline designed to analyze geological well pick data from the Volve Field and predict **Two-Way Travel Time (TWT)**. The project directly explores the intersection of traditional geophysical physical formulations and predictive Machine Learning regression architectures.

## Dataset

This project uses well pick data from the publicly available Volve Field dataset. The dataset contains geological formation picks and associated depth measurements used for time-to-depth modeling and TWT prediction.

**Source:**
- Volve Well Picks (modified): https://raw.githubusercontent.com/yohanesnuwara/MLGeo/main/well_logs/Volve_well_picks_modified.csv

The dataset is loaded directly from the source URL during execution, ensuring reproducibility and consistency across experiments.

## Objectives

- Compare traditional geophysical approaches against Machine Learning models.
- Predict Two-Way Travel Time (TWT) from subsurface well data.
- Evaluate model accuracy, robustness, and generalization capabilities.
- Establish a reproducible workflow for geoscience-focused predictive modeling.

---

## 📌 Project Overview & Achievements

In this project, we executed a complete end-to-end subsurface data science workflow across two primary phases:

1. **Geophysical Data Auditing & Diagnostics (Part 1 Questions):**
   * Processed raw structural data, isolated target profiles, and validated well counts, formation lists, and structural depth boundaries (`TVDSS`).
   * Evaluated missing value boundaries. Instead of using generic statistical imputation, a strict **Deterministic Logging Strategy** was chosen to analyze physical sub-surface correlations, subsequently dropping rows with missing target labels to maintain 100% predictive modeling purity.
   
2. **Machine Learning vs. Physics-Based Benchmarking (Part 2 Evaluation):**
   * Handled categorical engineering of geological formation horizons (`PICKS`).
   * Trained and optimized three separate Machine Learning architectures: **Linear Regression**, **Random Forest Regressor**, and **XGBoost Regressor**.
   * Constructed an analytical baseline model derived from the fundamental geophysical equation:
     $$TWT = \frac{2 \cdot |Z|}{V_{field\_avg}}$$
     where $V_{field\_avg}$ is extracted dynamically from the training matrix to prevent data leakage.
   * Compared performance metrics ($\text{R}^2$, $\text{RMSE}$, $\text{MAE}$) and visually validated physical consistency via cross-plots.

---

## 🗂 Repository Architecture

The project has been split into clean, single-responsibility modules to follow modular software engineering best practices:

```text
├── data/
│   └── volve_well_picks_modified.csv     # Raw subsurface input dataset
├── data_processing.py                     # Phase 1: Data cleaning, data-drop & geophysical logging
├── eda.py                                 # Exploratory Data Analysis (distributions & correlations)
├── modeling.py                            # Phase 2: Encoding, splitting, ML training, & physical scoring
├── main.py                                # System orchestrator and main entry-point execution
├── ml_vs_physics_metrics.txt             # Exported performance metrics table
└── ml_vs_physics_benchmark.png            # Saved Requirement #6 Multi-Validation Cross-Plot
```


## 📊 Core Performance Summary
Upon running the predictive pipeline, all evaluation models are evaluated on the exact same unseen test slice. The models yielded the following benchmarking matrix:

The complete benchmarking results are available in:

- [ml_vs_physics_metrics.txt](./ml_vs_physics_metrics.txt)


```text
=== PHASE 2: PREDICTIVE MODELING TO PREDICT TWT ===
Evaluating Performance Matrix:
Model Name                | R2 Score   | RMSE (ms)  | MAE (ms)  
-----------------------------------------------------------------
Linear Regression         | 0.9988     | 30.0598    | 21.5482   
Random Forest Regressor   | 0.9984     | 34.0675    | 19.7087   
XGBoost Regressor         | 0.9991     | 25.9258    | 14.4412   
Physical Formula (2Z/V)   | 0.9779     | 126.6816   | 99.1032   
-----------------------------------------------------------------
```


## 🧠 Key Findings & Physical Coherence

* **Machine Learning Superiority:** The XGBoost Regressor yielded the highest predictive performance ($R^2 = 0.9991$), followed closely by Linear Regression and Random Forest.
* **Physical Alignment:** The machine learning algorithms naturally learned the spatial and stratigraphical anomalies across the field without any external hints, significantly outperforming the simplified uniform field-wide Average Velocity equation ($2Z/V$).
* **Automated Outputs:** Every single run automatically dumps both this structured performance matrix to a flat file (`ml_vs_physics_metrics.txt`) and generates a validation scatter plot (`ml_vs_physics_benchmark.png`) ensuring immediate availability for project submission reports.

---

## 🚀 How to Install and Run

### 1. Prerequisites & Dependencies
Ensure you have Python 3.8+ installed along with the essential data science packages. You can install all required dependencies via pip:

```bash
pip install -r requirements.txt
```

### 2. Execution
To trigger the complete pipeline—which will read the data, calculate and display the structured answers to Part 1 of the assignment, run full visual EDAs, and complete the benchmarking modeling phase—simply execute the main.py entry point:


```bash
python main.py
```


### 3. Reviewing Artifacts

After the pipeline finishes, open your project directory to extract your saved report components:
* **Inspect `ml_vs_physics_metrics.txt`** for your formatted performance evaluation table.
* **Inspect `ml_vs_physics_benchmark.png`** to view the cross-plot comparing model prediction paths against the perfect $Y=X$ line of fit.