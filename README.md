# DS-3002 Data Mining — Assignment #4

## Heartbeat to Heatmap: Unsupervised Learning, Ensemble Methods, and Neural Networks

**Student:** Saif Shahzad  
**Roll No:** 23i-2634  
**Section:** DS-A  
**Date:** May 2026

---

## 📋 Assignment Overview

This assignment explores advanced machine learning techniques applied to heart disease prediction using the UCI Cleveland Heart Disease dataset (303 patients, 14 clinical features). The work spans five major components:

1. **Part A: Unsupervised Learning** — K-Means clustering, hierarchical clustering, PCA, t-SNE
2. **Part B: Ensemble Methods** — Random Forest, XGBoost with SHAP explainability, comparison with Naive Bayes
3. **Part C: Neural Networks** — Single-layer perceptron (SLP), multi-layer perceptron (MLP) with ablation study
4. **Part D: Convolutional Neural Networks** — MNIST digit recognition with filter visualization
5. **Part E: Clinical Dashboard** — Streamlit web app for risk prediction and interpretability

---

## 🎯 Key Results

| Model         | Accuracy   | AUC-ROC    | Disease Recall | Notes                                                   |
| ------------- | ---------- | ---------- | -------------- | ------------------------------------------------------- |
| **XGBoost**   | **0.9231** | **0.9387** | **0.9167**     | ⭐ Best performance; SHAP explainability                |
| MLP           | 0.9231     | 0.9302     | 0.9167         | Competitive; requires careful tuning                    |
| Random Forest | 0.8974     | 0.9286     | 0.9167         | Robust; good feature importances                        |
| Naive Bayes   | 0.8462     | 0.9103     | 0.9167         | Baseline; independence assumption violated              |
| SLP           | 0.7692     | 0.8077     | —              | Linear model; insufficient for non-linear relationships |
| CNN (MNIST)   | 0.9745     | —          | —              | Outperforms MLP baseline (0.9635) by 1.1%               |

**Deployment Recommendation:** XGBoost is recommended for production screening—highest AUC-ROC, high disease recall (minimizes missed diagnoses), native SHAP support for clinical interpretability.

---

## 📁 Project Structure

```
i23-2634-DM-Assignment4/
├── README.md                          ← You are here
├── COMPILATION_GUIDE.md               ← LaTeX compilation instructions
├── processed.cleveland.data           ← UCI Heart Disease dataset (303 rows, 14 cols)
│
├── notebook/
│   └── i232634_DS-A_Assignment 4_Notebook.ipynb  ← Main analysis (49 cells, all executed)
│
├── report/
│   └── report.pdf                     ← Compiled output (generated via pdflatex)
│
└── app/
    ├── app.py                         ← Streamlit dashboard
    ├── requirements.txt               ← Dashboard dependencies
    ├── model.pkl                      ← Serialized XGBoost model
    ├── scaler.pkl                     ← Feature scaler artifact
    ├── feature_columns.pkl            ← Feature column names
    └── sample_patient.json            ← Example patient for UI pre-population
```

---

## 🚀 Quick Start

### Option 1: Run the Complete Notebook

#### Prerequisites

```bash
# Python 3.8+
python --version

# Jupyter
pip install jupyter
```

#### Setup & Execution

```bash
# Navigate to workspace
cd "d:\University Work\Semester 6\DM\Assignments\4\i23-2634-DM-Assignment4"

# Install dependencies (inside notebook)
# First cell: !pip install xgboost shap imbalanced-learn lightgbm streamlit tensorflow -q

# Launch Jupyter and execute notebook
jupyter notebook notebook/i232634_DS-A_Assignment\ 4_Notebook.ipynb

# Execute all cells (Kernel → Restart & Run All)
# All 49 cells will run successfully, generating 23 PNG files in images/
```

**Expected Output:**

- ✅ 23 PNG images in `images/` directory
- ✅ Models trained and evaluated on test set
- ✅ Model artifacts saved to `app/`
- ✅ Summary table with all metrics printed

**Runtime:** ~10–15 minutes on CPU (GPU recommended for CNN cells)

---

### Option 2: Generate LaTeX Report

After running the notebook (which creates the `images/` folder), compile the LaTeX report:

```bash
cd report/

# Single-pass compilation
pdflatex report.tex

# OR multi-pass (recommended for references)
pdflatex report.tex
pdflatex report.tex
```

**Output:** `report/report.pdf` (300+ page academic document with all figures integrated)

---

### Option 3: Launch Interactive Dashboard

```bash
cd app/

# Install Streamlit dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run app.py

# Opens at http://localhost:8501
```

**Features:**

- Real-time XGBoost prediction for patient input
- SHAP feature importance visualization
- Risk confidence score (0–100%)
- Clinical interpretation in plain English

---

## 📖 File Descriptions

### Notebook: `notebook/i232634_DS-A_Assignment 4_Notebook.ipynb`

**Structure:** 49 cells organized into sections

| Section                 | Cells | Purpose                                                            |
| ----------------------- | ----- | ------------------------------------------------------------------ |
| Setup & Imports         | 1–4   | Load libraries, set random seeds                                   |
| Preprocessing           | 5–14  | Load data, handle missing values, encode, standardize, apply SMOTE |
| Part A: Unsupervised    | 15–30 | K-Means, hierarchical clustering, PCA, t-SNE, ARI metrics          |
| Part B: Ensemble        | 31–47 | Random Forest, XGBoost, SHAP, ROC curves, model comparison         |
| Part C: Neural Networks | 48–65 | SLP, MLP, 5-fold CV, ablation study                                |
| Part D: CNN             | 66–78 | MNIST data prep, CNN training, filter visualization, feature maps  |
| Part E: Dashboard       | 79–81 | Model serialization, Streamlit app code                            |
| Summary                 | 82    | Final results table                                                |

**Key Execution Notes:**

- Dependencies auto-installed in cell 2–3
- Images directory created automatically in cell 5 (`os.makedirs('images', exist_ok=True)`)
- All plotting cells include `savefig()` before `show()`
- SMOTE applied on training split only (prevents data leakage)
- Early stopping used in neural network training to prevent overfitting

---

### Report: `report/report.tex`

**Content:** 75KB LaTeX document with 7 main chapters

| Chapter                          | Pages | Content                                                                           |
| -------------------------------- | ----- | --------------------------------------------------------------------------------- |
| 1. Introduction                  | 2–3   | Dataset overview, preprocessing steps, correlation analysis                       |
| 2. Part A: Unsupervised Learning | 4–8   | K-Means results (ARI=0.41), hierarchical clustering, PCA (90% var in 6 PC), t-SNE |
| 3. Part B: Ensemble Methods      | 9–15  | RF grid search, XGBoost tuning, SHAP explainability, ROC comparison               |
| 4. Part C: Neural Networks       | 16–20 | SLP limitations, MLP architecture search, 5-fold CV, ablation results             |
| 5. Part D: CNNs                  | 21–24 | MNIST preprocessing, CNN vs MLP baseline, filter visualization                    |
| 6. Part E: Clinical Dashboard    | 25–26 | Streamlit architecture, deployment instructions                                   |
| 7. Summary                       | 27–28 | Results table, recommendations for deployment                                     |
| Appendix A: Math                 | 29–32 | K-Means objective, Silhouette, ARI, XGBoost, SHAP, CNN formulas                   |
| Bibliography                     | 33    | 10 academic references                                                            |

**Features:**

- 23 figures with captions and cross-references
- Tables with all metrics (accuracy, AUC, confusion matrices)
- Code snippets in lstlisting environments
- Professional formatting: headers, table of contents, hyperlinks

**Compilation:**

```bash
pdflatex report.tex   # Generate PDF with proper cross-references
```

See [COMPILATION_GUIDE.md](COMPILATION_GUIDE.md) for troubleshooting.

---

### Dashboard: `app/app.py`

**Purpose:** Streamlit web interface for clinical decision support

**User Inputs:**

- 14 patient features: age, sex, CP type, BP, cholesterol, max HR, ST depression, vessel count, etc.

**Outputs:**

- Binary prediction (disease present / absent)
- Risk score (0–100%)
- Confidence percentage
- Top 3 SHAP-based driving features (bar chart)
- Plain-English clinical interpretation

**Example Usage:**

```python
# Input: A 65-year-old male with chest pain type 2, thalach=120, oldpeak=2.8
# Output:
#   🔴 Disease Present — Confidence: 87.3%
#   Top 3 drivers: thalach, oldpeak, cp
#   Interpretation: "ST depression + reduced max HR → refer for stress echo"
```

---

## 📊 Data Dictionary

**UCI Cleveland Heart Disease Dataset** (303 patients → 297 after cleaning)

| Feature     | Type        | Range   | Description                                        |
| ----------- | ----------- | ------- | -------------------------------------------------- |
| age         | Continuous  | 29–77   | Patient age in years                               |
| sex         | Binary      | 0, 1    | 0=Female, 1=Male                                   |
| cp          | Categorical | 0–3     | Chest pain type (0=typical angina, 3=asymptomatic) |
| trestbps    | Continuous  | 94–200  | Resting blood pressure (mmHg)                      |
| chol        | Continuous  | 126–564 | Serum cholesterol (mg/dl)                          |
| fbs         | Binary      | 0, 1    | Fasting blood sugar > 120 mg/dl                    |
| restecg     | Categorical | 0–2     | Resting ECG results                                |
| **thalach** | Continuous  | 71–202  | **Max heart rate achieved (top predictor)**        |
| exang       | Binary      | 0, 1    | Exercise-induced angina                            |
| **oldpeak** | Continuous  | 0–6.2   | **ST depression (2nd top predictor)**              |
| slope       | Categorical | 0–2     | ST segment slope                                   |
| ca          | Continuous  | 0–3     | Number of major vessels (0–3)                      |
| thal        | Categorical | 1–3     | Thallium stress test result                        |
| **target**  | Binary      | 0, 1    | 0=No disease, 1=Disease present                    |

**Preprocessing Applied:**

1. Missing values: Dropped 6 rows with '?' markers → 297 retained
2. Target binarization: Levels 1–4 → 1 (disease present)
3. Encoding: One-hot for 4 categorical features → 37 features
4. Scaling: StandardScaler on 6 continuous features (fit on train only)
5. SMOTE: Applied on training split (50–50 class balance)
6. Split: Stratified 80/20 (157 train / 39 test after SMOTE)

---

## 🔧 Reproducibility Guide

### Full Reproduction (All Components)

```bash
# 1. Install dependencies
pip install jupyter pandas numpy scikit-learn xgboost shap imbalanced-learn \
            tensorflow tensorflow-keras matplotlib seaborn scipy streamlit joblib

# 2. Run notebook to generate images
cd notebook
jupyter notebook i232634_DS-A_Assignment\ 4_Notebook.ipynb
# Execute all cells (Kernel → Restart & Run All)
# Wait ~15 minutes for completion
cd ..

# 3. Verify images directory
ls -la images/ | wc -l  # Should show 24 items (. + .. + 23 PNG files)

# 4. Compile LaTeX report
cd report
pdflatex report.tex
pdflatex report.tex
cd ..

# 5. Launch dashboard (optional)
cd app
streamlit run app.py
# Navigate to http://localhost:8501
```

**Expected Outputs:**

- ✅ `images/01_class_distribution.png` through `23_mlp_ablation_study.png`
- ✅ `report/report.pdf` (300+ pages)
- ✅ Streamlit app running on `localhost:8501`

### Verify Results Match Assignment

**Check notebook execution:**

```python
# In notebook cell 82, the summary table should show:
# XGBoost Accuracy: 0.9231, AUC: 0.9387
# MLP Accuracy: 0.9231, AUC: 0.9302
# CNN MNIST Accuracy: 0.9745
```

**Check report generation:**

```bash
# In report/, verify PDF exists and contains all chapters
file report.pdf  # Should show "PDF document, version 1.4"
pdfinfo report.pdf | grep Pages  # Should show "Pages: 33" or similar
```

**Check dashboard:**

```bash
# In app/, verify all artifacts exist
ls -1 model.pkl scaler.pkl feature_columns.pkl sample_patient.json
# Should list 4 files without errors
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'xgboost'"

```bash
pip install xgboost shap tensorflow imbalanced-learn -q
# Then restart notebook kernel
```

### "FileNotFoundError: processed.cleveland.data"

Ensure the dataset file exists in the workspace root. If missing, download from:

- [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- Place as `processed.cleveland.data` in workspace root

### "Images directory not created"

The notebook cell:

```python
os.makedirs('images', exist_ok=True)
```

...automatically creates the directory. If it fails, manually create:

```bash
mkdir images
```

### "pdflatex: command not found"

Install LaTeX distribution:

- **Windows**: [MiKTeX](https://miktex.org/)
- **macOS**: [MacTeX](https://www.tug.org/mactex/)
- **Linux**: `sudo apt-get install texlive-full`

### "Images not found in LaTeX"

Ensure `images/` directory is at workspace root (not inside `report/`). The LaTeX file references:

```latex
\includegraphics[width=0.85\textwidth]{../images/01_*.png}
```

See [COMPILATION_GUIDE.md](COMPILATION_GUIDE.md) for detailed LaTeX troubleshooting.

---

## 📚 Key References

1. **UCI Heart Disease Dataset** — Dua & Graff (2019)
2. **XGBoost** — Chen & Guestrin (2016) — _Proceedings of KDD 2016_
3. **SHAP** — Lundberg & Lee (2017) — _Advances in NeurIPS_
4. **Dropout Regularization** — Srivastava et al. (2014) — _Journal of Machine Learning Research_
5. **CNNs** — LeCun, Bengio & Hinton (2015) — _Nature_
6. **t-SNE** — Van der Maaten & Hinton (2008) — _JMLR_

Full bibliography in `report/report.tex` (references.bib not used; embedded in document)

---

## 💡 Clinical Notes

**Why Disease Recall Matters:**

- **False Negative (missed disease):** Patient has heart disease but is not screened → delayed treatment → MI or death
- **False Positive (unnecessary test):** Patient is disease-free but flagged for follow-up → harmless extra test

Therefore, **maximizing recall for the disease class is clinically critical**, even at the cost of extra false positives.

**XGBoost Advantages for Clinical Deployment:**

1. **High recall:** 91.7% disease detection (only 1 false negative in test set)
2. **Explainability:** SHAP values enable cardiologists to understand individual predictions
3. **Speed:** <10 ms per prediction; no GPU required
4. **Robustness:** Regularization prevents overfitting on small datasets
5. **Interpretability:** Feature importances align with clinical knowledge (thalach, oldpeak, cp)

**Limitations:**

- Trained only on Cleveland cohort; may not generalize to other populations
- Should be used as decision-support tool, not autonomous diagnosis
- Regular retraining recommended as new patient data accumulates

---

## 📝 Assignment Checklist

- [x] **Part A: Unsupervised Learning** — K-Means, hierarchical, PCA, t-SNE
- [x] **Part B: Ensemble Methods** — RF, XGBoost, SHAP, comparison table
- [x] **Part C: Neural Networks** — SLP, MLP, 5-fold CV, ablation study
- [x] **Part D: CNNs** — MNIST, filter visualization, feature maps
- [x] **Part E: Clinical Dashboard** — Streamlit app with SHAP explainability
- [x] **Jupyter Notebook** — 49 cells, all executed, image outputs saved
- [x] **LaTeX Report** — 7 chapters, 23 figures, mathematical appendix, bibliography
- [x] **Reproducibility** — This README with step-by-step instructions

---

## 📞 Contact & Submission

**Student:** Saif Shahzad (23i-2634, Section DS-A)  
**Course:** DS-3002 Data Mining  
**Submission Date:** May 2026

**Files to Submit:**

1. ✅ `notebook/i232634_DS-A_Assignment 4_Notebook.ipynb`
2. ✅ `report/report.pdf` (or `report/report.tex` + `images/` folder)
3. ✅ `README.md` (this file)
4. ✅ `app/app.py` (optional; for demonstration)

---

**Last Updated:** May 6, 2026
