# Aiqon | AI Shipment ETA & Delay Propagation Engine

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App%20Online-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end AI system designed for **Aiqon's AI Internship Take-Home Assignment**: *Predicting Shipment ETA and Delay Propagation Using AI*.

This platform models cascading delays across multi-stage maritime and intermodal container journeys, resolves incomplete/missing intermediate telemetry via dynamic in-flight inference, and delivers transparent root-cause explainability with actionable operational mitigation playbooks.

---

## Key Highlights

1. **Multi-Stage Delay Propagation Modeling:**
   - Models stage-to-stage delay transfers across:
     - Stage 1: **Origin Gate-In & Departure**
     - Stage 2: **Ocean Transit & Sea Swell / Weather Throttling**
     - Stage 3: **Destination Port Anchorage, Berthing & Customs Dwell**
     - Stage 4: **Inland Final-Mile Drayage Delivery**
   - Captures non-linear logistics dynamics:
     - **Buffer Absorption:** Vessels speeding up by 1–1.5 knots during transit to recover modest origin delays ($\le 8\text{h}$).
     - **Berth Window Expiration Penalties:** Compounding 14–48h anchorage dwell if late arrival exceeds designated 12h terminal window.
     - **Customs Heavy-Tail Risk:** Green (auto-clear), Yellow (document audit), and Red (physical exam) channel distributions.

2. **Dynamic In-Flight Inference (Addressing Incomplete Intermediate Data):**
   - Live shipments dynamically update their predicted ETA and delay likelihood as journey milestones are achieved:
     - **Milestone 0:** Booking / Pre-Departure
     - **Milestone 1:** Origin Departed (Origin lag realized)
     - **Milestone 2:** Ocean Arrived (Sea transit realized, berth window determined)
     - **Milestone 3:** Customs Cleared (Port & customs dwell realized)
   - MAE shrinks from **9.81 hours** at Milestone 0 down to **3.42 hours** at Milestone 3!

3. **Interpretable Root-Cause AI & Playbooks:**
   - **Plotly Waterfall Breakdown:** Visualizes exact additive variance contributed by each factor from scheduled baseline to final net ETA.
   - **Natural Language Diagnostics:** Automated operational narratives explaining the primary and secondary delay drivers.
   - **Operational Playbooks:** Prescriptive actions (e.g., requesting off-peak secondary berthing windows 48h prior, electronic manifest pre-clearance, night drayage).

4. **Production Streamlit Frontend:**
   - **Tab 1: Fleet Operations Dashboard** — Global KPIs, delay histograms, root cause distributions, route heatmap.
   - **Tab 2: Live In-Flight Predictor** — Milestone tracker (0-3), custom scenario builder, live ETA & risk gauge.
   - **Tab 3: Delay Propagation & Explainability** — Waterfall charts, factor attribution, operational playbooks.
   - **Tab 4: Benchmark & Evaluation** — Cross-validated comparisons against baselines, milestone error convergence curve, feature importances.
   - **Tab 5: Dataset Simulator & Explorer** — Searchable data grid, CSV exporter, on-the-fly synthetic data generator.

---

## Benchmark & Quantitative Evaluation

Evaluated across 5-fold cross-validation on 500 multi-stage shipment journeys:

| Model Candidate | MAE (Hours) | RMSE (Hours) | $R^2$ Score | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Static Schedule (Zero Delay Baseline)** | 23.44 | 32.66 | -0.352 | 43.2% | 0.000 | 0.000 | 0.000 | 0.500 |
| **Historical Route Average** | 22.03 | 27.87 | 0.015 | 59.2% | 0.606 | 0.803 | 0.691 | 0.601 |
| **Standard Linear Regression** | 15.44 | 19.19 | 0.533 | 69.8% | 0.733 | 0.736 | 0.735 | 0.792 |
| **Proposed Delay Propagation AI (Ours)** | **8.77** | **12.26** | **0.810** | **90.4%** | **0.873** | **0.972** | **0.920** | **0.980** |

---

## Quickstart Guide

### Prerequisites
- Python 3.10+ (tested on Python 3.10, 3.11, 3.12, and 3.14 on macOS / Linux / Windows)
- `pip` or virtual environment manager

### 1. One-Click Launch (macOS / Linux)
```bash
chmod +x run.sh
./run.sh
```
This script will automatically configure the virtual environment, install dependencies, verify the dataset, and launch the Streamlit frontend.

### 2. Manual Setup
```bash
# Clone or navigate to the project directory
cd aiqon

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Regenerate synthetic dataset
python data/generator.py

# Run model benchmark evaluation
python models/evaluation.py

# Launch Streamlit Frontend
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Repository Structure

```
aiqon/
├── data/
│   ├── generator.py                  # Realistic synthetic logistics journey generator
│   └── shipments_dataset.csv         # 500 multi-stage container shipment records
├── models/
│   ├── __init__.py                   # Package initializer
│   ├── delay_propagation.py          # Multi-stage cascade predictive engine
│   ├── explainability.py             # Factor attribution, waterfall & operational playbooks
│   └── evaluation.py                 # Multi-metric benchmark comparator (5-fold CV)
├── app.py                            # Full Streamlit frontend application
├── generate_report_pdf.py            # Automated ReportLab PDF generator script
├── Aiqon_ETA_Delay_Propagation_Technical_Report.pdf  # Formal 2-3 page PDF report
├── REPORT.md                         # Formal technical report in Markdown
├── requirements.txt                  # Python dependencies
├── run.sh                            # One-click startup script
└── README.md                         # Project documentation and guide
```

---

## Deliverables Summary for Aiqon Evaluation

1. **Code Repository:** Clean, modular, fully runnable Python backend and Streamlit frontend.
2. **Technical Report:**
   - PDF format: [`Aiqon_ETA_Delay_Propagation_Technical_Report.pdf`](Aiqon_ETA_Delay_Propagation_Technical_Report.pdf)
   - Markdown format: [`REPORT.md`](REPORT.md)
3. **Interactive Demo:** Streamlit web application with fleet dashboard, in-flight predictor, waterfall diagnostics, benchmark evaluations, and synthetic data generator.
4. **Submission Target:** `info@aiqon.in`
