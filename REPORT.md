# Technical Report: Predicting Shipment ETA and Delay Propagation Using AI
**Aiqon AI Internship — Take-Home Assignment (3 Days)**  
**Author:** Candidate Submission  
**Tech Stack:** Python 3, Scikit-Learn, Streamlit, Plotly, Pandas, NumPy, ReportLab  
**Date:** November 2026  

---

## 1. Executive Summary & Problem Formulation
In modern container logistics and maritime freight, accurate Estimated Time of Arrival (ETA) predictions are vital for schedule reliability, carrier coordination, terminal berth scheduling, and warehouse labor planning. However, static scheduled timetables routinely fail in practice. Delays are rarely static or isolated; instead, they exhibit **cascading, non-linear propagation dynamics** across multi-stage shipment lifecycles:

1. **Origin Departure Lag:** Port gate-in congestion, feeder barge delays, or documentation bottlenecks.
2. **Ocean Transit Disruption:** Weather/sea-state shocks, heavy swell detours, or engine throttling.
3. **Buffer Absorption:** Under modest delays ($\le 8\text{h}$), vessels can accelerate by 1–1.5 knots to recover schedule slack.
4. **Berth Window Miss Compounding:** Ports operate fixed berthing windows (e.g. 12-hour tide slots). If a vessel arrives past its designated window, it forfeits its slot and is forced into outer anchorage for an additional 14–48 hours, non-linearly amplified by port congestion.
5. **Port & Customs Clearance Dwell:** Container discharge bottlenecks, terminal yard dwell, and customs risk tier examinations (Green auto-clear vs Red physical inspection).
6. **Inland Final-Mile Delivery:** Intermodal rail/truck chassis availability and road traffic congestion.

### Mathematical Formulation
For a shipment journey with $K$ sequential stages, the scheduled timetable provides milestone times $t_k^{\text{sched}}$ and expected stage durations $D_k^{\text{sched}}$. The actual final arrival timestamp is:

$$T_{\text{actual}} = t_0^{\text{sched}} + \Delta_{\text{origin}} + \sum_{k=1}^K D_k^{\text{actual}}$$

The net delay magnitude $\Delta_{\text{total}}$ (in hours) is defined as:

$$\Delta_{\text{total}} = T_{\text{actual}} - T_{\text{sched}}$$

Delay likelihood classification is framed as:

$$Y_{\text{delay}} = \begin{cases} 1, & \text{if } \Delta_{\text{total}} > \tau_{\text{SLA}} \ (6.0\text{ hours}) \\ 0, & \text{otherwise} \end{cases}$$

---

## 2. Dataset Construction & Logistics Mechanics
To reflect authentic maritime operations, we engineered a realistic multi-stage synthetic logistics generator ($N = 500$ shipments) across 5 high-volume global trade corridors:
- **R1: Shanghai (CNSHA) $\to$ Rotterdam (NLRTM):** 10,500 nm, 664h scheduled transit.
- **R2: Ningbo (CNNGB) $\to$ Los Angeles (USLAX):** 5,700 nm, 396h scheduled transit.
- **R3: Singapore (SGSIN) $\to$ Hamburg (DEHAM):** 8,400 nm, 542h scheduled transit.
- **R4: Nhava Sheva (INNSA) $\to$ Felixstowe (GBFXT):** 6,200 nm, 440h scheduled transit.
- **R5: Shenzhen (CNSZX) $\to$ New York (USNYC):** 11,200 nm, 730h scheduled transit.

### Logistics Modeling Logic:
1. **Buffer Absorption Mechanics:** Realistic simulation where vessel operators can make up time during open ocean transit when origin delays are moderate.
2. **Berth Window Expiration:** Non-linear cliff penalty applied if cumulative arrival lag at destination port exceeds threshold, forcing outer anchorage dwell.
3. **Customs Risk Inspection Tails:** Auto-clear (Green, 70%, 1–4h), Document Audit (Yellow, 22%, 8–24h), and Physical Inspection (Red, 8%, 36–96h).
4. **Stochastic External Shocks:** Beta-distributed terminal congestion indices ($0.0 - 1.0$), ocean storm severity indices, and inland hinterland traffic levels.

---

## 3. Modeling Architecture & In-Flight Inference

### Addressing Incomplete / Missing Intermediate Data
In live logistics operations, shipments are queried mid-voyage. When a container is at sea, downstream customs and inland delivery times are unknown. Our engine solves this using **Milestone-Conditioned Cascade Inference**:

- **Milestone 0 (Booking / Pre-Departure):** Inferences conditioned on route distance, scheduled durations, carrier historical reliability, and port congestion forecasts.
- **Milestone 1 (Origin Departed):** Assimilates realized origin lag $\Delta_{\text{origin}}$ and terminal dwell.
- **Milestone 2 (Ocean Arrived / Anchorage):** Assimilates realized ocean transit time, storm shocks, and evaluates berth-window forfeiture indicator.
- **Milestone 3 (Customs Cleared):** Assimilates realized port terminal dwell and customs inspection channel.

### Predictive Engine
The system employs **Gradient Boosted Decision Trees (GBDT)** for continuous delay magnitude regression and a **Calibrated Random Forest Classifier** for delay probability estimation.

Key engineered features include:
- `weather_x_distance`: Non-linear storm exposure index.
- `port_congestion_x_dwell`: Terminal congestion interaction with scheduled dwell.
- `berth_risk_factor`: Discrete indicator for cumulative delay $>10\text{h}$ interacting with port congestion.
- `cum_delay_at_port` & `cum_delay_post_customs`: Running stage delay accumulation.

---

## 4. Quantitative Results & Benchmark Comparison
We conducted a 5-fold cross-validation benchmarking against standard logistics baselines:

| Model Candidate | MAE (Hours) | RMSE (Hours) | $R^2$ Score | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Static Schedule (Zero Delay)** | 23.44 | 32.66 | -0.352 | 43.2% | 0.000 | 0.000 | 0.000 | 0.500 |
| **Historical Route Average** | 22.03 | 27.87 | 0.015 | 59.2% | 0.606 | 0.803 | 0.691 | 0.601 |
| **Standard Linear Regression** | 15.44 | 19.19 | 0.533 | 69.8% | 0.733 | 0.736 | 0.735 | 0.792 |
| **Proposed Delay Propagation AI (Ours)** | **8.77** | **12.26** | **0.810** | **90.4%** | **0.873** | **0.972** | **0.920** | **0.980** |

### In-Flight Error Convergence (Uncertainty Shrinkage)
As shipments progress through milestones, prediction uncertainty shrinks dramatically:
- **Milestone 0 (Pre-Booking):** $\text{MAE} = 9.81\text{ hours}$
- **Milestone 1 (Origin Departed):** $\text{MAE} = 5.93\text{ hours}$
- **Milestone 2 (Ocean Arrived):** $\text{MAE} = 4.78\text{ hours}$
- **Milestone 3 (Customs Cleared):** $\text{MAE} = 3.42\text{ hours}$

---

## 5. Interpretability & Actionable Operational Guidance
Moving beyond black-box predictions, the engine provides:
1. **Local Factor Waterfall:** Quantifies exact hours added or subtracted by Origin Departure, Transit Weather, Berth Miss Penalty, Customs Dwell, and Buffer Recovery.
2. **Automated Diagnostic Narrative:** Generates human-readable summaries describing the primary root cause and secondary contributors.
3. **Actionable Mitigation Playbooks:** Recommends specific operational actions, including:
   - Requesting secondary off-peak berthing windows 48h in advance.
   - Electronic manifest pre-clearance to bypass terminal yard dwell.
   - Off-peak nocturnal inland drayage dispatch.

---

## 6. System Architecture & Streamlit UI
The solution provides an end-to-end interactive Streamlit application structured into 5 tabs:
1. **Fleet Operations Dashboard:** High-level KPIs, delay distribution, root-cause pie charts, trade lane delay variances.
2. **Live Journey & In-Flight ETA Predictor:** Interactive milestone simulator (0 to 3), ETA calculator, and stage cascade timeline.
3. **Delay Propagation & Root-Cause AI:** Interactive Plotly waterfall chart, factor attribution ranking, and diagnostic playbooks.
4. **Model Benchmark & Evaluation:** Comparative benchmark table, milestone error convergence curve, feature importances, and residual calibration plot.
5. **Dataset Explorer & Synthetic Simulator:** Raw data filtering, CSV export, and on-the-fly dataset generation.

---

## 7. Submission Checklist & Deliverables
- [x] Python backend with multi-stage delay propagation model.
- [x] Streamlit frontend with interactive UI, in-flight milestone simulator, and waterfall charts.
- [x] Synthetic dataset generator capturing authentic maritime logistics physics.
- [x] Quantitative evaluation against static timetables and linear baselines.
- [x] Formal 2-3 page Technical Report in Markdown (`REPORT.md`) and PDF (`Aiqon_ETA_Delay_Propagation_Technical_Report.pdf`).
- [x] Production instructions and run scripts (`run.sh`, `README.md`, `requirements.txt`).
