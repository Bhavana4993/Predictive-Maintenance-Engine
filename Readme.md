# ⚙️ Industrial Predictive Maintenance & Downtime Cost Estimator

An end-to-end, physics-informed machine learning pipeline and interactive Streamlit web application designed to predict mechanical equipment failure modes and optimize preventive maintenance schedules based on asymmetric operational costs.

---

## 📌 Business & Financial Objective

Standard machine learning models evaluate predictions using symmetric classification metrics (Accuracy/F1-Score), treating all errors equally. In industrial manufacturing, cost asymmetry is severe:

| Operational Outcome | Classification Event | Financial Impact |
| :--- | :--- | :--- |
| **Preventive Maintenance** | True Positive / False Positive | **$1,200** (Scheduled inspection & part replacement) |
| **Unscheduled Downtime** | False Negative | **$10,000** (Catastrophic failure & production stoppage) |
| **Normal Operation** | True Negative | **$0** (No intervention required) |

By optimizing the decision probability threshold against this financial loss matrix, this engine minimizes total expected operational cost rather than just raw prediction error.

---

## 🛠️ Key System Features

* **Physics-Informed Feature Engineering:** Derives thermodynamic and kinematic variables ($\Delta T$ Heat Dissipation, Mechanical Power Output in $kW$, and Overstrain Load Factor) directly from raw telemetry metrics.
* **Class Imbalance Handling:** Trains an **XGBoost Classifier** configured with dynamic positive weight scaling (`scale_pos_weight`) to capture rare failure events (3.39% baseline incidence).
* **Cost-Aware Threshold Tuning:** Sweeps probability decision boundaries from $0.01$ to $0.99$ to identify the global minimum operational risk point.
* **Interactive Streamlit Dashboard:** Provides real-time telemetry control sliders, failure probability scoring, net risk mitigation savings calculations, and root-cause failure diagnostics.

---

## 🔬 Telemetry & Engineered Features

The dataset incorporates 10,000 operational records based on the **UCI AI4I 2020 Predictive Maintenance Dataset**:

* **Raw Telemetry:** Air Temperature ($K$), Process Temperature ($K$), Rotational Speed ($RPM$), Torque ($Nm$), and Tool Wear ($min$).
* **Engineered Variables:**
  * $\Delta T = T_{\text{process}} - T_{\text{air}}$ *(Heat Dissipation Capability)*
  * $\text{Power } (kW) = \tau \cdot \left(\text{RPM} \cdot \frac{2\pi}{60}\right) / 1000$ *(Mechanical Load Window)*
  * $\text{Overstrain Factor} = \tau \cdot \text{Tool Wear}$ *(Shear Stress Risk)*

---

## 📂 Repository Structure

```text
Predictive_Maintenance_Engine/
│── data/
│   └── predictive_maintenance.csv      # Raw operational telemetry data
│── artifacts/
│   ├── model.pkl                       # Trained XGBoost Classifier model
│   └── threshold.pkl                   # Cost-optimized decision threshold
│── notebooks/
│   └── 01_eda_and_cost_modeling.ipynb  # EDA, physics feature engineering & cost optimization
│── app.py                              # Interactive Streamlit Web Application
└── requirements.txt                    # Project dependencies