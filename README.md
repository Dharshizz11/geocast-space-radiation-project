# Space Radiation Forecasting: TPA-LSTM + Residual XGBoost

An end-to-end physics-informed machine learning pipeline for multi-horizon forecasting of high-energy (>2 MeV) electron flux at Geostationary Earth Orbit (GEO).

Forecasting space radiation environments around NOAA GOES satellites (**GOES-13, GOES-14, GOES-15**) driven by upstream solar wind and interplanetary magnetic field (IMF) measurements from NASA's **WIND** spacecraft. Predictions are cross-checked against ISRO's **GSAT / GRASP** radiation spectrometer payload.

---

## 🛰️ Key Highlights & Architecture

### 1. Dual-Stage Hybrid AI Engine: TPA-LSTM + Residual XGBoost
```
                   Wind Satellite (L1) + GOES 13, 14, 15 (GEO)
                  [IMF Bx, By, Bz, Vsw, Np, Tp, Proton/Electron Flux]
                                         │
                                         ▼
                   Physics-Informed Feature Engineering Layer
               [Newell Coupling, Akasofu ε, Pdyn, Clock Angle, MLT]
                                         │
                 ┌───────────────────────┴────────────────────────┐
                 │                                                │
                 │ (Historical Time Series: T=48)                 │ (Instantaneous Physics + Lags)
      ┌───────────────────────┐                        ┌─────────────────────┐
      │       TPA-LSTM        │                        │    Physics State    │
      │ (1D CNN Pattern Attn  │                        │   + Dynamic Trends  │
      │   over LSTM states)   │                        └─────────────────────┘
      └───────────────────────┘                                   │
                 │ ŷ_TPA-LSTM (Base Forecast)                     │
                 ├─────────────────────────────────┐              │
                 │                                 │              │
                 │                        ┌───────────────────────────────────┐
                 │                        │         Residual XGBoost          │
                 │                        │ (Predicts Error: e = y - ŷ_TPA)   │
                 │                        └───────────────────────────────────┘
                 │                                         │ ê_XGBoost (Residual Correction)
                 │                                         │
            ┌────┴─────────────────────────────────────────┴─────┐
            │              FINAL HYBRID PREDICTION:              │
            │     ŷ_final = ŷ_TPA-LSTM + ê_XGBoost (Residual)    │
            └──────────────────────┬─────────────────────────────┘
                                   │
           ┌───────────────────────┼─────────────────────┐
           │                       │                     │
    45-min Forecast         6-hour Forecast       12-hour Forecast
    (Immediate Risk)       (Orbit Transfer)      (Mission Planning)
```

- **TPA-LSTM**: 1D Convolutional filter banks applied across the rows of the LSTM hidden state matrix extract cross-time invariant wave patterns (CME shockwaves, diurnal solar magnetic rhythms) and dynamically weight them with attention.
- **Residual XGBoost**: Fits non-linear decision tree ensembles directly on the residual errors ($e = y - \hat{y}_{\text{TPA}}$) using instantaneous solar wind dynamic pressure $P_{dyn}$, Newell coupling, and local time to correct sudden storm commencement spikes.

### 2. Six-Model Benchmark Suite
Evaluated side-by-side at **45 minutes, 6 hours, and 12 hours**:
1. **Linear Regression** (Ridge Baseline)
2. **Standalone XGBoost** (Tabular GBDT Baseline)
3. **Standalone LSTM** (Standard Recurrent Sequence Baseline)
4. **Temporal Transformer** (Multi-Head Self-Attention Baseline)
5. **TPA-LSTM** (1D-CNN Temporal Pattern Attention)
6. **Proposed Hybrid** (TPA-LSTM + Residual XGBoost)

### 3. Space Weather Storm Stratification & Operational Skill Scores
Performance is reported across:
- **All Test Periods** (Overall Benchmark)
- **Geomagnetic Storm Periods** ($Dst < -50\text{ nT}$)
- **High-Flux Radiation Events** ($>10^3\text{ and }>10^4\text{ cm}^{-2}\text{s}^{-1}\text{sr}^{-1}$)
- Operational NOAA metrics: Probability of Detection (POD), False Alarm Ratio (FAR), Critical Success Index (CSI / Threat Score), Heidke Skill Score (HSS).

### 4. Zero-Shot Independent Cross-Validation (ISRO GSAT / GRASP)
Predictions are independently cross-checked against ISRO's GSAT / GRASP payload with physical instrument disparity notes.

---

## 💻 CLI Pipeline Execution

```bash
# 1. Clean raw CDF/CSV satellite data
python scripts/01_load_clean_raw.py

# 2. Resample to 5-min grid, engineer physics features & run anti-leakage checks
python scripts/02_sync_and_features.py

# 3. Train and benchmark all 6 models
python scripts/03_train_and_evaluate.py --epochs 20

# 4. Cross-validate against ISRO GSAT / GRASP data
python scripts/04_validate_grasp.py

# Run entire pipeline in one command:
python scripts/run_all.py
```

---

## 🚀 Launching the Interactive Operations Cockpit

```bash
streamlit run dashboard/app.py
```
Or open `dashboard/web/index.html` in any browser for the zero-dependency 3D orbital mission control viewer.
