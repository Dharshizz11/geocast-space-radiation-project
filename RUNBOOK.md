# Space Radiation Forecasting Runbook

### Quickstart Guide:
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute Full Pipeline:**
   ```bash
   python scripts/run_all.py
   ```

3. **Run Test Suite:**
   ```bash
   pytest tests/ -v
   ```

4. **Run Unified Ablation Study (Architecture & Physics Features):**
   ```bash
   python scripts/05_run_ablation_study.py --skip_existing
   # Or run with accelerated stride for quick iteration:
   python scripts/05_run_ablation_study.py --train_stride 6 --skip_existing
   ```

5. **Launch Streamlit Operations Cockpit:**
   ```bash
   streamlit run dashboard/app.py
   ```

6. **Standalone Browser Cockpit:**
   Open `dashboard/web/index.html` in your browser.
