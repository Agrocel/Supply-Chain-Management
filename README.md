# Mahalaabh SCM Project

## 🚀 Project Overview

**Mahalaabh SCM Project** is a Python-based supply chain management system that **optimizes inventory, forecasts demand, and automates data workflows**.

**Key benefits:**
- Predict product demand using historical data
- Reduce overstock and stockouts
- Automate data cleaning, processing, and model building
- Generate actionable insights with visual reports
- CI/CD

---

## 🏗️ Project Modules & Data Flow

The project follows a **modular workflow**:


**Module Details:**

### 1. Raw Data Ingestion
- Collect data from Excel, CSV, or databases
- Stored in `data/raw/`

### 2. Data Cleaning
- Handle missing values, duplicates, and inconsistencies
- Stored in `data/clean/`

### 3. Data Processing
- Transform, aggregate, and prepare data for modeling
- Stored in `data/processed/`

### 4. Feature Engineering
- Create derived variables (moving averages, lag features, seasonal indicators)
- Stored in `data/features/`

### 5. Model Building
- Train forecasting models (Prophet, LightGBM)
- Hybrid modeling with residuals if needed
- Scripts in `Source/Model/`

### 6. Evaluation
- Calculate metrics like MAE, RMSE
- Reports in `Output/Evaluation/`

### 7. Forecast Outputs
- Save forecasts, charts, and visualizations as HTML
- Stored in `Output/Forecasts/` & `Output/Visualizations/`

---

## 📁 Folder Structure

data/
├─ raw/ # Original unprocessed data
├─ clean/ # Cleaned & validated data
├─ processed/ # Processed datasets
└─ features/ # Feature-engineered datasets

Source/
├─ DataCleaning/ # Cleaning scripts
├─ Processing/ # Processing scripts
├─ Model/ # Model building scripts
└─ Evaluation/ # Evaluation scripts

Output/
├─ Forecasts/ # Forecast results
├─ Visualizations/ # Charts & plots
└─ Evaluation/ # Evaluation metrics & reports


## 🧠 Models Used & Future Plans

**Current Base Models:**
- **Time Series Model** – For trend and seasonality forecasting
- **Prophet Model** – Handles multiple seasonalities and holidays

**Future Enhancements:**
- Integrate **CI/CD pipelines** for automated model deployment
- Include **Deep Learning (DL) models** for advanced forecasting
- Add **Machine Learning (ML) models** for hybrid prediction

---
