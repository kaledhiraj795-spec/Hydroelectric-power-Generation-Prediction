# Hydroelectric Dam Power Generation Prediction ⚡

An end-to-end Machine Learning web application predicting dam electrical power generation (MW) based on hydraulic and operational sensor parameters using regularized Ridge Regression.

## 📌 Project Overview
Hydroelectric power plants rely on turbine efficiency and fluid dynamics. This project models the complex relationships between hydraulic head, water flow rate, reservoir water level, turbine gate opening, and ambient water temperature to forecast active megawatt generation.

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Statsmodels, Matplotlib, Seaborn
- **Framework & Deployment:** Streamlit
- **Model:** Ridge Regression ($L_2$ Regularization)

## 📊 ML Workflow
1. **Exploratory Data Analysis (EDA):** Analyzed correlation patterns across physical dam sensors.
2. **Multicollinearity Diagnostics (VIF):** Addressed significant multicollinearity between reservoir water level and hydraulic head ($VIF > 10$).
3. **Data Preprocessing:** Standardized inputs using `StandardScaler` to ensure balanced penalty attribution across features.
4. **Regularization Comparison:** Compared Multiple Linear Regression baseline against Ridge ($L_2$) and Lasso ($L_1$), validating with 5-Fold Cross-Validation.
5. **Deployment:** Packaged the trained pipeline with Streamlit for real-time turbine output estimation.

## 🚀 Running Locally

1. Clone repository:
```bash
git clone [https://github.com/kaledhiraj795-spec/Hydroelectric-power-Generation-Prediction.git](https://github.com/kaledhiraj795-spec/Hydroelectric-power-Generation-Prediction.git)
cd Hydroelectric-power-Generation-Prediction
