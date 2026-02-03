# Car Price Prediction

Project for Machine Learning course, Masters in Data Science and Advanced Analytics

This repository reflects a cleaned version of the project, with improved organization and documentation for clarity and reproducibility.

---

## Project Authors

Pedro Santos – 20250399 – [20250399@novaims.unl.pt](mailto:20250399@novaims.unl.pt)  
Miguel Correia – 20250381 – [20250381@novaims.unl.pt](mailto:20250381@novaims.unl.pt)  
Pedro Fernandes – 20250418 – [20250418@novaims.unl.pt](mailto:20250418@novaims.unl.pt)  
Tiago Duarte – 20250360 – [20250360@novaims.unl.pt](mailto:20250360@novaims.unl.pt)

---

## Project Overview

Cars 4 You is an online car resale platform that evaluates and resells cars. Due to growing inspection wait times, the goal of this project is to develop a predictive model that estimates car prices based on user-provided details, reducing reliance on in-person inspections.

---

## Project Goals

1. **Regression Benchmarking** – Develop and compare regression models to accurately predict car prices.
2. **Model Optimization** – Improve model performance through hyperparameter tuning, preprocessing, and feature selection.
3. **Additional Insights** – Optional analyses include feature importance, ablation studies, creating an analytics interface, evaluating general vs. brand-specific models, and exploring deep learning approaches.

---

## Usage

- Input car features (e.g., brand, model, year, mileage) to receive a predicted price.
- Models can be extended, optimized, and visualized for insights into pricing drivers.

---

## Outcome

A reliable, automated system to estimate car prices, improving efficiency and customer satisfaction while reducing inspection bottlenecks.

---

## Notebooks

The main notebook is **main.ipynb**, which contains the complete pipeline summary and provides links to auxiliary notebooks:

- **01_categorical_variables_fixing.ipynb** – handles categorical variable fixing
- **02_visualizations.ipynb** – contains exploratory data visualizations and statistical plots
- **03_preprocessing.ipynb** – handles the creation of folds and data processing
- **04_feature_selection.ipynb** – performs feature engineering and selection
- **05_model_creation.ipynb** – handles model training, hyperparameter tuning, and evaluation

---

## Setup

### Create environment
```bash
conda env create -f environment.yml
```

## Activate Environment
```bash
conda activate cars4you
```

## Run

Open notebooks in Jupyter or VSCode and select `cars4you` kernel.

## Deactivate
```bash
conda deactivate
```

---

---
# Cars 4 You - Streamlit UI

This repository also contains the Streamlit user interface for the Cars4You project. It provides an analytics interface where a user can enter vehicle details and obtain a price prediction from a pre-trained regression model.

This implementation corresponds to the "Additional Insights" objective (c): create an analytics interface that returns a prediction when new input data is provided.

## Important Notes!!

**1. Simplified Preprocessing:**
This is a **simplified version** of the original notebook's preprocessing code. In the UI, **ALL fields are MANDATORY** and there are **NO missing values to impute**. As such, functions like `guess_brand_model()`, `fix_empty_categorical()`, and `fix_empty_numerical()` have been removed, reducing the code from 604 to 255 lines (-58%) while maintaining the exact same preprocessing logic for complete inputs.

**2. Model Selection:**
The model used in this UI is **NOT the best model** we could train, but rather a **simpler and lighter model** that fits within the ZIP file size limit for submission. The best-performing models and full training pipeline can be found in the main project repository.

## Scope

**Included:**
- Streamlit form to collect vehicle attributes
- Preprocessing required to transform user input into model-ready format
- Loading a pre-trained model and returning a predicted price (£)

**Not included** (these can be found in the main scope - notebooks main and auxiliary):
- Model training, benchmarking, and optimisation notebooks 
- The full project report and analysis deliverables 

## Inputs and Output

**Inputs (examples):**
- Brand, model, year
- mileage, tax, mpg, engineSize
- transmission, fuelType
- previousOwners, hasDamage

**Output:**
- Predicted vehicle price in £

## Folder Structure

```
UI/
├── app.py                                          # Streamlit application (UI + inference)
├── preprocessing_utils.py                          # Preprocessing utilities (simplified for UI)
├── mapping_dicts/                                  # CSV mapping files for standardization
│   ├── brand_mapping.csv
│   ├── fueltype_mapping.csv
│   ├── model_mapping.csv
│   └── transmission_mapping.csv
├── preprocessing_results/full_dataset/
│   ├── scaler.pkl                                  # MinMaxScaler for feature scaling
│   └── encoding_maps.pkl                           # Target encoding maps for model
├── files/
│   └── model_exported.pkl                          # Trained model file (lightweight version)
└── requirements.txt                                # Python dependencies
```

## Run Locally

### 1) Enter the specified folder
```bash
cd UI
```

### 2) Create and activate a virtual environment
```bash
python -m venv venv

# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Windows CMD:
venv\Scripts\activate.bat

# macOS/Linux:
source venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Run the app
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## Online Demo

The application is also available online here:
- **Live Demo:** https://uicars4you-mlproject.streamlit.app/

---
