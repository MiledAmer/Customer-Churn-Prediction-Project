# Customer Churn Prediction Project

## 1. Project Overview
This project aims to predict customer churn using machine learning. It follows a modular structure to ensure reproducibility and scalability. 

The goal is to analyze customer behavior and build a classifier (Logistic Regression, Random Forest, etc.) to identify customers at risk of leaving. The project culminates in a **Deployment-Ready Web Dashboard** built with **FastAPI** that provides strategic interventions based on financial sensitivity analysis.

## 2. Key Features
- **Data Analysis**: Comprehensive Exploratory Data Analysis (EDA) to understand churn factors.
- **Machine Learning**: End-to-end pipeline includes data preprocessing, feature engineering, and model training.
- **Evaluation**: Detailed model performance metrics and visualization.
- **Web Interface**: Interactive dashboard for real-time predictions and sensitivity analysis using FastAPI and TailwindCSS.
- **Reproducibility**: Modular code structure with helper utilities.

## 3. Project Structure
The repository is organized as follows:

- `data/`: Contains the datasets.
  - `dataset.csv`: Raw data.
  - `processed/`: Processed training and testing sets.
- `notebooks/`: Jupyter notebooks for the ML workflow.
  - `01_EDA.ipynb`: Exploratory Data Analysis.
  - `02_data_preparation.ipynb`: Data cleaning and splitting.
  - `03_modeling.ipynb`: Model training and selection.
  - `04_evaluation.ipynb`: Performance evaluation.
- `web-app/`: The FastAPI application source code.
  - `main.py`: Backend logic and API endpoints.
  - `templates/`: Frontend HTML/TailwindCSS dashboard.
  - `*.pkl`: Serialized model and scaler objects used for inference.
- `models/`: Directory for storing model artifacts during training.
- `plots/`: Generated plots and figures.
- `utils/`: Helper scripts for preprocessing (`preprocessing.py`), EDA (`EDAUtils.py`), model evaluation (`evaluation.py`), and visualization (`visualization.py`).
- `report.html`: HTML report generated from the analysis.
- `Churn Prediction Analysis.pdf`: pdf report generated from report.html.
- `requirements.txt`: Python dependencies.

## 4. Installation & Setup
To set up the environment, follow these steps:

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

2. **Activate the environment**:
   - **Windows**:
     ```bash
     .\venv\Scripts\activate
     ```
   - **Mac/Linux**:
     ```bash
     source venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 5. Usage

### Running the Web App
To start the FastAPI dashboard:
```bash
cd web-app
uvicorn main:app --reload
```
The app will be available at `http://127.0.0.1:8000`.

### Running Notebooks
To explore the analysis and training process:
```bash
jupyter lab
```
Navigate to the `notebooks/` directory to run the files in order.  