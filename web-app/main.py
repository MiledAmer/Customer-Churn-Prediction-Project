import pickle
import pandas as pd
import os
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --- LOAD MODEL & SCALER ---
base_dir = os.path.dirname(os.path.abspath(__file__))

def load_asset(filename):
    with open(os.path.join(base_dir, filename), 'rb') as f:
        return pickle.load(f)

try:
    model = load_asset('churn_model_v1.pkl')
    scaler = load_asset('scaler.pkl')
    print("✅ System Ready.")
except Exception as e:
    print(f"❌ Error loading files: {e}")

# --- DEFINE FULL INPUT SCHEMA (19 Features) ---
class CustomerInput(BaseModel):
    gender: Optional[str] = None
    SeniorCitizen: Optional[int] = None
    Partner: Optional[str] = None
    Dependents: Optional[str] = None
    tenure: Optional[float] = None        
    PhoneService: Optional[str] = None
    MultipleLines: Optional[str] = None
    InternetService: Optional[str] = None
    OnlineSecurity: Optional[str] = None
    OnlineBackup: Optional[str] = None
    DeviceProtection: Optional[str] = None
    TechSupport: Optional[str] = None
    StreamingTV: Optional[str] = None
    StreamingMovies: Optional[str] = None
    Contract: Optional[str] = None
    PaperlessBilling: Optional[str] = None
    PaymentMethod: Optional[str] = None
    MonthlyCharges: Optional[float] = None
    TotalCharges: Optional[float] = None

# --- ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def analyze_profit_strategy(probability: float, data: dict):
    """
    Returns structured advice based on Profit Sensitivity Analysis.
    Output is a dict: {'title': str, 'message': str, 'style': str}
    """
    # SAFE EXTRACTION:
    # Use (data.get('Field') or 0) to ensure we never have None.
    # If the value is None, 'or 0' forces it to be 0 (or a default string).
    
    monthly = data.get('MonthlyCharges') or 0.0
    contract = data.get('Contract') or 'Month-to-month'
    tenure = data.get('tenure') or 0

    # 1. THE SAFE ZONE (< 37%)
    if probability < 0.37:
        return {
            "title": "✅ Low Priority / Safe",
            "message": f"Churn risk is {probability:.0%} (Below the 37% intervention threshold). "
                       "Spending budget on retention for this customer would reduce overall profit. "
                       "Action: Monitor only.",
            "style": "bg-emerald-900/30 text-emerald-300 border-emerald-700"
        }

    # 2. THE PROFIT ZONE (37% - 75%)
    elif 0.37 <= probability < 0.75:
        
        # SCENARIO A: POWER USER (High Revenue)
        if monthly > 90:
            return {
                "title": "💎 POWER USER DETECTED - ACT NOW",
                "message": (
                    f"This customer pays ${monthly}/mo (Top Tier). "
                    "Your Sensitivity Analysis shows that losing this customer is 3x more costly than a basic user. "
                    "You MUST offer a 'Premium Retention Bundle' to lock them in."
                ),
                "style": "bg-blue-900/40 text-blue-200 border-blue-500 border-2"
            }

        # SCENARIO B: FLIGHT RISK (Month-to-month)
        if contract == "Month-to-month":
             return {
                "title": "⚠️ Contract Volatility Warning",
                "message": (
                    "Customer is on a Month-to-month contract, making them highly sensitive to price. "
                    "According to the model, moving them to a 1-year contract reduces risk by 40%. "
                    "Action: Offer a $10/mo discount in exchange for a 1-year commitment."
                ),
                "style": "bg-yellow-900/30 text-yellow-200 border-yellow-600"
            }

        # SCENARIO C: STANDARD USER
        return {
            "title": "🔔 Opportunity for Intervention",
            "message": "Customer is in the optimal window for retention. "
                       "Standard email marketing offer is recommended.",
            "style": "bg-slate-700 text-slate-300 border-slate-500"
        }

    # 3. THE DANGER ZONE (> 75%)
    else:
        return {
            "title": "🚨 EXTREME CHURN RISK",
            "message": "Probability is critically high. "
                       "This customer has likely already decided to leave. "
                       "Only a 'Hail Mary' offer (e.g., 50% off for 3 months) might save them.",
            "style": "bg-red-900/40 text-red-200 border-red-500 animate-pulse"
        }

@app.post("/predict")
async def predict(data: CustomerInput):
    try:
        input_data = data.model_dump()
        df_input = pd.DataFrame([input_data])
        
        # ... (Standard Preprocessing & Prediction logic) ...
        # ...
        if hasattr(scaler, 'feature_names_in_'):
             df_input = df_input[scaler.feature_names_in_]
        
        input_processed = scaler.transform(df_input)
        
        # Get Probability
        prediction = model.predict(input_processed)[0]
        probability = model.predict_proba(input_processed)[0][1]

        # --- CALL NEW PROFIT STRATEGY FUNCTION ---
        strategy_message = analyze_profit_strategy(probability, input_data)

        return {
            "churn_prediction": int(prediction),
            "churn_probability": float(probability),
            "advice": strategy_message  # Matches the frontend 'result.advice'
        }

    except Exception as e:
        return {"error": str(e)}