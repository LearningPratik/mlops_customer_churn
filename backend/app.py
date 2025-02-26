from fastapi import FastAPI
from typing import Optional, List
from pydantic import BaseModel
import pandas as pd
import pickle

# Model and database paths (you'll need to adjust these)
MODEL_PATH = "../models/rf_model_1.pkl"
FEATURE_COLS = [

    'gender', 'seniorcitizen', 'partner', 'dependents', 'tenure', 'phoneservice', 'paperlessbilling',
    'multiplelines', 'internetservice', 'onlinesecurity', 'onlinebackup', 'deviceprotection', 
    'techsupport', 'streamingtv', 'streamingmovies',
    'contract', 'paymentmethod',
    'monthlycharges', 'totalcharges'
]

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn",
    version="1.0.0",
)

with open(MODEL_PATH, 'rb') as model_file_pkl:

        # Load the pipeline dictionary
        pipeline = pickle.load(model_file_pkl) 

        # Extract preprocessor
        preprocessor = pipeline['preprocessor'] 

        # Extract model
        model = pipeline['model'] 

class ChurnPredictionRequest(BaseModel):
    gender: str = 'Female'
    seniorcitizen: int = 0
    partner: str = 'Yes'
    dependents: str = 'No'
    tenure: int = 41
    phoneservice: str = 'Yes'
    paperlessbilling: str = 'No'
    multiplelines: str = 'No'
    internetservice: str = 'DSL'
    onlinesecurity: str = 'Yes'
    onlinebackup: str = 'No'
    deviceprotection: str = 'Yes'
    monthlycharges: float = 79.85
    totalcharges: float = 3320.75
    contract: str = 'One year'
    paymentmethod: str = 'Bank transfer (automatic)'
    techsupport: str = 'Yes'
    streamingtv: str = 'Yes'
    streamingmovies: str = 'Yes'

class ChurnPredictionResponse(BaseModel):
    churn_probability: float
    churn_risk: str 
    recommendations: List[str]

@app.post("/predict/", response_model=ChurnPredictionResponse)
async def predict_customer_churn(
    request: ChurnPredictionRequest
):
    # Extract features from the request
    features_dict = {
        'gender': request.gender,
        'seniorcitizen': request.seniorcitizen,
        'partner': request.partner,
        'dependents': request.dependents,
        'tenure': request.tenure,
        'phoneservice': request.phoneservice,
        'paperlessbilling': request.paperlessbilling,
        'multiplelines': request.multiplelines,
        'internetservice': request.internetservice,
        'onlinesecurity': request.onlinesecurity,
        'onlinebackup': request.onlinebackup,
        'deviceprotection': request.deviceprotection,
        'monthlycharges': request.monthlycharges,
        'totalcharges': request.totalcharges,
        'contract': request.contract,
        'paymentmethod': request.paymentmethod,
        'techsupport': request.techsupport,
        'streamingtv': request.streamingtv,
        'streamingmovies': request.streamingmovies
    }
    
    features = {col: [features_dict.get(col)] for col in FEATURE_COLS}
    new_data = pd.DataFrame(features)

    # Transform the evaluation data using the loaded preprocessor
    processed_data = preprocessor.transform(new_data)

    # Make predictions using the model from the pipeline on TRANSFORMED test data
    y_pred = model.predict(processed_data)
    probablity = model.predict_proba(processed_data)[:, 1]

    # Determine risk level
    if probablity < 0.3:
        risk = "Low"
        recommendations = ["Regular follow-up", "Cross-sell opportunities"]
    elif probablity < 0.6:
        risk = "Medium"
        recommendations = ["Offer contract upgrade", "Suggest bundled services"]
    else:
        risk = "High"
        recommendations = ["Immediate outreach needed", "Consider some discount", "Offer premium support"]
    
    
    # Return response
    return ChurnPredictionResponse(
        churn_probability = probablity,
        churn_risk = risk,
        recommendations = recommendations
    )