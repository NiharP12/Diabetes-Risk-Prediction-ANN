import torch
import numpy as np
import os
import sys
import pickle

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from model import DiabetesANN
from utils import load_scaler

def predict_sample():
    # Paths
    MODEL_PATH = 'models/diabetes_ann.pth'
    SCALER_PATH = 'models/scaler_diabetes.pkl'
    
    # Check if files exist
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Model or Scaler not found. Please run src/train.py first.")
        return
        
    # Load Scaler
    scaler = load_scaler(SCALER_PATH)
    
    # Load Model
    input_size = 8 # Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age
    model = DiabetesANN(input_size=input_size)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    # Sample Input (Based on the Pima Dataset)
    # 6,148,72,35,0,33.6,0.627,50 -> Diabetic (Outcome 1)
    sample_input = np.array([[6, 148, 72, 35, 125, 33.6, 0.627, 50]]) # Using a dummy median for insulin=125
    
    # Preprocess (Scale)
    sample_scaled = scaler.transform(sample_input)
    sample_tensor = torch.FloatTensor(sample_scaled)
    
    # Predict
    with torch.no_grad():
        output = model(sample_tensor)
        prediction = (output > 0.5).item()
        
    print("\n--- Prediction Logic ---")
    print(f"Sample Input: {sample_input}")
    print(f"Probability: {output.item():.4f}")
    
    if prediction:
        print("Result: Diabetic")
    else:
        print("Result: Not Diabetic")

if __name__ == "__main__":
    predict_sample()
