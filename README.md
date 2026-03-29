# Disease Risk Prediction using ANN

This project focuses on predicting the likelihood of diabetes in patients using an Artificial Neural Network (ANN). The model is trained on the Pima Indians Diabetes dataset using patient medical attributes and provides binary classification (Diabetic / Non-Diabetic).

---

## Aim

To develop an Artificial Neural Network (ANN) model that can accurately predict diabetes risk based on patient health parameters.

---

## Features

- Predicts diabetes risk using ANN  
- Uses real-world healthcare dataset (Pima Indians Diabetes Dataset)  
- Handles missing values and performs data preprocessing  
- End-to-end pipeline: Data → Training → Evaluation → Prediction  
- Saves trained model and scaler for reuse  

---

## Tech Stack

- Python  
- PyTorch  
- NumPy  
- Pandas  
- Scikit-learn  

---

## Dataset

- Dataset: Pima Indians Diabetes Dataset  
- Features:
  - Pregnancies  
  - Glucose  
  - Blood Pressure  
  - Skin Thickness  
  - Insulin  
  - BMI  
  - Diabetes Pedigree Function  
  - Age  
- Target:
  - Outcome (0 = Non-Diabetic, 1 = Diabetic)  

---

## Model Architecture

The ANN model consists of:

- Input Layer (8 features)  
- Hidden Layer 1: 16 neurons + ReLU + Dropout  
- Hidden Layer 2: 8 neurons + ReLU + Dropout  
- Output Layer: 1 neuron + Sigmoid  

---

## Data Preprocessing

- Replaced invalid zero values with median values  
- Feature scaling using StandardScaler  
- Train-test split (80:20)  
- Converted data into PyTorch tensors  

---

## Training

To train the model:

python src/train.py  

### Training Details

- Loss Function: Binary CrossEntropy (BCELoss)  
- Optimizer: Adam  
- Epochs: 100  
- Learning Rate: 0.001  

During training:
- Loss and accuracy are printed every 10 epochs  
- Model learns patterns from patient data  

---

## Results

Final evaluation on test dataset:

- **Accuracy:** 79.87%  
- **Precision:** 78.57%  
- **Recall:** 60.00%  
- **F1-Score:** 68.04%  

The model achieves a good balance between precision and recall for medical risk prediction.

## ⭐ Support

If you like this project, please give it a ⭐ on GitHub!
