import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import sys

# Add src to path just in case
sys.path.append(os.path.join(os.getcwd(), 'src'))

from model import DiabetesANN
from utils import load_and_preprocess, save_scaler

def train():
    # Paths
    DATA_PATH = 'data/diabetes.csv'
    MODEL_SAVE_PATH = 'models/diabetes_ann.pth'
    SCALER_SAVE_PATH = 'models/scaler_diabetes.pkl'
    
    # Load and Preprocess Data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess(DATA_PATH)
    
    # Initialize Model
    input_size = X_train.shape[1]
    model = DiabetesANN(input_size=input_size)
    
    # Loss and Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    epochs = 100
    print(f"Starting Training for {epochs} epochs...\n")
    
    for epoch in range(epochs):
        model.train()
        
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate Training Accuracy
        with torch.no_grad():
            train_preds = (outputs > 0.5).float()
            train_acc = accuracy_score(y_train, train_preds)
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {train_acc:.4f}')
            
    # Evaluation
    print("\n--- Final Evaluation ---")
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_preds = (test_outputs > 0.5).float()
        
        # Metrics
        acc = accuracy_score(y_test, test_preds)
        prec = precision_score(y_test, test_preds)
        rec = recall_score(y_test, test_preds)
        f1 = f1_score(y_test, test_preds)
        cm = confusion_matrix(y_test, test_preds)
        
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Precision:    {prec:.4f}")
        print(f"Recall:       {rec:.4f}")
        print(f"F1-score:     {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)
        
    # Save Model and Scaler
    if not os.path.exists('models'):
        os.makedirs('models')
        
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    save_scaler(scaler, SCALER_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")
    print(f"Scaler saved to {SCALER_SAVE_PATH}")

if __name__ == "__main__":
    train()
