import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

print("DEBUG: Training script started.")

DATA_PATH = "data/processed_customer_data.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "customer_satisfaction_model.pkl")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

def train_model():
    print("DEBUG: Loading processed data...")
    df = pd.read_csv(DATA_PATH)
    print(f"DEBUG: Data loaded successfully. Shape: {df.shape}")

    # Check for target column
    if "low_satisfaction" not in df.columns:
        print("ERROR: 'low_satisfaction' column not found.")
        print("Available columns:", df.columns.tolist())
        return

    # Prepare features
    features = [col for col in df.columns if col not in ["low_satisfaction", "feedback_text"]]
    X = df[features].select_dtypes(include=[np.number]).fillna(0)
    y = df["low_satisfaction"]

    print(f"DEBUG: Features selected ({len(features)} columns). Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    print("DEBUG: Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    print("DEBUG: Model training completed. Evaluating...")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 3))

    joblib.dump(model, MODEL_PATH)
    print(f"\n✅ Model saved successfully at {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
