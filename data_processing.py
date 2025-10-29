import pandas as pd
import numpy as np
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

print("DEBUG: Script started.")

try:
    nltk.download('vader_lexicon')
except:
    print("DEBUG: Could not download vader_lexicon (might already exist).")

DATA_DIR = "data"
OUTPUT_PATH = os.path.join(DATA_DIR, "processed_customer_data.csv")

def load_csvs(data_dir=DATA_DIR):
    print("DEBUG: Loading CSV files...")
    orders = pd.read_csv(f"{data_dir}/orders.csv")
    perf = pd.read_csv(f"{data_dir}/delivery_performance.csv")
    routes = pd.read_csv(f"{data_dir}/routes_distance.csv")
    fleet = pd.read_csv(f"{data_dir}/vehicle_fleet.csv")
    warehouse = pd.read_csv(f"{data_dir}/warehouse_inventory.csv")
    feedback = pd.read_csv(f"{data_dir}/customer_feedback.csv")
    costs = pd.read_csv(f"{data_dir}/cost_breakdown.csv")

    # Normalize column names
    orders.columns = [c.strip().lower() for c in orders.columns]
    feedback.columns = [c.strip().lower() for c in feedback.columns]
    perf.columns = [c.strip().lower() for c in perf.columns]
    routes.columns = [c.strip().lower() for c in routes.columns]
    fleet.columns = [c.strip().lower() for c in fleet.columns]
    warehouse.columns = [c.strip().lower() for c in warehouse.columns]
    costs.columns = [c.strip().lower() for c in costs.columns]

    print("DEBUG: All CSVs loaded and normalized.")
    return orders, perf, routes, fleet, warehouse, feedback, costs

def preprocess_and_merge():
    print("DEBUG: Starting preprocess_and_merge()...")
    orders, perf, routes, fleet, warehouse, feedback, costs = load_csvs()

    print("DEBUG: orders shape:", orders.shape)
    print("DEBUG: feedback shape:", feedback.shape)

    # Confirm shared key column
    if "order_id" not in orders.columns or "order_id" not in feedback.columns:
        print("ERROR: 'order_id' column not found even after normalization.")
        print("orders columns:", orders.columns.tolist())
        print("feedback columns:", feedback.columns.tolist())
        return None

    # Merge
    df = orders.merge(feedback, how="left", on="order_id", suffixes=("", "_fb"))
    print("DEBUG: merged with feedback ->", df.shape)

    # Ensure feedback_text and rating exist
    if "rating" not in df.columns:
        df["rating"] = 5  # neutral default
    if "feedback_text" not in df.columns:
        text_cols = [c for c in df.columns if "feedback" in c.lower() or "comment" in c.lower()]
        if text_cols:
            df.rename(columns={text_cols[0]: "feedback_text"}, inplace=True)
        else:
            df["feedback_text"] = ""

    sid = SentimentIntensityAnalyzer()
    df["sent_compound"] = df["feedback_text"].fillna("").apply(lambda x: sid.polarity_scores(str(x))["compound"])
    df["satisfaction_score"] = 0.6 * ((df["rating"] - 3) / 2.0) + 0.4 * df["sent_compound"]
    df["low_satisfaction"] = ((df["rating"] <= 3) | (df["sent_compound"] < -0.2)).astype(int)

    print("DEBUG: computed sentiment and satisfaction scores.")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved processed data to {OUTPUT_PATH}")
    print("DEBUG: Done.")
    return df

if __name__ == "__main__":
    preprocess_and_merge()

