import pandas as pd 
import numpy as np
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

print("DEBUG: Script started.")

# Download the sentiment analysis lexicon if not available
try:
    nltk.download('vader_lexicon')
except:
    print("DEBUG: Could not download vader_lexicon (might already exist).")

DATA_DIR = "data"
OUTPUT_PATH = os.path.join(DATA_DIR, "processed_customer_data.csv")

def load_csvs(data_dir=DATA_DIR):
    """Load only the necessary CSV files for the project."""
    print("DEBUG: Loading CSV files...")

    # ✅ Keep only these CSVs
    orders = pd.read_csv(f"{data_dir}/orders.csv")
    feedback = pd.read_csv(f"{data_dir}/customer_feedback.csv")
    perf = pd.read_csv(f"{data_dir}/delivery_performance.csv")
    costs = pd.read_csv(f"{data_dir}/cost_breakdown.csv")

    # Normalize column names
    for df in [orders, feedback, perf, costs]:
        df.columns = [c.strip().lower() for c in df.columns]

    print("DEBUG: All required CSVs loaded and normalized.")
    return orders, feedback, perf, costs


def preprocess_and_merge():
    """Clean, merge, and prepare data for modeling."""
    print("DEBUG: Starting preprocess_and_merge()...")
    orders, feedback, perf, costs = load_csvs()

    print("DEBUG: orders shape:", orders.shape)
    print("DEBUG: feedback shape:", feedback.shape)

    # Confirm key column exists
    if "order_id" not in orders.columns or "order_id" not in feedback.columns:
        print("ERROR: 'order_id' column not found even after normalization.")
        print("orders columns:", orders.columns.tolist())
        print("feedback columns:", feedback.columns.tolist())
        return None

    # Merge essential datasets
    df = (
        orders
        .merge(feedback, how="left", on="order_id", suffixes=("", "_fb"))
        .merge(perf, how="left", on="order_id", suffixes=("", "_perf"))
        .merge(costs, how="left", on="order_id", suffixes=("", "_cost"))
    )

    print("DEBUG: Merged dataset shape:", df.shape)

    # Ensure feedback_text and rating exist
    if "rating" not in df.columns:
        df["rating"] = 5  # neutral default
    if "feedback_text" not in df.columns:
        text_cols = [c for c in df.columns if "feedback" in c.lower() or "comment" in c.lower()]
        if text_cols:
            df.rename(columns={text_cols[0]: "feedback_text"}, inplace=True)
        else:
            df["feedback_text"] = ""

    # Sentiment analysis
    sid = SentimentIntensityAnalyzer()
    df["sent_compound"] = df["feedback_text"].fillna("").apply(lambda x: sid.polarity_scores(str(x))["compound"])

    # Satisfaction score
    df["satisfaction_score"] = 0.6 * ((df["rating"] - 3) / 2.0) + 0.4 * df["sent_compound"]

    # Flag low satisfaction customers
    df["low_satisfaction"] = ((df["rating"] <= 3) | (df["sent_compound"] < -0.2)).astype(int)

    print("DEBUG: Computed sentiment and satisfaction scores.")

    # Save final dataset
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved processed data to {OUTPUT_PATH}")
    print("DEBUG: Done.")

    return df


if __name__ == "__main__":
    preprocess_and_merge()



