import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
from pathlib import Path

from models.recsys import CollaborativeFilteringRecSys


def rmse(true_data, predictions):
    """
    Calculate Root Mean Squared Error (RMSE) between true and predicted ratings.

    Args:
    - true_data (DataFrame): DataFrame with columns 'movie_id' and 'rating' for true ratings.
    - predictions (DataFrame): DataFrame with columns 'movie_id' and 'pred_rating' for predicted ratings.

    Returns:
    - float: RMSE value.
    """
    combined = true_data.merge(predictions)
    true_ratings = combined["rating"].values
    pred_ratings = combined["pred_rating"].values
    return np.sqrt(mean_squared_error(true_ratings, pred_ratings))


def mae(true_data, predictions):
    """
    Calculate Mean Absolute Error (MAE) between true and predicted ratings.

    Args:
    - true_data (DataFrame): DataFrame with columns 'movie_id' and 'rating' for true ratings.
    - predictions (DataFrame): DataFrame with columns 'movie_id' and 'pred_rating' for predicted ratings.

    Returns:
    - float: MAE value.
    """
    combined = true_data.merge(predictions)
    true_ratings = combined["rating"].values
    pred_ratings = combined["pred_rating"].values
    return mean_absolute_error(true_ratings, pred_ratings)


def precision_recall(true_data, predictions, k=10, good_threshold=3.5):
    """
    Calculate precision and recall metrics for recommendations.

    Args:
    - true_data (DataFrame): DataFrame with columns 'movie_id' and 'rating' for true ratings.
    - predictions (DataFrame): DataFrame with columns 'movie_id' and 'pred_rating' for predicted ratings.
    - k (int): Number of top recommendations.
    - good_threshold (float): Threshold to consider ratings as 'good'.

    Returns:
    - float or None, float or None: Precision and Recall values, or None if no relevant items.
    """
    relevant_items = set(true_data["movie_id"][true_data["rating"] >= good_threshold].values)
    if len(relevant_items) == 0:
        return None, None
    recommended = predictions[predictions["pred_rating"] >= good_threshold].sort_values("pred_rating", ascending=False)[
                      "movie_id"].values[:k]
    hits = sum(rec in relevant_items for rec in recommended)
    return hits / len(recommended), hits / len(relevant_items)


# Load the pre-trained recommendation model from a file
with open("../models/recsys.pkl", "rb") as f:
    recsys = pickle.load(f)

data_folder = Path("data")  # Define the data folder using Pathlib
data_col_names = ["user_id", "movie_id", "rating", "timestamp"]  # Define column names for the dataset
test_df = pd.read_csv(str(data_folder / "ub.test"), sep="\t", names=data_col_names,
                      encoding='latin-1')  # Load test data
evaluation_user_list = test_df["user_id"].unique()  # Get unique user IDs for evaluation

ks = [10, 20, 50]  # Define values of k for precision and recall calculations
scores = {
    "RMSE": [],  # List to store RMSE scores
    "MAE": []  # List to store MAE scores
}

# Initialize lists for precision and recall scores at different k values
for k in ks:
    scores[f"Precision@{k}"] = []
    scores[f"Recall@{k}"] = []

# Iterate through each user for evaluation
for user_id in tqdm(evaluation_user_list):
    preds = recsys.predict_unseen(user_id, verbose=False)  # Get predictions for unseen items for the user
    true_data = test_df[["movie_id", "rating"]][test_df["user_id"] == user_id]  # Get true ratings for the user

    # Calculate and store RMSE and MAE scores
    scores["RMSE"].append(rmse(true_data, preds))
    scores["MAE"].append(mae(true_data, preds))

    # Calculate precision and recall at different k values and store the scores
    for k in ks:
        p, r = precision_recall(true_data, preds, k=k)
        if p is not None:
            scores[f"Precision@{k}"].append(p)
            scores[f"Recall@{k}"].append(r)

# Calculate and print mean scores for RMSE, MAE, Precision, and Recall
for score, score_data in scores.items():
    print(f"{score}:\t\t{np.mean(score_data):.3f}")
