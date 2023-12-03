import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
from pathlib import Path

from models.recsys import CollaborativeFilteringRecSys


def rmse(true_data, predictions):
    combined = true_data.merge(predictions)
    true_ratings = combined["rating"].values
    pred_ratings = combined["pred_rating"].values
    return np.sqrt(mean_squared_error(true_ratings, pred_ratings))


def mae(true_data, predictions):
    combined = true_data.merge(predictions)
    true_ratings = combined["rating"].values
    pred_ratings = combined["pred_rating"].values
    return mean_absolute_error(true_ratings, pred_ratings)


def presicion_recall(true_data, predictions, k=10, good_threshold=3.5):
    relevant_items = set(true_data["movie_id"][true_data["rating"] >= good_threshold].values)
    if len(relevant_items) == 0:
        return None, None
    recommended = predictions[predictions["pred_rating"] >= good_threshold].sort_values("pred_rating", ascending=False)[
                      "movie_id"].values[:k]
    hits = sum(rec in relevant_items for rec in recommended)
    return hits / len(recommended), hits / len(relevant_items)


with open("../models/recsys.pkl", "rb") as f:
    recsys = pickle.load(f)

data_folder = Path("data")
data_col_names = ["user_id", "movie_id", "rating", "timestamp"]
test_df = pd.read_csv(str(data_folder / "ub.test"), sep="\t", names=data_col_names, encoding='latin-1')
evaluation_user_list = test_df["user_id"].unique()

ks = [10, 20, 50]
scores = {
    "RMSE": [],
    "MAE": []
}

for k in ks:
    scores[f"Presicion@{k}"] = []
    scores[f"Recall@{k}"] = []

for user_id in tqdm(evaluation_user_list):
    preds = recsys.predict_unseen(user_id, verbose=False)
    true_data = test_df[["movie_id", "rating"]][test_df["user_id"] == user_id]

    scores["RMSE"].append(rmse(true_data, preds))
    scores["MAE"].append(mae(true_data, preds))
    for k in ks:
        p, r = presicion_recall(true_data, preds, k=k)
        if p is not None:
            scores[f"Presicion@{k}"].append(p)
            scores[f"Recall@{k}"].append(r)

for score, score_data in scores.items():
    print(f"{score}:\t\t{np.mean(score_data):.3f}")
