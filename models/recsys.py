from tqdm import tqdm
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors


class CollaborativeFilteringRecSys:
    """
    Collaborative Filtering Recommender System using PCA and k-Nearest Neighbors.
    """
    def __init__(self, pca_var=0.9, knn_neighbors=600):
        """
        Initialize CollaborativeFilteringRecSys.

        Args:
        - pca_var (float): Variance to be retained in PCA.
        - knn_neighbors (int): Number of neighbors for k-Nearest Neighbors.
        """
        self.pca_var = pca_var
        self.knn_neighbors = knn_neighbors
        self.distances = None
        self.neighbors = None
        self.rating_matrix = None

    def fit(self, train_df: pd.DataFrame, movie_data: pd.DataFrame = None):
        """
        Fit the model.

        Args:
        - train_df (pd.DataFrame): DataFrame containing user-item ratings.
        - movie_data (pd.DataFrame, optional): Additional movie data (columns: movie_id, title, release_date, etc.).
        """
        self.rating_matrix = train_df.pivot(index="movie_id", columns="user_id", values="rating").fillna(0)
        if movie_data is None:
            total_item_matrix = self.rating_matrix
        else:
            total_item_matrix = self.rating_matrix.merge(movie_data, on="movie_id")
            total_item_matrix = total_item_matrix.drop(
                columns=["movie_id", "title", "release_date", "video_release_data", "url"])

        pipeline = Pipeline([("scale", MinMaxScaler((-1, 1))), ("pca", PCA(n_components=self.pca_var))])
        X_train = pipeline.fit_transform(total_item_matrix.values)
        knn = NearestNeighbors(metric="cosine")
        knn.fit(X_train)
        self.distances, self.neighbors = knn.kneighbors(X_train, n_neighbors=self.knn_neighbors)

    def calculate_rating(self, neighbors_dists, user_rating_history):
        """
        Calculate rating based on user's rating history and neighbor distances.

        Args:
        - neighbors_dists (list): List of tuples (distance, index) for neighbors.
        - user_rating_history (pd.Series): User's rating history.

        Returns:
        - float: Predicted rating.
        """
        sum_sim = 0
        weighted_sum_ratings = 0
        for movie_dist, movie_index in neighbors_dists:
            cur_movie_id = movie_index + 1
            sim = 1 - movie_dist
            sum_sim += sim
            weighted_sum_ratings += user_rating_history[cur_movie_id] * sim

        if weighted_sum_ratings == 0:
            return 0
        return weighted_sum_ratings / sum_sim

    def predict_movie_rating(self, user_id, movie_id, rated_neighbors):
        """
        Predict rating for a movie not seen by the user.

        Args:
        - user_id (int): User ID.
        - movie_id (int): Movie ID.
        - rated_neighbors (int): Number of rated neighbors to consider.

        Returns:
        - float: Predicted rating for the movie.
        """
        movie_index = self.rating_matrix.index.tolist().index(movie_id)
        movie_dists = self.distances[movie_index].tolist()
        movie_neighbors = self.neighbors[movie_index].tolist()

        if movie_index in movie_neighbors:
            cur_ind = movie_neighbors.index(movie_index)
            movie_dists.pop(cur_ind)
            movie_neighbors.pop(cur_ind)

        user_rating_history = self.rating_matrix[user_id]
        seen_movies = set(user_rating_history[user_rating_history != 0].index)
        neighbors_dists = list(filter(lambda x: (x[1] + 1) in seen_movies,
                                      zip(movie_dists, movie_neighbors)))[:rated_neighbors]

        return self.calculate_rating(neighbors_dists, user_rating_history)

    def predict_unseen(self, user_id, verbose=True, rated_neighbors=5):
        """
        Predict ratings for unseen movies by a user.

        Args:
        - user_id (int): User ID.
        - verbose (bool, optional): Whether to show progress bar or not.
        - rated_neighbors (int, optional): Number of rated neighbors to consider.

        Returns:
        - pd.DataFrame: DataFrame with movie_id and predicted ratings for unseen movies.
        """
        user_rating_history = self.rating_matrix[user_id]
        unseen_movies = user_rating_history[user_rating_history == 0].index

        ratings = []
        if verbose:
            pbar = tqdm(unseen_movies, desc="Predicting ratings")
        else:
            pbar = unseen_movies
        for movie_id in pbar:
            rating = self.predict_movie_rating(user_id, movie_id, rated_neighbors)
            ratings.append(rating)

        return pd.DataFrame({"movie_id": unseen_movies, "pred_rating": ratings})

    def recommend_unseen(self, user_id, top_k=20):
        """
        Recommend unseen movies for a user.

        Args:
        - user_id (int): User ID.
        - top_k (int, optional): Number of top recommendations to return.

        Returns:
        - pd.DataFrame: DataFrame with top recommended movies and predicted ratings.
        """
        rating_predictions = self.predict_unseen(user_id)
        rating_predictions = rating_predictions.sort_values("pred_rating", ascending=False).head(top_k)
        return rating_predictions
