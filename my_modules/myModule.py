import os
import pickle
import logging
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz
from surprise import PredictionImpossible
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)


def load_data():
    """
    Load and prepare movie datasets from various sources.

    This function loads three main datasets:
    1. MovieLens ratings dataset
    2. TMDb links dataset
    3. Movie metadata dataset

    Returns:
        tuple: (ratings_df, links_df, new_df)
            - ratings_df (DataFrame): User ratings data
            - links_df (DataFrame): Movie ID mappings
            - new_df (DataFrame): Movie metadata
    """
    ratings_df = pd.read_csv(os.getenv("RATINGS_PATH"))
    links_df = pd.read_csv(os.getenv("LINKS_PATH"))
    new_df = pd.read_csv(os.getenv("MOVIES_PATH"))
    return ratings_df, links_df, new_df


# def load_similarity_matrix():
#     """
#     Load the precomputed content-based cosine similarity matrix.

#     Returns:
#     numpy.ndarray: Cosine similarity matrix.
#     """
#     return np.load(r"data/cosine_similarity2.npy")


def load_model():
    """
    Load the pre-trained SVD model for collaborative filtering.

    This function loads a pickled SVD model that was previously trained
    on the MovieLens dataset.

    Returns:
        object: The loaded SVD model

    Note:
        Requires SVD_MODEL_PATH environment variable to be set
    """
    with open(os.getenv("SVD_MODEL_PATH"), "rb") as file:
        return pickle.load(file)


# Load Count Matrix
def load_count_matrix():
    """
    Load the pre-computed count matrix for content-based similarity.

    This function loads a sparse matrix containing movie feature counts
    used for computing content-based similarity.

    Returns:
        sparse matrix: The loaded count matrix

    Note:
        Requires COUNT_MATRIX_PATH environment variable to be set
    """
    return load_npz(os.getenv("COUNT_MATRIX_PATH", "data/count_matrix.npz"))


# Create Movie Index Mapping (Title to Index)
def create_indices(new_df):
    """
    Create a mapping between movie titles and their indices.

    This function creates a Series mapping movie titles to their
    corresponding indices in the DataFrame.

    Parameters:
        new_df (DataFrame): Movie metadata DataFrame

    Returns:
        Series: Title to index mapping
    """
    return pd.Series(new_df.index, index=new_df["title"]).drop_duplicates()


def weighted_rating(x, C, m):
    """
    Compute IMDb-style weighted rating for a movie.

    This function implements the IMDb weighted rating formula:
    WR = (v / (v + m)) * R + (m / (v + m)) * C
    where:
    - v is the number of votes
    - m is the minimum votes required
    - R is the average rating
    - C is the mean vote across all movies

    Parameters:
        x (Series): Movie data containing 'vote_count' and 'vote_average'
        C (float): Mean vote across all movies
        m (float): Minimum votes required to be considered

    Returns:
        float: Weighted rating score between 0 and 10
    """
    v = x["vote_count"]
    R = x["vote_average"]
    return (v / (v + m) * R) + (m / (v + m) * C)


def match_tmdb_to_movielens(qualified_movies, links_df):
    """
    Match TMDb IDs with MovieLens IDs for SVD predictions.

    This function merges the qualified movies DataFrame with the links
    DataFrame to get the corresponding MovieLens IDs needed for SVD.

    Parameters:
        qualified_movies (DataFrame): Movies selected for recommendation
        links_df (DataFrame): Mapping of TMDb and MovieLens IDs

    Returns:
        DataFrame: Updated movies DataFrame with matched MovieLens IDs
    """
    qualified_movies = qualified_movies.merge(
        links_df[["tmdbId", "movieId"]], left_on="id", right_on="tmdbId", how="left"
    )
    return qualified_movies.dropna(subset=["movieId"]).astype({"movieId": "int"})


def handle_new_user(
    qualified_movies,
    top_n,
    popularity_weight,
    similarity_weight,
    recency_weight,
    new_df,
):
    """
    Generate recommendations for new users using content-based approach.

    This function handles the cold-start problem for new users by:
    1. Using content-based similarity
    2. Incorporating popularity scores
    3. Applying recency bias
    4. Handling cases with insufficient similar movies

    Parameters:
        qualified_movies (DataFrame): Movies selected for recommendation
        top_n (int): Number of recommendations to return
        popularity_weight (float): Weight for popularity factor
        similarity_weight (float): Weight for content similarity
        recency_weight (float): Weight for recency factor
        new_df (DataFrame): Complete movie dataset

    Returns:
        DataFrame: Recommended movies with final scores
    """
    if len(qualified_movies) < top_n:
        popular_movies = new_df.sort_values("popularity", ascending=False).head(top_n * 2)
        qualified_movies = pd.concat(
            [qualified_movies, popular_movies]
        ).drop_duplicates("id")

    # Compute Final Score for Sorting
    qualified_movies["final_score"] = (
        popularity_weight * qualified_movies["weighted_rating"]
    ) + (similarity_weight * qualified_movies["similarity_score"])

    # Combine recency into final_score
    qualified_movies["final_score"] = (
        qualified_movies["final_score"] * (1 - recency_weight)
        + qualified_movies["recency_score"] * recency_weight
    )

    return qualified_movies.sort_values("final_score", ascending=False).head(top_n)[
        ["id", "title", "release_date", "final_score", "poster_path"]
    ]


def predict_ratings(user_id, qualified_movies, best_svd_model, ratings_df, C):
    """
    Predict movie ratings using the SVD model.

    This function predicts ratings for movies using the trained SVD model,
    handling cases where the user or movie is not in the training set.

    Parameters:
        user_id (int): ID of the user requesting recommendations
        qualified_movies (DataFrame): Movies to predict ratings for
        best_svd_model (object): Trained SVD model
        ratings_df (DataFrame): Complete ratings dataset
        C (float): Global mean rating

    Returns:
        DataFrame: Updated movies DataFrame with predicted ratings
    """
    try:
        user_inner_id = best_svd_model.trainset.to_inner_uid(user_id)
    except ValueError:
        # User not in training set, use global average
        qualified_movies['predicted_rating'] = C
        return qualified_movies

    # Vectorized prediction
    global_mean = best_svd_model.trainset.global_mean
    user_bias = best_svd_model.bu[user_inner_id]
    item_biases = []
    item_factors = []

    for movie_id in qualified_movies['movieId']:
        try:
            item_inner_id = best_svd_model.trainset.to_inner_iid(movie_id)
            item_biases.append(best_svd_model.bi[item_inner_id])
            item_factors.append(best_svd_model.qi[item_inner_id])
        except ValueError:
            # Fallback to nearest neighbor average
            nearest_neighbors = ratings_df[ratings_df["movieId"] == movie_id]["rating"]
            fallback = nearest_neighbors.mean() if not nearest_neighbors.empty else C
            item_biases.append(fallback - global_mean - user_bias)
            item_factors.append(np.zeros_like(best_svd_model.qi[0]))

    item_factors = np.array(item_factors)
    predicted = global_mean + user_bias + np.array(item_biases) + np.dot(item_factors, best_svd_model.pu[user_inner_id])
    qualified_movies['predicted_rating'] = np.clip(predicted, 0.5, 5.0)
    return qualified_movies


def compute_hybrid_score(
    qualified_movies,
    user_ratings_count,
    recency_weight,
    top_n,
    svd_thresholds=(10, 50),
    svd_weights=(0.5, 0.6, 0.7),
):
    """
    Compute final hybrid recommendation scores.

    This function combines multiple factors to compute final scores:
    1. SVD predictions (weighted based on user history)
    2. IMDb-style ratings
    3. Recency bias
    4. Dynamic weighting based on user interaction

    Parameters:
        qualified_movies (DataFrame): Movies with individual scores
        user_ratings_count (int): Number of ratings by the user
        recency_weight (float): Weight for recency factor
        top_n (int): Number of recommendations to return
        svd_thresholds (tuple): Thresholds for SVD weight adjustment
        svd_weights (tuple): Weights for different user activity levels

    Returns:
        DataFrame: Top N movies sorted by final score
    """
    if user_ratings_count < svd_thresholds[0]:
        svd_weight = svd_weights[0]
    elif user_ratings_count < svd_thresholds[1]:
        svd_weight = svd_weights[1]
    else:
        svd_weight = svd_weights[2]

    imdb_weight = 1 - svd_weight
    qualified_movies["final_score"] = (
        svd_weight * qualified_movies["predicted_rating"]
        + imdb_weight * qualified_movies["weighted_rating"]
    )

    # Recency Boost
    qualified_movies = _apply_recency_boost(qualified_movies)

    # Combine recency with final_score
    qualified_movies["final_score"] = (
        qualified_movies["final_score"] * (1 - recency_weight)
        + qualified_movies["recency_score"] * recency_weight
    )

    return qualified_movies.sort_values("final_score", ascending=False).head(top_n)[
        ["id", "title", "release_date", "final_score", "poster_path"]
    ]


def get_top_similar_movies(movie_index, count_matrix):
    """
    Find top similar movies using content-based similarity.

    This function computes similarity scores between a movie and all
    other movies in the dataset using the count matrix.

    Parameters:
        movie_index (int): Index of the reference movie
        count_matrix (sparse matrix): Pre-computed count matrix

    Returns:
        tuple: (similar_indices, similarity_scores)
            - similar_indices (array): Indices of similar movies
            - similarity_scores (array): Corresponding similarity scores
    """
    # Compute similarity scores dynamically (only one movie vector at a time)
    movie_vector = count_matrix[movie_index]
    similarity_scores = cosine_similarity(movie_vector, count_matrix).flatten()

    # Get indices of top similar movies (excluding the movie itself)
    similar_indices = similarity_scores.argsort()[::-1][1:102]

    # Return similar movie indices and their similarity scores
    return similar_indices, similarity_scores[similar_indices]


def _apply_recency_boost(df):
    """
    Apply recency boost to movie scores based on release year.

    This function calculates a recency score that:
    1. Normalizes release years to a 0-1 scale
    2. Favors newer movies over older ones
    3. Handles missing or invalid dates

    Parameters:
        df (DataFrame): Movie dataset with 'release_date' column

    Returns:
        DataFrame: Dataset with additional 'recency_score' column
    """
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df["year"] = df["year"].fillna(df["year"].min())
    min_year = df["year"].min()
    max_year = df["year"].max()
    year_range = max_year - min_year if max_year != min_year else 1
    df["recency_score"] = (df["year"] - min_year) / year_range
    return df


def improved_hybrid_recommendations(
    user_id,
    title,
    top_n=10,
    ratings_df=None,
    links_df=None,
    new_df=None,
    best_svd_model=None,
    count_matrix=None,
    indices=None,
    popularity_weight=0.15,
    similarity_weight=0.85,
    recency_weight=0.2,
):
    """
    Generate hybrid movie recommendations combining content-based and collaborative filtering.

    This function implements a sophisticated recommendation system that:
    1. Uses content-based filtering to find similar movies
    2. Applies collaborative filtering using SVD for personalized recommendations
    3. Handles cold-start problems for new users
    4. Incorporates recency bias and popularity factors

    Parameters:
        user_id (int): The ID of the user requesting recommendations
        title (str): The title of the reference movie
        top_n (int, optional): Number of recommendations to return. Defaults to 10.
        ratings_df (DataFrame, optional): Movie ratings dataset. Defaults to None.
        links_df (DataFrame, optional): Movie ID mapping dataset. Defaults to None.
        new_df (DataFrame, optional): Movie metadata dataset. Defaults to None.
        best_svd_model (object, optional): Pre-trained SVD model. Defaults to None.
        count_matrix (sparse matrix, optional): Pre-computed count matrix. Defaults to None.
        indices (Series, optional): Movie title to index mapping. Defaults to None.
        popularity_weight (float, optional): Weight for popularity factor. Defaults to 0.15.
        similarity_weight (float, optional): Weight for content similarity. Defaults to 0.85.
        recency_weight (float, optional): Weight for recency factor. Defaults to 0.2.

    Returns:
        DataFrame: Recommended movies with columns:
            - id: TMDb movie ID
            - title: Movie title
            - release_date: Release date
            - final_score: Combined recommendation score
            - poster_path: Path to movie poster

    Raises:
        ValueError: If movie title is not found or required parameters are missing
    """
    if indices is None:
        raise ValueError("Indices mapping is required.")

    index = indices.get(title, None)
    if index is None:
        raise ValueError(f"Movie title '{title}' not found in the dataset.")

    # Get top content-based similar movies
    similar_movie_indices, similarity_scores = get_top_similar_movies(
        index, count_matrix
    )

    recommended_movies = new_df.iloc[similar_movie_indices][
        ["title", "id", "vote_count", "vote_average", "release_date", "poster_path"]
    ].copy()

    recommended_movies["vote_count"] = (
        recommended_movies["vote_count"].fillna(0).astype(int)
    )
    recommended_movies["vote_average"] = (
        recommended_movies["vote_average"].fillna(0).astype(float)
    )

    # Compute IMDb weighted rating
    C = recommended_movies["vote_average"].mean()
    m = recommended_movies["vote_count"].quantile(0.65)
    recommended_movies["weighted_rating"] = recommended_movies.apply(
        lambda x: weighted_rating(x, C, m), axis=1
    )

    qualified_movies = recommended_movies.copy()
    qualified_movies.reset_index(drop=True, inplace=True)

    # Merge similarity scores
    similarity_df = pd.DataFrame(
        {
            "id": new_df.iloc[similar_movie_indices]["id"].values,
            "similarity_score": similarity_scores,
        }
    )
    qualified_movies = recommended_movies.merge(similarity_df, on="id", how="left")
    qualified_movies["similarity_score"] = qualified_movies["similarity_score"].fillna(
        qualified_movies["similarity_score"].min()
    )

    # Cold-start handling for new users
    if user_id not in ratings_df["userId"].unique():
        qualified_movies = _apply_recency_boost(qualified_movies)
        return handle_new_user(
            qualified_movies,
            top_n,
            popularity_weight,
            similarity_weight,
            recency_weight,
            new_df,
        )

    # Match MovieLens IDs before using SVD
    qualified_movies = match_tmdb_to_movielens(qualified_movies, links_df)

    # Predict ratings using SVD
    user_ratings_count = ratings_df[ratings_df["userId"] == user_id].shape[0]
    qualified_movies = predict_ratings(
        user_id, qualified_movies, best_svd_model, ratings_df, C
    )

    # Compute hybrid recommendation score
    qualified_movies = compute_hybrid_score(
        qualified_movies, user_ratings_count, recency_weight, top_n
    )

    return qualified_movies

# TMDb API Key
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

def get_movie_poster(movie):
    """
    Fetch movie poster URL from TMDb API.

    This function attempts to get a movie poster URL in the following order:
    1. Use existing poster_path if available
    2. Fetch movie data from TMDb API if poster_path is missing
    3. Return None if no poster is found

    Parameters:
        movie (dict): Movie data containing 'id' and 'poster_path'

    Returns:
        str: URL to the movie poster image, or None if not found

    Note:
        Requires TMDB_API_KEY environment variable to be set
    """
    if movie["poster_path"]:
        return f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
    else:
        url = f"https://api.themoviedb.org/3/movie/{movie['id']}?api_key={TMDB_API_KEY}&language=en-US"
        response = requests.get(url)
        movie_data = response.json()
        if "poster_path" in movie_data and movie_data["poster_path"]:
            return f"https://image.tmdb.org/t/p/w500{movie_data['poster_path']}"
    return None

# Genre-Based Recommender
C = None  # Will be set after loading new_df
m = None  # Will be set after loading new_df
current_year = dt.datetime.now().year

def preprocess_movies(df):
    """
    Preprocess movie dataset for recommendation system.

    This function performs the following preprocessing steps:
    1. Creates a copy of the input DataFrame
    2. Converts genres string to lowercase list
    3. Extracts release year from release date
    4. Handles missing or invalid data

    Parameters:
        df (DataFrame): Raw movie dataset with 'genres' and 'release_date' columns

    Returns:
        DataFrame: Preprocessed dataset with additional columns:
            - genres_list: List of lowercase genres
            - release_year: Extracted year from release date
    """
    df = df.copy()
    df['genres_list'] = df['genres'].apply(lambda x: [g.lower() for g in eval(x)] if isinstance(x, str) else [])
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    return df

def genre_based_recommender(genre, df, top_n=100, quantile=0.95, age_penalty_rate=0.02):
    """
    Generate movie recommendations based on genre with weighted scoring.

    This function provides genre-specific recommendations by:
    1. Filtering movies by the specified genre
    2. Computing weighted ratings using IMDb formula
    3. Applying age penalty to favor newer movies
    4. Considering genre relevance and popularity

    Parameters:
        genre (str): The genre to base recommendations on
        df (DataFrame): Movie dataset with required columns
        top_n (int, optional): Number of recommendations to return. Defaults to 100.
        quantile (float, optional): Quantile for vote count threshold. Defaults to 0.95.
        age_penalty_rate (float, optional): Rate of age penalty. Defaults to 0.02.

    Returns:
        DataFrame: Recommended movies with columns:
            - id: TMDb movie ID
            - poster_path: Path to movie poster
            - title: Movie title
            - release_date: Release date

    Note:
        The function applies case-insensitive genre matching and handles
        movies with multiple genres by considering genre position in the list.
    """
    global C, m
    if C is None:
        C = df['vote_average'].mean()
    if m is None:
        m = df['vote_count'].quantile(quantile)
    
    genre = genre.lower()
    genre_movies = df[df['genres_list'].apply(lambda genres: genre in genres)].copy()
    genre_movies['weighted_rating'] = genre_movies.apply(lambda x: weighted_rating(x, C, m), axis=1)    
    genre_movies['genre_index'] = genre_movies['genres_list'].apply(lambda x: x.index(genre) if genre in x else float('inf'))
    genre_movies['age_penalty'] = (1 - age_penalty_rate * (current_year - genre_movies['release_year'])).clip(lower=0.5)
    genre_movies['adjusted_score'] = genre_movies['weighted_rating'] * genre_movies['age_penalty']
    top_movies = genre_movies.sort_values(
        by=['genre_index', 'adjusted_score', 'popularity'],
        ascending=[True, False, False]
    ).head(top_n)
    return top_movies[['id', 'poster_path', 'title', 'release_date']]
