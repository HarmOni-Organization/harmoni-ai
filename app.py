from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from my_modules.myModule import (
    improved_hybrid_recommendations,
    get_movie_poster,
    genre_based_recommender,
    load_data,
    load_model,
    load_count_matrix,
    create_indices,
    preprocess_movies,
)
import logging
from functools import lru_cache
import re

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask App
app = Flask(__name__)

# Load data and model once at startup
ratings_df, links_df, new_df = load_data()
best_svd1 = load_model()
count_matrix = load_count_matrix()
indices = create_indices(new_df)
new_df2 = preprocess_movies(new_df)

def validate_input(text):
    """
    Validate input text for special characters and unicode.

    This function checks if the input text contains only allowed characters:
    - Alphanumeric characters
    - Spaces
    - Common punctuation marks (-'":,.!?())

    Parameters:
        text (str): The input text to validate

    Returns:
        bool: True if text contains only allowed characters, False otherwise
    """
    # Allow only alphanumeric characters, spaces, and common punctuation
    pattern = r'^[a-zA-Z0-9\s\-\'\":,.!?()]+$'
    return bool(re.match(pattern, text))

# Caching for movie posters
@lru_cache(maxsize=1000)
def cached_get_movie_poster(movie_id, poster_path):
    """
    Retrieve and cache movie poster URLs.

    This function uses LRU caching to efficiently retrieve movie poster URLs,
    reducing API calls to TMDb. It handles missing posters gracefully.

    Parameters:
        movie_id (int): The TMDb ID of the movie
        poster_path (str): The path to the movie poster

    Returns:
        str: URL to the movie poster image, or empty string if not found
    """
    movie = {"id": movie_id, "poster_path": poster_path}
    return get_movie_poster(movie) or ""  # Return empty string if poster not found

@app.route("/")
def home():
    """
    Render the homepage of the application.

    This route serves the main landing page of the movie recommendation system.
    It displays the application title and provides access to the recommendation features.

    Returns:
        HTML: The rendered index.html template
    """
    return render_template("index.html")

@app.route("/recommend", methods=["GET"])
def recommend():
    """
    Generate personalized movie recommendations based on user ID and movie title.

    This endpoint combines content-based and collaborative filtering to provide
    personalized movie recommendations. It handles various edge cases and validates
    input parameters.

    Query Parameters:
        userId (int): The ID of the user requesting recommendations
        title (str): The title of the reference movie
        topN (int, optional): Number of recommendations to return. Defaults to 10.

    Returns:
        JSON: Response containing:
            - status (bool): Success status of the request
            - data (dict): Contains userId, title, and recommendedMovies
            - message (str, optional): Error message if status is False

    Status Codes:
        200: Successful recommendation
        400: Invalid input parameters
        500: Internal server error
    """
    try:
        userId = request.args.get("userId")
        title = request.args.get("title")
        topN = request.args.get("topN", "10")

        if not userId or not title:
            return jsonify({"status": False, "message": "userId and title are required"}), 400

        # Validate title for special characters
        if not validate_input(title):
            return jsonify({"status": False, "message": "Invalid characters in title"}), 400

        try:
            userId = int(userId)
            if userId < 0:
                return jsonify({"status": False, "message": "userId must be non-negative"}), 400
        except ValueError:
            return jsonify({"status": False, "message": "userId must be a valid integer"}), 400

        try:
            topN = max(1, min(int(topN), 50))  # Ensure minimum of 1, maximum of 50 for performance
        except ValueError:
            topN = 10  # Default to 10 if invalid

        recommendations = improved_hybrid_recommendations(
            user_id=userId,
            title=title,
            top_n=topN,
            ratings_df=ratings_df,
            links_df=links_df,
            new_df=new_df,
            best_svd_model=best_svd1,
            count_matrix=count_matrix,
            indices=indices,
        )
        
        if recommendations is None or recommendations.empty:
            return jsonify({"status": True, "data": {"userId": userId, "title": title, "recommendedMovies": []}}), 200

        result = recommendations.to_dict(orient="records")
        for movie in result:
            movie["poster_url"] = cached_get_movie_poster(movie["id"], movie["poster_path"])
        
        logging.info(f"Recommendations generated for user {userId}, title {title}")
        return jsonify({"status": True, "data": {"userId": userId, "title": title, "recommendedMovies": result}})
    except ValueError as ve:
        logging.error(f"Validation error: {ve}")
        return jsonify({"status": False, "message": f"Invalid input: {ve}"}), 400
    except Exception as e:
        logging.error(f"Server error: {e}")
        return jsonify({"status": False, "message": "Internal server error"}), 500

@app.route("/genreBasedRecommendation", methods=["GET"])
def genreBasedRecommendation():
    """
    Generate movie recommendations based on a specific genre.

    This endpoint provides genre-specific movie recommendations using weighted
    ratings and recency bias. It handles genre validation and case sensitivity.

    Query Parameters:
        genre (str): The genre to base recommendations on
        topN (int, optional): Number of recommendations to return. Defaults to 100.

    Returns:
        JSON: Response containing:
            - status (bool): Success status of the request
            - data (dict): Contains genre and recommendedMovies
            - message (str, optional): Error message if status is False

    Status Codes:
        200: Successful recommendation
        400: Invalid genre or no movies found
        500: Internal server error
    """
    try:
        genre = request.args.get("genre")
        topN = request.args.get("topN", "100")

        if not genre:
            return jsonify({"status": False, "message": "Movie genre is required"}), 400

        # Validate genre for special characters
        if not validate_input(genre):
            return jsonify({"status": False, "message": "Invalid characters in genre"}), 400

        try:
            topN = max(1, min(int(topN), 100))  # Ensure minimum of 1, maximum of 100
        except ValueError:
            topN = 100  # Default to 100 if invalid

        # Convert genre to title case for consistency
        genre = genre.title()

        recommendations = genre_based_recommender(genre=genre, df=new_df2, top_n=topN)
        
        if recommendations is None or recommendations.empty:
            return jsonify({"status": False, "message": f"No movies found for genre: {genre}"}), 400

        result = recommendations.to_dict(orient="records")
        for movie in result:
            movie["poster_url"] = cached_get_movie_poster(movie["id"], movie["poster_path"])
        
        return jsonify(
            {
                "status": True,
                "message": "",
                "data": {
                    "genre": genre,
                    "recommendedMovies": result,
                },
            }
        )
    except ValueError as ve:
        logging.error(f"Validation error: {ve}")
        return jsonify({"status": False, "message": f"Invalid input: {ve}"}), 400
    except Exception as e:
        logging.error(f"Server error: {e}")
        return jsonify({"status": False, "message": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True)