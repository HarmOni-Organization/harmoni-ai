# Standard Library Imports
import os
import time
import logging
from functools import lru_cache

# Third-Party Imports
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Local Module Imports
from my_modules import (
    improved_hybrid_recommendations,
    get_movie_poster,
    genre_based_recommender,
    load_data,
    load_model,
    load_count_matrix,
    preprocess_movies,
    create_response,
    setup_logging
)


# Load environment variables from .env file
load_dotenv()

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS) for handling requests from different origins

# Configure logging
setup_logging(app)

# Get logger for this module
logger = logging.getLogger(__name__)

# Load datasets and models
logger.info("Loading datasets and models...")
ratings_df, links_df, new_df = load_data()  # Load movie ratings and metadata
best_svd1 = load_model()  # Load the trained Singular Value Decomposition (SVD) model
count_matrix = load_count_matrix()  # Load precomputed count-based feature matrix
new_df2 = preprocess_movies(new_df)  # Preprocess the movie dataset for recommendations
logger.info("Datasets and models loaded successfully")

# Define cache size for poster retrieval, with a default value of 1000 if not specified in .env
cache_size = int(os.getenv("POSTER_CACHE_SIZE", 1000))


@lru_cache(maxsize=1000)
def cached_get_movie_poster(movie_id, poster_path):
    """
    Retrieves and caches the movie poster URL to optimize repeated requests.

    This function uses LRU (Least Recently Used) caching to store up to 1000 
    movie poster URLs. If the poster URL is not available, it returns an empty string.

    Args:
        movie_id (int): The unique identifier of the movie.
        poster_path (str): The relative path to the movie's poster image.

    Returns:
        str: The URL of the movie's poster or an empty string if unavailable.
    """

    # Create a dictionary containing the movie ID and poster path
    movie = {"id": movie_id, "poster_path": poster_path}
    
    # Fetch and return the movie poster URL; return an empty string if not found
    return get_movie_poster(movie) or ""


@app.route("/")
def home():
    """
    Handles the root endpoint ("/") of the Hybrid Recommendation System API.

    Logs the incoming request details, including the client's IP address and HTTP method.
    Returns a welcome message.

    Returns:
        str: A welcome message for the API.
    """
    logger.info(
        f"Received request from: [{request.remote_addr}] to  [{request.method}]'/'"
    )
    return "Welcome to Hybrid Recommendation System API"


@app.route("/recommend", methods=["GET"])
def recommend():
    """
    Handles movie recommendations based on user and movie input.

    This endpoint processes a request containing `userId`, `movieId`, and an optional `topN` parameter.
    It validates input parameters, logs request details, and generates movie recommendations using
    an improved hybrid recommendation system.

    Query Parameters:
       - userId (str): The ID of the user requesting recommendations. Required.
       - movieId (str): The ID of the movie for generating recommendations. Required.
       - topN (str, optional): The number of top recommendations to return (default is 10, max is 50).

    Returns:
        Response (JSON): A JSON response containing:
            - status (bool): Indicates success or failure of the request.
            - message (str): A message describing the response.
            - data (dict): Contains `userId`, `movieId`, and a list of `recommendedMovies` if successful.

    Possible Response Status Codes:
        - 200: Successfully generated recommendations.
        - 400: Bad request (e.g., missing or invalid parameters).
        - 500: Internal server error.

    Logging:
        - Logs request metadata, including client IP, user agent, and query parameters.
        - Logs warnings for missing or invalid inputs.
        - Logs errors if the recommendation process encounters an exception.
        - Logs the number of generated recommendations and response times.

    Example:
        Request: GET /recommend?userId=123&movieId=456&topN=5
        Response:
        {
            "status": True,
            "message": "Recommendations generated successfully",
            "data": {
                "userId": "123",
                "movieId": "456",
                "recommendedMovies": [
                    {"id": 789, "title": "Movie Title", "poster_url": "URL"}
                ]
            }
        }
    """
    
    # Track request processing time
    start_time = time.time()
    
    # Extract query parameters from the request
    userId = request.args.get("userId")
    movieId = request.args.get("movieId")
    topN = request.args.get("topN", "10")

    # Log incoming request details
    logger.info(
        f"Received request from: [{request.remote_addr}] to  [{request.method}]:'/rcommend'",
        extra={
            "http_method": request.method,
            "remote_ip": request.remote_addr,
            "user_agent": request.user_agent.string,
            "query_params": request.args.to_dict(),
        },
    )
    
    # Create request info dictionary for logging
    request_info = {
        "http_method": request.method,
        "remote_ip": request.remote_addr,
        "user_agent": request.user_agent.string,
        "query_params": request.args.to_dict(),
    }

    # Validate required parameters
    if not userId or not movieId:
        logger.warning(
            "Missing userId or movieId",
            extra=request_info
        )
        return create_response(
            status=False,
            message="userId and movieId are required",
            status_code=400,
            start_time=start_time,
            request_info=request_info
        )

    try:
        # Ensure movieId is a valid non-negative integer
        movieId = int(movieId)
        if movieId < 0:
            logger.warning(
                "Negative movieId",
                extra=request_info
            )
            return create_response(
                status=False,
                message="movieId must be non-negative",
                status_code=400,
                start_time=start_time,
                request_info=request_info
            )
    except ValueError:
        logger.warning(
            "Invalid movieId format",
            extra=request_info
        )
        return create_response(
            status=False,
            message="movieId must be valid integer",
            status_code=400,
            start_time=start_time,
            request_info=request_info
        )

    try:
        # Ensure topN is a valid integer within limits (1 to 50)
        topN = max(1, min(int(topN), 50))
    except ValueError:
        # Default value in case of invalid input
        topN = 10

    try:
        # Generate recommendations using the hybrid recommendation system
        recommendations, error_message = improved_hybrid_recommendations(
            user_id=userId,
            movie_id=movieId,
            top_n=topN,
            ratings_df=ratings_df,
            links_df=links_df,
            new_df=new_df,
            best_svd_model=best_svd1,
            count_matrix=count_matrix,
        )
    except Exception as e:
        # Log and return an internal server error response
        logger.error(f"Recommendation error: {str(e)}", exc_info=True)
        return create_response(
            status=False,
            message="Internal server error occurred",
            status_code=500,
            start_time=start_time,
            request_info=request_info
        )

    # Handle errors returned by the recommendation system
    if error_message:
        logger.warning(
            f"Recommendation failed: {error_message}", 
            extra={"error": error_message, **request_info}
        )
        return create_response(
            status=False,
            message=error_message,
            status_code=400,
            start_time=start_time,
            request_info=request_info
        )

    # If no recommendations were found, return an empty list
    if recommendations is None or recommendations.empty:
        logger.info(
            f"No recommendations found for userId-{userId}, movieId-{movieId}",
            extra=request_info
        )
        return create_response(
            status=True,
            message="",
            data={
                "userId": userId,
                "movieId": movieId,
                "recommendedMovies": [],
            },
            status_code=200,
            start_time=start_time,
            request_info=request_info
        )

    # Convert recommendations to a dictionary format and fetch movie posters
    result = recommendations.to_dict(orient="records")
    for movie in result:
        movie["poster_url"] = cached_get_movie_poster(movie["id"], movie["poster_path"])

    # Log successful recommendations
    logger.info(
        f"Generated {len(result)} recommendations for user {userId}, movieId {movieId}",
        extra=request_info
    )
    
    # Return recommendations as a JSON response using the response handler
    return create_response(
        status=True,
        message="Recommendations generated successfully",
        data={
            "userId": userId,
            "movieId": movieId,
            "recommendedMovies": result,
        },
        status_code=200,
        start_time=start_time,
        request_info=request_info
    )


@app.route("/genreBasedRecommendation", methods=["GET"])
def genreBasedRecommendation():
    """
    Provides movie recommendations based on a specified genre.

    This endpoint processes a request containing a `genre` parameter and an optional `topN` parameter.
    It validates input parameters, logs request details, and generates a list of recommended movies
    based on the specified genre.

    Query Parameters:
        genre (str): The movie genre for which recommendations are requested. Required.
        topN (str, optional): The number of top recommendations to return (default is 100, max is 5000).

    Returns:
        Response (JSON): A JSON response containing:
            - status (bool): Indicates success or failure of the request.
            - message (str): A message describing the response.
            - data (dict): Contains `genre` and a list of `recommendedMovies` if successful.

    Possible Response Status Codes:
        - 200: Successfully generated genre-based recommendations.
        - 400: Bad request (e.g., missing or invalid parameters, no movies found for the given genre).
        - 500: Internal server error.

    Logging:
        - Logs request metadata, including client IP, user agent, and query parameters.
        - Logs warnings for missing genre input or if no movies match the specified genre.
        - Logs errors if the recommendation process encounters an exception.
        - Logs the number of generated recommendations and response times.

    Example:
        Request: GET /genreBasedRecommendation?genre=Action&topN=5
        Response:
        {
            "status": True,
            "message": "Recommendations generated successfully",
            "data": {
                "genre": "Action",
                "recommendedMovies": [
                    {"id": 123, "title": "Action Movie", "poster_url": "URL"}
                ]
            }
        }
    """
    # Start measuring response time
    start_time = time.time()
    
    # Retrieve request parameters
    genre = request.args.get("genre")
    topN = request.args.get("topN", 100)

    # Create request info dictionary for logging
    request_info = {
        "http_method": request.method,
        "remote_ip": request.remote_addr,
        "user_agent": request.user_agent.string,
        "query_params": request.args.to_dict(),
    }

    # Log request details
    logger.info(
        f"Received request from: [{request.remote_addr}] to  [{request.method}]:'/genreBasedRecommendation'",
        extra=request_info
    )

    # Validate the genre parameter
    if not genre:
        logger.warning(
            "Missing genre",
            extra=request_info
        )
        return create_response(
            status=False,
            message="Movie genre is required",
            status_code=400,
            start_time=start_time,
            request_info=request_info
        )

    # Validate and convert `topN` to an integer within the allowed range (1 to 5000)
    try:
        topN = max(1, min(int(topN), 5000))
    except ValueError:
        topN = 100

    try:
        # Format the genre input for consistency
        genre = genre.title()
        
        # Generate genre-based recommendations
        recommendations = genre_based_recommender(genre=genre, df=new_df2, top_n=topN)
    except Exception as e:
        logger.error(f"Genre recommendation error: {str(e)}", exc_info=True)
        return create_response(
            status=False,
            message="Internal server error occurred",
            status_code=500,
            start_time=start_time,
            request_info=request_info
        )

    # Handle case where no recommendations are found
    if recommendations is None or recommendations.empty:
        logger.warning(
            f"No movies found for genre {genre}",
            extra=request_info
        )
        return create_response(
            status=False,
            message=f"No movies found for genre: {genre}",
            status_code=400,
            start_time=start_time,
            request_info=request_info
        )

    # Convert recommendations to a dictionary format for JSON response
    result = recommendations.to_dict(orient="records")
    
    # Fetch movie poster URLs for each recommended movie
    for movie in result:
        movie["poster_url"] = cached_get_movie_poster(movie["id"], movie["poster_path"])

    # Log successful recommendation generation
    logger.info(
        f"Generated {len(result)} genre recommendations for genre {genre}",
        extra=request_info
    )
    
    # Return response using the response handler
    return create_response(
        status=True,
        message="Recommendations generated successfully",
        data={
            "genre": genre,
            "recommendedMovies": result,
        },
        status_code=200,
        start_time=start_time,
        request_info=request_info
    )


if __name__ == "__main__":
    app.run(debug=True)
