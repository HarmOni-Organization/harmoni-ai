# Standard Library Imports
import os
import time
import logging
from functools import lru_cache

# Third-Party Imports
from flask import Flask, request, jsonify, g
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
from my_modules.auth import require_auth


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
@require_auth  # Add authentication requirement
def recommend():
    """
    Handles movie recommendations based on user and movie input.

    This endpoint processes a request containing `movieId` and an optional `topN` parameter.
    It extracts the userId from the authentication token, validates input parameters,
    logs request details, and generates movie recommendations using an improved hybrid recommendation system.

    Query Parameters:
       - movieId (str): The ID of the movie for generating recommendations. Required.
       - topN (str, optional): The number of top recommendations to return (default is 10, max is 50).

    Returns:
        Response (JSON): A JSON response containing:
            - status (bool): Indicates success or failure of the request.
            - message (str): A message describing the response.
            - data (dict): Contains `userId`, `movieId`, and a list of `recommendedMovies` if successful.
    """
    
    # Track request processing time
    start_time = time.time()
    
    # Extract userId from the authentication token
    userId = g.user.get('userId')
    
    # Extract query parameters from the request
    movieId = request.args.get("movieId")
    topN = request.args.get("topN", "10")

    # Log incoming request details
    logger.info(
        f"Received request from: [{request.remote_addr}] to  [{request.method}]:'/recommend'",
        extra={
            "http_method": request.method,
            "remote_ip": request.remote_addr,
            "user_agent": request.user_agent.string,
            "query_params": request.args.to_dict(),
            "user_id": userId
        },
    )
    
    # Create request info dictionary for logging
    request_info = {
        "http_method": request.method,
        "remote_ip": request.remote_addr,
        "user_agent": request.user_agent.string,
        "query_params": request.args.to_dict(),
        "user_id": userId
    }

    # Validate required parameters
    if not userId:
        logger.warning(
            "Missing userId in token",
            extra=request_info
        )
        return create_response(
            status=False,
            message="User identification not found in token",
            status_code=400,
            start_time=start_time,
            request_info=request_info
        )
        
    if not movieId:
        logger.warning(
            "Missing movieId",
            extra=request_info
        )
        return create_response(
            status=False,
            message="movieId is required",
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
@require_auth  # Add authentication requirement
def genreBasedRecommendation():
    """
    Handles movie recommendations based on a specific genre.
    
    This endpoint processes a request containing a `genre` parameter and an optional `topN` parameter.
    It extracts the userId from the authentication token, validates input parameters,
    logs request details, and generates movie recommendations for the specified genre.
    
    Query Parameters:
        - genre (str): The movie genre for generating recommendations. Required.
        - topN (str, optional): The number of top recommendations to return (default is 100, max is 5000).
    
    Returns:
        Response (JSON): A JSON response containing:
            - status (bool): Indicates success or failure of the request.
            - message (str): A message describing the response.
            - data (dict): Contains `userId`, `genre`, and a list of `recommendedMovies` if successful.
            - response_time (float): The time taken to process the request (if enabled).
    
    Raises:
        Exception: If an error occurs during the recommendation process.
    """
    # Start measuring response time
    start_time = time.time()
    
    # Extract userId from the authentication token
    userId = g.user.get('userId')
    
    # Retrieve request parameters
    genre = request.args.get("genre")
    topN = request.args.get("topN", 100)

    # Create request info dictionary for logging
    request_info = {
        "http_method": request.method,
        "remote_ip": request.remote_addr,
        "user_agent": request.user_agent.string,
        "query_params": request.args.to_dict(),
        "user_id": userId
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
            "userId": userId,
            "genre": genre,
            "recommendedMovies": result,
        },
        status_code=200,
        start_time=start_time,
        request_info=request_info
    )


# Add a test endpoint to verify authentication is working
@app.route("/auth-test")
@require_auth
def auth_test():
    """
    Test endpoint to verify authentication is working.
    
    This endpoint requires valid authentication and returns the user information from the token.
    It logs the request details and handles potential edge cases with the user data.
    
    Returns:
        Response (JSON): A JSON response containing the user information from the token.
    """
    # Start measuring response time
    start_time = time.time()
    
    # Create request info dictionary for logging
    request_info = {
        "http_method": request.method,
        "remote_ip": request.remote_addr,
        "user_agent": request.user_agent.string,
        "path": request.path
    }
    
    # Log the authentication test request
    logger.info(
        f"Authentication test request from: [{request.remote_addr}] to [{request.method}]:'/auth-test'",
        extra=request_info
    )
    
    # Check if user data exists in g object
    if not hasattr(g, 'user') or not g.user:
        logger.warning(
            "Authentication test failed: No user data available",
            extra=request_info
        )
        return create_response(
            status=False,
            message="Authentication successful but no user data available",
            status_code=500,
            start_time=start_time,
            request_info=request_info
        )
    
    # Sanitize user data for logging (remove sensitive information)
    safe_user_data = {k: v for k, v in g.user.items() if k.lower() not in ('password', 'token', 'secret')}
    
    # Log successful authentication with user info
    logger.info(
        f"Authentication test successful for user: {safe_user_data.get('username', safe_user_data.get('sub', 'unknown'))}",
        extra={**request_info, "user_id": safe_user_data.get('userId', safe_user_data.get('sub', 'unknown'))}
    )
    
    # Return successful response with user data
    return create_response(
        status=True,
        message="Authentication successful",
        data={"user": g.user},
        status_code=200,
        start_time=start_time,
        request_info=request_info
    )


# Add a 404 error handler for routes that don't exist
@app.errorhandler(404)
def not_found(e):
    """
    Handles requests to non-existent routes.
    
    This function is triggered when a client attempts to access a route that doesn't exist.
    It returns a standardized JSON response with a 404 status code.
    
    Args:
        e: The error object passed by Flask.
        
    Returns:
        Response (JSON): A JSON response with a 404 status code and error message.
    """
    # Log the 404 error
    logger.warning(
        f"404 Not Found: {request.path}",
        extra={
            "http_method": request.method,
            "remote_ip": request.remote_addr,
            "user_agent": request.user_agent.string,
            "path": request.path
        }
    )
    
    # Return a standardized response
    return create_response(
        status=False,
        message=f"Endpoint not found: {request.path}",
        status_code=404,
        start_time=time.time(),
        request_info={
            "http_method": request.method,
            "remote_ip": request.remote_addr,
            "user_agent": request.user_agent.string,
            "path": request.path
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
