# Standard Library Imports
import os
import time
import logging
from functools import lru_cache

# Third-Party Imports
from flask import Blueprint, request, g

# Local Module Imports
from my_modules import (
    improved_hybrid_recommendations,
    get_movie_poster,
    genre_based_recommender,
    create_response
)
from my_modules.auth import require_auth

# Get logger for this module
logger = logging.getLogger(__name__)

# Define cache size for poster retrieval, with a default value of 1000 if not specified in .env
cache_size = int(os.getenv("POSTER_CACHE_SIZE", 1000))

# Create a blueprint for movie-related endpoints
movies_bp = Blueprint('movies', __name__, url_prefix='/api/v1/movies')

@movies_bp.route("/recommend", methods=["GET"])
@require_auth
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
        f"Received request from: [{request.remote_addr}] to  [{request.method}]:'/api/v1/movies/recommend'",
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

    # Import these at function level to avoid circular imports
    from app import ratings_df, links_df, new_df, best_svd1, count_matrix

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

    # Convert recommendations to a dictionary format
    result = recommendations.to_dict(orient="records")
    
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


@movies_bp.route("/genre-based", methods=["GET"])
@require_auth
def genre_based_recommendation():
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
        f"Received request from: [{request.remote_addr}] to  [{request.method}]:'/api/v1/movies/genre-based'",
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

    # Import at function level to avoid circular imports
    from app import new_df2

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


@movies_bp.route("/poster", methods=["GET"])
@require_auth
def get_poster():
    """
    Retrieves a movie poster URL based on movie ID and poster path.
    
    This endpoint accepts movieId and posterPath parameters and returns
    the corresponding poster URL. It uses caching to optimize repeated requests.
    
    Query Parameters:
        - movieId (str): The ID of the movie. Required.
        - posterPath (str): The relative path to the movie's poster image. Required.
        
    Returns:
        Response (JSON): A JSON response containing:
            - status (bool): Indicates success or failure of the request.
            - message (str): A message describing the response.
            - data (dict): Contains the poster URL if successful.
    """
    # Start measuring response time
    start_time = time.time()
    
    # Extract query parameters
    movie_id = request.args.get("movieId")
    poster_path = request.args.get("posterPath")
    
    # Create request info dictionary for logging
    request_info = {
        "http_method": request.method,
        "remote_ip": request.remote_addr,
        "user_agent": request.user_agent.string,
        "query_params": request.args.to_dict()
    }
    
    # Log request details
    logger.info(
        f"Received request from: [{request.remote_addr}] to [{request.method}]:'/api/v1/movies/poster'",
        extra=request_info
    )
    
    # Validate required parameters
    if not movie_id:
        logger.warning("Missing movieId", extra=request_info)
        return create_response(
            status=False,
            message="movieId is required",
            status_code=400,
            start_time=start_time,
            request_info=request_info
        )
    
    if not poster_path:
        logger.warning("Missing posterPath", extra=request_info)
        return create_response(
            status=False,
            message="posterPath is required",
            status_code=400,
            start_time=start_time,
            request_info=request_info
        )
    
    try:
        # Ensure movieId is a valid integer
        movie_id = int(movie_id)
    except ValueError:
        logger.warning("Invalid movieId format", extra=request_info)
        return create_response(
            status=False,
            message="movieId must be a valid integer",
            status_code=400,
            start_time=start_time,
            request_info=request_info
        )
    
    # Get the poster URL
    poster_url = cached_get_movie_poster(movie_id, poster_path)
    
    # Return the poster URL
    return create_response(
        status=True,
        message="Poster URL retrieved successfully",
        data={"posterUrl": poster_url},
        status_code=200,
        start_time=start_time,
        request_info=request_info
    )
