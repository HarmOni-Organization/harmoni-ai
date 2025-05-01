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
from routes import blueprints


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


@app.route("/api/v1/")
def home():
    """
    Handles the root endpoint ("/") of the Hybrid Recommendation System API.
    
    This endpoint provides general information about the API, including version,
    available endpoints, and basic usage instructions. It logs the incoming request
    details and returns a structured JSON response.
    
    Returns:
        Response (JSON): A JSON response containing API information and documentation.
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
    
    # Log the request
    logger.info(
        f"Received request from: [{request.remote_addr}] to [{request.method}]:'/'",
        extra=request_info
    )
    
    # API information
    api_info = {
        "name": "Harmoni AI - Movie Recommendation System API",
        "version": "1.0.0",
        "description": "A sophisticated movie recommendation system that leverages hybrid recommendation techniques to provide personalized movie suggestions.",
        "endpoints": [
            {
                "path": "/api/v1/movies/recommend",
                "method": "GET",
                "description": "Get personalized movie recommendations based on user ID and movie ID",
                "authentication": "Required",
                "parameters": [
                    {"name": "movieId", "type": "integer", "required": True, "description": "ID of the movie to base recommendations on"},
                    {"name": "topN", "type": "integer", "required": False, "default": 10, "max": 50, "description": "Number of recommendations to return"}
                ]
            },
            {
                "path": "/api/v1/movies/genre-based",
                "method": "GET",
                "description": "Get movie recommendations based on a specific genre",
                "authentication": "Required",
                "parameters": [
                    {"name": "genre", "type": "string", "required": True, "description": "Movie genre to get recommendations for"},
                    {"name": "topN", "type": "integer", "required": False, "default": 100, "max": 5000, "description": "Number of recommendations to return"}
                ]
            },
            {
                "path": "/api/v1/movies/poster",
                "method": "GET",
                "description": "Get a movie poster URL",
                "authentication": "Not required",
                "parameters": [
                    {"name": "movieId", "type": "integer", "required": True, "description": "ID of the movie"},
                    {"name": "posterPath", "type": "string", "required": True, "description": "Relative path to the movie's poster image"}
                ]
            },
            {
                "path": "/auth-test",
                "method": "GET",
                "description": "Test endpoint to verify authentication",
                "authentication": "Required",
                "parameters": []
            }
        ],
        "authentication": "This API uses token-based authentication. Include the authentication token in the request header: Authorization: Bearer <your_token>"
    }
    
    # Return API information as a JSON response
    return create_response(
        status=True,
        message="Welcome to Harmoni AI - Movie Recommendation System API",
        data=api_info,
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


# Register all blueprints
for blueprint in blueprints:
    app.register_blueprint(blueprint)

if __name__ == "__main__":
    app.run(debug=True)
