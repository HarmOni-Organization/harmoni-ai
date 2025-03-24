from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from my_modules import (
    improved_hybrid_recommendations,
    get_movie_poster,
    genre_based_recommender,
    load_data,
    load_model,
    load_count_matrix,
    preprocess_movies,
    fuzzy_title_match,
)
import os
import logging
from functools import lru_cache
import time

load_dotenv()

app = Flask(__name__)

# Configure root logger to handle ALL logs
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Remove any existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Create and configure handler
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s","message": "%(message)s", "path": "%(pathname)s", "line": "%(lineno)d"}'
)
handler.setFormatter(formatter)

# Add handler to root logger
logger.addHandler(handler)

# Specifically configure Flask's logger to propagate to root
app.logger.propagate = True
app.logger.handlers.clear()

# Add this to ensure third-party libraries don't add extra handlers
logging.getLogger("werkzeug").handlers.clear()
logging.getLogger("werkzeug").propagate = True


ratings_df, links_df, new_df = load_data()
best_svd1 = load_model()
count_matrix = load_count_matrix()
new_df2 = preprocess_movies(new_df)

cache_size = int(os.getenv("POSTER_CACHE_SIZE", 1000))


@lru_cache(maxsize=1000)
def cached_get_movie_poster(movie_id, poster_path):
    movie = {"id": movie_id, "poster_path": poster_path}
    return get_movie_poster(movie) or ""


@app.route("/")
def home():
    logger.info(
        f"Received request from: [{request.remote_addr}] to  [{request.method}]'/'"
    )
    return "Welcome to Hybrid Recommendation System API"


# @app.route("/movies", methods=["GET"])
# def get_movies():
#     start_time = time.time()
#     logger.info(
#         f"Received request from: [{request.remote_addr}] to  [{request.method}]:'/movies'",
#         extra={
#             "http_method": request.method,
#             "remote_ip": request.remote_addr,
#             "user_agent": request.user_agent.string,
#             "query_params": request.args.to_dict(),
#         },
#     )
#     search_query = request.args.get("search", "").strip()
#     page = int(request.args.get("page", 1))
#     per_page = 10

#     if not search_query or page < 1:
#         response = jsonify([])
#         response.status_code = 200
#         end_time = time.time()
#         logger.info(
#             f"Response Sent - Status Code:[{response.status_code}] - Response Time: {end_time - start_time:.2f}s",
#             extra={
#                 "status_code": response.status_code,
#                 "response_time": end_time - start_time,
#             },
#         )
#         return response

#     try:
#         results = fuzzy_title_match(search_query, page, per_page, new_df)
#         response = jsonify(results)
#         response.status_code = 200
#         end_time = time.time()
#         logger.info(
#             f"Response Sent - Status Code:[{response.status_code}] - Response Time: {end_time - start_time:.2f}s",
#             extra={
#                 "status_code": response.status_code,
#                 "response_time": end_time - start_time,
#             },
#         )
#         return response
#     except Exception as e:
#         app.logger.error(f"Error in movie search: {str(e)}", extra={"error": str(e)})
#         response = jsonify([])
#         response.status_code = 500
#         end_time = time.time()
#         logger.info(
#             f"Response Sent - Status Code:[{response.status_code}] - Response Time: {end_time - start_time:.2f}s",
#             extra={
#                 "status_code": response.status_code,
#                 "response_time": end_time - start_time,
#             },
#         )
#         return response


@app.route("/recommend", methods=["GET"])
def recommend():
    start_time = time.time()
    userId = request.args.get("userId")
    movieId = request.args.get("movieId")
    topN = request.args.get("topN", "10")

    logger.info(
        f"Received request from: [{request.remote_addr}] to  [{request.method}]:'/rcommend'",
        extra={
            "http_method": request.method,
            "remote_ip": request.remote_addr,
            "user_agent": request.user_agent.string,
            "query_params": request.args.to_dict(),
        },
    )

    if not userId or not movieId:
        logger.warning(
            "Missing userId or movieId",
            extra={
                "http_method": request.method,
                "remote_ip": request.remote_addr,
                "user_agent": request.user_agent.string,
                "query_params": request.args.to_dict(),
            },
        )
        response = jsonify(
            {"status": False, "message": "userId and movieId are required"}
        )
        response.status_code = 400
        end_time = time.time()
        logger.info(
            f"Response Sent - Status Code:[{response.status_code}] - Response Time: {(end_time - start_time):0.2f}s",
            extra={
                "status_code": response.status_code,
                "response_time": end_time - start_time,
            },
        )
        return response, 400

    try:
        movieId = int(movieId)
        if movieId < 0:
            logger.warning(
                "Negative movieId",
                extra={
                    "http_method": request.method,
                    "remote_ip": request.remote_addr,
                    "user_agent": request.user_agent.string,
                    "query_params": request.args.to_dict(),
                },
            )
            response = jsonify(
                {
                    "status": False,
                    "message": "movieId must be non-negative",
                }
            )
            response.status_code = 400
            end_time = time.time()
            logger.info(
                f"Response Sent - Status Code:[{response.status_code}] - Response Time: {end_time - start_time:.2f}s",
                extra={
                    "status_code": response.status_code,
                    "response_time": end_time - start_time,
                },
            )
            return response, 400
    except ValueError:
        logger.warning(
            "Invalid movieId format",
            extra={
                "http_method": request.method,
                "remote_ip": request.remote_addr,
                "user_agent": request.user_agent.string,
                "query_params": request.args.to_dict(),
            },
        )
        response = jsonify(
            {
                "status": False,
                "message": "movieId must be valid integer",
            }
        )
        response.status_code = 400
        end_time = time.time()
        logger.info(
            f"Response Sent - Status Code:[{response.status_code}] - Response Time: {end_time - start_time:.2f}s",
            extra={
                "status_code": response.status_code,
                "response_time": end_time - start_time,
            },
        )
        return response, 400

    try:
        topN = max(1, min(int(topN), 50))
    except ValueError:
        topN = 10

    try:
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
        logger.error(f"Recommendation error: {str(e)}", exc_info=True)
        return (
            jsonify({"status": False, "message": "Internal server error occurred"}),
            500,
        )

    if error_message:
        logger.warning(
            f"Recommendation failed: {error_message}", extra={"error": error_message}
        )
        response = jsonify({"status": False, "message": error_message})
        response.status_code = 400
        end_time = time.time()
        logger.info(
            f"Response Sent - Status Code:[{response.status_code}] - Response Time: {end_time - start_time:.2f}s",
            extra={
                "status_code": response.status_code,
                "response_time": end_time - start_time,
            },
        )
        return response, 400

    if recommendations is None or recommendations.empty:
        logger.info(
            f"No recommendations found for userId-{userId}, movieId-{movieId}",
            extra={
                "http_method": request.method,
                "remote_ip": request.remote_addr,
                "user_agent": request.user_agent.string,
                "query_params": request.args.to_dict(),
            },
        )
        response = jsonify(
            {
                "status": True,
                "message": "",
                "data": {
                    "userId": userId,
                    "movieId": movieId,
                    "recommendedMovies": [],
                },
            }
        )
        response.status_code = 200
        end_time = time.time()
        logger.info(
            f"Response Sent - Status Code:[{response.status_code}] - Response Time: {end_time - start_time:.2f}s",
            extra={
                "status_code": response.status_code,
                "response_time": end_time - start_time,
            },
        )
        return response, 200

    result = recommendations.to_dict(orient="records")
    for movie in result:
        movie["poster_url"] = cached_get_movie_poster(movie["id"], movie["poster_path"])

    logger.info(
        f"Generated {len(result)} recommendations for user {userId}, movieId {movieId}",
        extra={
            "http_method": request.method,
            "remote_ip": request.remote_addr,
            "user_agent": request.user_agent.string,
            "query_params": request.args.to_dict(),
        },
    )
    response = jsonify(
        {
            "status": True,
            "message": "Recommendations generated successfully",
            "data": {
                "userId": userId,
                "movieId": movieId,
                "recommendedMovies": result,
            },
        }
    )
    response.status_code = 200
    end_time = time.time()
    logger.info(
        f"Response Sent - Status Code:[{response.status_code}] - Response Time: {end_time - start_time:.2f}s",
        extra={
            "status_code": response.status_code,
            "response_time": end_time - start_time,
        },
    )
    return response, 200


@app.route("/genreBasedRecommendation", methods=["GET"])
def genreBasedRecommendation():
    start_time = time.time()
    genre = request.args.get("genre")
    topN = request.args.get("topN", 100)

    logger.info(
        f"Received request from: [{request.remote_addr}] to  [{request.method}]:'/genreBasedRecommendation'",
        extra={
            "http_method": request.method,
            "remote_ip": request.remote_addr,
            "user_agent": request.user_agent.string,
            "query_params": request.args.to_dict(),
        },
    )

    if not genre:
        logger.warning(
            "Missing genre",
            extra={
                "http_method": request.method,
                "remote_ip": request.remote_addr,
                "user_agent": request.user_agent.string,
                "query_params": request.args.to_dict(),
            },
        )
        response = jsonify({"status": False, "message": "Movie genre is required"})
        response.status_code = 400
        end_time = time.time()
        logger.info(
            f"Response Sent - Status Code:[{response.status_code}] - Response Time: {end_time - start_time:.2f}s",
            extra={
                "status_code": response.status_code,
                "response_time": end_time - start_time,
            },
        )
        return response, 400

    try:
        topN = max(1, min(int(topN), 5000))
    except ValueError:
        topN = 100

    try:
        genre = genre.title()
        recommendations = genre_based_recommender(genre=genre, df=new_df2, top_n=topN)
    except Exception as e:
        logger.error(f"Genre recommendation error: {str(e)}", exc_info=True)
        return (
            jsonify({"status": False, "message": "Internal server error occurred"}),
            500,
        )

    if recommendations is None or recommendations.empty:
        logger.warning(
            f"No movies found for genre {genre}",
            extra={
                "http_method": request.method,
                "remote_ip": request.remote_addr,
                "user_agent": request.user_agent.string,
                "query_params": request.args.to_dict(),
            },
        )
        response = jsonify(
            {"status": False, "message": f"No movies found for genre: {genre}"}
        )
        response.status_code = 400
        end_time = time.time()
        logger.info(
            f"Response Sent - Status Code:[{response.status_code}] - Response Time: {end_time - start_time:.2f}s",
            extra={
                "status_code": response.status_code,
                "response_time": end_time - start_time,
            },
        )
        return response, 400

    result = recommendations.to_dict(orient="records")
    for movie in result:
        movie["poster_url"] = cached_get_movie_poster(movie["id"], movie["poster_path"])

    logger.info(
        f"Generated {len(result)} genre recommendations for genre {genre}",
        extra={
            "http_method": request.method,
            "remote_ip": request.remote_addr,
            "user_agent": request.user_agent.string,
            "query_params": request.args.to_dict(),
        },
    )
    response = jsonify(
        {
            "status": True,
            "message": "Recommendations generated successfully",
            "data": {"genre": genre, "recommendedMovies": result},
        }
    )
    response.status_code = 200
    end_time = time.time()
    logger.info(
        f"Response Sent - Status Code:[{response.status_code}] - Response Time: {end_time - start_time:.2f}s",
        extra={
            "status_code": response.status_code,
            "response_time": end_time - start_time,
        },
    )
    return response, 200


if __name__ == "__main__":
    app.run(debug=True)
