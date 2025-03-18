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
    fuzzy_title_match
)
import os
import logging
from functools import lru_cache
from rapidfuzz import process, fuzz

load_dotenv()
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

ratings_df, links_df, new_df = load_data()
best_svd1 = load_model()
count_matrix = load_count_matrix()
new_df2 = preprocess_movies(new_df)

cache_size = int(os.getenv("POSTER_CACHE_SIZE", 1000))


@lru_cache(
    maxsize=1000
)  # Adjust based on memory usage and request patterns; monitor in production
def cached_get_movie_poster(movie_id, poster_path):
    movie = {"id": movie_id, "poster_path": poster_path}
    return get_movie_poster(movie) or ""


@app.route("/")
def home():
    movies = new_df[["id", "title"]].to_dict(orient="records")
    return render_template("index.html", movies=movies)

@app.route("/movies", methods=["GET"])
def get_movies():
    search_query = request.args.get("search", "").strip()
    page = int(request.args.get("page", 1))
    per_page = 10

    if not search_query or page < 1:
        return jsonify([])
    
    try:
        results = fuzzy_title_match(search_query, page, per_page, new_df)
        return jsonify(results)
    except Exception as e:
        app.logger.error(f"Error in movie search: {str(e)}")
        return jsonify([])

@app.route("/recommend", methods=["GET"])
def recommend():
    try:
        userId = request.args.get("userId")
        movieId = request.args.get("movieId")
        topN = request.args.get("topN", "10")

        logging.info(
            f"Received recommendation request: userId={userId}, movieId={movieId}, topN={topN}"
        )

        if not userId or not movieId:
            logging.warning("Missing userId or movieId")
            return (
                jsonify({"status": False, "message": "userId and movieId are required"}),
                400,
            )

        try:
            userId = int(userId)
            movieId = int(movieId)
            if userId < 0 or movieId < 0:
                logging.warning("Negative userId or movieId")
                return (
                    jsonify(
                        {
                            "status": False,
                            "message": "userId and movieId must be non-negative",
                        }
                    ),
                    400,
                )
        except ValueError:
            logging.warning("Invalid userId or movieId format")
            return (
                jsonify(
                    {
                        "status": False,
                        "message": "userId and movieId must be valid integers",
                    }
                ),
                400,
            )

        try:
            topN = max(1, min(int(topN), 50))
        except ValueError:
            topN = 10

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

        if error_message:
            logging.info(f"Recommendation failed: {error_message}")
            return jsonify({"status": False, "message": error_message}), 400

        if recommendations is None or recommendations.empty:
            logging.info(f"No recommendations found for user {userId}, movieId {movieId}")
            return (
                jsonify(
                    {
                        "status": True,
                        "data": {
                            "userId": userId,
                            "movieId": movieId,
                            "recommendedMovies": [],
                        },
                    }
                ),
                200,
            )

        result = recommendations.to_dict(orient="records")
        for movie in result:
            movie["poster_url"] = cached_get_movie_poster(
                movie["id"], movie["poster_path"]
            )

        logging.info(
            f"Generated {len(result)} recommendations for user {userId}, movieId {movieId}"
        )
        return jsonify(
            {
                "status": True,
                "message": "Recommendations generated successfully",
                "data": {"userId": userId, "movieId": movieId, "recommendedMovies": result},
            }
        )

    except Exception as e:
        logging.error(f"Server error in recommend route: {e}")
        return jsonify({"status": False, "message": "Internal server error"}), 500


@app.route("/genreBasedRecommendation", methods=["GET"])
def genreBasedRecommendation():
    try:
        genre = request.args.get("genre")
        topN = request.args.get("topN", 100)

        logging.info(
            f"Received genre recommendation request: genre={genre}, topN={topN}"
        )

        if not genre:
            logging.warning("Missing genre")
            return jsonify({"status": False, "message": "Movie genre is required"}), 400

        try:
            topN = max(1, min(int(topN), 5000))
        except ValueError:
            topN = 100

        genre = genre.title()
        recommendations = genre_based_recommender(genre=genre, df=new_df2, top_n=topN)

        if recommendations is None or recommendations.empty:
            logging.info(f"No movies found for genre {genre}")
            return (
                jsonify(
                    {"status": False, "message": f"No movies found for genre: {genre}"}
                ),
                400,
            )

        result = recommendations.to_dict(orient="records")
        for movie in result:
            movie["poster_url"] = cached_get_movie_poster(
                movie["id"], movie["poster_path"]
            )

        logging.info(f"Generated {len(result)} genre recommendations for genre {genre}")
        return jsonify(
            {
                "status": True,
                "message": "Recommendations generated successfully",
                "data": {"genre": genre, "recommendedMovies": result},
            }
        )
    except Exception as e:
        logging.error(f"Server error in genreBasedRecommendation: {e}")
        return jsonify({"status": False, "message": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True)
