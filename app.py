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

# Caching for movie posters
@lru_cache(maxsize=1000)
def cached_get_movie_poster(movie_id, poster_path):
    movie = {"id": movie_id, "poster_path": poster_path}
    return get_movie_poster(movie)


@app.route("/")
def home():
    """
    Render the homepage.

    Returns:
    HTML page: The index.html template.
    """
    return render_template("index.html")


@app.route("/recommend", methods=["GET"])
def recommend():
    """
    Generate movie recommendations for a given user and movie title.
    """
    try:
        userId = request.args.get("userId")
        title = request.args.get("title")
        topN = request.args.get("topN", "10")

        if not userId or not title:
            return jsonify({"status": False, "message": "userId and title are required"}), 400

        userId = int(userId)
        topN = min(int(topN), 50)  # Cap at 50 for performance

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
    Generate movie recommendations for a given genre.
    """
    try:
        genre = request.args.get("genre")
        topN = int(request.args.get("topN", 100))

        if not genre:
            return jsonify({"error": "Movie genre is required"}), 400

        recommendations = genre_based_recommender(genre=genre, df=new_df2, top_n=topN)
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