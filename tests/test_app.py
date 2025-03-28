import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from unittest.mock import patch, MagicMock
from app import app, cached_get_movie_poster


# Register pytest markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "basic: Basic route tests")
    config.addinivalue_line("markers", "recommendations: Movie recommendation tests")
    config.addinivalue_line("markers", "genre: Genre-based recommendation tests")
    config.addinivalue_line("markers", "posters: Movie poster tests")
    config.addinivalue_line("markers", "errors: Error handling tests")
    config.addinivalue_line("markers", "edge: Edge case tests")
    config.addinivalue_line("markers", "edge: Edge case tests")


# Fixtures
@pytest.fixture
def client():
    """Creates a test client for the Flask application."""
    with app.test_client() as client:
        yield client


# Basic Route Tests
@pytest.mark.basic
class TestBasicRoutes:
    """
    Test suite for basic application routes and endpoints.

    Tests cover:
    - Home page loading and content
    - Basic endpoint availability
    - Response format validation
    """

# Movie Recommendation Tests
@pytest.mark.recommendations
class TestMovieRecommendations:
    """
    Test suite for movie recommendation functionality.

    Tests cover:
    - Valid recommendation requests
    - Parameter validation
    - Response format and content
    - Edge cases and error conditions
    - TopN parameter handling
    """

    def test_valid_recommendation(self, client):
        """Verify that movie recommendations work with valid inputs."""
        response = client.get("/recommend?userId=1&movieId=27205&topN=5")  # Updated to use movieId
        assert response.status_code == 200
        data = response.json
        assert data["status"] is True
        assert len(data["data"]["recommendedMovies"]) == 5

    def test_missing_parameters(self, client):
        """Verify error handling for missing parameters."""
        # Test missing movieId
        response = client.get("/recommend?userId=1&topN=5")
        assert response.status_code == 400
        assert b"userId and movieId are required" in response.data

        # Test missing userId
        response = client.get("/recommend?movieId=27205&topN=5")
        assert response.status_code == 400
        assert b"userId and movieId are required" in response.data

    def test_invalid_parameters(self, client):
        """Verify error handling for invalid parameters."""

        # Test invalid movieId
        response = client.get("/recommend?userId=1&movieId=abc")
        assert response.status_code == 400
        assert b"movieId must be valid integer" in response.data

        # Test negative movieId
        response = client.get("/recommend?userId=1&movieId=-27205&topN=5")
        assert response.status_code == 400
        assert b"movieId must be non-negative" in response.data

    def test_topn_parameter_handling(self, client):
        """Verify handling of different topN parameter values."""
        # Test large topN (should be capped)
        response = client.get("/recommend?userId=1&movieId=27205&topN=1000")
        assert response.status_code == 200
        data = response.json
        assert len(data["data"]["recommendedMovies"]) <= 50

        # Test zero topN (should default to minimum of 1)
        response = client.get("/recommend?userId=1&movieId=27205&topN=0")
        assert response.status_code == 200
        data = response.json
        assert len(data["data"]["recommendedMovies"]) > 0

        # Test negative topN (should default to minimum of 1)
        response = client.get("/recommend?userId=1&movieId=27205&topN=-5")
        assert response.status_code == 200
        data = response.json
        assert len(data["data"]["recommendedMovies"]) > 0

        # Test invalid topN (should default to 10)
        response = client.get("/recommend?userId=1&movieId=27205&topN=abc")
        assert response.status_code == 200
        data = response.json
        assert len(data["data"]["recommendedMovies"]) > 0


@pytest.mark.edge
class TestEdgeCases:
    def test_invalid_movie_id(self, client):
        """Test handling of invalid movie IDs."""
        response = client.get("/recommend?userId=1&movieId=999999999&topN=5")
        assert response.status_code == 400
        assert b"Movie ID 999999999 not found" in response.data

    def test_extreme_user_ids(self, client):
        """Test handling of extreme user IDs."""
        response = client.get("/recommend?userId=999999999&movieId=27205&topN=5")
        assert response.status_code == 200
        data = response.json
        assert "recommendedMovies" in data["data"]

    def test_empty_response_handling(self, client):
        """Test handling of empty responses."""
        response = client.get("/genreBasedRecommendation?genre=NonexistentGenre&topN=5")
        assert response.status_code == 400
        assert b"No movies found for genre" in response.data


# Genre-based Recommendation Tests
@pytest.mark.genre
class TestGenreRecommendations:
    """
    Test suite for genre-based recommendation functionality.

    Tests cover:
    - Valid genre recommendations
    - Genre parameter validation
    - Case sensitivity handling
    - Response format and content
    - Error conditions
    """

    def test_valid_genre_recommendation(self, client):
        """Verify that genre-based recommendations work with valid inputs."""
        response = client.get("/genreBasedRecommendation?genre=Action&topN=5")
        assert response.status_code == 200
        data = response.json
        assert data["status"] is True
        assert len(data["data"]["recommendedMovies"]) == 5
        assert data["data"]["genre"] == "Action"

    def test_invalid_genre_parameters(self, client):
        """Verify error handling for invalid genre parameters."""
        # Test missing genre
        response = client.get("/genreBasedRecommendation?topN=5")
        assert response.status_code == 400
        assert b"Movie genre is required" in response.data

        # Test empty genre
        response = client.get("/genreBasedRecommendation?genre=&topN=5")
        assert response.status_code == 400
        assert b"Movie genre is required" in response.data

        # Test invalid genre
        response = client.get("/genreBasedRecommendation?genre=InvalidGenre&topN=5")
        assert response.status_code == 400
        assert b"No movies found for genre" in response.data

    def test_genre_case_sensitivity(self, client):
        """Test genre case sensitivity handling."""
        # Test lowercase genre
        response = client.get("/genreBasedRecommendation?genre=action&topN=5")
        assert response.status_code == 200
        data = response.json
        assert data["data"]["genre"] == "Action"

        # Test mixed case genre
        response = client.get("/genreBasedRecommendation?genre=AcTiOn&topN=5")
        assert response.status_code == 200
        data = response.json
        assert data["data"]["genre"] == "Action"


# Movie Poster Tests
@pytest.mark.posters
class TestMoviePosters:
    """
    Test suite for movie poster functionality.

    Tests cover:
    - Poster URL generation
    - Caching behavior
    - Missing poster handling
    - API integration
    """

    def test_movie_poster_retrieval(self, client):
        """Verify that movie poster URLs are correctly generated."""
        with patch("app.cached_get_movie_poster") as mock_get_poster:
            mock_get_poster.return_value = (
                "https://image.tmdb.org/t/p/w500/test_poster.jpg"
            )
            response = client.get("/recommend?userId=1&movieId=27205&topN=5")
            assert response.status_code == 200
            data = response.json
            assert (
                data["data"]["recommendedMovies"][0]["poster_url"]
                == "https://image.tmdb.org/t/p/w500/test_poster.jpg"
            )

    def test_cached_poster_function(self):
        """Verify that the cached poster function works correctly."""
        movie_id = 123
        poster_path = "/test_path.jpg"
        result = cached_get_movie_poster(movie_id, poster_path)
        assert isinstance(result, str)
        assert "image.tmdb.org" in result

    def test_missing_poster_handling(self, client):
        """Test handling of missing movie posters."""
        with patch("app.cached_get_movie_poster") as mock_get_poster:
            mock_get_poster.return_value = ""
            response = client.get("/recommend?userId=1&movieId=27205&topN=5")
            assert response.status_code == 200
            data = response.json
            assert "recommendedMovies" in data["data"]
            assert data["data"]["recommendedMovies"][0]["poster_url"] == ""


# Error Handling Tests
@pytest.mark.errors
class TestErrorHandling:
    """
    Test suite for error handling and edge cases.

    Tests cover:
    - Server errors
    - Invalid requests
    - Rate limiting
    - Malformed data
    - API failures
    """

    def test_recommendation_server_error(self, client):
        """Verify server error handling in recommendation endpoint."""
        with patch("app.improved_hybrid_recommendations") as mock_recommend:
            mock_recommend.side_effect = Exception("Unexpected error")
            response = client.get("/recommend?userId=1&movieId=27205&topN=5")
            assert response.status_code == 500
            assert b"Internal server error" in response.data

    def test_genre_recommendation_server_error(self, client):
        """Verify server error handling in genre recommendation endpoint."""
        with patch("app.genre_based_recommender") as mock_genre:
            mock_genre.side_effect = Exception("Unexpected error")
            response = client.get("/genreBasedRecommendation?genre=Action&topN=5")
            assert response.status_code == 500
            assert b"Internal server error" in response.data

    def test_malformed_request_handling(self, client):
        """Test handling of malformed requests."""
        # Test malformed URL
        response = client.get(
            "/recommend?userId=1&movieId=27205&topN=5&invalid=param"
        )
        assert response.status_code == 200  # Extra parameters should be ignored

        # Test invalid HTTP method
        response = client.post("/recommend")
        assert response.status_code == 405  # Method not allowed

    def test_rate_limiting(self, client):
        """Test rate limiting behavior (if implemented)."""
        # Make multiple rapid requests
        for _ in range(50):
            response = client.get("/recommend?userId=1&movieId=27205&topN=5")
            assert response.status_code == 200  # No rate limiting implemented yet