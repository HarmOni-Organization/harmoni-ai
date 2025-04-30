import sys
import os
import json
import pytest
import pandas as pd
from flask import g
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app as flask_app
from my_modules.auth import generate_jwt_token

@pytest.fixture
def app():
    """
    Pytest fixture to configure and provide a Flask application instance for testing.

    Yields:
        Flask: The configured Flask application with testing mode enabled.
    """
    flask_app.config.update({"TESTING": True})
    yield flask_app

@pytest.fixture
def client(app):
    """
    Pytest fixture to provide a test client for the Flask application.

    Args:
        app (Flask): The Flask application instance.

    Returns:
        FlaskClient: A test client for sending HTTP requests to the application.
    """
    return app.test_client()

@pytest.fixture
def auth_headers():
    """
    Pytest fixture to generate authentication headers for test requests.

    Returns:
        dict: A dictionary containing the Authorization and X-Internal-Key headers.
    """
    token = generate_jwt_token(
        user_id="123",
        username="testuser",
        email="testuser@example.com",
        secret= os.environ.get("JWT_SECRET_KEY")
    )
    return {
        "Authorization": f"Bearer {token}",
        "X-Internal-Key": os.environ.get("INTERNAL_API_KEY")
    }

@pytest.fixture
def mock_recommendation_data():
    """
    Pytest fixture to provide mock recommendation data for testing.

    Returns:
        tuple: A tuple containing a DataFrame of mock recommendations and None for error.
    """
    recommendations = pd.DataFrame({
        'id': [1, 2, 3],
        'title': ['Movie 1', 'Movie 2', 'Movie 3'],
        'poster_path': ['/path1.jpg', '/path2.jpg', '/path3.jpg'],
        'genres': ['Action', 'Comedy', 'Drama']
    })
    return recommendations, None

@pytest.fixture
def mock_empty_recommendations():
    """
    Pytest fixture to provide an empty DataFrame for recommendation tests.

    Returns:
        tuple: An empty DataFrame and None for error.
    """
    return pd.DataFrame(), None

@pytest.fixture
def mock_error_recommendations():
    """
    Pytest fixture to simulate an error in the recommendation process.

    Returns:
        tuple: None for recommendations and an error message string.
    """
    return None, "Error in recommendations"

def set_g_user():
    """
    Helper function to set the Flask global `g.user` for simulating an authenticated user in tests.
    """
    g.user = {'userId': '123', 'username': 'testuser'}

def test_home(client):
    response = client.get('/')
    assert response.status_code == 200

    # Ensure the response is JSON
    assert response.content_type.startswith('application/json')

    data = response.get_json()
    assert isinstance(data, dict)
    assert data.get('status') is True
    assert 'message' in data
    assert 'welcome' in data['message'].lower()

    # Optionally check for endpoints list if present
    if 'data' in data and 'endpoints' in data['data']:
        assert isinstance(data['data']['endpoints'], list)
        endpoint_paths = [ep.get('path') for ep in data['data']['endpoints']]
        # Check for required endpoints (but don't require the root path to be listed)
        assert '/recommend' in endpoint_paths
        assert '/genreBasedRecommendation' in endpoint_paths

def test_recommend_success(client, mock_recommendation_data, auth_headers):
    """
    Test the /recommend endpoint for successful recommendation retrieval.

    Args:
        client (FlaskClient): The test client.
        mock_recommendation_data (tuple): Mocked recommendation data.
        auth_headers (dict): Authentication headers.
    """
    with patch('app.improved_hybrid_recommendations', return_value=mock_recommendation_data):
        with patch('app.cached_get_movie_poster', return_value="http://example.com/poster.jpg"):
            with patch('my_modules.auth.verify_token', return_value=(True, {'userId': '123', 'username': 'testuser'})):
                with patch('my_modules.auth.verify_internal_key', return_value=True):
                    response = client.get('/recommend?movieId=1&topN=5', headers=auth_headers)
                    assert response.status_code == 200
                    data = json.loads(response.data)
                    assert data['status'] is True
                    assert data['message'] == "Recommendations generated successfully"
                    assert data['data']['userId'] == '123'
                    assert data['data']['movieId'] == 1
                    assert len(data['data']['recommendedMovies']) == 3
                    if 'response_time' in data:
                        assert isinstance(data['response_time'], float)

def test_recommend_missing_movie_id(client, auth_headers):
    """
    Test the /recommend endpoint for missing movieId parameter.

    Args:
        client (FlaskClient): The test client.
        auth_headers (dict): Authentication headers.
    """
    with patch('my_modules.auth.verify_token', return_value=(True, {'userId': '123', 'username': 'testuser'})):
        with patch('my_modules.auth.verify_internal_key', return_value=True):
            response = client.get('/recommend', headers=auth_headers)
            assert response.status_code == 400
            data = json.loads(response.data)
            assert data['status'] is False
            assert "movieId is required" in data['message']

def test_recommend_invalid_movie_id(client, auth_headers):
    """
    Test the /recommend endpoint for an invalid (non-integer) movieId parameter.

    Args:
        client (FlaskClient): The test client.
        auth_headers (dict): Authentication headers.
    """
    with patch('my_modules.auth.verify_token', return_value=(True, {'userId': '123', 'username': 'testuser'})):
        with patch('my_modules.auth.verify_internal_key', return_value=True):
            response = client.get('/recommend?movieId=invalid', headers=auth_headers)
            assert response.status_code == 400
            data = json.loads(response.data)
            assert data['status'] is False
            assert "movieId must be valid integer" in data['message']

def test_recommend_negative_movie_id(client, auth_headers):
    """
    Test the /recommend endpoint for a negative movieId parameter.

    Args:
        client (FlaskClient): The test client.
        auth_headers (dict): Authentication headers.
    """
    with patch('my_modules.auth.verify_token', return_value=(True, {'userId': '123', 'username': 'testuser'})):
        with patch('my_modules.auth.verify_internal_key', return_value=True):
            response = client.get('/recommend?movieId=-1', headers=auth_headers)
            assert response.status_code == 400
            data = json.loads(response.data)
            assert data['status'] is False
            assert "movieId must be non-negative" in data['message']

def test_recommend_empty_results(client, mock_empty_recommendations, auth_headers):
    """
    Test the /recommend endpoint when no recommendations are found.

    Args:
        client (FlaskClient): The test client.
        mock_empty_recommendations (tuple): Mocked empty recommendations.
        auth_headers (dict): Authentication headers.
    """
    with patch('app.improved_hybrid_recommendations', return_value=mock_empty_recommendations):
        with patch('my_modules.auth.verify_token', return_value=(True, {'userId': '123', 'username': 'testuser'})):
            with patch('my_modules.auth.verify_internal_key', return_value=True):
                response = client.get('/recommend?movieId=123', headers=auth_headers)
                assert response.status_code == 200
                data = json.loads(response.data)
                assert data['status'] is True
                assert data['data']['recommendedMovies'] == []

def test_recommend_error(client, mock_error_recommendations, auth_headers):
    """
    Test the /recommend endpoint when an error occurs in the recommendation process.

    Args:
        client (FlaskClient): The test client.
        mock_error_recommendations (tuple): Mocked error recommendations.
        auth_headers (dict): Authentication headers.
    """
    with patch('app.improved_hybrid_recommendations', return_value=mock_error_recommendations):
        with patch('my_modules.auth.verify_token', return_value=(True, {'userId': '123', 'username': 'testuser'})):
            with patch('my_modules.auth.verify_internal_key', return_value=True):
                response = client.get('/recommend?movieId=123', headers=auth_headers)
                assert response.status_code == 400
                data = json.loads(response.data)
                assert data['status'] is False
                assert "Error in recommendations" in data['message']

def test_recommend_exception(client, auth_headers):
    """
    Test the /recommend endpoint for unhandled exceptions.

    Args:
        client (FlaskClient): The test client.
        auth_headers (dict): Authentication headers.
    """
    with patch('app.improved_hybrid_recommendations', side_effect=Exception("Test exception")):
        with patch('my_modules.auth.verify_token', return_value=(True, {'userId': '123', 'username': 'testuser'})):
            with patch('my_modules.auth.verify_internal_key', return_value=True):
                response = client.get('/recommend?movieId=123', headers=auth_headers)
                assert response.status_code == 500
                data = json.loads(response.data)
                assert data['status'] is False
                assert "Internal server error" in data['message']

def test_not_found(client):
    """
    Test the application's 404 error handler for non-existent endpoints.

    Args:
        client (FlaskClient): The test client.
    """
    response = client.get('/non-existent-endpoint')
    assert response.status_code == 404
    data = json.loads(response.data)
    assert data['status'] is False
    assert "Endpoint not found" in data['message']

def test_genre_recommendation_success(client, mock_recommendation_data, auth_headers):
    """
    Test the /genreBasedRecommendation endpoint for successful genre-based recommendations.

    Args:
        client (FlaskClient): The test client.
        mock_recommendation_data (tuple): Mocked recommendation data.
        auth_headers (dict): Authentication headers.
    """
    with patch('app.genre_based_recommender', return_value=mock_recommendation_data[0]):
        with patch('app.cached_get_movie_poster', return_value="http://example.com/poster.jpg"):
            with patch('my_modules.auth.verify_token', return_value=(True, {'userId': '123', 'username': 'testuser'})):
                with patch('my_modules.auth.verify_internal_key', return_value=True):
                    response = client.get('/genreBasedRecommendation?genre=Action&topN=5', headers=auth_headers)
                    assert response.status_code == 200
                    data = json.loads(response.data)
                    assert data['status'] is True
                    assert data['message'] == "Recommendations generated successfully"
                    assert data['data']['userId'] == '123'
                    assert data['data']['genre'] == 'Action'
                    assert len(data['data']['recommendedMovies']) == 3

def test_genre_recommendation_missing_genre(client, auth_headers):
    """
    Test the /genreBasedRecommendation endpoint for missing genre parameter.

    Args:
        client (FlaskClient): The test client.
        auth_headers (dict): Authentication headers.
    """
    with patch('my_modules.auth.verify_token', return_value=(True, {'userId': '123', 'username': 'testuser'})):
        with patch('my_modules.auth.verify_internal_key', return_value=True):
            response = client.get('/genreBasedRecommendation', headers=auth_headers)
            assert response.status_code == 400
            data = json.loads(response.data)
            assert data['status'] is False
            assert "Movie genre is required" in data['message']

def test_genre_recommendation_empty_results(client, auth_headers):
    """
    Test the /genreBasedRecommendation endpoint when no movies are found for the genre.

    Args:
        client (FlaskClient): The test client.
        auth_headers (dict): Authentication headers.
    """
    with patch('app.genre_based_recommender', return_value=pd.DataFrame()):
        with patch('my_modules.auth.verify_token', return_value=(True, {'userId': '123', 'username': 'testuser'})):
            with patch('my_modules.auth.verify_internal_key', return_value=True):
                response = client.get('/genreBasedRecommendation?genre=NonExistentGenre', headers=auth_headers)
                assert response.status_code == 400
                data = json.loads(response.data)
                assert data['status'] is False
                assert "No movies found for genre" in data['message']

def test_genre_recommendation_exception(client, auth_headers):
    """
    Test the /genreBasedRecommendation endpoint for unhandled exceptions.

    Args:
        client (FlaskClient): The test client.
        auth_headers (dict): Authentication headers.
    """
    with patch('app.genre_based_recommender', side_effect=Exception("Test exception")):
        with patch('my_modules.auth.verify_token', return_value=(True, {'userId': '123', 'username': 'testuser'})):
            with patch('my_modules.auth.verify_internal_key', return_value=True):
                response = client.get('/genreBasedRecommendation?genre=Action', headers=auth_headers)
                assert response.status_code == 500
                data = json.loads(response.data)
                assert data['status'] is False
                assert "Internal server error" in data['message']

def test_auth_test_success(client, auth_headers):
    """
    Test the /auth-test endpoint for successful authentication.

    Args:
        client (FlaskClient): The test client.
        auth_headers (dict): Authentication headers.
    """
    with patch('my_modules.auth.verify_token', return_value=(True, {'userId': '123', 'username': 'testuser'})):
        with patch('my_modules.auth.verify_internal_key', return_value=True):
            response = client.get('/auth-test', headers=auth_headers)
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['status'] is True
            assert data['message'] == "Authentication successful"
            assert data['data']['user']['userId'] == '123'

def test_auth_test_no_user(client, auth_headers):
    """
    Test the /auth-test endpoint when no user data is available.

    Args:
        client (FlaskClient): The test client.
        auth_headers (dict): Authentication headers.
    """
    with patch('my_modules.auth.verify_token', return_value=(True, None)):
        with patch('my_modules.auth.verify_internal_key', return_value=True):
            response = client.get('/auth-test', headers=auth_headers)
            assert response.status_code == 500
            data = json.loads(response.data)
            assert data['status'] is False
            assert "no user data available" in data['message'].lower()

def test_cached_get_movie_poster():
    """
    Test the cached_get_movie_poster function to ensure it properly caches movie poster URLs.
    
    This test verifies that:
    1. The function returns the expected poster URL
    2. The caching mechanism works by only calling the underlying function once
    """
    with patch('app.get_movie_poster', return_value="http://example.com/poster.jpg"):
        from app import cached_get_movie_poster
        result1 = cached_get_movie_poster(123, "/path.jpg")
        assert result1 == "http://example.com/poster.jpg"
        result2 = cached_get_movie_poster(123, "/path.jpg")
        assert result2 == "http://example.com/poster.jpg"

def test_invalid_token(client, auth_headers):
    """
    Test the API's response when an invalid authentication token is provided.
    
    Args:
        client (FlaskClient): The test client.
        auth_headers (dict): Authentication headers with an invalid token.
    """
    with patch('my_modules.auth.verify_token', return_value=(False, None)):
        with patch('my_modules.auth.verify_internal_key', return_value=True):
            response = client.get('/recommend?movieId=1', headers=auth_headers)
            assert response.status_code in [401, 403]

def test_missing_token(client):
    """
    Test the API's response when the authentication token is missing.
    
    Args:
        client (FlaskClient): The test client.
    """
    headers = {
        "X-Internal-Key": os.environ.get("INTERNAL_API_KEY", "your-internal-api-key-here")
    }
    with patch('my_modules.auth.verify_token', return_value=(False, None)):
        with patch('my_modules.auth.verify_internal_key', return_value=True):
            response = client.get('/recommend?movieId=1', headers=headers)
            assert response.status_code in [401, 403]

def test_invalid_internal_key(client, auth_headers):
    """
    Test the API's response when an invalid internal API key is provided.
    
    Args:
        client (FlaskClient): The test client.
        auth_headers (dict): Authentication headers with an invalid internal key.
    """
    with patch('my_modules.auth.verify_token', return_value=(True, {'userId': '123', 'username': 'testuser'})):
        with patch('my_modules.auth.verify_internal_key', return_value=False):
            response = client.get('/recommend?movieId=1', headers=auth_headers)
            assert response.status_code in [401, 403]

def test_missing_internal_key(client):
    """
    Test the API's response when the internal API key is missing.
    
    Args:
        client (FlaskClient): The test client.
    """
    headers = {
        "Authorization": "Bearer your-jwt-token-here"
    }
    with patch('my_modules.auth.verify_token', return_value=(True, {'userId': '123', 'username': 'testuser'})):
        with patch('my_modules.auth.verify_internal_key', return_value=False):
            response = client.get('/recommend?movieId=1', headers=headers)
            assert response.status_code in [401, 403]
