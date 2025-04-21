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
    flask_app.config.update({"TESTING": True})
    yield flask_app

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def auth_headers():
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
    recommendations = pd.DataFrame({
        'id': [1, 2, 3],
        'title': ['Movie 1', 'Movie 2', 'Movie 3'],
        'poster_path': ['/path1.jpg', '/path2.jpg', '/path3.jpg'],
        'genres': ['Action', 'Comedy', 'Drama']
    })
    return recommendations, None

@pytest.fixture
def mock_empty_recommendations():
    return pd.DataFrame(), None

@pytest.fixture
def mock_error_recommendations():
    return None, "Error in recommendations"

def set_g_user():
    g.user = {'userId': '123', 'username': 'testuser'}

def test_home(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Welcome to Hybrid Recommendation System API" in response.data

def test_recommend_success(client, mock_recommendation_data, auth_headers):
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
    with patch('my_modules.auth.verify_token', return_value=(True, {'userId': '123', 'username': 'testuser'})):
        with patch('my_modules.auth.verify_internal_key', return_value=True):
            response = client.get('/recommend', headers=auth_headers)
            assert response.status_code == 400
            data = json.loads(response.data)
            assert data['status'] is False
            assert "movieId is required" in data['message']

def test_recommend_invalid_movie_id(client, auth_headers):
    with patch('my_modules.auth.verify_token', return_value=(True, {'userId': '123', 'username': 'testuser'})):
        with patch('my_modules.auth.verify_internal_key', return_value=True):
            response = client.get('/recommend?movieId=invalid', headers=auth_headers)
            assert response.status_code == 400
            data = json.loads(response.data)
            assert data['status'] is False
            assert "movieId must be valid integer" in data['message']

def test_recommend_negative_movie_id(client, auth_headers):
    with patch('my_modules.auth.verify_token', return_value=(True, {'userId': '123', 'username': 'testuser'})):
        with patch('my_modules.auth.verify_internal_key', return_value=True):
            response = client.get('/recommend?movieId=-1', headers=auth_headers)
            assert response.status_code == 400
            data = json.loads(response.data)
            assert data['status'] is False
            assert "movieId must be non-negative" in data['message']

def test_recommend_empty_results(client, mock_empty_recommendations, auth_headers):
    with patch('app.improved_hybrid_recommendations', return_value=mock_empty_recommendations):
        with patch('my_modules.auth.verify_token', return_value=(True, {'userId': '123', 'username': 'testuser'})):
            with patch('my_modules.auth.verify_internal_key', return_value=True):
                response = client.get('/recommend?movieId=123', headers=auth_headers)
                assert response.status_code == 200
                data = json.loads(response.data)
                assert data['status'] is True
                assert data['data']['recommendedMovies'] == []

def test_recommend_error(client, mock_error_recommendations, auth_headers):
    with patch('app.improved_hybrid_recommendations', return_value=mock_error_recommendations):
        with patch('my_modules.auth.verify_token', return_value=(True, {'userId': '123', 'username': 'testuser'})):
            with patch('my_modules.auth.verify_internal_key', return_value=True):
                response = client.get('/recommend?movieId=123', headers=auth_headers)
                assert response.status_code == 400
                data = json.loads(response.data)
                assert data['status'] is False
                assert "Error in recommendations" in data['message']

def test_recommend_exception(client, auth_headers):
    with patch('app.improved_hybrid_recommendations', side_effect=Exception("Test exception")):
        with patch('my_modules.auth.verify_token', return_value=(True, {'userId': '123', 'username': 'testuser'})):
            with patch('my_modules.auth.verify_internal_key', return_value=True):
                response = client.get('/recommend?movieId=123', headers=auth_headers)
                assert response.status_code == 500
                data = json.loads(response.data)
                assert data['status'] is False
                assert "Internal server error" in data['message']

def test_not_found(client):
    response = client.get('/non-existent-endpoint')
    assert response.status_code == 404
    data = json.loads(response.data)
    assert data['status'] is False
    assert "Endpoint not found" in data['message']

def test_genre_recommendation_success(client, mock_recommendation_data, auth_headers):
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
    with patch('my_modules.auth.verify_token', return_value=(True, {'userId': '123', 'username': 'testuser'})):
        with patch('my_modules.auth.verify_internal_key', return_value=True):
            response = client.get('/genreBasedRecommendation', headers=auth_headers)
            assert response.status_code == 400
            data = json.loads(response.data)
            assert data['status'] is False
            assert "Movie genre is required" in data['message']

def test_genre_recommendation_empty_results(client, auth_headers):
    with patch('app.genre_based_recommender', return_value=pd.DataFrame()):
        with patch('my_modules.auth.verify_token', return_value=(True, {'userId': '123', 'username': 'testuser'})):
            with patch('my_modules.auth.verify_internal_key', return_value=True):
                response = client.get('/genreBasedRecommendation?genre=NonExistentGenre', headers=auth_headers)
                assert response.status_code == 400
                data = json.loads(response.data)
                assert data['status'] is False
                assert "No movies found for genre" in data['message']

def test_genre_recommendation_exception(client, auth_headers):
    with patch('app.genre_based_recommender', side_effect=Exception("Test exception")):
        with patch('my_modules.auth.verify_token', return_value=(True, {'userId': '123', 'username': 'testuser'})):
            with patch('my_modules.auth.verify_internal_key', return_value=True):
                response = client.get('/genreBasedRecommendation?genre=Action', headers=auth_headers)
                assert response.status_code == 500
                data = json.loads(response.data)
                assert data['status'] is False
                assert "Internal server error" in data['message']

def test_auth_test_success(client, auth_headers):
    with patch('my_modules.auth.verify_token', return_value=(True, {'userId': '123', 'username': 'testuser'})):
        with patch('my_modules.auth.verify_internal_key', return_value=True):
            response = client.get('/auth-test', headers=auth_headers)
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['status'] is True
            assert data['message'] == "Authentication successful"
            assert data['data']['user']['userId'] == '123'

def test_auth_test_no_user(client, auth_headers):
    with patch('my_modules.auth.verify_token', return_value=(True, None)):
        with patch('my_modules.auth.verify_internal_key', return_value=True):
            response = client.get('/auth-test', headers=auth_headers)
            assert response.status_code == 500
            data = json.loads(response.data)
            assert data['status'] is False
            assert "no user data available" in data['message'].lower()

def test_cached_get_movie_poster():
    with patch('app.get_movie_poster', return_value="http://example.com/poster.jpg"):
        from app import cached_get_movie_poster
        result1 = cached_get_movie_poster(123, "/path.jpg")
        assert result1 == "http://example.com/poster.jpg"
        result2 = cached_get_movie_poster(123, "/path.jpg")
        assert result2 == "http://example.com/poster.jpg"

def test_invalid_token(client, auth_headers):
    with patch('my_modules.auth.verify_token', return_value=(False, None)):
        with patch('my_modules.auth.verify_internal_key', return_value=True):
            response = client.get('/recommend?movieId=1', headers=auth_headers)
            assert response.status_code in [401, 403]

def test_missing_token(client):
    headers = {
        "X-Internal-Key": os.environ.get("INTERNAL_API_KEY", "your-internal-api-key-here")
    }
    with patch('my_modules.auth.verify_token', return_value=(False, None)):
        with patch('my_modules.auth.verify_internal_key', return_value=True):
            response = client.get('/recommend?movieId=1', headers=headers)
            assert response.status_code in [401, 403]

def test_invalid_internal_key(client, auth_headers):
    with patch('my_modules.auth.verify_token', return_value=(True, {'userId': '123', 'username': 'testuser'})):
        with patch('my_modules.auth.verify_internal_key', return_value=False):
            response = client.get('/recommend?movieId=1', headers=auth_headers)
            assert response.status_code in [401, 403]

def test_missing_internal_key(client):
    headers = {
        "Authorization": "Bearer your-jwt-token-here"
    }
    with patch('my_modules.auth.verify_token', return_value=(True, {'userId': '123', 'username': 'testuser'})):
        with patch('my_modules.auth.verify_internal_key', return_value=False):
            response = client.get('/recommend?movieId=1', headers=headers)
            assert response.status_code in [401, 403]
