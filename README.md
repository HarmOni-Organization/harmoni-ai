# Harmoni AI - Hybrid Movie Recommendation System

A sophisticated movie recommendation system that combines content-based and collaborative filtering approaches to provide personalized movie recommendations. The system uses a hybrid approach that leverages both user preferences and movie content to generate accurate and diverse recommendations.

## Features

- **Hybrid Recommendation Engine**

  - Content-based filtering using movie features
  - Collaborative filtering using SVD (Singular Value Decomposition)
  - Dynamic weighting based on user interaction
  - Cold-start handling for new users

- **Genre-Based Recommendations**

  - Genre-specific movie suggestions
  - Weighted scoring system
  - Recency bias for newer movies
  - Popularity consideration

- **Movie Poster Integration**

  - Automatic poster fetching from TMDb
  - Caching for improved performance
  - Fallback handling for missing posters

- **RESTful API**
  - Simple and intuitive endpoints
  - Input validation and error handling
  - Rate limiting and caching
  - Comprehensive error messages

## Prerequisites

- Python 3.8+
- Docker and Docker Compose
- TMDb API key

## Environment Variables

Create a `.env` file in the project root with the following variables:

```env
FLASK_APP=app.py
FLASK_ENV=development
TMDB_API_KEY=your_tmdb_api_key
DATA_DIR=data
MODEL_DIR=models
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/harmoni-ai.git
cd harmoni-ai
```

2. Build and start the Docker containers:

```bash
docker-compose up --build
```

The application will be available at `http://localhost:5000`

## API Endpoints

### 1. Hybrid Movie Recommendations

```http
GET /recommend
```

Generate personalized movie recommendations based on user ID and a reference movie.

**Query Parameters:**

- `userId` (required): Integer ID of the user
- `title` (required): Title of the reference movie
- `topN` (optional): Number of recommendations (default: 10, max: 50)

**Example Response:**

```json
{
  "status": true,
  "data": {
    "userId": 1,
    "title": "The Matrix",
    "recommendedMovies": [
      {
        "id": 603,
        "title": "The Matrix",
        "release_date": "1999-03-31",
        "final_score": 0.95,
        "poster_url": "https://image.tmdb.org/t/p/w500/..."
      }
    ]
  }
}
```

### 2. Genre-Based Recommendations

```http
GET /genreBasedRecommendation
```

Get movie recommendations based on a specific genre.

**Query Parameters:**

- `genre` (required): Movie genre (e.g., "Action", "Drama")
- `topN` (optional): Number of recommendations (default: 100, max: 100)

**Example Response:**

```json
{
  "status": true,
  "message": "",
  "data": {
    "genre": "Action",
    "recommendedMovies": [
      {
        "id": 550,
        "title": "Fight Club",
        "release_date": "1999-10-15",
        "poster_url": "https://image.tmdb.org/t/p/w500/..."
      }
    ]
  }
}
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- `200`: Successful request
- `400`: Invalid input parameters
- `500`: Internal server error

**Example Error Response:**

```json
{
  "status": false,
  "message": "Invalid characters in title"
}
```

## Development

### Running Tests

```bash
docker-compose run test
```

### Project Structure

```
harmoni-ai/
├── app.py                 # Main Flask application
├── my_modules/
│   └── myModule.py       # Core recommendation logic
├── tests/
│   └── test_app.py       # Test suite
├── data/                 # Movie datasets
├── models/              # Trained models
├── templates/           # HTML templates
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose configuration
└── requirements.txt    # Python dependencies
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MovieLens dataset for providing the movie ratings data
- TMDb API for movie metadata and posters
- Scikit-learn and Surprise libraries for machine learning components
