# Harmoni AI - Hybrid Movie Recommendation System

HarmOni Movie Recommendation Service is a sophisticated movie recommendation system that provides personalized movie suggestions based on user preferences, content, and popularity. Developed using Python and Flask, it employs a hybrid recommendation approach integrating content-based filtering, Singular Value Decomposition (SVD) collaborative filtering, and popularity filtering from the Internet Movie Database ([IMDB](https://www.imdb.com/)).

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **`Hybrid Recommendation Engine`**: Integrates content-based, popularity filtering, and collaborative filtering techniques for improved recommendation accuracy.
- **`User-Friendly Interface`**: Accessible via a web-based interface built with Flask, allowing users to interact seamlessly with the recommendation system.
- **`Extensible Architecture`**: Modular design facilitates easy updates and integration of additional features.

## Project Structure

The repository is organized as follows:

```
harmoni-ai/
├── app.py                 # Main Flask application
├── my_modules/
│   └── myModule.py        # Core recommendation logic
├── tests/
│   └── test_app.py        # Test suite
├── data/                  # Movie datasets
├── models/                # Trained models
├── templates/             # HTML templates for the web interface
├── static/                # Static files (e.g., CSS, JavaScript)
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Installation

To set up the Harmoni AI system locally, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/HarmOni-Organization/harmoni-ai.git
   cd harmoni-ai
   ```
   
2. **Set up environment variables**:
   Copy the environment template and configure it.
   ```bash
      cp .env.example .env
   ```

3. **Prepare the dataset**:

   Ensure the movie dataset is placed in the data/ directory. Update paths in the code if necessary.

4. **Build and Run the Docker Container**:
   ```bash
   docker build -t harmoni-ai .
   docker run -p 5000:5000 harmoni-ai
   ```
   The service will now be running locally. Access it in your web browser at `http://172.17.0.2:5000/`.

Alternatively, you can use **`Docker Compose`** for easier management:

   ```bash
   docker-compose up --build
   ```

## Usage

- **Homepage**: Enter a User ID and a Movie Title to receive recommendations.
- **API Endpoint**:
   - Use `/recommend` with the following query parameters:
     - `userId`: The user ID.
     - `title`: The movie title.
     - `topN`: The number of recommendations (default: 10).

     Example API request:
     ```
     http://127.0.0.1:5000/recommend?userId=1&title=Inception&topN=3
     ```
     Example API response:
     ```
     {
     "data": {
        "recommendedMovies": [
            {
                "final_score": 4.257097358991,
                "id": 603,
                "poster_path": "/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg",
                "poster_url": "https://image.tmdb.org/t/p/w500/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg",
                "release_date": "1999-03-30",
                "title": "The Matrix"
            },
            {
                "final_score": 3.870021302510938,
                "id": 8681,
                "poster_path": "/y5Va1WXDX6nZElVirPrGxf6w99B.jpg",
                "poster_url": "https://image.tmdb.org/t/p/w500/y5Va1WXDX6nZElVirPrGxf6w99B.jpg",
                "release_date": "2008-02-18",
                "title": "Taken"
            },
            {
                "final_score": 3.8335548353137865,
                "id": 1672,
                "poster_path": "/dhAZ8T0tTlTNw3D7Cn8wc87sk1w.jpg",
                "poster_url": "https://image.tmdb.org/t/p/w500/dhAZ8T0tTlTNw3D7Cn8wc87sk1w.jpg",
                "release_date": "1981-10-21",
                "title": "The Professional"
            }
        ],
        "title": "Inception",
        "userId": 1
     },
     "message": "Recommendations generated successfully",
     "status": true
     }
     ```
   - Use `/genreBasedRecommendation` with the following query parameters:
      - `genre`: The genre you like.
      - `topN`:  The number of recommendations (default: 100).
     
     Example API request:
     ```
     http://127.0.0.1:5000/genreBasedRecommendation?genre=Action&topN=3
     ```
     Example API response:
     ```
     {
     "data": {
        "genre": "Action",
        "recommendedMovies": [
            {
                "id": 361743,
                "poster_path": "/62HCnUTziyWcpDaBO2i1DX17ljH.jpg",
                "poster_url": "https://image.tmdb.org/t/p/w500/62HCnUTziyWcpDaBO2i1DX17ljH.jpg",
                "release_date": "2022-05-24",
                "title": "Top Gun: Maverick"
            },
            {
                "id": 791373,
                "poster_path": "/tnAuB8q5vv7Ax9UAEje5Xi4BXik.jpg",
                "poster_url": "https://image.tmdb.org/t/p/w500/tnAuB8q5vv7Ax9UAEje5Xi4BXik.jpg",
                "release_date": "2021-03-18",
                "title": "Zack Snyder's Justice League"
            },
            {
                "id": 634649,
                "poster_path": "/5weKu49pzJCt06OPpjvT80efnQj.jpg",
                "poster_url": "https://image.tmdb.org/t/p/w500/5weKu49pzJCt06OPpjvT80efnQj.jpg",
                "release_date": "2021-12-15",
                "title": "Spider-Man: No Way Home"
            }
        ]
     },
     "message": "Recommendations generated successfully",
     "status": true
     }
     ```

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/YourFeatureName`.
3. Implement your changes.
4. Commit the changes: `git commit -m 'Add feature: YourFeatureName'`.
5. Push to the branch: `git push origin feature/YourFeatureName`.
6. Open a pull request for review.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions, please contact the project maintainers through the repository's issue tracker or by email at [Othman M. O. Shbeir](mailto:uthmanshbeir@gmail.com).
