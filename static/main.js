document.addEventListener("DOMContentLoaded", function () {
  const recommendationType = document.getElementById("recommendationType");
  const movieBasedFields = document.getElementById("movieBasedFields");
  const genreBasedFields = document.getElementById("genreBasedFields");

  // Toggle input fields based on recommendation type
  recommendationType.addEventListener("change", function () {
    if (this.value === "movie") {
      movieBasedFields.style.display = "block";
      genreBasedFields.style.display = "none";
    } else {
      movieBasedFields.style.display = "none";
      genreBasedFields.style.display = "block";
    }
  });

  document
    .getElementById("recommendationForm")
    .addEventListener("submit", function (event) {
      event.preventDefault();

      const recommendationType =
        document.getElementById("recommendationType").value;
      const topN = document.getElementById("topN").value;
      let fetchUrl = "";

      if (recommendationType === "movie") {
        const userId = document.getElementById("userId").value.trim();
        const title = document.getElementById("title").value.trim();

        if (!userId || !title) {
          alert("Please enter both User ID and Movie Title.");
          return;
        }

        fetchUrl = `/recommend?userId=${userId}&title=${encodeURIComponent(
          title
        )}&topN=${topN}`;
      } else {
        const genre = document.getElementById("genre").value.trim();

        if (!genre) {
          alert("Please enter a genre.");
          return;
        }

        fetchUrl = `/genreBasedRecommendation?genre=${encodeURIComponent(
          genre
        )}&topN=${topN}`;
      }

      fetch(fetchUrl)
        .then((response) => response.json())
        .then((data) => {
          if (
            data &&
            data.status &&
            Array.isArray(data.data?.recommendedMovies)
          ) {
            const movies = data.data.recommendedMovies;
            const moviesContainer = document.getElementById("moviesGrid");
            moviesContainer.innerHTML = ""; // Clear previous results

            movies.forEach((movie) => {
              const posterUrl =
                movie.poster_url ||
                "https://via.placeholder.com/200x300?text=No+Image";

              // Create movie card elements safely
              const movieCard = document.createElement("div");
              movieCard.classList.add("movie-card");

              const img = document.createElement("img");
              img.src = posterUrl;
              img.alt = movie.title;
              img.classList.add("movie-poster");

              const titleElement = document.createElement("h3");
              titleElement.textContent = movie.title;

              movieCard.appendChild(img);
              movieCard.appendChild(titleElement);
              moviesContainer.appendChild(movieCard);
            });
          } else {
            alert("No recommendations found.");
          }
        })
        .catch((error) => {
          console.error("Fetch error:", error);
          alert("Failed to fetch recommendations.");
        });
    });
});