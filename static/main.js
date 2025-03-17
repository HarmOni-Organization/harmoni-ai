document.addEventListener("DOMContentLoaded", function () {
  // Get DOM Elements
  const loader = document.querySelector(".loader");
  const errorMessage = document.getElementById("errorMessage");
  const recommendationType = document.getElementById("recommendationType");
  const movieBasedFields = document.getElementById("movieBasedFields");
  const genreBasedFields = document.getElementById("genreBasedFields");

  // Initialize Select2 with AJAX
  $("#title").select2({
    placeholder: "Search for a movie...",
    allowClear: true,
    ajax: {
      url: "/movies",
      dataType: "json",
      delay: 250,
      data: function (params) {
        return {
          search: params.term,
          page: params.page || 1,
        };
      },
      processResults: function (data) {
        return {
          results: data.map((movie) => ({
            id: movie.id,
            text: movie.title,
          })),
          pagination: { more: false },
        };
      },
      cache: true,
    },
    minimumInputLength: 1,
  });

  // Toggle fields based on recommendation type
  recommendationType.addEventListener("change", function () {
    if (this.value === "movie") {
      movieBasedFields.style.display = "block";
      genreBasedFields.style.display = "none";
    } else {
      movieBasedFields.style.display = "none";
      genreBasedFields.style.display = "block";
    }
  });

  // Form submission handler
  document
    .getElementById("recommendationForm")
    .addEventListener("submit", function (event) {
      event.preventDefault();
      errorMessage.style.display = "none"; // Reset error message

      const type = recommendationType.value;
      const topN = document.getElementById("topN").value;
      let fetchUrl = "";

      if (type === "movie") {
        const userId = document.getElementById("userId").value.trim();
        const movieId = document.getElementById("title").value.trim();

        if (!userId || !movieId) {
          errorMessage.textContent =
            "Please enter both User ID and select a Movie.";
          errorMessage.style.display = "block";
          return;
        }

        fetchUrl = `/recommend?userId=${encodeURIComponent(
          userId
        )}&movieId=${encodeURIComponent(movieId)}&topN=${topN}`;
      } else if (type === "genre") {
        const genre = document.getElementById("genre").value.trim();
        if (!genre) {
          errorMessage.textContent = "Please enter a genre.";
          errorMessage.style.display = "block";
          return;
        }
        fetchUrl = `/genreBasedRecommendation?genre=${encodeURIComponent(
          genre
        )}&topN=${topN}`;
      }

      // Show loader
      loader.style.display = "block";

      fetch(fetchUrl)
        .then((response) => response.json())
        .then((data) => {
          loader.style.display = "none";

          if (
            data &&
            data.status &&
            Array.isArray(data.data?.recommendedMovies)
          ) {
            const movies = data.data.recommendedMovies;
            const moviesContainer = document.getElementById("moviesGrid");
            moviesContainer.innerHTML = "";

            if (movies.length === 0) {
              errorMessage.textContent = "No recommendations found.";
              errorMessage.style.display = "block";
              return;
            }

            movies.forEach((movie) => {
              const posterUrl =
                movie.poster_url ||
                "https://via.placeholder.com/200x300?text=No+Image";
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
            errorMessage.textContent =
              data.message || "No recommendations found.";
            errorMessage.style.display = "block";
          }
        })
        .catch((error) => {
          loader.style.display = "none";
          console.error("Fetch error:", error);
          errorMessage.textContent = "Failed to fetch recommendations.";
          errorMessage.style.display = "block";
        });
    });
});
