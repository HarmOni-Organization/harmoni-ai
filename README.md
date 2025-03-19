# 🎬 HarmOni AI - Hybrid Movie Recommendation System

The **HarmOni AI Recommendation System** is a **Flask-powered API** that delivers **personalized movie recommendations** using a **hybrid recommendation approach**. It integrates **content-based filtering, collaborative filtering (SVD), and popularity-based filtering from IMDb** to enhance accuracy.

This system is part of **HarmOni**, an all-in-one entertainment hub designed to enrich the media experience.

---

## 📌 Features

👉 **Hybrid Recommendation Engine** (Content-based, Collaborative Filtering & Popularity-based)  
👉 **User-Friendly API** (Interact with the recommendation system via RESTful API)  
👉 **Genre-Based Suggestions** (Find movies based on genre preferences)  
👉 **Extensible Architecture** (Easily integrates with additional features & datasets)  
👉 **Docker Support** (Run seamlessly in a containerized environment)  

---

## 📂 Project Structure

```
harmoni-ai/
├── app.py                 # Main Flask application
├── my_modules/
│   └── myModule.py        # Core recommendation logic
├── tests/
│   └── test_app.py        # Unit tests
├── data/                  # Movie datasets
├── models/                # Pre-trained recommendation models
├── templates/             # HTML templates for the web interface
├── static/                # Static assets (CSS, JS)
├── Dockerfile             # Docker build configuration
├── docker-compose.yml     # Docker Compose setup
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🚀 Getting Started

### **1️⃣ Installation**

Ensure you have the following installed:

- **Python 3.8+**
- **Docker & Docker Compose** (optional, but recommended)
- **Flask** (installed via `requirements.txt`)

Clone the repository:

```sh
git clone https://github.com/HarmOni-Organization/harmoni-ai.git
cd harmoni-ai
```

Install dependencies:

```sh
pip install -r requirements.txt
```

### **2️⃣ Configuration**

Copy the environment template and configure it:

```sh
cp .env.example .env
```

Ensure the **movie dataset** is placed inside the `data/` directory.

### **3️⃣ Running the Project**

#### **Using Docker (Recommended)**

```sh
docker-compose up --build
```

#### **Manual Run**

```sh
python app.py
```

The service will be available at:  
📌 **http://127.0.0.1:5000/**

---

## 🔗 API Endpoints

### **🎥 Movie Recommendations**

| Method | Endpoint               | Description                                  |
|--------|------------------------|----------------------------------------------|
| `GET`  | `/recommend`           | Get personalized movie recommendations      |
| `GET`  | `/genreBasedRecommendation` | Get movie recommendations by genre |

#### **Example API Usage**

**User-Based Recommendation**
```sh
http://127.0.0.1:5000/recommend?userId=2000&movieId=286217&topN=3
```
**Response**
```json
{
  "data": {
    "userId": 2000,
    "movieId": 286217,
    "recommendedMovies": [
      {
        "id": 157336,
        "title": "Interstellar",
        "release_date": "2014-11-05",
        "poster_url": "https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",
        "final_score": 1.38
      },
      {
        "id": 118340,
        "title": "Guardians of the Galaxy",
        "release_date": "2014-07-30",
        "poster_url": "https://image.tmdb.org/t/p/w500/r7vmZjiyZw9rpJMQJdXpjgiCOk9.jpg",
        "final_score": 1.31
      }
    ]
  },
  "message": "Recommendations generated successfully",
  "status": true
}
```

**Genre-Based Recommendation**
```sh
http://127.0.0.1:5000/genreBasedRecommendation?genre=Action&topN=3
```
**Response**
```json
{
  "data": {
    "genre": "Action",
    "recommendedMovies": [
      {
        "id": 361743,
        "title": "Top Gun: Maverick",
        "release_date": "2022-05-24",
        "poster_url": "https://image.tmdb.org/t/p/w500/62HCnUTziyWcpDaBO2i1DX17ljH.jpg"
      }
    ]
  },
  "message": "Recommendations generated successfully",
  "status": true
}
```

---

## 🛠️ Development & Testing

### **Run Tests**
```sh
pytest
```

### **Lint & Format Code**
```sh
black .
```

### **Run in Debug Mode**
```sh
FLASK_ENV=development python app.py
```
---

### 🎉 **Contribute to HarmOni AI!**

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch:  
   ```sh
   git checkout -b feature/new-feature
   ```
3. Implement and test your changes.
4. Commit with a meaningful message:  
   ```sh
   git commit -m "feat: Add new recommendation logic"
   ```
5. Push to your fork and create a pull request.
   
---

## 📝 License

HarmOni AI is **open-source** and distributed under the **MIT License**.

