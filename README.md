# Title: Building a website to search and recommendation movies
## Team:
### Võ Hà Nhật Tân - 18133047
### Phạm Đình Nhiên - 18133038
## List of project requirements:
### Search and recommendation functions
### Recommend movies that match the searched movie
### Show detailed movie information, actors, reviews
### Tools used: Python, Jupyter notebook, Front-end: HTML.
### Algorithm to use: Content-based filtering, Matrix Similarity, Knn, search algorithm Okapi-BM25

# How to install system
## Prepare:
- Clone this project
- Download File Posters in https://github.com/vohanhattan/RecommendationSystem/tree/main/backup and extract to ./movieseach/static
- Download File nlkt_data in https://github.com/vohanhattan/RecommendationSystem/tree/main/backup extract to C:\ or D:\
- Download Review file aclImdb in https://github.com/vohanhattan/RecommendationSystem/tree/main/backup extract to ./movieseach/utils/
- Extract file movie csv in /utils/movies
- You need to install the necessary libraries [pip install -r requirement]
- You should upload_movies and upload_reviews (run file python in ./moviesearch/utils)
- You can upload movie, delete, edit by run file menu.py in ./moviesearch/utils base on add_movie.csv

## Run Project:
- Run server: python main.py runserver
- http://localhost:8000/
