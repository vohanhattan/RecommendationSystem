from pymongo import MongoClient
from functions import *
from create_dataset import *
import pandas as pd

# utilities
reverse_index = dict()


def fill_reverse_index(tokens, movie_id):
    """Populate the reverse index with the given words and movie_id
    word -> [idx1, idx2, ...] the indices to the movies containing that word"""
    tokens = tokens
    for w in tokens:
        if w in reverse_index.keys():
            reverse_index[w].append(movie_id)
        else:
            reverse_index[w] = [movie_id]


if __name__ == '__main__':

    client = MongoClient('localhost', 27017)  # mongodb://localhost:27017/
    db = client.movie_search  # getting a database
    if "movies_data" in db.list_collection_names():
        db.movies_data.drop()  # dropping a collection
    if "word_movies" in db.list_collection_names():
        db.word_movies.drop()  # dropping a collection
    movies_db = db.movies_data  # getting a collection
    word_movies = db.word_movies  # getting a collection

    n_neighbors = 31
    n_recommendations = 15
    credits = "movies/tmdb_5000_credits.csv"
    movies = "movies/tmdb_5000_movies.csv"
    movies_imdb = "movies/movies_metadata.csv"
    df = clean_dataset(movies, credits, movies_imdb)

    for i, (index, row) in enumerate(df.iterrows()):
        movie_idx = row["id"]
        if not exists(row["overview"]):  # ignore movies without description
            continue
        description = row["overview"]
        genres = row["genres"].split("|")  # no missing values for this field
        production_countries = None
        if exists(row["production_countries"]):
            production_countries = row["production_countries"].split("|")
        production_companies = None
        if exists(row["production_companies"]):
            production_companies = row["production_companies"].split("|")
        actors = [row["actor_1_name"], row["actor_2_name"], row["actor_3_name"]]
        actors = actors[:3-pd.isna(actors).sum()]

        runtime = row["runtime"]
        if not exists(row["runtime"]):
            runtime = None

        director_name = row["director_name"]
        if pd.isna(row["director_name"]):
            director_name = None

        similar_movies = find_similars(df, i, n_neighbors, n_recommendations, verbose=False)

        movie = {"movie_id": movie_idx,
                 "title": row["title"],
                 "description": description,
                 "genres": genres,
                 "imdb_id": row["imdb_id"],
                 "spoken_languages": row["spoken_languages"],
                 "production_companies": production_companies,
                 "production_countries": production_countries,
                 "release_date": row["release_date"],
                 "runtime": runtime,
                 "vote_average": row["vote_average"],
                 "vote_count": row["vote_count"],
                 "budget": row["budget"],
                 "popularity": row["popularity"],
                 "revenue": row["revenue"],
                 "director_name": director_name,
                 "actors": actors,
                 "similar_movies": similar_movies
                 }

        movies_db.insert_one(movie)
        fill_reverse_index(preprocess_text(description), movie_idx)
        fill_reverse_index(preprocess_text(row["title"]), movie_idx)
        if actors is not None:
            fill_reverse_index(preprocess_list(actors), movie_idx)
        if production_companies is not None:
            fill_reverse_index(preprocess_list(production_companies), movie_idx)
        if director_name is not None:
            fill_reverse_index(preprocess_text(director_name), movie_idx)
        print(i, row["title"])

    for key, item in reverse_index.items():
        word_movies.insert_one({"word": key, "movies": item})
    print("Inserted {} movies".format(movies_db.count_documents({})))
    print("Reverse index contains {} words".format(len(reverse_index.keys())))
