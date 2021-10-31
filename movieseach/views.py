from django.shortcuts import render
import os
from . utils.functions import *
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import numpy as np
from pymongo import MongoClient
import locale

dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
client = MongoClient('localhost', 27017)  # mongodb://localhost:27017/
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')  # used to format currency values

cache = dict()


def clean_movie(movie, cut_description=None):
    """Clean movie fields"""
    if cut_description is not None and len(movie["description"]) > cut_description:  # cut overview
        movie["description"] = movie["description"][:cut_description] + "..."
    if type(movie["imdb_id"]) != int and type(movie["imdb_id"]) != float:  # extract imdb ID
        movie["imdb_id"] = int(movie["imdb_id"][2:])
    movie["runtime"] = int(movie["runtime"])
    movie["release_date"] = int(movie["release_date"])
    if type(movie["budget"]) != str or not movie["budget"].startswith("$"):
        movie["budget"] = locale.currency(int(movie["budget"]), grouping=True)[:-3]
    if type(movie["revenue"]) != str or not movie["revenue"].startswith("$"):
        movie["revenue"] = locale.currency(int(movie["revenue"]), grouping=True)[:-3]
    movie["img_path"] = os.path.join("posters", str(movie["imdb_id"]) + ".jpg")
    if not os.path.exists((os.path.join(dir_path, "moviesearch", "static", movie["img_path"]))) \
            or os.stat(os.path.join(dir_path, "moviesearch", "static", movie["img_path"])).st_size < 1:
        movie["img_path"] = "missing_image.png"
    return movie


def connect_db():
    """Connect to the db and return movies table (movies_db) and reverse index (word_movies)"""
    db = client.movie_search  # getting a database
    movies_db = db.movies_data  # movies table
    word_movies = db.word_movies  # reverse index table
    reviews_db = db.movies_review  # reviews table
    return movies_db, word_movies, reviews_db


def retrieve_movie(movies_db, idx):
    """If movie with index idx in cache return it,
    else download it from movies_db, store it in the cache and return it"""
    if idx in cache:
        return cache[idx]
    else:
        doc = movies_db.find_one({"movie_id": idx}, {"_id": 0})
        cache[idx] = doc
        return doc


def index(request):
    if request.POST:
        print("Raw query:", request.POST['query'])
        query_terms = drop_duplicates(preprocess_text(request.POST['query']))
        print("Query terms:", query_terms)
        movies_db, word_movies, _ = connect_db()
        movies_ids = []
        for term in query_terms:  # get possible movies
            doc = word_movies.find_one({"word": term}, {"movies": 1})
            if doc is None:
                continue
            movies_ids += doc["movies"]
        if len(movies_ids) == 0:
            print("No movies found")
            return render(request, os.path.join('templates/index.html'),
                          {"info": "Sorry, no movies matched your query :( "})
        movies_ids = drop_duplicates(movies_ids)
        # get the id of all movies containing at least one words in the query in their title or description
        print("Retrieved {} movies".format(len(movies_ids)))

        movies = []  # list of dicts
        corpus = []  # list of title+description for each movie
        print("Retrieving movies data...")
        for i in tqdm(movies_ids):
            doc = retrieve_movie(movies_db, i)
            movies.append(doc)
            title = preprocess_text(doc["title"])
            description = preprocess_text(doc["description"])
            local_corpus = title + title + description  # title repeated 2 times to give more importance ot the tile
            if doc["actors"] is not None:
                actors = preprocess_list(doc["actors"])
                local_corpus += actors
            if doc["production_companies"] is not None:
                companies = preprocess_list(doc["production_companies"])
                local_corpus += companies
            if doc["director_name"] is not None:
                director = preprocess_text(doc["director_name"])
                local_corpus += director
            corpus.append(local_corpus)  # words in title count as two

        print("Ranking movies...")
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query_terms)  # get the score of each document given the query
        n = 20
        top_n = np.argsort(scores)[::-1][:n]  # indexes of the top n documents
        top_n_movies = [movies[i] for i in top_n]
        for movie in top_n_movies:
            clean_movie(movie, 400)
        print("Top-{} movies".format(n))
        for i, movie in enumerate(top_n_movies):
            print(i+1, movie["title"])

        return render(request, os.path.join('templates/index.html'), {'movies': top_n_movies})

    return render(request, os.path.join('templates/index.html'))


def details(request):
    if request.POST:
        print("Movie id:", request.POST['movie_id'])
        movies_db, _, reviews_db = connect_db()
        movie = retrieve_movie(movies_db, int(request.POST['movie_id']))
        reviews = reviews_db.find_one({"imbd_id": movie["imdb_id"]}, {"_id": 0})
        print("Imdb ID", movie["imdb_id"])

        reviews_list = []
        if reviews is not None:
            for (review, label) in zip(reviews["reviews"], reviews["labels"]):
                reviews_list.append({"review": review, "label": label})

        movie = clean_movie(movie)
        similar_movies = [clean_movie(retrieve_movie(movies_db, int(idx)), 400) for idx in movie["similar_movies"]]

        return render(request, os.path.join('templates/details.html'),
                      {'movie': movie, 'similar_movies': similar_movies, 'reviews': reviews_list})
    return render(request, os.path.join('templates/index.html'))
