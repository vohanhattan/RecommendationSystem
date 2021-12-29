import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pymongo import MongoClient
from functions import *
from create_dataset import *
import sys

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

def add_is_valid(myDF):
    client = MongoClient('localhost', 27017)  
    db = client.movie_search  
    movies_db = db.movies_data  # getting a collection
    word_movies = db.word_movies  # getting a collection 
    list_imdb_id=[]
    list_title=[]
    list_id=[]
    #for i, (index, row) in enumerate(myDF.iterrows()):
    find_imdb_id_db=movies_db.find({'imdb_id':myDF["imdb_id"]})
    for imdb_id in find_imdb_id_db:
        list_imdb_id.append(imdb_id['imdb_id'])

    find_title_db=movies_db.find({'title':myDF["title"]})
    for title in find_title_db:
        list_title.append(title['title'])

    find_id=movies_db.find({'movie_id':myDF["id"]})
    for id_movies in find_id:
        list_id.append(id_movies['movie_id'])

    if(len(list_id)):
        print("The id of the movie",list_id, "is duplicate ==> [PASS]")
        return False
    if(len(list_title)):
        print("The title of the movie",list_title, "is duplicate ==> [PASS]")
        return False
    if(len(list_imdb_id)):
        print("The imdb id of the movie",list_imdb_id, "is duplicate ==> [PASS]")
        return False
    return True



if __name__ == '__main__':
    
    cleanDF = pd.read_csv("movies/cleaned_data.csv", encoding="ISO-8859-1")
    myDF = pd.read_csv("movies/add_movie.csv",encoding="ISO-8859-1")
    totalDF= cleanDF.append(myDF,ignore_index=True)
    for i, (index, row) in enumerate(myDF.iterrows()):
        if pd.isna(row["keywords"]) or pd.isna(row["genres"]) or pd.isna(row["id"]) or pd.isna(row["genres"]) or pd.isna(row["overview"]) or pd.isna(row["title"]) or pd.isna(row["production_countries"]) or pd.isna(row["production_companies"]) or pd.isna(row["actor_1_name"]) or pd.isna(row["actor_2_name"]) or pd.isna(row["actor_3_name"]) or pd.isna(row["runtime"]) or pd.isna(row["director_name"]) or pd.isna(row["imdb_id"]) or pd.isna(row["budget"]) or pd.isna(row["release_date"]) or pd.isna(row["spoken_languages"]) or pd.isna(row["vote_average"]) or pd.isna(row["vote_count"]) or pd.isna(row["popularity"]) or pd.isna(row["revenue"]):
            print("The data in the CSV file cannot be empty, Please check the CSV file and try again!")
            sys.exit()
        if not isinstance(row["id"],int):
            print("The Movie id is not of type int please edit and run again")
            sys.exit()
    count_movie_valid=0
    list_id_movie=[]
    
    

    client = MongoClient('localhost', 27017)  # mongodb://localhost:27017/
    db = client.movie_search  
    movies_db = db.movies_data  # getting a collection
    word_movies = db.word_movies  # getting a collection 
    n_neighbors = 31
    n_recommendations = 15
    
    for i, (index, row) in enumerate(myDF.iterrows()):
        if(add_is_valid(row)):
            count_movie_valid=+1
            movie_idx = row["id"]
            description = row["overview"]
            genres = row["genres"].split("|")  
            production_countries = row["production_countries"].split("|")
            production_companies = row["production_companies"].split("|")
            actors = [row["actor_1_name"], row["actor_2_name"], row["actor_3_name"]]
            actors = actors[:3-pd.isna(actors).sum()]

            runtime = row["runtime"]
            if not exists(row["runtime"]):
                runtime = None

            director_name = row["director_name"]
            if pd.isna(row["director_name"]):
                director_name = None
            

            
            similar_movies = find_similars(totalDF, i+4807, n_neighbors, n_recommendations, verbose=False)

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
#Tạo các Keyword từ phần mô tả, tên tiêu đề, diễn viên, đạo diễn, nhà sản xuất
            fill_reverse_index(preprocess_text(description), movie_idx)
            fill_reverse_index(preprocess_text(row["title"]), movie_idx)
            if actors is not None:
                fill_reverse_index(preprocess_list(actors), movie_idx)
            if production_companies is not None:
                fill_reverse_index(preprocess_list(production_companies), movie_idx)
            if director_name is not None:
                fill_reverse_index(preprocess_text(director_name), movie_idx)
            print(i, row["title"])
        else:
            continue
    print("Inserted",count_movie_valid,"movies")
        
#------------------------------------- Word movie -------------------------------------------------
    for key, item in reverse_index.items():
        print("The word", key ,"has been updated in the database ==> [Success]")
        word_movies.update_one({"word": key},{'$addToSet':{"movies":{'$each':list(dict.fromkeys(item))}}},upsert=True)
    print("Reverse index contains {} words".format(len(reverse_index.keys())))
    print("Reverse contains {} words".format(reverse_index))
            

    
