import pandas as pd
import numpy as np
from pymongo import MongoClient
from functions import *
from create_dataset import *
import sys

def check_null_value(myDF):
    for i, (index, row) in enumerate(myDF.iterrows()):
        if pd.isna(row["keywords"]) or pd.isna(row["genres"]) or pd.isna(row["id"]) or pd.isna(row["genres"]) or pd.isna(row["overview"]) or pd.isna(row["title"]) or pd.isna(row["production_countries"]) or pd.isna(row["production_companies"]) or pd.isna(row["actor_1_name"]) or pd.isna(row["actor_2_name"]) or pd.isna(row["actor_3_name"]) or pd.isna(row["runtime"]) or pd.isna(row["director_name"]) or pd.isna(row["imdb_id"]) or pd.isna(row["budget"]) or pd.isna(row["release_date"]) or pd.isna(row["spoken_languages"]) or pd.isna(row["vote_average"]) or pd.isna(row["vote_count"]) or pd.isna(row["popularity"]) or pd.isna(row["revenue"]):
            print("The data in the CSV file cannot be empty, Please check the CSV file and try again!")
            return False
        if not isinstance(row["id"],int):
            print("The Movie id is not of type int please edit and run again")
            return False
    return True

def add_is_valid(myDF):
    client = MongoClient('localhost', 27017)  
    db = client.movie_search  
    movies_db = db.movies_data  # getting a collection  
    list_id=[]
    find_id=movies_db.find({'movie_id':myDF["id"]})
    for id_movies in find_id:
        list_id.append(id_movies['movie_id'])

    if(len(list_id)):
        return True
    print("The id movie",myDF["id"] ,":", myDF["title"] ,"doesn't exist in database, try again!")
    return False

def check_input_in_csv(myInput, myDF):
    for i, (index, row) in enumerate(myDF.iterrows()):
            if(myInput==row["id"]):
                return True            
    return False

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
    db = client.movie_search  
    movies_db = db.movies_data  # getting a collection
    word_movies = db.word_movies  # getting a collection 
    myDF = pd.read_csv("movies/add_movie.csv",encoding="ISO-8859-1")
    cleanDF = pd.read_csv("movies/cleaned_data.csv", encoding="ISO-8859-1")
    totalDF= cleanDF.append(myDF,ignore_index=True)
    total_movies=myDF["id"].sum()
    count_movie_valid=0
    n_neighbors = 31
    n_recommendations = 15
    if(check_null_value(myDF)==False):
        sys.exit()
    print('Enter movie_id to update one:')
    userInput = input()
    try:
        myInput = int(userInput)
        if(check_input_in_csv(myInput,myDF)):
            for i, (index, row) in enumerate(myDF.iterrows()):
                if(myInput==row["id"]):
                    myDF=myDF.iloc[i]
                    break
            
        else:
            print("movie_id does not exist in csv file, try again");
            sys.exit()
        
                
                
    except ValueError:
        print("movie_id has integer data type, so please enter integer!")
        sys.exit()

    myDF["id"]=myDF["id"].item()
    myDF["release_date"]=myDF["release_date"].item()
    myDF["runtime"]=myDF["runtime"].item()

    if(add_is_valid(myDF)):
        count_movie_valid=+1
        movie_idx = myDF["id"]
        description = myDF["overview"]
        genres = myDF["genres"].split("|")  
        production_countries = myDF["production_countries"].split("|")
        production_companies = myDF["production_companies"].split("|")
        actors = [myDF["actor_1_name"], myDF["actor_2_name"], myDF["actor_3_name"]]
        actors = actors[:3-pd.isna(actors).sum()]

        runtime = myDF["runtime"]
        if not exists(myDF["runtime"]):
            runtime = None

        director_name = myDF["director_name"]
        if pd.isna(myDF["director_name"]):
            director_name = None
            

        for i, (index, row) in enumerate(totalDF.iterrows()):
                if(myInput==row["id"]):
                    id_entry=i
                    break
        similar_movies = find_similars(totalDF, id_entry, n_neighbors, n_recommendations, verbose=False)

        movie = {"movie_id": movie_idx,
                "title": myDF["title"],
                "description": description,
                "genres": genres,
                "imdb_id": myDF["imdb_id"],
                "spoken_languages": myDF["spoken_languages"],
                "production_companies": production_companies,
                "production_countries": production_countries,
                "release_date": myDF["release_date"],
                "runtime": runtime,
                "vote_average": myDF["vote_average"],
                "vote_count": myDF["vote_count"].item(),
                "budget": myDF["budget"].item(),
                "popularity": myDF["popularity"],
                "revenue": myDF["revenue"].item(),
                "director_name": director_name,
                "actors": actors,
                "similar_movies": similar_movies
                }
        movies_db.delete_one({"movie_id":myInput})
        word_movies.update({},{'$pull':{"movies":myInput}},multi=True,upsert=False)
        movies_db.insert_one(movie)
#Tạo các Keyword từ phần mô tả, tên tiêu đề, diễn viên, đạo diễn, nhà sản xuất
        fill_reverse_index(preprocess_text(description), movie_idx)
        fill_reverse_index(preprocess_text(myDF["title"]), movie_idx)
        if actors is not None:
            fill_reverse_index(preprocess_list(actors), movie_idx)
        if production_companies is not None:
            fill_reverse_index(preprocess_list(production_companies), movie_idx)
        if director_name is not None:
            fill_reverse_index(preprocess_text(director_name), movie_idx)
        print(i, myDF["title"])
        print("Add one movie by movie_id ==> [SUCCESS]")
    else:
        sys.exit()
    # print(type(myDF["id"]))
    print("Inserted",count_movie_valid,"movies")

#------------------------------------- Word movie -------------------------------------------------
    for key, item in reverse_index.items():
        print("The word", key ,"has been updated in the database ==> [Success]")
        word_movies.update_one({"word": key},{'$addToSet':{"movies":{'$each':list(dict.fromkeys(item))}}},upsert=True)
    print("Reverse index contains {} words".format(len(reverse_index.keys())))
    print("Reverse contains {} words".format(reverse_index))