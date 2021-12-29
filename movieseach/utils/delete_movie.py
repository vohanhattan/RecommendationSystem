import pandas as pd
import numpy as np
from pymongo import MongoClient
from functions import *
from create_dataset import *
import sys

if __name__ == '__main__':
    client = MongoClient('localhost', 27017)  # mongodb://localhost:27017/
    db = client.movie_search 
    movies_db = db.movies_data  # getting a collection
    word_movies = db.word_movies  # getting a collection 
    myDF = pd.read_csv("movies/add_movie.csv",encoding="ISO-8859-1")
    find=False
    list_id=[]
    print('Enter movie_id to delete:')
    userInput = input()
    try:
        myInput = int(userInput)
        for i, (index, row) in enumerate(myDF.iterrows()):
            find_id=movies_db.find({'movie_id':myInput})
            for i in find_id:
                list_id.append(i['movie_id'])
            if(len(list_id)):
                find=True
#--------------------------Delete function--------------------------------
                movies_db.delete_one({"movie_id":myInput})
                word_movies.update({},{'$pull':{"movies":myInput}},multi=True,upsert=False)
                movies_db.update({},{'$pull':{"similar_movies":myInput}},multi=True,upsert=False)
            else:
                find=False
                
                
    except ValueError:
        print("movie_id has integer data type, so please enter integer!")
        sys.exit()
    
    if find:
        print("Deleted movie by movie_id ==> [SUCCESS]")
    else:
        print("movie_id does not exist, try again later ==> [FAIL]")
    

    
