from pymongo import MongoClient
from functions import *
import glob
from tqdm import tqdm


if __name__ == '__main__':

    client = MongoClient('localhost', 27017)  # mongodb://localhost:27017/
    db = client.movie_search  # getting a database
    if "movies_review" in db.list_collection_names():
        db.movies_review.drop()  # dropping a collection
    reviews_db = db.movies_review  # getting a collection

    train_positive = "aclImdb/train/pos"
    train_negative = "aclImdb/train/neg"
    test_positive = "aclImdb/test/pos"
    test_negative = "aclImdb/test/neg"
    paths = [train_positive, train_negative, test_positive, test_negative]

    train_positive_urls = "aclImdb/train/urls_pos.txt"
    train_negative_urls = "aclImdb/train/urls_neg.txt"
    test_positive_urls = "aclImdb/test/urls_pos.txt"
    test_negative_urls = "aclImdb/test/urls_neg.txt"
    urls = [train_positive_urls, train_negative_urls, test_positive_urls, test_negative_urls]

    d = dict()
    for j, file in enumerate(urls):
        f = open(file, "r")
        lines = f.readlines()
        file_list = glob.glob(paths[j] + "/*.txt")
        for i, review in tqdm((enumerate(file_list))):
            review_text = open(review, "r", encoding="utf-8").readlines()[0]
            review_text = review_text.replace("<br />", "").replace("/", "").replace("\\", "")
            review_id = int(review.split("\\")[-1].split("_")[0])
            imdb_id = lines[review_id].split("/")[-2]
            if j % 2 == 1:
                label = "neg"
            else:
                label = "pos"
            if imdb_id in d.keys():
                d[imdb_id].append((review_text, label))
            else:
                d[imdb_id] = [(review_text, label)]
        f.close()

    for (key, item) in tqdm(d.items()):
        review = {"imbd_id": int(key[3:]),
                  "reviews": [i[0] for i in item],
                  "labels": [i[1] for i in item]
                  }
        reviews_db.insert_one(review)
