from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import pandas as pd


ps = PorterStemmer()
table = str.maketrans('', '', string.punctuation)
stop_words = set(stopwords.words('english'))



def preprocess_text(line):
    
#    Split mỗi từ cách nhau một khoảng trắng
#    bỏ các dấu câu.
#    biến đổi thành chữ viết thường
#    loại bỏ các stopword
#    bỏ những từ nhỏ hơn 2 ký tự
    tokens = line.split()
    tokens = [w.translate(table).lower() for w in tokens]
    tokens = [ps.stem(w) for w in tokens if w not in stop_words and len(w) > 1]
    return tokens  


def drop_duplicates(l):
#Bỏ các ký tự trùng
    return list(dict.fromkeys(l))

def exists(value):
    if pd.isna(value) or pd.isnull(value) or value is None:
        return False
    return True

def preprocess_list(array):
    tokens = []
    if array is None:
        return tokens
    for s in array:
        if s is not None:
            tokens += preprocess_text(s)
    return tokens


if __name__ == '__main__':
    print("")
