import json
import pandas as pd
import numpy as np
import math
import nltk
from nltk.corpus import wordnet
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
import os
from fuzzywuzzy import fuzz

PS = nltk.stem.PorterStemmer()
gaussian_filter = lambda x, y, sigma: math.exp(-(x - y) ** 2 / (2 * sigma ** 2))

#-------------------------------------------------CLEAN DATA FROM tmdb_5000_movies.csv,tmdb_5000_credits.csv,movies_metadata.csv-------------------------------------
def load_tmdb_movies(path):
    """Load movie database"""
    df = pd.read_csv(path, encoding="utf-8")
    df['release_date'] = pd.to_datetime(df['release_date']).apply(lambda x: x.date())
    json_columns = ['genres', 'keywords', 'production_countries',
                    'production_companies', 'spoken_languages']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df


def load_tmdb_credits(path):
    """Load credits database"""
    df = pd.read_csv(path, encoding="utf-8")
    json_columns = ['cast', 'crew']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df


def safe_access(container, index_values):
    """return nan value avoid error"""
    result = container
    try:
        for idx in index_values:
            result = result[idx]
        return result
    except IndexError or KeyError:
        return np.nan


def get_director(crew_data):
    """get director name for each movie"""
    directors = [x['name'] for x in crew_data if x['job'] == 'Director']
    return safe_access(directors, [0])


def pipe_flatten_names(keywords):
    """Separate keywords by '|'"""
    return '|'.join([x['name'] for x in keywords])


def convert_to_original_format(tmdb_movies, credits):
    """Create the full dataset, merge 2 datasets to complete datasetfinal"""
    tmdb_movies['release_date'] = pd.to_datetime(tmdb_movies['release_date']).apply(lambda x: x.year)
    tmdb_movies['production_countries'] = tmdb_movies['production_countries'].apply(
        lambda x: safe_access(x, [0, 'name']))
    tmdb_movies['production_companies'] = tmdb_movies['production_companies'].apply(
        lambda x: safe_access(x, [0, 'name']))
    tmdb_movies['spoken_languages'] = tmdb_movies['spoken_languages'].apply(lambda x: safe_access(x, [0, 'name']))
    tmdb_movies['genres'] = tmdb_movies['genres'].apply(pipe_flatten_names)
    tmdb_movies['keywords'] = tmdb_movies['keywords'].apply(pipe_flatten_names)
    # new columns
    tmdb_movies['director_name'] = credits['crew'].apply(get_director)
    tmdb_movies['actor_1_name'] = credits['cast'].apply(lambda x: safe_access(x, [1, 'name']))
    tmdb_movies['actor_2_name'] = credits['cast'].apply(lambda x: safe_access(x, [2, 'name']))
    tmdb_movies['actor_3_name'] = credits['cast'].apply(lambda x: safe_access(x, [3, 'name']))
    return tmdb_movies

"""Đối với mỗi bộ phim sẽ được mô tả bằng các từ khóa, khi truy xuất không cần phải truy xuất hết csdl mà chỉ cần truy xuất các từ khóa từ đó 
ta sẽ được bộ phim tương ứng ==> tiết kiệm thời gian truy xuất nâng cao hiệu quả của chương trình"""


def count_word(df, ref_col, t_list):
    """keyword = all words in t_list
       keyword_count: keyword -> #keyboard appears in df[ref_col]
       keyword_sorted: (same as keyword_count but ordered by count)
    """
    keyword_count = dict()
    for s in t_list:
        keyword_count[s] = 0
    for list_keywords in df[ref_col].str.split('|'):
        if type(list_keywords) == float and pd.isnull(list_keywords):
            continue
        for s in [s for s in list_keywords if s in t_list]:
            if pd.notnull(s):
                keyword_count[s] += 1
    # chuyển đổi dictionary trong danh sách để sắp xếp các từ khóa theo tần suất
    keyword_sorted = []
    for k, v in keyword_count.items():
        keyword_sorted.append([k, v])
    keyword_sorted.sort(key=lambda x: x[1], reverse=True)
    return keyword_sorted, keyword_count

"""Collect các từ khóa có trong biến keywords sau đó được làm sạch bằng NLTK, Finally tìm kiếm số lần xuất hiện các từ khóa"""
def keywords_inventory(dataframe, colonne='keywords'):
    """Collect the keywords"""
    keywords_roots = dict()  
    keywords_select = dict()  
    category_keys = []
    for s in dataframe[colonne]:  
        if pd.isnull(s):
            continue
        for t in s.split('|'):  
            t = t.lower()
            root = PS.stem(t)  
            if root in keywords_roots:
                keywords_roots[root].add(t)
            else:
                keywords_roots[root] = {t}
    for s in keywords_roots.keys():
        if len(keywords_roots[s]) > 1:
            min_length = 1000  
            for k in keywords_roots[s]:  
                if len(k) < min_length:
                    clef = k  
                    min_length = len(k)
            category_keys.append(clef)
            keywords_select[s] = clef
        else:
            category_keys.append(list(keywords_roots[s])[0])
            keywords_select[s] = list(keywords_roots[s])[0]
    return category_keys, keywords_roots, keywords_select


def replacement_df_keywords(df, replacement, roots=False):
    """Replace all keywords by the main keyword"""
    df_new = df.copy(deep=True)
    for index, row in df_new.iterrows():
        keywords = row['keywords']
        if pd.isnull(keywords):
            continue
        new_keywords = []
        for s in keywords.split('|'):
            word = PS.stem(s) if roots else s
            if word in replacement.keys():
                new_keywords.append(replacement[word])
            else:
                new_keywords.append(s)
        df_new.at[index, 'keywords'] = '|'.join(new_keywords)
    return df_new


"""Groups of synonyms """
def get_synonyms(word):
    """get the synonyms of 'word'"""
    lemma = set()
    for ss in wordnet.synsets(word):
        for w in ss.lemma_names():
            # We just get the 'nouns':
            index = ss.name().find('.')+1
            if ss.name()[index] == 'n':
                lemma.add(w.lower().replace('_', ' '))
    return lemma


def test_keyword(word, key_count, threshold):
        """check if 'word' is a key of 'key_count' with a test on the number of occurrences"""
        return (False, True)[key_count.get(word, 0) >= threshold]


def replacement_df_low_frequency_keywords(df, keyword_occurrences, thr=3):
    """delete of keywords with low frequenci"""
    df_new = df.copy(deep=True)
    key_count = dict()
    for s in keyword_occurrences:
        key_count[s[0]] = s[1]
    for index, row in df_new.iterrows():
        keywords = row['keywords']
        if pd.isnull(keywords):
            continue
        new_list = []
        for s in keywords.split('|'):
            if key_count.get(s, 4) > thr:
                new_list.append(s)
        df_new.at[index, 'keywords'] = '|'.join(new_list)
    return df_new


def fill_year(df):
    """Dự đoán năm sản xuất nếu trống bằng hoạt động gần nhất của các diễn viên"""
    col = ['director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name']
    usual_year = [0 for _ in range(4)]
    var = [0 for _ in range(4)]
    
    for i in range(4):
        usual_year[i] = df.groupby(col[i])['release_date'].mean()
    actor_year = dict()
    for i in range(4):
        for s in usual_year[i].index:
            if s in actor_year.keys():
                if pd.notnull(usual_year[i][s]) and pd.notnull(actor_year[s]):
                    actor_year[s] = (actor_year[s] + usual_year[i][s]) / 2
                elif pd.isnull(actor_year[s]):
                    actor_year[s] = usual_year[i][s]
            else:
                actor_year[s] = usual_year[i][s]
    missing_year_info = df[df['release_date'].isnull()]
    count_replaced = 0
    for index, row in missing_year_info.iterrows():
        value = [np.NaN for _ in range(4)]
        count = 0
        sum_year = 0
        for i in range(4):
            var[i] = df.loc[index][col[i]]
            if pd.notnull(var[i]):
                value[i] = actor_year[var[i]]
            if pd.notnull(value[i]):
                count += 1
                sum_year += actor_year[var[i]]
        if count != 0:
            sum_year = sum_year / count

        if int(sum_year) > 0:
            count_replaced += 1
            df.at[index, 'release_date'] = int(sum_year)

    return


def variable_linreg_imputation(df, col_to_predict, ref_col):
    """đưa ra giá trị còn thiếu từ sự phù hợp tuyến tính của dữ liệu"""
    regr = linear_model.LinearRegression()
    test = df[[col_to_predict, ref_col]].dropna(how='any', axis=0)
    X = np.array(test[ref_col])
    Y = np.array(test[col_to_predict])
    X = X.reshape(len(X), 1)
    Y = Y.reshape(len(Y), 1)
    regr.fit(X, Y)

    test = df[df[col_to_predict].isnull() & df[ref_col].notnull()]
    for index, row in test.iterrows():
        value = float(regr.predict(row[ref_col]))
        df.at[index, col_to_predict] = value


def clean_dataset(movies_path, credits_path, imdb_path):
    credits = load_tmdb_credits(credits_path)
    movies = load_tmdb_movies(movies_path)
    movies_imdb = pd.read_csv(imdb_path, encoding="utf-8")[["id", "imdb_id"]].astype({'id': 'int'})
    df_initial = convert_to_original_format(movies, credits)
    df_initial = df_initial.join(movies_imdb.set_index('id'), on='id')
    print("Movies:", df_initial.shape[0])
    print("Features:", df_initial.shape[1])

    # set keywords
    set_keywords = set()
    for list_keywords in df_initial['keywords'].str.split('|').values:
        if isinstance(list_keywords, float):
            continue  
        set_keywords = set_keywords.union(list_keywords)
    # remove null 
    set_keywords.remove('')
    keyword_occurrences, _ = count_word(df_initial, 'keywords', set_keywords)
    print("Original total keywords:", len(keyword_occurrences))

    # set all genres
    genre_labels = set()
    for s in df_initial['genres'].str.split('|').values:
        genre_labels = genre_labels.union(set(s))
    genres_occurrences, dum = count_word(df_initial, 'genres', genre_labels)
    print("Genres: ", len(genres_occurrences))

    df_duplicate_cleaned = df_initial
    
    keywords, keywords_roots, keywords_select = keywords_inventory(df_duplicate_cleaned, colonne='keywords')
    # replacement of the keywords by the main keyword
    df_keywords_cleaned = replacement_df_keywords(df_duplicate_cleaned, keywords_select, roots=True)
    # count số lần xuất hiện của keyword
    keywords.remove('')
    keyword_occurrences, keywords_count = count_word(df_keywords_cleaned, 'keywords', keywords)
    print("Cleaned total keywords:", len(keyword_occurrences))

    key_count = dict()
    for s in keyword_occurrences:
        key_count[s[0]] = s[1]
    # tạo dic để lưu keyword có tần suất cao
    replacement_mot = dict()

    # thay thế các từ khóa xuất hiện ít hơn 5 lần bằng từ đồng nghĩa xuất hiện nhiều lần hơn
    for index, [mot, nb_apparitions] in enumerate(keyword_occurrences):
        if nb_apparitions > 5:
            continue  # chỉ những từ khóa xuất hiện ít hơn 5 lần
        lemma = get_synonyms(mot)
        if len(lemma) == 0:
            continue     
        list_mots = [(s, key_count[s]) for s in lemma if test_keyword(s, key_count, key_count[mot])]
        list_mots.sort(key=lambda x: (x[1], x[0]), reverse=True)
        if len(list_mots) <= 1:
            continue       
        if mot == list_mots[0][0]:
            continue    
        
        replacement_mot[mot] = list_mots[0][0]

    print('The replacement concerned {} keywords.'.format(len(replacement_mot)))

    # thay thế các keyword bằng mainKEyword
    df_keywords_synonyms = replacement_df_keywords(df_keywords_cleaned, replacement_mot, roots=False)
    keywords, keywords_roots, keywords_select = keywords_inventory(df_keywords_synonyms, colonne='keywords')

    # số lần xuất hiện từ khóa mới
    keywords.remove('')
    new_keyword_occurrences, keywords_count = count_word(df_keywords_synonyms, 'keywords', keywords)

    # Create dataframe trong đó các từ khóa có tần số thấp bị loại bỏ
    df_keywords_occurrence = \
        replacement_df_low_frequency_keywords(df_keywords_synonyms, new_keyword_occurrences)
    keywords, keywords_roots, keywords_select = \
        keywords_inventory(df_keywords_occurrence, colonne='keywords')

    # new keywords count
    keywords.remove('')

    df_cleaned = df_keywords_occurrence.copy(deep=True)
    df_filling = df_cleaned.copy(deep=True)

    # fill missing year values
    fill_year(df_filling)

    # fill missing values in the plot_keywords
    for index, row in df_filling[df_filling['keywords'].isnull()].iterrows():
        list_mot = row['title'].strip().split()
        new_keyword = []
        for s in list_mot:
            lemma = get_synonyms(s)
            for t in list(lemma):
                if t in keywords:
                    new_keyword.append(t)
        if new_keyword:
            df_filling.at[index, 'keywords'] = '|'.join(new_keyword)

    # fill revenue value given vote_count
    variable_linreg_imputation(df_filling, 'revenue', 'vote_count')
    df = df_filling.copy(deep=True)

    # final dataset
    df.reset_index(inplace=True, drop=True)
    return df


#-------------------------------------------------RECOMMENDATION SYSTEM-------------------------------------------------------------------------------------------------

def entry_variables(df, id_entry):
    """Function collecting some variables content base id_entry trả về giá trị tên đạo diễn, tên các diễn viên, keywords"""
    col_labels = []
    if pd.notnull(df['director_name'].iloc[id_entry]):
        for s in df['director_name'].iloc[id_entry].split('|'):
            col_labels.append(s)

    for i in range(3):
        column = 'actor_NUM_name'.replace('NUM', str(i + 1))
        if pd.notnull(df[column].iloc[id_entry]):
            for s in df[column].iloc[id_entry].split('|'):
                col_labels.append(s)

    if pd.notnull(df['keywords'].iloc[id_entry]):
        for s in df['keywords'].iloc[id_entry].split('|'):
            col_labels.append(s)
    return col_labels


def add_variables(df, ref_var):
    """add một list biến vào dataframe đã cho trong input và khởi tạo các biến 0 or 1 tùy thuộc vào phần mô tả phim và nội dung biến ref_var"""
    for s in ref_var:
        df[s] = pd.Series([0 for _ in range(len(df))])
    columns = ['genres', 'actor_1_name', 'actor_2_name',
               'actor_3_name', 'director_name', 'keywords']
    for categories in columns:
        for index, row in df.iterrows():
            if pd.isnull(row[categories]):
                continue
            for s in row[categories].split('|'):
                if s in ref_var:
                    df.at[index, s] = 1
    return df


def recommend(df, id_entry, n_neighbors):
    """Recommend a list of N films similar to the film selected by the user."""
    df_copy = df.copy(deep=True)
    list_genres = set()
    for s in df['genres'].str.split('|').values:
        list_genres = list_genres.union(set(s))
    # Tạo các biến bổ sung để kiểm tra sự giống nhau
    variables = entry_variables(df_copy, id_entry)
    variables += list(list_genres)
    df_new = add_variables(df_copy, variables)
    # xác định các láng giềng gần nhất sử dụng euclidean
    X = df_new[variables].values
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='euclidean').fit(X)
    x_test = df_new.iloc[id_entry][variables].values
    x_test = x_test.reshape(1, -1)
    distances, indices = nbrs.kneighbors(x_test)
    return indices[0][:]


def extract_parameters(df, list_films):
    """trích xuất một số biến của dataframe input và trả về danh sách này cho một lựa chọn gồm N phim. 
    Danh sách này được sắp xếp theo các tiêu chí được thiết lập trong hàm criteria_selection()'"""
    parameters_films = []
    max_users = -1
    max_popularity = -1
    for i, index in enumerate(list_films):
        parameters_films.append(list(df.iloc[index].reindex(['title', 'release_date',
                                                             'vote_average', 'popularity',
                                                             'vote_count', 'id'])))
        max_users = max(max_users, parameters_films[i][4])
        max_popularity = max(max_popularity, parameters_films[i][3])
    title_main = parameters_films[0][0]
    year_ref = parameters_films[0][1]
    parameters_films.sort(key=lambda x: criteria_selection(title_main, max_users, max_popularity,
                                                           year_ref, x[0], x[1], x[2], x[3], x[4]), reverse=True)
    return parameters_films


def sequel(title_1, title_2):
    """So sánh 2 tiêu đề phim có giống nhau không"""
    if fuzz.ratio(title_1, title_2) > 50 or fuzz.token_set_ratio(title_1, title_2) > 50:
        return True
    return False


def criteria_selection(title_main, max_users, max_popularity, year_ref, title, year, score, popularity, votes):
    """Chọn ra các bộ phim được đề xuất tốt nhất dựa trên số lượng vote, năm sản xuất, độ nổi tiếng, điểm imdb."""
    if pd.notnull(year_ref):
        factor_1 = gaussian_filter(year_ref, year, 20)
    else:
        factor_1 = 0.1
    sigma = max_users * 1.0
    if pd.notnull(votes):
        factor_2 = gaussian_filter(votes, max_users, sigma)
    else:
        factor_2 = 0.1
    sigma = max_popularity * 1.0
    if pd.notnull(popularity):
        factor_3 = gaussian_filter(popularity, max_popularity, sigma)
    else:
        factor_3 = 0.1
    if sequel(title_main, title):
        note = -1
    else:
        note = score ** 2 * factor_1 * factor_2 * factor_3
    return note


def find_similars(df, id_entry, n_neighbors, n_recommendations, verbose=False):
    """Return the indices of the n_recommendations movies most similar to the movies whose id is id_entry"""
    if verbose:
        print(90*'_' + '\n' + "QUERY: films similar to id={} -> '{}'".format(id_entry,
              df.iloc[id_entry]['title']))
    # find N similar movies
    list_films = recommend(df, id_entry, n_neighbors)
    # extract useful parameters from the recommended movies
    parameters_films = extract_parameters(df, list_films)
    # select n_recommendations films from this list
    film_selection = parameters_films[0:n_recommendations]
    film_idxs = []
    for i, s in enumerate(film_selection):
        if verbose:
            print("nº{:<2}     -> {:<30}".format(i + 1, s[0]))
        film_idxs.append(int(s[-1]))
    return film_idxs


if __name__ == '__main__':

    n_neighbors = 31
    n_recommendations = 15

    credits = "movies/tmdb_5000_credits.csv"
    movies = "movies/tmdb_5000_movies.csv"
    movies_imdb = "movies/movies_metadata.csv"
    df = clean_dataset(movies, credits, movies_imdb)
    sim_indices = find_similars(df, 2, n_neighbors, n_recommendations, verbose=True)
    print(sim_indices)


    train_positive_urls = "aclImdb/train/urls_pos.txt"
    train_negative_urls = "aclImdb/train/urls_neg.txt"
    test_positive_urls = "aclImdb/test/urls_pos.txt"
    test_negative_urls = "aclImdb/test/urls_neg.txt"
    urls = [train_positive_urls, train_negative_urls, test_positive_urls,test_negative_urls]
    list1 = df['imdb_id'].tolist()
    set1 = set(list1)
    list2 = []
    for j, file in enumerate(urls):
        f = open(file, "r")
        lines = f.readlines()
        for l in lines:
            l1 = l.split("/")[-2]
            if l1 not in list2:
                list2.append(l1)
        f.close()
    set2 = set(list2)
    print("Movies with review:", len(set1.intersection(set2)))  # 500+
