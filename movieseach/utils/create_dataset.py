# https://www.kaggle.com/fabiendaniel/film-recommendation-engine

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
    """return missing value rather than an error upon indexing/key failure"""
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
    """Create the full dataset"""
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
    # convert the dictionary in a list to sort the keywords by frequency
    keyword_sorted = []
    for k, v in keyword_count.items():
        keyword_sorted.append([k, v])
    keyword_sorted.sort(key=lambda x: x[1], reverse=True)
    return keyword_sorted, keyword_count


def keywords_inventory(dataframe, colonne='keywords'):
    """Collect the keywords"""
    keywords_roots = dict()  # root -> keyword
    keywords_select = dict()  # root -> shortest_keyword
    category_keys = []
    for s in dataframe[colonne]:  # for each column
        if pd.isnull(s):
            continue
        for t in s.split('|'):  # for each keyword
            t = t.lower()
            root = PS.stem(t)  # get the root
            if root in keywords_roots:
                keywords_roots[root].add(t)
            else:
                keywords_roots[root] = {t}
    for s in keywords_roots.keys():
        if len(keywords_roots[s]) > 1:
            min_length = 1000  # length of the shortest derived words given a root
            for k in keywords_roots[s]:  # for each derived word
                if len(k) < min_length:
                    clef = k  # shortest derived word
                    min_length = len(k)
            category_keys.append(clef)
            keywords_select[s] = clef
        else:
            category_keys.append(list(keywords_roots[s])[0])
            keywords_select[s] = list(keywords_roots[s])[0]
    return category_keys, keywords_roots, keywords_select


def replacement_df_keywords(df, replacement, roots=False):
    """Replace all keywords with their main form and eventually stem them"""
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
    """deletion of keywords with low frequencies (<=thr)"""
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
    """Fill missing years of movies given their cast"""
    col = ['director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name']
    usual_year = [0 for _ in range(4)]
    var = [0 for _ in range(4)]
    # I get the mean years of activity for the actors and director
    for i in range(4):
        usual_year[i] = df.groupby(col[i])['release_date'].mean()
    # I create a dictionary collecting this info
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
    # identification of missing title years
    missing_year_info = df[df['release_date'].isnull()]
    # filling of missing values
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
            #if count_replaced < 10:
                #print("{:<45} -> {:<20}".format(df.loc[index]['title'], int(sum_year)))
    return


def variable_linreg_imputation(df, col_to_predict, ref_col):
    """impute the missing value from a linear fit of the data"""
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
    # load the dataset
    credits = load_tmdb_credits(credits_path)
    movies = load_tmdb_movies(movies_path)
    movies_imdb = pd.read_csv(imdb_path, encoding="utf-8")[["id", "imdb_id"]].astype({'id': 'int'})
    # create full dataset
    df_initial = convert_to_original_format(movies, credits)
    df_initial = df_initial.join(movies_imdb.set_index('id'), on='id')
    print("Movies:", df_initial.shape[0])
    print("Features:", df_initial.shape[1])

    # set of all keywords
    set_keywords = set()
    for list_keywords in df_initial['keywords'].str.split('|').values:
        if isinstance(list_keywords, float):
            continue  # exclude NaN values
        set_keywords = set_keywords.union(list_keywords)
    # remove null chain entry
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
    # creates keywords inventory
    keywords, keywords_roots, keywords_select = keywords_inventory(df_duplicate_cleaned, colonne='keywords')
    # replacement of the keywords by the main keyword
    df_keywords_cleaned = replacement_df_keywords(df_duplicate_cleaned, keywords_select, roots=True)
    # count of the keywords occurrences
    keywords.remove('')
    keyword_occurrences, keywords_count = count_word(df_keywords_cleaned, 'keywords', keywords)
    print("Cleaned total keywords:", len(keyword_occurrences))

    key_count = dict()
    for s in keyword_occurrences:
        key_count[s[0]] = s[1]
    # creation of a dictionary to replace keywords by higher frequency keywords
    replacement_mot = dict()

    # substitute keywords that appear less than 5 times with the synonym that appears more times
    for index, [mot, nb_apparitions] in enumerate(keyword_occurrences):
        if nb_apparitions > 5:
            continue  # only the keywords that appear less than 5 times
        lemma = get_synonyms(mot)
        if len(lemma) == 0:
            continue     # case of the plurals
        list_mots = [(s, key_count[s]) for s in lemma if test_keyword(s, key_count, key_count[mot])]
        list_mots.sort(key=lambda x: (x[1], x[0]), reverse=True)
        if len(list_mots) <= 1:
            continue       # no replacement
        if mot == list_mots[0][0]:
            continue    # replacement by himself
        #print('{:<12} -> {:<12} (init: {})'.format(mot, list_mots[0][0], list_mots))
        replacement_mot[mot] = list_mots[0][0]

    print('The replacement concerned {} keywords.'.format(len(replacement_mot)))

    # replacement of keyword varieties by the main keyword
    df_keywords_synonyms = replacement_df_keywords(df_keywords_cleaned, replacement_mot, roots=False)
    keywords, keywords_roots, keywords_select = keywords_inventory(df_keywords_synonyms, colonne='keywords')

    # new count of keyword occurrences
    keywords.remove('')
    new_keyword_occurrences, keywords_count = count_word(df_keywords_synonyms, 'keywords', keywords)

    # Creation of a dataframe where keywords of low frequencies are suppressed
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
        #print('{:<50} -> {:<30}'.format(row['title'], str(new_keyword)))
        if new_keyword:
            df_filling.at[index, 'keywords'] = '|'.join(new_keyword)

    # fill revenue value given vote_count
    variable_linreg_imputation(df_filling, 'revenue', 'vote_count')
    df = df_filling.copy(deep=True)

    # final dataset
    df.reset_index(inplace=True, drop=True)
    return df


# RECOMMENDATION SYSTEM-------------------------------------------------------------------------------------------------

def entry_variables(df, id_entry):
    """Function collecting some variables content given an id_entry"""
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
    """Function adding variables values to the dataframe,
    ref_var are the variable values we take as reference"""
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
    # Create additional variables to check the similarity
    variables = entry_variables(df_copy, id_entry)
    variables += list(list_genres)
    df_new = add_variables(df_copy, variables)
    # determination of the closest neighbors: the distance is calculated / new variables
    X = df_new[variables].values
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='euclidean').fit(X)
    x_test = df_new.iloc[id_entry][variables].values
    x_test = x_test.reshape(1, -1)
    distances, indices = nbrs.kneighbors(x_test)
    return indices[0][:]


def extract_parameters(df, list_films):
    """Given N movies, order them according to 'criteria_selection'"""
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
    """compares the 2 titles passed in input and defines if these titles are similar or not."""
    if fuzz.ratio(title_1, title_2) > 50 or fuzz.token_set_ratio(title_1, title_2) > 50:
        return True
    return False


def criteria_selection(title_main, max_users, max_popularity, year_ref, title, year, score, popularity, votes):
    """gives a mark to a film depending on its IMDB score, the title year
     and the number of users who have voted for this film."""
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
    urls = [train_positive_urls, train_negative_urls, test_positive_urls]
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
