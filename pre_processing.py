import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk import everygrams
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def create_base_df():
    df = pd.read_csv('concat_uncleaned_recipes.csv').dropna()
    df = filter_cuisines(df, [2,3,5,6,7,8,9,13])

    cuisine_map = {2: 'Mexican', 3 : 'Italian', 5 : 'French', 6 : 'American', 7 : 'British', 8 : 'Chinese', 9 : 'Indian', 13 : 'Japanese'}
    #cuisine_map = {'2': 'Mexican', '3' : 'Italian', '5' : 'French', '6' : 'American', '7' : 'British', '8' : 'Chinese', '9' : 'Indian', '13' : 'Japanese', }
    df['name'] = df['Cuisine'].replace(cuisine_map)

    return df

def clean_strings(string):
    #tokenize
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    tokenized_string = tokenizer.tokenize(string)

    #lower
    lowered_string = [word.lower() for word in tokenized_string]

    #lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_string = [lemmatizer.lemmatize(word) for word in lowered_string]

    #filter stop words
    stop_words=set(stopwords.words("english"))
    filtered_string = [word for word in lemmatized_string if word not in stop_words]

    #remove custom stop words
    custom_stop_words = ['inch', 'ounce', 'pound', 'teaspoon', 'tablespoon', 'cup', 'small', 'medium', 'large', 'g', 'oz', 'lb', 'gram', 'cm', 'tbsp', 'tsp', 'ml', 'fl', 'chopped']
    cleaned_string = [word for word in filtered_string if word not in custom_stop_words]

    return cleaned_string

def create_basic_doc_term_matrix(df):

    df['Ingredients'] = df.apply(lambda row: ' '.join(clean_strings(row['Ingredients'])), axis=1)

    vec = CountVectorizer(stop_words=None)
    X = vec.fit_transform(df['Ingredients'])

    df_features = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())

    return df_features

def create_tf_idf(df):

    df['Ingredients'] = df.apply(lambda row: ' '.join(clean_strings(row['Ingredients'])), axis=1)
    #clean recipes and then join them back into one string

    tf=TfidfVectorizer()
    X = tf.fit_transform(df['Ingredients'])
    #init and fit tfidf

    idf_df = pd.DataFrame(X.toarray().transpose(), index = tf.get_feature_names())
    print(idf_df.transpose())

#create_tf_idf(df)

def process_data(df):
    df['Ingredients'] = df.apply(lambda row: ' '.join(clean_strings(row['Ingredients'])), axis=1)

    X_train_lem, X_test_lem, y_train_lem, y_test_lem = train_test_split(df['Ingredients'], df['Cuisine'], test_size=0.20, random_state=1)

    tfidf=TfidfVectorizer()

    tfidf_X_train_lem = tfidf.fit_transform(X_train_lem)
    tfidf_X_test_lem = tfidf.transform(X_test_lem)

    return tfidf_X_train_lem, tfidf_X_test_lem, y_train_lem, y_test_lem

def filter_cuisines(df, cuisines_to_filter):

    df = df[df['Cuisine'].isin(cuisines_to_filter)]

    return df
