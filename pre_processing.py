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

df = pd.read_csv('overall_bbc.csv')
df = df.dropna()

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

    df['recipe'] = df.apply(lambda row: ' '.join(clean_strings(row['recipe'])), axis=1)

    vec = CountVectorizer(stop_words=None)
    X = vec.fit_transform(df['recipe'])

    df_features = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())

    return df_features

def create_tf_idf(df):

    df['recipe'] = df.apply(lambda row: ' '.join(clean_strings(row['recipe'])), axis=1)

    tf=TfidfVectorizer()
    X = tf.fit_transform(df['recipe'])

    idf_df = pd.DataFrame(X.toarray().transpose(), index = tf.get_feature_names())
    print(idf_df.transpose())

#create_tf_idf(df)

def process_data(df):
    df['recipe'] = df.apply(lambda row: ' '.join(clean_strings(row['recipe'])), axis=1)

    X_train_lem, X_test_lem, y_train_lem, y_test_lem = train_test_split(df['recipe'], df['type'], test_size=0.20, random_state=1)

    tfidf=TfidfVectorizer()

    tfidf_X_train_lem = tfidf.fit_transform(X_train_lem)
    tfidf_X_test_lem = tfidf.transform(X_test_lem)

    return tfidf_X_train_lem, tfidf_X_test_lem, y_train_lem, y_test_lem

def fit_random_forest(df):

    tfidf_X_train_lem, tfidf_X_test_lem, y_train_lem, y_test_lem = process_data(df)

    rf_classifier_lem = RandomForestClassifier(n_estimators=100, random_state=0)

    rf_classifier_lem.fit(tfidf_X_train_lem, y_train_lem)

    rf_test_preds_lem = rf_classifier_lem.predict(tfidf_X_test_lem)

    rf_acc_score_lem = accuracy_score(y_test_lem, rf_test_preds_lem)
    rf_f1_score_lem = f1_score(y_test_lem, rf_test_preds_lem, average='macro')

    print(rf_acc_score_lem)
    print(rf_f1_score_lem)

#fit_random_forest(df)

def freq_by_cuisine(df):
    df['recipe'] = df.apply(lambda row: ' '.join(clean_strings(row['recipe'])), axis=1)
    df_freq_mex = df[df['type']==2]
    df_freq_ital = df[df['type']==3]
    df_freq_afri = df[df['type']==4]
    df_freq_fren = df[df['type']==5]
    df_freq_amer = df[df['type']==6]
    df_freq_brit = df[df['type']==7]
    df_freq_ch = df[df['type']==8]
    df_freq_ind = df[df['type']==9]
    df_freq_irish = df[df['type']==10]
    df_freq_nord = df[df['type']==11]
    df_freq_pak = df[df['type']==12]
    df_freq_japan = df[df['type']==13]

    df_list = [df_freq_mex, df_freq_ital, df_freq_afri, df_freq_fren, df_freq_amer, df_freq_brit, df_freq_ch, df_freq_ind, df_freq_irish, df_freq_nord, df_freq_pak, df_freq_japan]

    # for item in df_list:
    #     #print(item.head)
    #     data = item['recipe']
    #     total_vocab_sat = set()
    #     for recipe in data:
    #         #print(recipe)
    #         word_list = recipe.split(' ')
    #         for word in word_list:
    #             #print(word)
    #             total_vocab_sat.update({word})
    #
    #     print(len(total_vocab_sat))


    for cuisine in df_list:
        #print(item.head)
        data = cuisine['recipe'].apply(lambda row: list(everygrams(row.split(' '),min_len = 4, max_len = 4)))
        flat_data = [item for sublist in data for item in sublist]
        data_freq = FreqDist(flat_data)

        print(data_freq.most_common(20))


#freq_by_cuisine(df)


# tfidf_X_train_lem, tfidf_X_test_lem = process_data(df)
#
# non_zero_cols = tfidf_X_train_lem.nnz / float(tfidf_X_train_lem.shape[0])
#
# percent_sparse = 1 - (non_zero_cols / float(tfidf_X_train_lem.shape[1]))
# print(percent_sparse)
# print(non_zero_cols)
#
# print(tfidf_X_train_lem)
