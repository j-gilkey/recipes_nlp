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
import pre_processing



def fit_random_forest():

    df = pd.read_csv('concat_uncleaned_recipes.csv')
    df = df.dropna()

    df = pre_processing.filter_cuisines(df, [2, 3, 5, 6, 7, 8, 9, 13])

    tfidf_X_train_lem, tfidf_X_test_lem, y_train_lem, y_test_lem = pre_processing.process_data(df)

    rf_classifier_lem = RandomForestClassifier(n_estimators=100, random_state=0)

    rf_classifier_lem.fit(tfidf_X_train_lem, y_train_lem)

    rf_test_preds_lem = rf_classifier_lem.predict(tfidf_X_test_lem)

    rf_acc_score_lem = accuracy_score(y_test_lem, rf_test_preds_lem)
    rf_f1_score_lem = f1_score(y_test_lem, rf_test_preds_lem, average='macro')

    print(rf_acc_score_lem)
    print(rf_f1_score_lem)


fit_random_forest()
