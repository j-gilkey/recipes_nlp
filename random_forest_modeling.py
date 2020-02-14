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
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score
import pre_processing



def fit_random_forest(df, max_gram):

    tfidf_X_train_lem, tfidf_X_test_lem, y_train_lem, y_test_lem = pre_processing.process_data(df, max_gram)
    #great tfidf train test split

    rf_classifier_lem = RandomForestClassifier(n_estimators=100, random_state=0)
    #instantiate random forest

    rf_classifier_lem.fit(tfidf_X_train_lem, y_train_lem)
    #fit that forest

    rf_test_preds_lem = rf_classifier_lem.predict(tfidf_X_test_lem)
    #create predictions

    rf_acc_score_lem = accuracy_score(y_test_lem, rf_test_preds_lem)
    rf_f1_score_lem = f1_score(y_test_lem, rf_test_preds_lem, average='macro')
    #get accuracy and F1

    class_rep = classification_report(y_test_lem, rf_test_preds_lem)

    print('Accuracy:')
    print(rf_acc_score_lem)
    print('F1 Score:')
    print(rf_f1_score_lem)
    #print(class_rep)


def refine_forest_model():
    #run random forest on different refinements
    df = pd.read_csv('concat_uncleaned_recipes.csv')
    df = df.dropna()

    df = pre_processing.filter_cuisines(df, [2, 3, 5, 6, 7, 8, 9, 13])
    fit_random_forest(df, 1)
    #filter down to only the cuisines we're interested in

    df = pre_processing.filter_cuisines(df, [2, 3, 5, 7, 8, 9, 13])
    fit_random_forest(df, 1)
    #remove american cuisine type

    df = pre_processing.filter_cuisines(df, [2, 5, 7, 8, 9, 13])
    fit_random_forest(df, 1)
    #remove french cuisine type


#refine_forest_model()
