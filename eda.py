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
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pre_processing

pd.set_option('display.max_columns', None)


def freq_by_cuisine():

    df = pd.read_csv('concat_uncleaned_recipes.csv').dropna()


    df['Ingredients'] = df.apply(lambda row: ' '.join(pre_processing.clean_strings(row['Ingredients'])), axis=1)
    df_freq_mex = df[df['Cuisine']==2]
    df_freq_ital = df[df['Cuisine']==3]
    df_freq_fren = df[df['Cuisine']==5]
    df_freq_amer = df[df['Cuisine']==6]
    df_freq_brit = df[df['Cuisine']==7]
    df_freq_ch = df[df['Cuisine']==8]
    df_freq_ind = df[df['Cuisine']==9]
    df_freq_japan = df[df['Cuisine']==13]

    df_list = [df_freq_mex, df_freq_ital, df_freq_fren, df_freq_amer, df_freq_brit, df_freq_ch, df_freq_ind, df_freq_japan]


    for cuisine in df_list:
        #print(item.head)
        data = cuisine['Ingredients'].apply(lambda row: list(everygrams(row.split(' '),min_len = 2, max_len = 2)))
        #data = cuisine['Ingredients'].apply(lambda row: row.split(' '))
        flat_data = [item for sublist in data for item in sublist]
        fdist = FreqDist(flat_data)

        print(fdist.most_common(20))
        word_distro_plot(fdist)

def box_whisk_by_cuisine():
    df = pre_processing.create_base_df()

    #df['Ingredients'] = df.apply(lambda row: ' '.join(pre_processing.clean_strings(row['Ingredients'])), axis=1)
    df['Ingredients'] = df.apply(lambda row: pre_processing.clean_strings(row['Ingredients']), axis=1)
    #df['word_count'] = len(df.Ingredients)

    df['word_count'] = df.Ingredients.str.len()

    fig = plt.figure()
    plt.style.use('seaborn')
    sns.set_palette('colorblind')
    ax = sns.boxplot(x= 'name', y = 'word_count', data = df)
    ax.set_ylabel('Word Count')
    ax.set_xlabel('Cuisine')
    plt.show()


def word_distro_plot(fdist):
    fig = plt.figure()
    plt.style.use('seaborn')
    sns.set_palette('colorblind')
    fdist.plot(30)
    plt.show()

def count_plot():
    df = pre_processing.create_base_df()
    df['Ingredients'] = df.apply(lambda row: pre_processing.clean_strings(row['Ingredients']), axis=1)
    fig = plt.figure()
    plt.style.use('seaborn')
    sns.set_palette('colorblind')

    ax = sns.countplot(x='name', data=df)
    ax.set_ylabel('Recipe Count')
    ax.set_xlabel('Cuisine')
    plt.show()

#box_whisk_by_cuisine()


count_plot()


#freq_by_cuisine()
