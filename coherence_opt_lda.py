import gensim
import warnings
import numpy as np
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, ward
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import pre_processing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.sklearn
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def calculate_coherence( w2v_model, term_rankings ):
    overall_coherence = 0.0
    for topic_index in range(len(term_rankings)):
        # check each pair of terms
        pair_scores = []
        for pair in combinations( term_rankings[topic_index], 2 ):
            #print(pair)
            pair_scores.append( w2v_model.similarity(pair[0], pair[1]) )
        # get the mean for all pairs in this topic
        topic_score = sum(pair_scores) / len(pair_scores)
        overall_coherence += topic_score
    # get the mean score across all topics
    return overall_coherence / len(term_rankings)

def get_descriptor( all_terms, H, topic_index, top ):
    # reverse sort the values to sort the indices
    top_indices = np.argsort( H[topic_index,:] )[::-1]
    #print(top_indices)
    # now get the terms corresponding to the top-ranked indices
    top_terms = []
    for term_index in top_indices[0:top]:
        #print(term_index)
        top_terms.append( all_terms[term_index] )
        #top_terms.append( all_terms.loc[term_index,:] )
    return top_terms

def init_lda_models(df, X, max_clusters):

    topic_models = []

    for k in range(1,max_clusters):
        print("Applying LDA for k=%d ..." % k )
        model = LatentDirichletAllocation(n_components=k, random_state=0)
        W = model.fit(X)
        H = model.components_
        # store for later
        topic_models.append( (k,W,H) )

    return topic_models

def plot_coherence(k_values, coherences):
    fig = plt.figure()
    plt.style.use('seaborn')
    sns.set_palette('colorblind')
    plt.xlabel("Number of Topics")
    plt.ylabel("Mean Coherence")
    # add the points
    plt.plot( k_values, coherences, linestyle='-')
    plt.show()


def assess_coherence():

    df = pd.read_csv('concat_uncleaned_recipes.csv').dropna()
    df = pre_processing.filter_cuisines(df, [2, 3, 5, 7, 8, 9, 13])

    df['Ingredients_string'] = df.apply(lambda row: ' '.join(pre_processing.clean_strings(row['Ingredients'])), axis=1)
    df['Ingredients_list'] = df.apply(lambda row: pre_processing.clean_strings(row['Ingredients']), axis=1)

    secondary_stop_words = ['ingredient', 'list', 'add', 'finely', 'available']

    vec = CountVectorizer(stop_words=secondary_stop_words)
    X = vec.fit_transform(df['Ingredients_string'])
    w2v_model = gensim.models.Word2Vec(df['Ingredients_list'], size=500, min_count=20, sg=1)
    terms = vec.get_feature_names()


    topic_models = init_lda_models(df, X, 15)

    k_values = []
    coherences = []

    for (k,W,H) in topic_models:
        term_rankings = []
        for topic_index in range(k):
            term_rankings.append( get_descriptor( terms, H, topic_index, 10 ) )
        # Now calculate the coherence based on our Word2vec model
        k_values.append( k )
        coherences.append( calculate_coherence( w2v_model, term_rankings ) )
        print("K=%02d: Coherence=%.4f" % ( k, coherences[-1] ) )

    plot_coherence(k_values, coherences)

assess_coherence()
