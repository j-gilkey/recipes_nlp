import numpy as np
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


def do_lda():
    df = pd.read_csv('concat_uncleaned_recipes.csv').dropna()
    df['Ingredients'] = df.apply(lambda row: ' '.join(pre_processing.clean_strings(row['Ingredients'])), axis=1)

    #text_data = list(df['Ingredients'])

    lda_tf = LatentDirichletAllocation(n_components=8, random_state=0)

    vec = CountVectorizer(stop_words=None)
    X = vec.fit_transform(df['Ingredients'])

    lda_tf.fit(X)

    pyLDAvis.sklearn.prepare(lda_tf, X, vec)





    # dictionary = corpora.Dictionary(text_data)
    # corpus = [dictionary.doc2bow(text) for text in text_data]
    # ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 8, id2word=dictionary, passes=15)
    #
    # pyLDAvis.enable_notebook()
    # lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
    # pyLDAvis.display(lda_display)

    # topics = ldamodel.print_topics(num_words=4)
    # for topic in topics:
    #     print(topic)


do_lda()

def create_linkage():

    df = pd.read_csv('concat_uncleaned_recipes.csv').dropna()
    df['Ingredients'] = df.apply(lambda row: ' '.join(pre_processing.clean_strings(row['Ingredients'])), axis=1)

    vec = CountVectorizer(stop_words=None)

    X = vec.fit_transform(df['Ingredients'])
    df_features = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())

    Z = linkage(df_features, 'single')

    plot_dendrogram(Z)

def plot_dendrogram(Z):
    plt.title('Hierarchical Clustering Dendrogram (truncated)')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        truncate_mode='lastp',  # show only the last p merged clusters
        p=30,  # show only the last p merged clusters
        show_leaf_counts=False,  # otherwise numbers in brackets are counts
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,  # to get a distribution impression in truncated branches
    )
    plt.show()

#create_linkage()
