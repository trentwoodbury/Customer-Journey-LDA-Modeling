from collections import OrderedDict
import csv
from gensim import corpora, models
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import numpy as np
import pandas as pd
from pprint import pprint
import pymc as pm
from sklearn.feature_extraction.text import TfidfVectorizer


def data_to_path(filepath, qty):
    #INPUT: filepath, path to data
    #INPUT: shortener, quantity to cut data (to run faster)

    #import data
    pre_df = pd.read_csv(filepath, header=1)

    #Create a numpy array of user journeys
    paths = np.array([ 'Path'])
    for i in range(2, qty):
        #select random row without replacement
        #range starts at row 3 to not include headers
        row_ind = np.random.choice(range(3, len(pre_df)), replace = False)
        #extract path from row
        path = list(str(pre_df.iloc[row_ind, :]).split())[1]
        #add path to paths numpy array
        paths = np.vstack((paths, path))

    for journey in range(len(paths)):
        paths[journey] = paths[journey][0].replace('->', ' ')
    #transpose data so that each journey is no longer a new column
    #after this transpose, each journey is a row
    paths = np.transpose(paths)
    return paths

def paths_to_docs(path):
    #INPUT: path, output of data_to_paths() function
    #OUTPUT: words, a list of documents (list of lists of words)
    words = []
    for val in path:
        for string in val:
            word_list = string.split()
            #treat journey.entry as stopword.
            words.append(string.split()[1:-1])
    return words

def words_to_corpus(words):
    #INPUT: list of lists of words
    #OUTPUT: Corpus of words matched with frequency and dictionary
    dictionary = corpora.Dictionary(words)
    corpus = [dictionary.doc2bow(text) for text in words]
    return corpus, dictionary

def gen_lda_model(corpus, dictionary, topic_qty = 10, word_qty=4):
    #INPUT: corpus and dictionary.
    #INPUT: topic_qty: how many topics to cluster
    #INPUT: word_qty: how many words
    #OUPUT: lda model in gensim print format

    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=10, id2word = dictionary, passes=20)
    return ldamodel.print_topics(num_topics=topic_qty, num_words = word_qty)

def split_nums_names(topics_list):
    #INPUT: LDA model in gensim printed format
    #OUTPUT: num_vals, list of percents of topic explained by each term
    #OUTPUT: name_vals, list of terms
    num_vals = []
    name_vals = []
    for idx, topic in enumerate(topics_list):
        # * sign splits number and term, hence we split on it
        topic_split = topic[1].split('*')
        # There is always 1 num val before the names
        #we are simultaneously instantiating this num_vals list
        num_vals.append([topic_split[0]])
        #instantiate name_vals list
        name_vals.append([])
        #for loop to add values to num_vals and name_vals lists
        for word_num in topic_split[1:]:
            word_num =  word_num.split('+')
            #we test if word_num > 1 to make sure we have a pair of word and number (we do not always)
            if len(word_num) > 1:
                num_vals[idx].append(word_num[1])
            name_vals[idx].append(word_num[0])
    return num_vals, name_vals


def pandas_visualization(num_vals, name_vals, word_qty= 4, topic_qty= 10):
    #INPUT: ouput of split_num_names
    #INPUT: word_qty: quantity of words (columns) to show in dataframe
    #INPUT: topic_qty: number of topics (rows) to show in dataframe
    n_themes = []
    for i in range(topic_qty):
        #current series is always the current row
        current_series = []
        names = [name_vals[i][0] for i in range(word_qty)]
        nums = [num_vals[i][0] for i in range(word_qty)]
        #alternatingly append name and value
        for i in range(word_qty):
            current_series.append(names[i])
            current_series.append(nums[i])
        n_themes.append(current_series)

    return pd.DataFrame(n_themes)

def main():
    filepath = '../../data/Top_Traversals_demo-1daybehavior_20140401.csv'
    path = data_to_path(filepath, 10000)
    words = paths_to_docs(path)
    corpus, dictionary = words_to_corpus(words)
    lda_model = gen_lda_model(corpus, dictionary)
    num_vals, name_vals = split_nums_names(lda_model)
    print pandas_visualization(num_vals, name_vals)


if __name__ == "__main__":
    main()
