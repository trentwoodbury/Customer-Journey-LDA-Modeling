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


def data_to_path(filepath, shortener):
    #INPUT: filepath, path to data
    #INPUT: shortener, quantity to cut data (to run faster)

    #import data
    pre_df = pd.read_csv(filepath, header=1)

    #Create a numpy array of user journeys
    paths = np.array([ 'Path'])
    for row_ind in range(2, len(pre_df)-shortener):
        #extract path from row
        path = list(str(pre_df.iloc[row_ind, :]).split())[1]
        #add path to paths numpy array
        paths = np.vstack((paths, path))

    #remove header (paths[0] is the header) and separate words
    paths = paths[1:]

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
    dictionary = corpora.Dictionary(unique_words)
    corpus = [dictionary.doc2bow(text) for text in words]
    return corpus

def gen_lda_model(corpus, topic_qty = 10, word_qty=4):
    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=10, id2word = dictionary, passes=20)
    return ldamodel.print_topics(num_topics=topic_qty, num_words = word_qty)

def split_nums_names(topics_list):
    num_vals = []
    name_vals = []
    for idx, topic in enumerate(topics_list):
        topic_split = topic[1].split('*')
        num_vals.append([topic_split[0]])
        name_vals.append([])
        for word_num in topic_split[1:]:
            word_num =  word_num.split('+')
            if len(word_num) > 1:
                num_vals[idx].append(word_num[1])
            name_vals[idx].append(word_num[0])
    return num_vals, name_vals


def pandas_visualization(num_vals, name_vals, word_qty):
    ten_themes = pd.DataFrame()
    for i in range(10):
        new_df = pd.Series([name_vals[i][0], num_vals[i][0] for i in range(word_qty)])

        ten_themes = ten_themes.append(new_df, ignore_index = True)

    ten_themes.columns = ['Word {0}', 'Word {1} Theme'.format(i, i) for i in range(word_qty)]

    return ten_themes


def main():
    filepath = '../../data/Top_Traversals_demo-1daybehavior_20140401.csv'
    path = data_to_path(filepath, 400000)
    words = paths_to_docs(path)
    corpus = words_to_corpus(words)
    lda_model = gen_lda_model(corpus)
    num_vals, name_vals = split_nums_names(lda_model)
    print pandas_visualization(num_vals, name_vals, 4)

    
if __name__ == "__main__":
    main()
