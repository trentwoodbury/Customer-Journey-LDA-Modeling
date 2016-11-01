#This program is to be run after word_transition_model.py has run

from functools import partial
from gensim import corpora, models
from matplotlib import font_manager
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable
import multiprocessing
import numpy as np
import os
import pandas as pd
from timeit import default_timer as timer

def paths_to_docs(path):
    #INPUT: path, output of data_to_paths() function
    #OUTPUT: words, a list of documents (list of lists of words)
    words = []
    for string in path[0][0]:
        word_list = string.split()
        words.append(string.split())
    return words

def doc_combine(words_list):
    #INPUT: list list of words (output of paths_to_docs() function)
    #OUTPUT: list of list of word transitions
    for doc_i, doc in enumerate(words_list):
        #Check to see if document contains more than 1 word
        if len(doc) > 1:
            #if doc contains 2+ words, iterate through all except last word
            for word_i in range(len(doc)-1):
                #convert each word to that word + the next word
                words_list[doc_i][word_i] = str(words_list[doc_i][word_i])+ ' ' + str(words_list[doc_i][word_i + 1])
    return words_list

def words_to_set(words):
    #INPUT: list of lists of words
    #OUTPUT: Set of unique words
    word_set = set()
    for doc in words:
        for word in doc:
            word_set.add(word)
    return word_set

def make_vectors(word_set, word_df):
    #INPUT: word_set, output of words_to_set
    #INPUT: word_df, dataframe of top 50 words for all 30 subjects
    #OUTPUT: numpy array where columns are each unique word in word_set and
    #rows
    lda_vec = np.empty((len(word_df),len(word_set)))
    words = 


def main():
    #Read in numpy array of
    path = np.load('path.npz')
    words = doc_combine(paths_to_docs(path['arr_0']))
    word_set = words_to_set(words)

    #Read in word correlation results
    word_df = pd.read_csv('transitions_df.csv')



if __name__ == "__main__":
    start_time = timer()
    main()
    print "Load time:", timer() - start_time
