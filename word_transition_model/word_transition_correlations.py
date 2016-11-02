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
from scipy import sparse
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

def make_vectors(word_list, word_df):
    #INPUT: word_set, output of words_to_set
    #INPUT: word_df, dataframe of top 50 words for all 30 subjects
    #OUTPUT: numpy array where columns are each unique word in word_set and
    #rows
    lda_vec = np.empty((len(word_df),len(word_set)))
    for word_i, word in enumerate(word_list):
        for row_i, row in enumerate(word_df):
            for cell_i, cell in enumerate(row):
                if cell == word:
                    lda_vec[row_i, word_i] = word_df.iloc[row_i, cell_i+1]
    return lda_vec





def main():
    #Read in numpy array of
    path = np.load('path.npz')
    words = doc_combine(paths_to_docs(path['arr_0']))
    word_set = words_to_set(words)

    #Read in word correlation results
    word_df = pd.read_csv('transitions_df.csv')
    #remove column of row numbers.
    #Whether this line is required depends on the nature of the dataframe.
    #Make sure to look at the dataframe before running this line of code.
    word_df = word_df.iloc[:, 1:]

    #Make subject vectors
    word_list = list(word_set)
    subject_array = make_vectors(word_list, word_df)




if __name__ == "__main__":
    start_time = timer()
    main()
    print "Load time:", timer() - start_time
