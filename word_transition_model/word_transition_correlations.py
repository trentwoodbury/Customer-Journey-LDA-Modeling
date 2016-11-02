#This program is to be run after word_transition_model.py has run

from functools import partial
import multiprocessing
import numpy as np
import os
import pandas as pd
from scipy import sparse
from scipy import spatial


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
    lda_vec = np.empty((len(word_df),len(word_list)))
    dict_list = [{word[:-1]: corr for word, corr in zip(word_df.iloc[row][::2], word_df.iloc[row][1::2])}\
                 for row in range(len(word_df))]
    lda_vectors = pd.DataFrame(dict_list, columns = word_list).fillna(0).values
    for row_i in range(len(lda_vectors)):
        for col_i in range(len(lda_vectors[0])):
            try:
                lda_vectors[row_i, col_i] = float(lda_vectors[row_i, col_i])
            except:
                lda_vectors[row_i, col_i] = 0

    return lda_vectors

def get_distances(lda_vec):
    #INPUT: lda_vec, output of make_vectors function
    #OUTPUT: matrix of distances between rows of lda_vec
    distance_mat = np.empty((435, 3))
    row_count = 0
    for row_i in range(len(lda_vec)-1):
        for row_j in range(row_i+1, len(lda_vec)):
            distance_mat[row_count, 0] = int(row_i)
            distance_mat[row_count, 1] = int(row_j)
            try:
                distance_mat[row_count, 2] = spatial.distance.pdist([lda_vec[row_i], lda_vec[row_j]], 'euclidean')
            except:
                print 'row 1:', row_i, 'row 2:', row_j
            row_count += 1
    return distance_mat

def make_dist_mat(distances):
    #INPUT: distances, output of get_distances()
    #OUTPUT: properly formatted distance matrix
    distance_matrix = np.empty((30, 30))
    for row in distances:
        distance_matrix[row[0]][row[1]] = row[2]
        distance_matrix[row[1]][row[0]] = row[2]
    return distance_matrix



if __name__ == "__main__":

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
    lda_vec = make_vectors(word_list, word_df)
    distances = get_distances(lda_vec)

    #Make distance matrix
    dist_mat = make_dist_mat(distances)
    print dist_mat
