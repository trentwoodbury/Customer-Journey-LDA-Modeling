#This program is to be run after word_transition_model.py has run

from functools import partial
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pandas as pd
from scipy import sparse
from scipy import spatial
from wordcloud import WordCloud

def doc_combine(words_list):
    #INPUT: list list of words
    #OUTPUT: list of list of word transitions
    result_list = []
    for  doc in words_list:
        #Check to see if document contains more than 1 word
        if len(doc) > 1:
            zip_list = zip(doc, doc[1:])
            result_list.append([val[0] + ' ' + val[1] for val in zip_list])
    return result_list

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
    distances = np.empty((435, 3))
    row_count = 0
    for row_i in range(len(lda_vec)-1):
        for row_j in range(row_i+1, len(lda_vec)):
            distances[row_count, 0] = int(row_i)
            distances[row_count, 1] = int(row_j)
            try:
                distances[row_count, 2] = spatial.distance.pdist([lda_vec[row_i], lda_vec[row_j]], 'euclidean')
            except:
                print 'row 1:', row_i, 'row 2:', row_j
            row_count += 1
    return distances

def make_dist_mat(distances):
    #INPUT: distances, output of get_distances()
    #OUTPUT: properly formatted distance matrix
    distance_matrix = np.empty((30, 30))
    for row in distances:
        distance_matrix[row[0]][row[1]] = row[2]
    return distance_matrix

def heat_map(dist_mat):
    #INPUT: distance matrix
    #OUTPUT: heat map of distances

    # #Normalize data to size that can be read by matplotlib
    for row_i, row in enumerate(dist_mat):
        for col_i, cell in enumerate(row):
            dist_mat[row_i][col_i] = dist_mat[row_i][col_i]

    #Main Plot Body
    fig = plt.figure(figsize = (16,12))
    ax = fig.add_subplot(111)
    ax.pcolor(dist_mat, cmap='Blues')
    ax.grid(True)

    #Y axis
    ax.set_ylabel('Theme Number')
    ax.set_yticks(np.arange(30))
    ax.set_yticklabels(np.arange(30))

    #X axis
    ax.set_xlabel('Theme Number')
    ax.set_xticks(np.arange(30))
    ax.set_xticklabels(np.arange(30))


    #Title
    ax.set_title('Distance Matrix', fontsize = 18)

    plt.savefig('visualizations/distance_matrix.png')


def word_cloud(word_df_row, theme_num):
    #INPUT: row of word_df dataframe
    #OUTPUT: Word Cloud

    #Make repeating string to represent correlations
    repeated_string = ''
    for cell in range(50):
        if type(word_df_row[(2*cell)+1]) == np.float64:
            repeated_string += int(100*word_df_row[(2*cell)+1]) * \
            (' ' + word_df_row[2*cell].replace('.', '_').replace(' ', '___'))
    wordcloud = WordCloud().generate(repeated_string)

    #Plot Word Cloud
    fig = plt.figure(figsize = (20, 20))
    ax = fig.add_subplot(111)
    ax.imshow(wordcloud)
    ax.set_title('Word Cloud {}'.format(theme_num), fontsize = 20)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig('visualizations/word_cloud_{}.png'.format(theme_num))


if __name__ == "__main__":
    #Read in numpy array of
    path = np.load('data/path.npz')['arr_0']
    words = doc_combine(path)
    word_set = words_to_set(words)

    #Read in word correlation results
    word_df = pd.read_csv('data/transitions_df.csv')
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
    heat_map(dist_mat)

    # #Make word clouds
    # for i in range(len(word_df)):
    #     word_cloud(word_df.iloc[i], i+1)
