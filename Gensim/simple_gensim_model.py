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

def format_data():
    #This function builds a list lists: a list of documents of words
    #in this case a document is a user journey and a word is a single user
    #action

    #import data
    filepath = '../../data/Top_Traversals_demo-1daybehavior_20140401.csv'
    pre_df = pd.read_csv(filepath, header=1)

    #Create a numpy array of user journeys
    paths = np.array([ 'Path'])
    for row_ind in range(2, len(pre_df)):
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

    #words is a list of documents
    words = []
    for val in paths:
        for string in val:
            word_list = string.split()
            #treat journey.entry as stopword.
            words.append(string.split()[1:-1])

    return words



if __name__ == "__main__":
    print format_data()
