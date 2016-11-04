import cPickle
from functools import partial
from gensim import corpora, models
from matplotlib import font_manager
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable
import multiprocessing
import numpy as np
import os
import pandas as pd
import sys
from timeit import default_timer as timer


def load_data(filepath):
    #INPUT: filepath to csv file
    #OUTPUT: returns dataframe of filepath's csv file

    pre_df = pd.read_csv(filepath, header = 0)
    return pre_df
    # pre_df should have length of 438,982

def data_to_path(pre_df, qty):
    #INPUT: pre_df, dataframe
    #INPUT: quantity of rows to read in, 438982 for total dataframe.
    #OUTPUT: paths, list os list of words

    paths = []
    for i in range(0, qty):
         paths.append(pre_df['Path'][i].replace(' ', '.').replace('->', ' ').split())
    return paths

def doc_combine(words_list):
    #INPUT: list of  list of words (output of data_to_path() function)
    #OUTPUT: list of list of word transitions

    result_list = []
    for  doc in words_list:
        #Check to see if document contains more than 1 word
        if len(doc) > 1:
            zip_list = zip(doc, doc[1:])
            result_list.append([val[0] + '->' + val[1] for val in zip_list])
    return result_list

def words_to_corpus(words):
    #INPUT: list of lists of words (output of doc_combine() function)
    #OUTPUT: Corpus of words matched with frequency and dictionary

    dictionary = corpora.Dictionary(words)
    corpus = [dictionary.doc2bow(text) for text in words]
    return corpus, dictionary

def gen_lda_model(corpus, dictionary, topic_qty = 10, word_qty=50):
    #INPUT: corpus and dictionary.
    #INPUT: topic_qty: how many topics to cluster
    #INPUT: word_qty: how many words
    #OUTPUT: lda model in gensim print format

    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=topic_qty, id2word = dictionary, passes=20)
    cPickle.dump(ldamodel, open('data/ldamodel.pkl', 'w'))
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
    #OUPUT: Dataframe of results (for readability)
    n_themes = []
    for i in range(topic_qty):
        #current series is always the current row
        current_series = []
        names = [name_vals[i][j] for j in range(word_qty)]
        nums = [num_vals[i][j] for j in range(word_qty)]
        #alternatingly append name and value
        for i in range(word_qty):
            current_series.append(names[i])
            current_series.append(nums[i])
        n_themes.append(current_series)

    return pd.DataFrame(n_themes)


def graph_term_import(df_row, theme_num, word_qty = 50):
    #INPUT: df_row, a row from the output of pandas_visualization
    #INPUT: theme_num, the theme number
    #OUPUT: Horizontal Bar Chart of term import in theme

    #x is labels, y is values
    x = [df_row[i*2+1] for i in range(1, word_qty)]
    y = [float(df_row[i*2]) if str(df_row[i*2]) > '0' else 0 for i in range(1, word_qty)]
    x_pos = np.arange(word_qty-1)
    ticks_font = font_manager.FontProperties(style='normal',
    size=7, weight='normal', stretch='normal')

    fig = plt.figure(figsize = (16,12))
    ax = fig.add_subplot(111)
    ax.barh(x_pos, y, align='center', alpha=0.4)
    ax.set_yticks(x_pos)
    ax.set_yticklabels(x)
    for label in ax.get_yticklabels():
        label.set_fontproperties(ticks_font)

    ax.set_xlabel('Word Probability')
    ax.set_ylabel('Terms')
    ax.set_title('Theme {}'.format(theme_num))
    make_axes_area_auto_adjustable(ax)
    plt.savefig('visualizations/visualization{}.png'.format(theme_num))


def main():
    #this conditional allows us to skip the computationally intensive
    #parts of the code for running the code multiple times
    if not os.path.exists('data/path.npz'):
        filepath = '../../data/Top_Traversals_demo-1daybehavior_20140401.csv'
        pre_df = load_data(filepath)
        path = data_to_path(pre_df, 100)
        np.savez_compressed('data/path.npz', path)

    #These lines should only be included if you have path.npz and not
    #transitions_df.csv
    # else:
    #     path = np.load('data/path.npz')['arr_0']

    #this conditional allows us to skip more of the computationally intensive
    #parts of the code for running the code multiple times
    if not os.path.exists('data/transitions_df.csv'):
        words = doc_combine(path)
        corpus, dictionary = words_to_corpus(words)
        lda_model = gen_lda_model(corpus, dictionary, topic_qty = 30)
        num_vals, name_vals = split_nums_names(lda_model)
        print "Terms of Importance by Topic (each row is a topic) \n"
        word_df = pandas_visualization(num_vals, name_vals, word_qty = 50, topic_qty=30)
        print word_df
        word_df.to_csv('data/transitions_df.csv')
        # for i in range(len(word_df)):
        #     graph_term_import(word_df.iloc[i, :], i)

    else:
        word_df = pd.read_csv('data/transitions_df.csv')
        print "Terms of Importance by Topic (each row is a topic) \n"
        print word_df
        for i in range(len(word_df)):
            graph_term_import(word_df.iloc[i, :], i)



if __name__ == "__main__":
    np.set_printoptions(threshold=2000)
    start_time = timer()
    main()
    print "Load time:", timer() - start_time
