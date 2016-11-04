import cPickle
from gensim import corpora
import numpy as np
import pandas as pd


def get_model(filepath):
    #INPUT: filepath to model's pickle file
    #OUTPUT: LDA Model
    model = cPickle.load(open(filepath, 'r'))
    return model

def format_input_string(input_string):
    #INPUT: string in format of original journey csv
    #OUTPUT: string formatted like "event->event event2->event2 etc."
    string_list = input_string.replace(' ', '.').replace('->', ' ').split()
    zip_list = zip(string_list, string_list[1:])
    formatted_list = [val[0] + '->' + val[1] for val in zip_list]
    return formatted_list

def predict_text(formatted_list, model):
    #INPUT: journey to predit in list of strings format.
    #The journey will have to be
    #formatted like ['event->event', 'event2->event2'] 
    #OUTPUT: model predictions for input
    text = [formatted_list]
    dictionary = corpora.Dictionary(text)
    corpus = [dictionary.doc2bow(phrase) for phrase in text]
    return model.get_document_topics(corpus[0], minimum_probability=.05)
