import cPickle
from gensim import corpora
import numpy as np
import pandas as pd


def load_data(filepath):
    #INPUT: filepath to csv file
    #OUTPUT: returns dataframe of filepath's csv file

    pre_df = pd.read_csv(filepath, header = 0)
    return pre_df
    # pre_df should have length of 438,982

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

def make_results_df(string_list, model):
    #INPUT: list of strings to be converted and predicts
    #OUTPUT: dataframe of strings and predictions
    df = pd.DataFrame(columns=['Journey', 'Prediction'])
    for string in string_list:
        formatted_list = format_input_string(string)
        prediction = predict_text(formatted_list, model)
        df = df.append({'Journey': string, 'Prediction': prediction}, ignore_index = True)
    return df


if __name__ == '__main__':

    pre_df = load_data('../../data/Top_Traversals_demo-1daybehavior_20140401.csv')

    model = get_model('data/ldamodel.pkl')

    #Make, print, and save predictions dataframe
    df = make_results_df(pre_df['Path'][:100], model)
    print df
    df.to_csv('data/predictions.csv')
