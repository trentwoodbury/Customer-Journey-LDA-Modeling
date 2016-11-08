import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def get_a_dict(filepath):
    df = pd.read_csv(filepath).iloc[:, 1:13]
    theme_dict = {}
    interesting_theme_idx = [3, 6, 11, 15, 16]
    theme_names = ['Horrendous IVR', 'Mobile Disengagement', "Couldn't Find it Online", "Mobile Users", "Just Show Me the Summary"]
    counter = 0
    for row_num in interesting_theme_idx:
        theme_dict[theme_names[counter]] = [df.iloc[row_num, ::2], df.iloc[row_num, 1::2]]
        counter += 1
    return theme_dict

if __name__ == "__main__":
    filepath = '../word_transition_model/data/transitions_df.csv'
    df = get_a_dict(filepath)
    print df
