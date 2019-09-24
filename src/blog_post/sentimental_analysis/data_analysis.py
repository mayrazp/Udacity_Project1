import logging
import pandas as pd
from blog_post.settings.settings import Settings
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from textblob import TextBlob
import numpy as np
from blog_post.visualization.visualization_result import PlotModel


class AnalysisData():
    def __init__(self):
        self.__logger = logging.getLogger('data_analysis.AnalysisData')

    def score_sentimental(self):
        data = pd.read_csv(Settings.cleaned_dataset_file)
        clean_data = self.steam_word(data)
        polarity_score_list = [round(TextBlob(word).polarity,1) for word in clean_data['comments']]
        data['polarity_score'] = polarity_score_list
        data['label_polarity'] = 'neutral'
        data.loc[data[data['polarity_score'] == 0].index, ['label_polarity']] = 'neutral'
        data.loc[data[(data['polarity_score'] > 0) & (data['polarity_score'] < 0.6)].index,
                 ['label_polarity']] = 'positive'
        data.loc[data[(data['polarity_score'] >= 0.6) & (data['polarity_score'] <= 1.0)].index,
                 ['label_polarity']] = 'highly positive'
        data.loc[data[(data['polarity_score'] < 0) & (data['polarity_score'] > -0.6)].index,
                 ['label_polarity']] = 'negative'

        array_polarity_score = np.unique(polarity_score_list, return_counts=True)
        neighborhood_df = data[['polarity_score', 'neighbourhood']].groupby(['neighbourhood']).agg(['mean', 'count'])
        neighborhood_df.columns = neighborhood_df.columns.droplevel(0)
        handle_plot = PlotModel()
        handle_plot.review_polarity_level(data)
        handle_plot.review_polarity(array_polarity_score)
        handle_plot.review_neighborhood(neighborhood_df)

    def steam_word(self, data):
        data.dropna(subset=['comments'], axis=0, inplace=True)
        porter = PorterStemmer()
        data.loc[:, 'comments'] = data['comments'].apply(
            lambda x: ' '.join([porter.stem(word) for word in word_tokenize(x)]))
        return data