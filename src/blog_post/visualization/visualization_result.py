import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from blog_post.settings.settings import Settings
from wordcloud import WordCloud, STOPWORDS
import logging


class PlotModel():
    def __init__(self):
        self.__logger = logging.getLogger('PlotModel')

    def review_polarity(self, array_polarity_score):
        plt.figure(figsize=(8, 4))
        ax = sns.barplot(array_polarity_score[0], array_polarity_score[1], color="salmon", saturation=.5, label='Boston')
        ax.set_xlabel('Airbnb review', fontsize=14)
        ax.set_ylabel('Number of reviews', fontsize=14)
        plt.savefig(Settings.polarity_score_plot, transparent=True,
                    bbox_inches='tight', pad_inches=0)
        plt.show()


    def review_neighborhood(self, neighborhood_df):
        ax = neighborhood_df[['mean']].plot.bar(figsize=(10, 6), legend=False, fontsize=6)
        ax.set_xlabel("Neighbourhood", fontsize=10)
        ax.set_ylabel("Average polarity score", fontsize=10)
        plt.savefig(Settings .average_polarity_neighborhood, transparent=True,
                    bbox_inches='tight', pad_inches=0)
        plt.show()
        ax = neighborhood_df[['count']].plot.bar(figsize=(10, 6), legend=False, fontsize=6)
        ax.set_xlabel("Neighbourhood", fontsize=10)
        ax.set_ylabel("Number of review", fontsize=10)
        plt.savefig(Settings.count_polarity_neighborhood, transparent=True,
                    bbox_inches = 'tight', pad_inches = 0)
        plt.show()

    def show_topic_classification(self, feature_names, model,ntopics):
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        for j in range(0, ntopics):
            topic = model.components_[j]
            topic_words = [feature_names[i] for i in topic.argsort()[:-25 - 1:-1]]
            topic_cloud = WordCloud(
                stopwords=STOPWORDS,
                background_color='black',
                width=4000,
                height=2800
            ).generate(" ".join(topic_words))
            ax = fig.add_subplot(2,2, j+1)
            ax.imshow(topic_cloud)
            ax.axis('off')
        plt.savefig(Settings.topic_model, transparent=True,
                        bbox_inches='tight', pad_inches=0)
        plt.show()

    def show_topics(self, lda):
        plt.figure(figsize=(8, 6))
        num_col = 3
        num_row = 5
        for i in range(15):
            df = pd.DataFrame(lda.show_topic(i), columns=['term', 'prob']).set_index('term')
            plt.subplot(num_row, num_col, i + 1)
            ax = sns.barplot(x='prob', y=df.index, data=df, palette='Blues')
            ax.set_title('topic ' + str(i), fontdict={'fontsize': 8, 'fontweight': 'medium'})
            ax.set_xlabel('probability', fontsize=8)
            ax.set_ylabel('term', fontsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.tick_params(axis="x", labelsize=8)
        plt.savefig(Settings.topics_plot, transparent=True,
                    bbox_inches='tight', pad_inches=0)
        plt.show()

