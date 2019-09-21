import pandas as pd
import logging
from blog_post.visualization.visualization_result import PlotModel
from blog_post.settings.settings import Settings


class GetTopic():
    def __init__(self):
        self.__logger = logging.getLogger('user_topic.GetTopic')
        self.n_topics = 4
        self.num_words_topic = 25
        self.df = pd.read_csv(Settings.cleaned_dataset_file)
        self.df.dropna(subset=['comments'], axis=0, inplace=True)
        self.num_rows = self.df['comments'].size

    def execute_lda_model(self):
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer
        # the vectorizer object will be used to transform text to vector form
        vectorizer = CountVectorizer(max_df=0.9, min_df=25, token_pattern='\w+|\$[\d\.]+|\S+')
        # apply transformation
        tf = vectorizer.fit_transform(self.df['comments'])
        # tf_feature_names tells us what word each column in the matric represents
        tf_feature_names = vectorizer.get_feature_names()
        number_of_topics = self.n_topics
        model = LatentDirichletAllocation(batch_size=128,
                        n_components=number_of_topics, learning_method='online',
                       random_state=0).fit(tf)
        self.display_topics(model, tf_feature_names)

    def display_topics(self, model, feature_names):
        no_top_words = self.num_words_topic
        ntopics = self.n_topics
        topic_dict = {}
        for topic_idx, topic in enumerate(model.components_):
            topic_dict["Topic %d words" % (topic_idx)] = ['{}'.format(feature_names[i])
                                                          for i in topic.argsort()[:-no_top_words - 1:-1]]
            topic_dict["Topic %d weights" % (topic_idx)] = ['{:.1f}'.format(topic[i])
                                                            for i in topic.argsort()[:-no_top_words - 1:-1]]
        #print(pd.DataFrame(topic_dict))
        handle_plot = PlotModel()
        handle_plot.show_topic_classification(feature_names, model,ntopics)