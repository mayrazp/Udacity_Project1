import pandas as pd
from blog_post.settings.settings import Settings
from nltk.corpus import stopwords
import nltk
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import logging
nltk.download('wordnet')


class CleanUp():
    def __init__(self):
        self.__logger = logging.getLogger('data_pre_process.CleanUp')

    def build(self):
        data = self.merge_csv()
        clean_data = self.clean_dataset(data)
        self.__write_clean_dataset(clean_data, Settings.cleaned_dataset_file)
        data_without_stop_word = self.remove_stopwords(clean_data)
        data_without_stop_word['comments'] = data_without_stop_word['comments'].apply(self.lemmatize_text)
        self.__write_clean_dataset(data_without_stop_word, Settings.data_without_stopword_file)
        self.__logger.info('clean up was completed')

    def __read_dataset(self, dataset_file):
        data = pd.read_csv(dataset_file)
        return data

    def merge_csv(self):
        listings_data = self.__read_dataset(Settings.listings_file)
        tmp_listings_data = listings_data.loc[:, ['id', 'neighbourhood']]
        reviews_data = self.__read_dataset(Settings.reviews_file)
        merged_dataframe = pd.merge(left=reviews_data,right=tmp_listings_data, \
                                    how='left', left_on='listing_id', right_on='id')
        return merged_dataframe

    def __write_clean_dataset(self, data, output_file):
        data.to_csv(output_file, index=False)

    def clean_dataset(self, data):
        data.columns = map(str.lower, data.columns)
        data.drop(columns=['id_y'], inplace=True)
        self.__logger.info(data.isnull().sum())
        # removing NA from column comments
        data.dropna(subset=['comments'], axis=0, inplace=True)
        # removing everything except alphabets`
        data['comments'] = data['comments'].str.replace("[^A-Za-z0-9]", " ")
        data['comments'] = data['comments'].str.replace('\d+', '')
        data['comments'] = data['comments'].str.replace('[^\w\s]', '')
        data['comments'] = \
            data['comments'].str.split().map(lambda sl: " ".join(s for s in sl if len(s) > 2))
        # remove extra white spaces
        data['comments'] = data['comments'].replace('\s+', ' ', regex=True)
        return data

    def lemmatize_text(self, text):
        lemma = WordNetLemmatizer()
        return [lemma.lemmatize(w) for w in word_tokenize(text)]

    def remove_stopwords(self, data):

        stop_words = stopwords.words('english')
        additional_stopwords = ['would','could', 'will', 'francisco','lauren','kevin', 'can', 'may', 'might', 'must',
                                'and', 'i', 'a', 'and', 'so', 'arnt', 'this', 'when', 'did',  'there', 'street',
                                'all', 'front', 'there', 'traci', 'jon', 'jasmine', 'kerry', 'jarrett', 'steve',
                                'abiel', 'moreover','zoe','every']
        stop_words.extend(additional_stopwords)

        data['comments'] = \
            data['comments'].str.split().map(lambda sl: " ".join(s for s in sl if len(s) > 4))

        data['comments'] = data['comments'].replace('\s+', ' ', regex=True)

        data.dropna(subset=['comments'], axis=0, inplace=True)

        data.loc[:, 'comments'] = data['comments'].apply(
            lambda x: ' '.join([word for word in word_tokenize(x.lower()) if word not in stop_words]))

        data['comments'] = data['comments'].str.strip()
        return data