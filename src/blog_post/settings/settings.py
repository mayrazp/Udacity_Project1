import os


class Settings():
    """
       Class for project settings, it is necessary to set BOSTON_AIRBNB_BASE_DATA_DIR as directory
       to folder data on user's environment variables.
    """
    # Class attributes for paths
    base_dir = os.getenv("BOSTON_AIRBNB_BASE_DATA_DIR")
    # parser file paths
    listings_file = base_dir + '/input/Oakland/listings.csv'
    reviews_file = base_dir + '/input/Oakland/reviews.csv'
    # clean up paths
    cleaned_dataset_file = base_dir + '/output/clean_data_boston.csv'

    #topics
    user_topic_file = base_dir + '/output/user_topics.csv'
    # plots
    polarity_score_plot = base_dir +  '/output/polarity_score.jpg'
    topics_plot = base_dir +  '/output/topics.jpg'
    average_polarity_neighborhood = base_dir + '/output/average_polarity_score_neighborhood.jpg'
    count_polarity_neighborhood = base_dir + '/output/count_polarity_score_neighborhood.jpg'
    topic_model = base_dir + '/output/topic_1.jpg'
    # log file
    output_log_file = base_dir + '/output/sent_analysis.log'

