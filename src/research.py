from blog_post.settings.settings import Settings
from blog_post.data_pre_process.data_cleanup import CleanUp
from blog_post.sentimental_analysis.data_analysis import AnalysisData
from blog_post.extract_topic.user_topic import GetTopic
import logging

logging.basicConfig(filename=Settings.output_log_file, level=logging.DEBUG,
                    format='%(levelname)s %(asctime)s %(name)s %(funcName)s > %(message)s')
logger = logging.getLogger('research')


def build_pre_processing():
    logger.info('pre processing was called')
    preprocess_obj = CleanUp()
    preprocess_obj.build()
    logger.info('data pre-processing was completed')

def execute_analysis():
    logger.info('data analysis was called')
    analysis_obj = AnalysisData()
    analysis_obj.score_sentimental()
    topic_obj = GetTopic()
    topic_obj.execute_lda_model()


def main():
    build_pre_processing()
    execute_analysis()


if __name__ == '__main__':
    main()
