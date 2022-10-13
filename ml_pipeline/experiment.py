import logging
import sys

from tasks import vua_format as vf
from ml_pipeline import utils, cnn, preprocessing, pipeline_with_lexicon
from ml_pipeline import pipelines
from ml_pipeline.cnn import CNN, evaluate

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
#handler = logging.FileHandler('experiment.log')
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def run(task_name,
        data_dir,
        pipeline_name,
        print_predictions,
       training_file, test_file):
    logger.info('>> Running {} experiment'.format(task_name))
    tsk = task(task_name)
    logger.info('>> Loading data...')
#     tsk.load(data_dir)
    tsk.load_2nd(data_dir,training_file,test_file)
    logger.info('>> retrieving train/data instances...')
    train_X, train_y, test_X, test_y = utils.get_instances(tsk, split_train_dev=False)
    test_X_ref = test_X

    if pipeline_name.startswith('cnn'):
        pipe = cnn(pipeline_name)
        train_X, train_y, test_X, test_y = pipe.encode(train_X, train_y, test_X, test_y)
        logger.info('>> testing...')
    else:
        pipe = pipeline(pipeline_name)
  
    logger.info('>> training pipeline ' + pipeline_name)
    pipe.fit(train_X, train_y)
    if pipeline_name == 'naive_bayes_counts_lex':
        logger.info("   -- Found {} tokens in lexicon".format(pipe.tokens_from_lexicon))

    logger.info('>> testing...')
    sys_y = pipe.predict(test_X)

    logger.info('>> evaluation...')
    logger.info(utils.eval(test_y, sys_y))
    
    logger.info('>> confusion matrix...')
    logger.info(utils.plot_confusion_matrix(pipe, test_y, sys_y))
    
    if print_predictions:
        logger.info('>> predictions')
        utils.print_all_predictions(test_X_ref, test_y, sys_y, logger)

# ------------------- RUN naive bayes count bgram------------------------
def run_experiment(task_name,
                    data_dir,
                    pipeline_name,
                    print_predictions,
                    min_df,
                    ngram_range):
    logger.info('>> Running {} experiment'.format(task_name))
    tsk = task(task_name)
    logger.info('>> Loading data...')
    tsk.load(data_dir)
    logger.info('>> retrieving train/data instances...')
    train_X, train_y, test_X, test_y = utils.get_instances(tsk, split_train_dev=False)
    test_X_ref = test_X



    pipe = pipeline_experiment(pipeline_name,min_df,ngram_range)

    logger.info('>> training pipeline ' + pipeline_name)

    pipe.fit(train_X, train_y)

    logger.info('>> testing...')
    sys_y = pipe.predict(test_X)

    logger.info('>> evaluation...')
    logger.info(utils.eval(test_y, sys_y))

    logger.info('>> confusion matrix...')
    logger.info(utils.plot_confusion_matrix(pipe, test_y, sys_y))


    if print_predictions:
        logger.info('>> predictions')
        utils.print_all_predictions(test_X_ref, test_y, sys_y, logger)

# ------------------- RUN naive bayes count bgram------------------------


def task(name):
    if name == 'vua_format':
        return vf.VuaFormat()
    else:
        raise ValueError("task name is unknown. You can add a custom task in 'tasks'")


def cnn(name):
    if name == 'cnn_raw':
        return CNN()
    elif name == 'cnn_prep':
        return CNN(preprocessing.std_prep())
    else:
        raise ValueError("pipeline name is unknown.")

# ------------------- RUN naive bayes count bgram------------------------
def pipeline_experiment(name,
                        min_df,
                        ngram_range):
    if name == 'naive_bayes_counts_bgram':
        return pipelines.naive_bayes_counts_bgram(min_df,ngram_range)
    elif name == 'naive_bayes_tfidf_bgram':
        return pipelines.naive_bayes_tfidf_bgram(min_df,ngram_range)
    elif name == 'naive_bayes_counts_lex_bgram':
        return pipeline_with_lexicon.naive_bayes_counts_lex_bgram(min_df,ngram_range)
# ------------------- RUN naive bayes count bgram------------------------


def pipeline(name):
    if name == 'naive_bayes_counts':
        return pipelines.naive_bayes_counts()
    elif name == 'naive_bayes_tfidf':
        return pipelines.naive_bayes_tfidf()
    elif name == 'naive_bayes_counts_lex':
        return pipeline_with_lexicon.naive_bayes_counts_lex()
    elif name == 'svm_libsvc_counts':
        return pipelines.svm_libsvc_counts()
    elif name == 'svm_libsvc_tfidf':
        return pipelines.svm_libsvc_tfidf()
    elif name == 'svm_libsvc_embed':
        return pipelines.svm_libsvc_embed()
    elif name == 'svm_sigmoid_embed':
        return pipelines.svm_sigmoid_embed()
    else:
        raise ValueError("pipeline name is unknown. You can add a custom pipeline in 'pipelines'")




