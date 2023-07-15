import os
import logging

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub

import warnings
warnings.filterwarnings('ignore')

# DATA_PATH = os.getcwd() + '/../data/'
DATA_PATH = '/usr/src/app/data/'

DOC_COUNTER = 0

LOG_FILE = DATA_PATH + f'logs/feature_extraction_tracker_log.log'
logging.basicConfig(handlers=[logging.FileHandler(filename=LOG_FILE, 
                                                 encoding='utf-8', mode='a+')],
                    level=logging.INFO)

# tf.saved_model.LoadOptions(
#     allow_partial_checkpoint=False,
#     experimental_io_device='/job:localhost',
#     experimental_skip_checkpoint=False
# )

# tf_model = tf.keras.models.load_model(
#     DATA_PATH + 'USE_model/'
# )

tf_model = hub.load(os.path.join(DATA_PATH, 'models/USE_model'))

def get_modified_vectors(vec_data):
    
    new_data = []
    for val in vec_data:
        new_data.append(val)
    
    new_data = np.array(new_data).reshape(-1, 512)
    return new_data

def get_avg_token_vector(token_list):
    
    global DOC_COUNTER
    DOC_COUNTER += 1

    avg_token_vec = []
    for token in token_list:
        avg_token_vec.append(tf_model(token)['outputs'].numpy()[0].reshape(1, -1))

    logging.info(f'finished document {DOC_COUNTER} ... ')
        
    return get_modified_vectors(np.mean(avg_token_vec, axis=0))

if  __name__ == '__main__':

    logging.info('starting feature extractor on unlabeled data ... ')

    try:
        pickle_filepath = DATA_PATH+'unlabeled_df_features.pkl'

        unlabeled_df = pd.read_pickle(pickle_filepath)
        unlabeled_df['nc_vec'] = unlabeled_df.apply(lambda x:get_avg_token_vector(x['text_tokens'][0]), axis=1)
        unlabeled_df.to_pickle(pickle_filepath)
    except Exception as e:
        logging.info(e)

    logging.info('Ending feature extractor on unlabeled data ... ')

