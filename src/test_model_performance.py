import os
import json
import logging

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub

from sentence_transformers import SentenceTransformer

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support, classification_report, precision_recall_curve

import warnings
warnings.filterwarnings('ignore')

# transformer model: https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3
# roberta model: https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1

DATA_PATH = os.getcwd() + '/../data/'
# DATA_PATH = '/usr/src/app/data/'

DOC_COUNTER = 0

LOG_FILE = DATA_PATH + f'logs/performance_tracker_log.log'
logging.basicConfig(handlers=[logging.FileHandler(filename=LOG_FILE, 
                                                 encoding='utf-8', mode='a+')],
                    level=logging.INFO)

def get_svm_classifier():
    return SVC(kernel='rbf', gamma='auto', class_weight='balanced', probability=True, random_state=122)

def get_modified_vectors(vec_data):
    
    new_data = []
    for val in vec_data:
        new_data.append(val)
    
    new_data = np.array(new_data).reshape(-1, embedding_len)
    return new_data

def get_threshold_output(preds, threshold):
    
    y_preds = []
    for val in preds:
        if val >= threshold:
            y_preds.append(1)
        else:
            y_preds.append(0)
        
    y_preds = np.array(y_preds)
    
    return y_preds

def get_f1_score_binary(y, preds, threshold, print_report=False):
    
    y_preds = get_threshold_output(preds, threshold)
    
    metrics = precision_recall_fscore_support(y, y_preds)
    f1_score_tech = metrics[2][1]
    precision_tech = metrics[0][1]
    recall_tech = metrics[1][1]
    
    if print_report:
        print(classification_report(y, y_preds))
        return f1_score_tech, precision_tech, recall_tech
    
    return f1_score_tech

def get_best_threshold(y, preds):
    
    threshold_vals = np.arange(0.1, 1, 0.001)
    f1_score_list = []
    
    for val in threshold_vals:
        f1_score_list.append(get_f1_score_binary(y, preds, val))

    max_idx = np.nanargmax(f1_score_list)
    thre_max = threshold_vals[max_idx]
    fscore = f1_score_list[max_idx]
    
    print(fscore)
    print(thre_max)
    
    return thre_max    

def get_trained_model_binary(X, y):
    
#     skf_f1score = perform_cross_validation(X, y, fold_cnt=5)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=123)
    
    model = get_svm_classifier()
    clf = make_pipeline(StandardScaler(with_mean=False), model)
    clf.fit(X_train, y_train)
    
    pred_probs = clf.predict_proba(X_test)[:,1]
    threshold = get_best_threshold(y_test, pred_probs)
    
    skf_f1score = 0
    
    return clf, skf_f1score, threshold

def get_test_f1score_binary(model, X_test, y_test, threshold):
    
    preds = model.predict_proba(X_test)
    preds = preds[:,1]

    f1_score = get_f1_score_binary(y_test, preds, threshold, print_report=True)
    
    return f1_score

def get_multi_class_metrics(y_test, preds, pr_flag=False):
    
    metrics = precision_recall_fscore_support(y_test, preds)
    
    precision_tech = metrics[0][1]
    precision_milt = metrics[0][2]
    
    recall_tech = metrics[1][1]
    recall_milt = metrics[1][2]
    
    f1_score_tech = metrics[2][1]
    f1_score_milt = metrics[2][2]
    
#     f1_score_milt = metrics[2][1]
    
    f1_score = (f1_score_tech+f1_score_milt)/2
    
    if pr_flag:
        print(classification_report(y_test, preds))
        return f1_score, preds, f1_score_milt, precision_milt, recall_milt
    
    return f1_score

def perform_cross_validation(X, y, fold_cnt=5):
    
    skf = StratifiedKFold(n_splits=fold_cnt, shuffle=True, random_state=123)
    f1_scores_list = []
    
    for train_idx, test_idx in skf.split(X, y):
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = get_svm_classifier()
        clf = make_pipeline(StandardScaler(with_mean=False), model)
        
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        f1_score = get_multi_class_metrics(y_test, preds, pr_flag=False)
        f1_scores_list.append(f1_score)
    
    return sum(f1_scores_list)/fold_cnt

def get_trained_model(X, y):
    
    skf_f1score = perform_cross_validation(X, y, fold_cnt=5)
    
    model = get_svm_classifier()
    clf = make_pipeline(StandardScaler(with_mean=False), model)
    clf.fit(X, y)
    
    return clf, skf_f1score

def get_test_f1score(model, X_test, y_test):
    
    preds = model.predict(X_test)
    f1_score, preds, f1_score_milt, precision_milt, recall_milt = get_multi_class_metrics(y_test, preds, pr_flag=True)
    
    return f1_score, preds, f1_score_milt, precision_milt, recall_milt

def get_performance_metrics(X_train, y_train, X_test, y_test, binary=False):
    
    if binary:
        model, cv_score, threshold = get_trained_model_binary(X_train, y_train)
        test_f1_score, precision_tech, recall_tech = get_test_f1score_binary(model, X_test, y_test,threshold)
        
        return model, threshold, test_f1_score, precision_tech, recall_tech
    else:    
        model, test_cv_score = get_trained_model(X_train, y_train)
        test_f1_score, preds, f1_score_milt, precision_milt, recall_milt = get_test_f1score(model, X_test, y_test)
        
        test_df['pred_label'] = preds

        return model, test_cv_score, test_f1_score, f1_score_milt, precision_milt, recall_milt


def get_avg_token_vector(token_list, testing_model_type):

    global DOC_COUNTER
    DOC_COUNTER += 1
    
    avg_token_vec = []
    for token in token_list:
        avg_token_vec.append(get_embeddings(model, token, testing_model_type))
        
    return np.mean(avg_token_vec, axis=0)

def write_data_to_file(filepath, data):

    with open(filepath, "a") as f:
        f.write(data+'\n')

def perform_analysis(model_type, feature_type):

    if feature_type == 'noun_chunks':
        multi_model, test_cv_score, test_f1_score, f1_score_milt, precision_milt, recall_milt = get_performance_metrics(X_train_nc, y_train_new, X_test_nc, y_test_new)
    elif feature_type == 'document_vectors':
        multi_model, test_cv_score, test_f1_score, f1_score_milt, precision_milt, recall_milt = get_performance_metrics(X_train_use, y_train_new, X_test_use, y_test_new)

    test_df_new = test_df[test_df['pred_label'].isin([0,1,3])]
    y_train_tech = np.array([0 if val!=1 else 1 for val in y_train]).astype('int32')
    y_test_new_tech = np.array([0 if val!=1 else 1 for val in test_df_new.label.values]).astype('int32')

    if feature_type == 'noun_chunks':
        X_test_new = get_modified_vectors(test_df_new.nc_vec.values)
        bin_model, threshold, f1_score_tech, precision_tech, recall_tech = get_performance_metrics(X_train_nc, y_train_tech, X_test_new, y_test_new_tech, binary=True)

    elif feature_type == 'document_vectors':
        X_test_new = get_modified_vectors(test_df_new.doc_vec.values)
        bin_model, threshold, f1_score_tech, precision_tech, recall_tech = get_performance_metrics(X_train_use, y_train_tech, X_test_new, y_test_new_tech, binary=True)

    results_list = [test_cv_score, test_f1_score,f1_score_milt,  precision_milt, recall_milt, f1_score_tech, precision_tech, recall_tech]
    # filter_unlabeled_data(multi_model, bin_model)

    results_list = [str(round(val, 2)) for val in results_list]
    results_list.append(model_type)
    results_list.append(feature_type)

    write_data_to_file(DATA_PATH+'two_stage_classification_results.txt', '|'.join(results_list))

def get_embeddings(model, text , model_type):

    if model_type == 'xlm_roberta':
        return model.encode(text).reshape(1, -1)
    elif model_type == 'use_xlm_transformer':
        return model(text).numpy()

if __name__ == "__main__":

    logging.info('Starting test_model_performance ....')

    try:
        # model_tests = ['use_xlm_transformer', 'xlm_roberta']
        model_tests = ['xlm_roberta']

        for testing_model_type in model_tests:

            logging.info(f'Staring model testing type: {testing_model_type}')

            if testing_model_type == 'use_xlm_transformer':
                model = hub.load(os.path.join(DATA_PATH, 'use_multilingual_transformer'))
                embedding_len = 512
            elif testing_model_type == 'xlm_roberta':
                model = SentenceTransformer(os.path.join(DATA_PATH, 'paraphrase-xlm-r-multilingual-v1'))
                embedding_len = 768

            train_df = pd.read_pickle(DATA_PATH + 'train_df_features.pkl')  ## train_df
            test_df = pd.read_pickle(DATA_PATH + 'test_df_features.pkl') ## test_df

            train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

            logging.info('starting document_vector_generation ....')

            train_df['doc_vec'] = train_df.apply(lambda x:get_embeddings(model, x['text'], testing_model_type), axis=1)
            test_df['doc_vec'] = test_df.apply(lambda x:get_embeddings(model, x['text'], testing_model_type), axis=1)

            logging.info('ending document_vector_generation ....')

            logging.info('starting noun chunk generation ....')

            train_df['nc_vec'] = train_df.apply(lambda x:get_avg_token_vector(x['text_tokens'][0], testing_model_type), axis=1)
            test_df['nc_vec'] = test_df.apply(lambda x:get_avg_token_vector(x['text_tokens'][0], testing_model_type), axis=1)

            logging.info('ending noun chunk generation ....')

            train_df.to_pickle(DATA_PATH + 'train_df_features.pkl')  ## train_df
            test_df.to_pickle(DATA_PATH + 'test_df_features.pkl') ## test_df        

            logging.info('Dataframes updated ...')

            y_train = train_df.label.values
            y_test = test_df.label.values

            y_train_new = np.array([val if val!=3 else 0 for val in y_train]).astype('int32')
            y_test_new = np.array([val if val!=3 else 0 for val in y_test]).astype('int32')

            X_train_use = get_modified_vectors(train_df.doc_vec.values)
            X_test_use = get_modified_vectors(test_df.doc_vec.values)

            X_train_nc = get_modified_vectors(train_df.nc_vec.values)
            X_test_nc = get_modified_vectors(test_df.nc_vec.values)

            perform_analysis(model_type=testing_model_type, feature_type='noun_chunks')
            perform_analysis(model_type=testing_model_type, feature_type='document_vectors')

            logging.info(f'Ending model testing type: {testing_model_type}')


    except Exception as e:
        logging.error(e)

    logging.info('Ending test_model_performance ....')