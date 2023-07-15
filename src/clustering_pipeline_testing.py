import os
from pydoc import doc
import pandas as pd
import string
import numpy as np
import hdbscan
import umap
import logging
import xlsxwriter
import time
import itertools
import traceback
import json

import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

import redis
import msgpack
import msgpack_numpy as m
m.patch()

from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '/usr/src/app/data/'
tf_model = None

LOG_FILE = DATA_PATH + f'logs/candidate_pool_generation_log.log'
logging.basicConfig(handlers=[logging.FileHandler(filename=LOG_FILE, 
                                                 encoding='utf-8', mode='a+')],
                    level=logging.INFO)

rdb = redis.StrictRedis(
    host='redis_cache',
    port=6379,
)

mean_cluster_cnt = None
unique_sub_topic_ir5_cnt = None
unique_sub_topic_ir6_cnt = None
unique_document_list_ir2 = []
unique_document_list_ir3 = []
unique_document_list_ir4 = []
unique_document_list_ir5 = []
unique_document_list_ir6 = []
unique_document_ir5_cnt = None
unique_document_ir6_cnt = None


def write_document_data(data, filepath):

    with open(filepath, 'w') as f:
        json.dump(data, f)

def get_modified_vectors(vec_data):
    
    new_data = []
    for val in vec_data:
        new_data.append(val)
    
    new_data = np.array(new_data).reshape(-1, 512)
    return new_data

def get_pool_vec(doc_vec_list, pool):
    
    doc_vec_list = get_modified_vectors(doc_vec_list)
    if pool == 'mean':
        return np.nanmean(doc_vec_list, axis=0)
    elif pool == 'max':
        return np.nanmax(doc_vec_list, axis=0)

def get_umap_output(vec_array, dim_size=5):
    
    umap_obj = umap.UMAP(n_neighbors=30, 
                        n_components=dim_size, 
                        min_dist=0.01,
                        metric='cosine',
                        random_state=123).fit(vec_array) 
    
    umap_output = umap_obj.transform(vec_array) 
    return umap_output, umap_obj

def get_hdbscan_output(data_points, min_cluster_size, min_samples, cluster_size=7):
    
    hdbscan_output = hdbscan.HDBSCAN(
                            min_cluster_size=min_cluster_size,
                            min_samples=min_samples,
                            metric='euclidean',
                            cluster_selection_method='eom').fit(data_points)
    return hdbscan_output

def get_dbscan_output(data_points, cluster_size=7):
    
    dbscan_output = DBSCAN(
        #eps=3,
#                                       min_samples=2,
                                      metric='euclidean').fit(data_points)
    return dbscan_output

def project_on_2Dplane(umap_output, cluster_ids):
    
    umap_df = pd.DataFrame(np.column_stack((umap_output, cluster_ids)), columns=['x', 'y', 'cluster ids'])
    # grid = sns.FacetGrid(umap_df, hue='cluster ids', height=7)
    # grid.map(plt.scatter, 'x', 'y').add_legend()

def remove_noise(data_points, labels):
    
    data_points_filtered = []
    labels_filtered = []
    
    for dp, l in zip(data_points, labels):
        if l != -1:
            data_points_filtered.append(dp)
            labels_filtered.append(l)
            
    return data_points_filtered, labels_filtered

def get_clustering_analysis(cluster_df, final_candidate_pool_vecs, min_cluster_size, min_samples, dimen_size=5, cluster_size=7, clustering='hdbscan'):
    
    umap_output_5, umap_5 = get_umap_output(final_candidate_pool_vecs, dim_size=dimen_size)
    if clustering == 'hdbscan':
        clustering_output = get_hdbscan_output(umap_output_5, min_cluster_size, min_samples, cluster_size=cluster_size)
    elif clustering == 'dbscan':
        clustering_output = get_dbscan_output(umap_output_5, cluster_size=cluster_size)
    
    cluster_df['cluster_id'] = clustering_output.labels_
    data_points, labels = remove_noise(umap_output_5, clustering_output.labels_)
    sil_score = silhouette_score(np.array(data_points),np.array(labels),metric="euclidean",random_state=200)
    print(f'Silhouette score ... {sil_score}')

#     cluster_df.cluster_id.hist(bins=150)
    
    # umap_output_2, umap_2 = get_umap_output(final_candidate_pool_vecs, dim_size=2)
    # project_on_2Dplane(umap_output_2, cluster_df['cluster_id'])
    
    return cluster_df, sil_score

def get_nearest_keyword(keywords, keyword_vecs, mean_vec):
    
    query_distances = cosine_similarity([mean_vec], list(keyword_vecs))
    subtopic_keywords_dict = dict()
    for index in query_distances.argsort()[0]: 
        
        subtopic_keywords_dict[keywords[index]] = query_distances[0][index]
    
    subtopic_keywords_dict = sorted(subtopic_keywords_dict.items(), key=lambda x: x[1], reverse=True)
    return subtopic_keywords_dict[0][0]

def get_topics(cluster_data_df, candidate_pool):
    
    topic_list = []
    
    for idx, row in cluster_data_df.iterrows():
        
        candidate_words = row['candidate_words']
        topic = row['topic']
        
        for cp in candidate_pool:
            if cp in candidate_words:
                topic_list.append(topic)
                break
    
    return topic_list

def get_topic_documents(topic_words, final_df):
    
    doc_id_list = []
    for idx, row in final_df.iterrows():

        candidate_pool = row['candidate_pool']
        doc_id = row['page_id']

        for tw in topic_words:
            if tw in candidate_pool:
                doc_id_list.append(doc_id)

    return list(set(doc_id_list))

def get_meta_data(cluster_id, df, page_id_list):
    
    label_1 = label_2 = label_3 = label_4 = label_5 = 0
    
    for page_id in page_id_list:
        
        label = int(df[df['page_id'] == page_id]['label'].values[0])
        
        if label == 1:
            label_1 += 1
        elif label == 2:
            label_2 += 1
        elif label == 3:
            label_3 += 1
        elif label == 4:
            label_4 += 1
        elif label == 5:
            label_5 += 1

    return {
        'cluster_id': cluster_id,
        'label_1': label_1,
        'label_2': label_2,
        'label_3': label_3,
        'label_4': label_4,  
        'label_5': label_5,      
    }

def get_sub_topic_modelling(query_df, umap_dim, min_cluster_size, min_samples, clustering='hdbscan'):
    
    final_candidate_pool = []

    for idx, row in query_df.iterrows():
        final_candidate_pool.extend(row['candidate_pool'])
        
    final_candidate_pool = list(set(final_candidate_pool))

    final_candidate_pool_vecs = [m.unpackb(rdb.get(nc)) for nc in final_candidate_pool]
    # final_candidate_pool_vecs = [tf_model(nc)['outputs'].numpy()[0] for nc in final_candidate_pool]

    df_data = []
    for word, vec in zip(final_candidate_pool, final_candidate_pool_vecs):
        df_data.append((word, vec))

    cluster_df = pd.DataFrame(df_data, columns= ['candidate_words', 'candidate_vecs'])
    cluster_df, sil_score = get_clustering_analysis(cluster_df, final_candidate_pool_vecs, min_cluster_size, min_samples, dimen_size=umap_dim, cluster_size=8, clustering='hdbscan')

    cluster_data = []

    for cluster_id in set(cluster_df.cluster_id.values):

        if cluster_id != -1:
            df = cluster_df[cluster_df['cluster_id'] == cluster_id]
            cluster_data.append((cluster_id, df.candidate_words.values, df.candidate_vecs.values))

    cluster_data_df = pd.DataFrame(cluster_data, columns=['cluster_id', 'candidate_words', 'candidate_vecs'])
    cluster_data_df['mean_vec'] = cluster_data_df.apply(lambda x:get_pool_vec(x['candidate_vecs'], 'mean'), axis=1)
    cluster_data_df['topic'] = cluster_data_df.apply(lambda x:get_nearest_keyword(x['candidate_words'], x['candidate_vecs'], x['mean_vec']), axis=1)

    cluster_data_df['page_id_list'] = cluster_data_df.apply(lambda x:get_topic_documents(x['candidate_words'], query_df), axis=1)
    cluster_data_df['doc_cnt'] = cluster_data_df.apply(lambda x:len(x['page_id_list']), axis=1)

    global mean_cluster_cnt

    final_page_id_list = []
    for page_id_list in cluster_data_df.page_id_list.values:
        final_page_id_list.extend(page_id_list)

    final_page_id_list = list(set(final_page_id_list))

    document_cnt_list = []
    for page_id in final_page_id_list:

        page_id_cnt = 0
        for page_id_list in cluster_data_df.page_id_list.values:
            if page_id in page_id_list:
                page_id_cnt += 1

        document_cnt_list.append(page_id_cnt)

    mean_cluster_cnt.append(round(sum(document_cnt_list)/len(document_cnt_list), 2))

    meta_data = []
    for idx, row in cluster_data_df.iterrows():
        meta_data.append(get_meta_data(row['cluster_id'], query_df, row['page_id_list']))
        
    entity_df = pd.DataFrame(meta_data)
    cluster_data_df = pd.concat([cluster_data_df.set_index('cluster_id'), entity_df.set_index('cluster_id')], axis=1, join='inner').reset_index()
      
    return cluster_data_df, sil_score

def get_document_labels(page_id_list, df):
    
    label_1 = label_2 = label_3 = label_4 = label_5 = 0
    for page_id in list(page_id_list):
        
        label = df[df['page_id'] == page_id]['label'].values[0]
        
        if label == 1:
            label_1 += 1
        elif label == 2:
            label_2 += 1
        elif label == 3:
            label_3 += 1
        elif label == 4:
            label_4 += 1
        elif label == 5:
            label_5 += 1
            
    return (label_1+label_2+label_3+label_4), label_1, label_2, label_3, label_4, label_5

def get_covered_doc_data(page_id_list, df):
    
    covered_df = df[df['page_id'].isin(page_id_list)]
    covered_df = covered_df[covered_df['label'].isin([1, 2])]    
    
    return len(covered_df.index)

def get_expection_average_ir0(data_distribution, target_ranks_list):

    query_avg_exp = []
    pos_doc_cnt = 0

    for idx, row in enumerate(data_distribution):
        if row in target_ranks_list:
            pos_doc_cnt += 1
            query_avg_exp.append(round((pos_doc_cnt/(idx+1)), 3))

    return sum(query_avg_exp)/len(query_avg_exp)

def get_expectation_average(df, target_ranks_list, query, system_name):

    query_avg_exp = {}
    pos_doc_cnt = 0

    query_updated = query.lower().replace(' ', '_')
    avg_prec_folder = DATA_PATH +f'average_precision_query/{system_name}/{query_updated}_avg_prec.json'
    
    for idx, row in df.iterrows():
        if row['label'] in target_ranks_list:
            pos_doc_cnt += 1
            query_avg_exp[idx+1] = round((pos_doc_cnt/(idx+1)), 3)

    avg_exp_val = list(query_avg_exp.values())
    if len(avg_exp_val) == 0:
        avg_exp = 0
    else:
        avg_exp = sum(avg_exp_val)/len(avg_exp_val)

    query_avg_exp['query'] = query
    if system_name != "IR5":
        write_document_data(query_avg_exp, filepath=avg_prec_folder)
    
    return avg_exp

def get_expectation_score_ordered_df(df, target_ranks_list, query, system_name):

    exp_score_5 = 0
    exp_score_10 = 0
    exp_score_15 = 0
    exp_score_20 = 0
    exp_score_25 = 0
    
    for idx, row in df.iterrows():
        
        if row['label'] in target_ranks_list:
            if idx < 5:
                exp_score_5 += 1
            elif idx < 10:
                exp_score_10 += 1
            elif idx < 15:
                exp_score_15 += 1
            elif idx < 20:
                exp_score_20 += 1
            elif idx < 25:
                exp_score_25 += 1
    
    exp_score_10 += exp_score_5
    exp_score_15 += exp_score_10
    exp_score_20 += exp_score_15
    exp_score_25 += exp_score_20

    exp_score_avg = get_expectation_average(df, target_ranks_list, query, system_name)
    
    # exp_score_5 /= 5
    # exp_score_10 /= 10
    # exp_score_15 /= 15
    # exp_score_20 /= 20
    # exp_score_25 /= 25

    exp_data =  {
        'exp_5': round(exp_score_5, 3),
        'exp_10': round(exp_score_10, 3),
        'exp_15': round(exp_score_15, 3),       
        'exp_20': round(exp_score_20, 3),
        'exp_25': round(exp_score_25, 3),
        'exp_25': round(exp_score_25, 3),
        'exp_avg': round(exp_score_avg, 3),
    }

    return exp_data
    
def calculate_expectation(cluster_df, cluster_dict, df, target_ranks_list, system_name):
    
    query = df['query'].values[0]
    query_vec = get_modified_vectors(tf_model(query)['outputs'].numpy()[0])
    query_rank_df = get_ranked_df(df, query_vec, 'mean_nc_vec')

    cluster_data = np.array(list(cluster_dict.keys()))    
    all_combinations = []

    # if len(cluster_dict) < 7:
    #     all_combinations = list(itertools.permutations(cluster_data))
    # else:
    #     while len(set(all_combinations)) < 10:
    #         all_combinations.append(tuple(np.random.permutation(cluster_data)))

    while len(set(all_combinations)) < 1000:
        all_combinations.append(tuple(np.random.permutation(cluster_data)))
            
    list_of_found_relevant_doc_5 = []
    list_of_found_relevant_doc_10 = []
    list_of_found_relevant_doc_15 = []
    list_of_found_relevant_doc_20 = []
    list_of_found_relevant_doc_25 = []
    list_of_found_relevant_doc_avg = []

    for comb in all_combinations:
        
        cluster_list = []
        for cluster_id in comb:
            cluster_list.append(cluster_df[cluster_df['cluster_id']==cluster_id])

        cluster_ordered_df = pd.concat(cluster_list)
        cluster_ordered_df = cluster_ordered_df.reset_index(drop=True)

        ranked_document_list = get_cluster_ranked_documents(cluster_ordered_df, query_rank_df)

        df_list = []
        for page_id in ranked_document_list:
            df_list.append(df[df['page_id']==page_id])

        ordered_df = pd.concat(df_list)
        ordered_df = ordered_df.reset_index(drop=True)

        exp_comb_data = get_expectation_score_ordered_df(ordered_df, target_ranks_list, query, system_name)
        list_of_found_relevant_doc_5.append(exp_comb_data['exp_5'])
        list_of_found_relevant_doc_10.append(exp_comb_data['exp_10'])
        list_of_found_relevant_doc_15.append(exp_comb_data['exp_15'])
        list_of_found_relevant_doc_20.append(exp_comb_data['exp_20'])
        list_of_found_relevant_doc_25.append(exp_comb_data['exp_25'])
        list_of_found_relevant_doc_avg.append(exp_comb_data['exp_avg'])
        
    exp_data =  {
        'exp_5': round(sum(list_of_found_relevant_doc_5) / len(list_of_found_relevant_doc_5), 3),
        'exp_10': round(sum(list_of_found_relevant_doc_10) / len(list_of_found_relevant_doc_10), 3),
        'exp_15': round(sum(list_of_found_relevant_doc_15) / len(list_of_found_relevant_doc_15), 3),      
        'exp_20': round(sum(list_of_found_relevant_doc_20) / len(list_of_found_relevant_doc_20), 3),
        'exp_25': round(sum(list_of_found_relevant_doc_25) / len(list_of_found_relevant_doc_25), 3),
        'exp_avg': round(sum(list_of_found_relevant_doc_avg) / len(list_of_found_relevant_doc_avg), 3),
    }
                
    return exp_data

def get_cluster_expectation_scores(cluster_df, df, target_ranks_list, system_name):
    
    cluster_dict = dict()
    
    for idx, row in cluster_df.iterrows():
        cluster_dict[row['cluster_id']] = row['page_id_list']
        
    ir2_exp_data =  calculate_expectation(cluster_df, cluster_dict, df, target_ranks_list, system_name)
    return ir2_exp_data

def get_cluster_ranked_documents(cluster_df, query_df):

    document_list = []
    for idx, row in cluster_df.iterrows():

        page_id_list = row['page_id_list']
        page_id_df = query_df[query_df['page_id'].isin(page_id_list)]
        final_page_id_ranked = list(page_id_df.page_id.values)

        for page_id in final_page_id_ranked:
            if page_id not in document_list:
                document_list.append(page_id)

    return document_list

def get_ranked_df(df, query_vec, col_name):

    rank_df = df.copy()
    if col_name == 'mean_vec':
        rank_df['query_sim'] = rank_df.apply(lambda x:cosine_similarity(query_vec, get_modified_vectors(x[col_name]))[0][0], axis=1)
    elif col_name == 'mean_nc_vec':
        rank_df['query_sim'] = rank_df.apply(lambda x:cosine_similarity(query_vec, x[col_name])[0][0], axis=1)
    elif col_name == 'doc_cnt':
        rank_df['query_sim'] = rank_df.apply(lambda x:len(x['page_id_list']), axis=1)

    rank_df = rank_df.sort_values(by=['query_sim'], ascending=False)
    rank_df = rank_df.reset_index(drop=True)

    return rank_df

def get_ordered_df(cluster_df, df, rank_type, cluster_rank_type, user_query, system_name):

    if cluster_rank_type == 'query':
        query = df['query'].values[0]
    elif cluster_rank_type == 'template':
        query = "Innovation and Technology"

    user_query = user_query.lower().replace(' ', '_')

    query_vec = get_modified_vectors(tf_model(query)['outputs'].numpy()[0])

    if rank_type == 'query_sim':
        ir2_df = get_ranked_df(cluster_df, query_vec, 'mean_vec')
    elif rank_type == 'doc_cnt':
        ir2_df = get_ranked_df(cluster_df, query_vec, rank_type)
    
    return ir2_df, query_vec, query

def get_cluster_ranking_expectation_scores(cluster_df, df, rank_type, cluster_rank_type, target_ranks_list, user_query, system_name):

    ir2_df, query_vec, query = get_ordered_df(cluster_df, df, rank_type, cluster_rank_type, user_query, system_name)    

    query_rank_df = get_ranked_df(df, query_vec, 'mean_nc_vec')

    ranking_system_filepath = DATA_PATH +f'dataframes/ranking_data/{user_query}_{system_name}_ranking.pkl'
    ir2_df.to_pickle(ranking_system_filepath)

    ranked_document_list = get_cluster_ranked_documents(ir2_df, query_rank_df)

    if rank_type == 'query_sim' and cluster_rank_type == 'query':
        global unique_document_list_ir2
        unique_document_list_ir2.append(ranked_document_list)
    elif rank_type == 'query_sim' and cluster_rank_type == 'template':
        global unique_document_list_ir3
        unique_document_list_ir3.append(ranked_document_list)
    elif rank_type == 'doc_cnt' and cluster_rank_type == 'query':
        global unique_document_list_ir4
        unique_document_list_ir4.append(ranked_document_list)

    df_list = []
    for page_id in ranked_document_list:
        df_list.append(df[df['page_id']==page_id])

    ordered_df = pd.concat(df_list)
    ordered_df = ordered_df.reset_index(drop=True)

    return get_expectation_score_ordered_df(ordered_df, target_ranks_list, query, system_name)

# def combine_rows(df, window_size):

#     total_len = len(df.index)
#     new_data_list = []

#     for lower_idx in range(0, total_len, window_size):

#         upper_idx = lower_idx + window_size

#         if upper_idx >= total_len:
#             upper_idx = total_len

#         new_data_list.append(merge_df_rows(df, lower_idx, upper_idx))

#     combined_df = df
#     return combined_df
        

def combine_ir_df_2(ir_df_1, ir_df_2):

    new_ranking_list = []

    ranking_list_1 = list(ir_df_1.cluster_id.values)
    ranking_list_2 = list(ir_df_2.cluster_id.values)

    # logging.info(f'Ranking list 1: {ranking_list_1}')
    # logging.info(f'Ranking list 2: {ranking_list_2}')

    for rank_1, rank_2 in zip(ranking_list_1, ranking_list_2):

        if rank_1 not in new_ranking_list:
            new_ranking_list.append(rank_1)

        if rank_2 not in new_ranking_list:
            new_ranking_list.append(rank_2)

    # logging.info(f'New ranking list: {new_ranking_list}')

    global unique_sub_topic_ir5_cnt
    if new_ranking_list != ranking_list_1 and new_ranking_list != ranking_list_2:
        unique_sub_topic_ir5_cnt += 1


    ranked_df = ir_df_1.sort_values(by="cluster_id", key=lambda column: column.map(lambda e: new_ranking_list.index(e)))
    ranked_df = ranked_df.reset_index()

    # logging.info(f'New ranking list ranked_df: {ranked_df.cluster_id.values}\n')

    return ranked_df

def combine_ir_df_3(ir_df_1, ir_df_2, ir_df_3):

    new_ranking_list = []

    ranking_list_1 = list(ir_df_1.cluster_id.values)
    ranking_list_2 = list(ir_df_2.cluster_id.values)
    ranking_list_3 = list(ir_df_3.cluster_id.values)

    # logging.info(f'Ranking list 1: {ranking_list_1}')
    # logging.info(f'Ranking list 2: {ranking_list_2}')
    # logging.info(f'Ranking list 3: {ranking_list_3}')

    for rank_1, rank_2, rank_3 in zip(ranking_list_1, ranking_list_2, ranking_list_3):

        if rank_1 not in new_ranking_list:
            new_ranking_list.append(rank_1)

        if rank_2 not in new_ranking_list:
            new_ranking_list.append(rank_2)

        if rank_3 not in new_ranking_list:
            new_ranking_list.append(rank_3)

    # logging.info(f'New ranking list: {new_ranking_list}')

    global unique_sub_topic_ir6_cnt
    if new_ranking_list != ranking_list_1 and new_ranking_list != ranking_list_2 and new_ranking_list != ranking_list_3:
        unique_sub_topic_ir6_cnt += 1

    ranked_df = ir_df_1.sort_values(by="cluster_id", key=lambda column: column.map(lambda e: new_ranking_list.index(e)))
    ranked_df = ranked_df.reset_index()

    # logging.info(f'New ranking list ranked_df: {ranked_df.cluster_id.values}\n')

    return ranked_df

def get_cluster_expectation_scores_combined_two(cluster_df, df, cluster_rank_type, target_ranks_list, user_query, system_name):

    ir2_df_query, query_vec, query = get_ordered_df(cluster_df, df, 'query_sim', cluster_rank_type, user_query, system_name)    
    ir2_df_cnt, query_vec, query = get_ordered_df(cluster_df, df, 'doc_cnt', cluster_rank_type, user_query, system_name) 

    ir2_df = combine_ir_df_2(ir2_df_cnt, ir2_df_query)
        
    query_rank_df = get_ranked_df(df, query_vec, 'mean_nc_vec')

    ranking_system_filepath = DATA_PATH +f'dataframes/ranking_data/{user_query}_{system_name}_ranking.pkl'
    ir2_df.to_pickle(ranking_system_filepath)

    ranked_document_list = get_cluster_ranked_documents(ir2_df, query_rank_df)

    global unique_document_list_ir5
    unique_document_list_ir5.append(ranked_document_list)

    df_list = []
    for page_id in ranked_document_list:
        df_list.append(df[df['page_id']==page_id])

    ordered_df = pd.concat(df_list)
    ordered_df = ordered_df.reset_index(drop=True)

    return get_expectation_score_ordered_df(ordered_df, target_ranks_list, query, system_name)

def get_cluster_expectation_scores_combined_three(cluster_df, df, cluster_rank_type, target_ranks_list, user_query, system_name):

    ir2_df_query, query_vec, query = get_ordered_df(cluster_df, df, 'query_sim', cluster_rank_type, user_query, system_name)    
    ir2_df_template, query_vec, query = get_ordered_df(cluster_df, df, 'query_sim', 'template', user_query, system_name)    
    ir2_df_cnt, query_vec, query = get_ordered_df(cluster_df, df, 'doc_cnt', cluster_rank_type, user_query, system_name) 

    ir2_df = combine_ir_df_3(ir2_df_cnt, ir2_df_query, ir2_df_template)
        
    query_rank_df = get_ranked_df(df, query_vec, 'mean_nc_vec')

    ranking_system_filepath = DATA_PATH +f'dataframes/ranking_data/{user_query}_{system_name}_ranking.pkl'
    ir2_df.to_pickle(ranking_system_filepath)

    ranked_document_list = get_cluster_ranked_documents(ir2_df, query_rank_df)

    global unique_document_list_ir6
    unique_document_list_ir6.append(ranked_document_list)

    df_list = []
    for page_id in ranked_document_list:
        df_list.append(df[df['page_id']==page_id])

    ordered_df = pd.concat(df_list)
    ordered_df = ordered_df.reset_index(drop=True)

    return get_expectation_score_ordered_df(ordered_df, target_ranks_list, query, system_name)


def get_query_expectation_scores(df, target_ranks_list, system_name):

    query = df['query'].values[0]
    query_vec = get_modified_vectors(tf_model(query)['outputs'].numpy()[0])

    ir1_query_df = get_ranked_df(df, query_vec, 'mean_nc_vec')

    exp_system_filepath = DATA_PATH +f'dataframes/expectation_data/expectation_{query}_ranking.pkl'
    ir1_query_df.to_pickle(exp_system_filepath)
    
    return get_expectation_score_ordered_df(ir1_query_df, target_ranks_list, query, system_name)

def get_mean_values(df, system_name):

    map_score = 0
    for idx in ["5", "10", "15", "20", "25"]:
        column_name = f'{system_name} {idx}'
        map_score += df[column_name].mean()

    return round(map_score/5, 3)

def write_map_to_file(clustering_output_df, write_data, cdd_type, targeted_rank):

    if cdd_type == 'small_cdd':
        map_filename = DATA_PATH + f'dataframes/{clustering_type}_cluster_dataframes_updated_nc_small_cdd/map_results_hdbscan_{targeted_rank}_small_cdd.csv'
    elif cdd_type == 'large_cdd':
        map_filename = DATA_PATH + f'dataframes/{clustering_type}_cluster_dataframes_updated_nc_large_cdd/map_results_hdbscan_{targeted_rank}_large_cdd.csv'
    
    # map_scores_1 = get_mean_values(clustering_output_df, 'S1 IR1 exp score')
    # map_scores_2 = get_mean_values(clustering_output_df, 'S2 IR2 exp score')
    # map_scores_3 = get_mean_values(clustering_output_df, 'S3 IR2 exp score')
    # map_scores_4 = get_mean_values(clustering_output_df, 'S4 IR2 exp score')
    # map_scores_5 = get_mean_values(clustering_output_df, 'S5 IR2 exp score')

    # map_scores_list = [min_cluster_size, min_samples, mean_clusters_cnt, mean_targeted_document_ratio, map_scores_1, map_scores_2, map_scores_3, map_scores_4, map_scores_5]

    map_scores = np.array(write_data).reshape(1, len(write_data))

    with open(map_filename, "ab") as f:
        np.savetxt(f, map_scores, delimiter=",", fmt="%s")

def get_expectation_probability(g, b, d):
    
    exp_score = ((g/(g+b)) * d)
    if exp_score > g:
        return g
    else:
        return exp_score

def get_normal_distributed_ranking(pos_cnt, total_doc_cnt):
    
    half_cnt = int(total_doc_cnt / pos_cnt)
    split_rem = pos_cnt % 2
    split_itr_cnt = int(pos_cnt/2)

    sample_distribution = [0] * total_doc_cnt
    low_idx = 0
    high_idx = total_doc_cnt - 1
    mid_idx = int(total_doc_cnt/2)

    if split_rem > 0:
        sample_distribution[mid_idx] = 1

    if pos_cnt > 1:
            
        for idx in range(split_itr_cnt):

            idx_1 = low_idx + half_cnt
            idx_2 = high_idx - half_cnt
            
            if idx_1 == idx_2 or (idx_2 < idx_1 and pos_cnt == 2):
                idx_1 -= int(half_cnt/2)
                idx_2 += int(half_cnt/2)
            
            sample_distribution[idx_1] = 1
            sample_distribution[idx_2] = 1
                
            low_idx = idx_1
            high_idx = idx_2
            
    return sample_distribution

def get_query_expectation_scores_random(df, target_ranks_list):

    label_counts = df['label'].value_counts().to_dict()
    label_1 = label_counts.get(1)
    label_2 = label_counts.get(2)
    label_3 = label_counts.get(3)
    label_4 = label_counts.get(4)

    label_1 = 0 if label_1 is None else label_1
    label_2 = 0 if label_2 is None else label_2
    label_3 = 0 if label_3 is None else label_3
    label_4 = 0 if label_4 is None else label_4

    g = label_1
    b = (label_2 + label_3 + label_4)

    exp_5 = round(get_expectation_probability(g, b, d=5), 3)
    exp_10 = round(get_expectation_probability(g, b, d=10), 3)
    exp_15 = round(get_expectation_probability(g, b, d=15), 3)       
    exp_20 = round(get_expectation_probability(g, b, d=20), 3)
    exp_25 = round(get_expectation_probability(g, b, d=25), 3)

    sample_distribution = get_normal_distributed_ranking(pos_cnt=g, total_doc_cnt=(g+b))
    ap_ir0 = get_expection_average_ir0(sample_distribution, target_ranks_list)

    return {
        'exp_5': round(get_expectation_probability(g, b, d=5), 3),
        'exp_10': round(get_expectation_probability(g, b, d=10), 3),
        'exp_15': round(get_expectation_probability(g, b, d=15), 3),       
        'exp_20': round(get_expectation_probability(g, b, d=20), 3),
        'exp_25': round(get_expectation_probability(g, b, d=25), 3),
        'exp_avg': round((exp_5 + exp_10 + exp_15 + exp_20 + exp_25)/5, 3),
        'ap_ir0': round(ap_ir0, 3),
    }

def get_elimination_cluster_cnt(cluster_df, df, sil_score, target_ranks_list=[1], targeted_rank='1'):
    
    cluster_cnt_eliminated = 0
    neg_doc_eliminated = []
    query = df['query'].values[0]
    
    negative_page_id_list = []
    for page_id, label in zip(df.page_id.values, df.label.values):
        if label == 3 or label == 4:
            negative_page_id_list.append(page_id)
        
    total_doc_cnt = len(set(negative_page_id_list))
    
    for idx, row in cluster_df.iterrows():
        
        label_1 = row['label_1']
        label_2 = row['label_2']
        label_3 = row['label_3']
        label_4 = row['label_4']
        label_5 = row['label_5']

        criteria = (2*label_2) - (label_3+label_4)
        if label_1 == 0 and criteria < 0:
            
            cluster_cnt_eliminated += 1
            common_neg_docs = list(set(negative_page_id_list) & set(row['page_id_list']))
            neg_doc_eliminated.extend(common_neg_docs)
    
    targeted_doc_cnt = len(set(neg_doc_eliminated))
    
    clustered_page_id_list = []
    for id_list in cluster_df.page_id_list.values:
        clustered_page_id_list.extend(id_list)
    
    cluster_page_ids = set(clustered_page_id_list)
    all_page_ids = set(df.page_id.values)
    
    clustering_loss = all_page_ids - cluster_page_ids
    missed_documents = get_document_labels(clustering_loss, df)

    ir0_expectation_data_0 = get_query_expectation_scores_random(df, target_ranks_list)
    ir1_expectation_data_1 = get_query_expectation_scores(df, target_ranks_list, system_name='IR1')
    ir2_expectation_data_2 = get_cluster_ranking_expectation_scores(cluster_df, df, 'query_sim', 'query', target_ranks_list, query, system_name='IR2')
    ir2_expectation_data_3 = get_cluster_ranking_expectation_scores(cluster_df, df, 'query_sim', 'template', target_ranks_list, query, system_name='IR3')
    ir2_expectation_data_4 = get_cluster_ranking_expectation_scores(cluster_df, df, 'doc_cnt', 'query', target_ranks_list, query, system_name='IR4')

    ir2_expectation_data_5 = get_cluster_expectation_scores_combined_two(cluster_df, df, 'query', target_ranks_list, query, system_name='IR6')
    ir2_expectation_data_6 = get_cluster_expectation_scores_combined_three(cluster_df, df, 'query', target_ranks_list, query, system_name='IR7')

    ir2_expectation_data_7 = get_cluster_expectation_scores(cluster_df, df, target_ranks_list, system_name='IR5')

    return {'Query': query,
            'Cluster cnt': len(cluster_df.index),
            'Silhouette score': sil_score,
            'Targeted cnt': cluster_cnt_eliminated,
            'Targeted document cnt': targeted_doc_cnt,
            'Total document cnt': total_doc_cnt,
            'Ratio of targeted doc': round((targeted_doc_cnt/total_doc_cnt) * 100, 3),
            'Missed total doc': missed_documents[0],
            'Missed perfekt doc': missed_documents[1],
            'Missed relevant doc': missed_documents[2],
            'Missed irrelevant doc': missed_documents[3],
            'Missed negative doc': missed_documents[4],
            'Missed un-labelled doc': missed_documents[5],
            'S0 IR0 exp score 5': ir0_expectation_data_0['exp_5'],
            'S0 IR0 exp score 10': ir0_expectation_data_0['exp_10'],
            'S0 IR0 exp score 15': ir0_expectation_data_0['exp_15'],
            'S0 IR0 exp score 20': ir0_expectation_data_0['exp_20'],
            'S0 IR0 exp score 25': ir0_expectation_data_0['exp_25'],
            'S0 IR0 exp score avg': ir0_expectation_data_0['exp_avg'],
            'S0 IR0 exp score ap': ir0_expectation_data_0['ap_ir0'],
            'S1 IR1 exp score 5': ir1_expectation_data_1['exp_5'],
            'S1 IR1 exp score 10': ir1_expectation_data_1['exp_10'],
            'S1 IR1 exp score 15': ir1_expectation_data_1['exp_15'],
            'S1 IR1 exp score 20': ir1_expectation_data_1['exp_20'],
            'S1 IR1 exp score 25': ir1_expectation_data_1['exp_25'],
            'S1 IR1 exp score avg': ir1_expectation_data_1['exp_avg'],
            'S2 IR2 exp score 5': ir2_expectation_data_2['exp_5'],
            'S2 IR2 exp score 10': ir2_expectation_data_2['exp_10'],
            'S2 IR2 exp score 15': ir2_expectation_data_2['exp_15'],
            'S2 IR2 exp score 20': ir2_expectation_data_2['exp_20'],
            'S2 IR2 exp score 25': ir2_expectation_data_2['exp_25'],
            'S2 IR2 exp score avg': ir2_expectation_data_2['exp_avg'],
            'S3 IR2 exp score 5': ir2_expectation_data_3['exp_5'],
            'S3 IR2 exp score 10': ir2_expectation_data_3['exp_10'],
            'S3 IR2 exp score 15': ir2_expectation_data_3['exp_15'],
            'S3 IR2 exp score 20': ir2_expectation_data_3['exp_20'],
            'S3 IR2 exp score 25': ir2_expectation_data_3['exp_25'],
            'S3 IR2 exp score avg': ir2_expectation_data_3['exp_avg'],
            'S4 IR2 exp score 5': ir2_expectation_data_4['exp_5'],
            'S4 IR2 exp score 10': ir2_expectation_data_4['exp_10'],
            'S4 IR2 exp score 15': ir2_expectation_data_4['exp_15'],
            'S4 IR2 exp score 20': ir2_expectation_data_4['exp_20'],
            'S4 IR2 exp score 25': ir2_expectation_data_4['exp_25'],
            'S4 IR2 exp score avg': ir2_expectation_data_4['exp_avg'],
            'S5 IR2 exp score 5': ir2_expectation_data_5['exp_5'],
            'S5 IR2 exp score 10': ir2_expectation_data_5['exp_10'],
            'S5 IR2 exp score 15': ir2_expectation_data_5['exp_15'],
            'S5 IR2 exp score 20': ir2_expectation_data_5['exp_20'],
            'S5 IR2 exp score 25': ir2_expectation_data_5['exp_25'],
            'S5 IR2 exp score avg': ir2_expectation_data_5['exp_avg'],
            'S6 IR2 exp score 5': ir2_expectation_data_6['exp_5'],
            'S6 IR2 exp score 10': ir2_expectation_data_6['exp_10'],
            'S6 IR2 exp score 15': ir2_expectation_data_6['exp_15'],
            'S6 IR2 exp score 20': ir2_expectation_data_6['exp_20'],
            'S6 IR2 exp score 25': ir2_expectation_data_6['exp_25'],
            'S6 IR2 exp score avg': ir2_expectation_data_6['exp_avg'],
            'S7 IR2 exp score 5': ir2_expectation_data_7['exp_5'],
            'S7 IR2 exp score 10': ir2_expectation_data_7['exp_10'],
            'S7 IR2 exp score 15': ir2_expectation_data_7['exp_15'],
            'S7 IR2 exp score 20': ir2_expectation_data_7['exp_20'],
            'S7 IR2 exp score 25': ir2_expectation_data_7['exp_25'],
            'S7 IR2 exp score avg': ir2_expectation_data_7['exp_avg'],
            }


def write_clusteroutput_data(clustering_output, hyperparameters_set, cdd_type, targeted_rank):

    clustering_output_df = pd.DataFrame(clustering_output)
    mean_targeted_document_ratio = round(clustering_output_df['Ratio of targeted doc'].mean(), 3)
    max_clusters_cnt = round(clustering_output_df['Cluster cnt'].max(), 3)
    min_clusters_cnt = round(clustering_output_df['Cluster cnt'].min(), 3)
    mean_clusters_cnt = round(clustering_output_df['Cluster cnt'].mean(), 3)
    mean_sil_score = round(clustering_output_df['Silhouette score'].mean(), 3)
    mean_missed_docs = round(clustering_output_df['Missed total doc'].mean(), 3)

    write_data =  list(hyperparameters_set)
    write_data.extend([max_clusters_cnt, min_clusters_cnt, mean_clusters_cnt, mean_sil_score, mean_targeted_document_ratio, mean_missed_docs])

    # logging.info(f"\n Mean targeted document ratio - {targeted_rank} .... {mean_targeted_document_ratio} ")

    # write_map_to_file(clustering_output_df, write_data, cdd_type, targeted_rank)   
    if cdd_type == 'large_cdd':
        clustering_output_df.to_excel(clustering_df_folderpath + f'{clustering_type}_clustering_output_df_large_cdd_{targeted_rank}.xlsx', engine='xlsxwriter', index=False)
    elif cdd_type == 'small_cdd':
        clustering_output_df.to_excel(clustering_df_folderpath + f'{clustering_type}_clustering_output_small_cdd_df_{targeted_rank}.xlsx', engine='xlsxwriter', index=False)


def get_candidate_pool(subtopic_keywords_list, cp_threshold = 0.4):
    
    sim_values = []
    for key, value in subtopic_keywords_list:
        sim_values.append(value)
            
    upper_limit = round(np.percentile(sim_values, cp_threshold), 3)
    candidate_pool = []

    for key, value in subtopic_keywords_list:
        
        if value < upper_limit:
            candidate_pool.append(key)
            
    return candidate_pool

if __name__ == '__main__':

    logging.info("Started clustering pipeline testing ....")
    try:

        start = time.time()
        tf_model = hub.load(os.path.join(DATA_PATH, 'models/USE_model'))
        idx = 0
        cdd_type = 'small_cdd'
        clustering_type = 'hdbscan'
        target_queries = ['Kryptologie', 'Defense', 'Cyber Attack', 'Data Centric Warfare', 'unbemannte Wirksysteme', ]

        query_dataframe_folderpath = DATA_PATH+'dataframes/query_dataframes_updated_nc_small_cdd/'
        clustering_df_folderpath = DATA_PATH+f'dataframes/{clustering_type}_cluster_dataframes_updated_nc_small_cdd/'
        
        if cdd_type == 'large_cdd':
            query_dataframe_folderpath = DATA_PATH+'dataframes/query_dataframes_updated_nc_large_cdd/'
            clustering_df_folderpath = DATA_PATH+f'dataframes/{clustering_type}_cluster_dataframes_updated_nc_large_cdd/'

        # hyperparameters_dict = {
        #             'candidate_pool_thresholds': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],  
        #             'umap_dimensions': [5, 10],
        #             'min_cluster_size_list': [20, 25, 30, 35, 40, 45, 50, 55, 60],
        #             'min_samples_list': [1, 3, 5, 7, 10]
        # }

        hyperparameters_dict = {
                    'candidate_pool_thresholds': [85],
                    'umap_dimensions': [5],
                    'min_cluster_size_list': [20],
                    'min_samples_list': [10]
        }

        unique_sub_topic_ir5_cnt = 0
        unique_sub_topic_ir6_cnt = 0
        unique_document_ir5_cnt = 0
        unique_document_ir6_cnt = 0

        for cp_threshold in hyperparameters_dict['candidate_pool_thresholds']:

            mean_cluster_cnt = []

            for umap_dim in hyperparameters_dict['umap_dimensions']:

                for min_cluster_size in hyperparameters_dict['min_cluster_size_list']:

                    for min_samples in hyperparameters_dict['min_samples_list']:

                        hyperparameters_set = (cp_threshold, umap_dim, min_cluster_size, min_samples)
                        start_itr = time.time()

                        clustering_output_1 = []
                        # clustering_output_2 = []

                        idx += 1
                        try:
                            for query_df_filename in os.listdir(query_dataframe_folderpath):

                                query_df = pd.read_pickle(query_dataframe_folderpath+query_df_filename)
                                query_df['candidate_pool'] = query_df.apply(lambda x:get_candidate_pool(x['keywords_use'], cp_threshold = cp_threshold), axis=1)
                                query = query_df['query'].values[0]

                                if query in target_queries:
                                    continue

                                if cdd_type == 'large_cdd':
                                    query_df = query_df.rename(columns={'nc_vec':'mean_nc_vec'})

                                # cluster_data_df = pd.read_pickle(clustering_df_folderpath+query_df_filename)
                                cluster_data_df, sil_score = get_sub_topic_modelling(query_df, umap_dim, min_cluster_size, min_samples, clustering=clustering_type)


                                clustering_output_1.append(get_elimination_cluster_cnt(cluster_data_df, query_df, sil_score, target_ranks_list=[1], targeted_rank='1'))
                                # clustering_output_2.append(get_elimination_cluster_cnt(cluster_data_df, query_df, sil_score, target_ranks_list=[1, 2], targeted_rank='12'))

                                cluster_data_df.to_pickle(clustering_df_folderpath+query_df_filename)
                                # logging.info(f'finished query: {query}')

                            mean_cluster_repetition_cnt = round(sum(mean_cluster_cnt)/len(mean_cluster_cnt), 2)
                            logging.info(f'Candidate pool: {cp_threshold}, repetition count: {mean_cluster_repetition_cnt}')
                            write_clusteroutput_data(clustering_output_1, hyperparameters_set, cdd_type, targeted_rank='1')
                            # write_clusteroutput_data(clustering_output_2, min_cluster_size, min_samples, cdd_type, targeted_rank='12') 

                        except Exception as e:
                            logging.error(e)
                            logging.error(traceback.format_exc())

                            logging.info(f'Failed iteration cp_threshold: {cp_threshold}, umap_dim: {umap_dim}, min_cluster_size: {min_cluster_size}, min_samples: {min_samples} .... {idx}')
                            continue

                        end_itr = time.time()
                        time_itr =  round((end_itr-start_itr) , 3)
                        logging.info(f'Finished iteration cp_threshold: {cp_threshold}, umap_dim: {umap_dim}, min_cluster_size: {min_cluster_size}, min_samples: {min_samples}, Time taken .... {time_itr} secs.... {idx}')


        for idx in range(len(unique_document_list_ir3)):

            rank_ir2 = unique_document_list_ir2[idx]
            rank_ir3 = unique_document_list_ir3[idx]
            rank_ir4 = unique_document_list_ir4[idx]
            rank_ir5 = unique_document_list_ir5[idx]
            rank_ir6 = unique_document_list_ir6[idx]

            if rank_ir5 != rank_ir4 and rank_ir5 != rank_ir2:
                unique_document_ir5_cnt += 1
            
            if rank_ir6 != rank_ir4 and rank_ir6 != rank_ir2 and rank_ir6 != rank_ir3:
                unique_document_ir6_cnt += 1

        # unique_sub_topic_ir5_cnt = round(unique_sub_topic_ir5_cnt/17, 2)
        # unique_sub_topic_ir6_cnt = round(unique_sub_topic_ir5_cnt/17, 2)
        # unique_document_ir5_cnt = round(unique_sub_topic_ir5_cnt/17, 2)
        # unique_document_ir6_cnt = round(unique_sub_topic_ir5_cnt/17, 2)

        logging.info(f'Unique sub-topic count IR5: {unique_sub_topic_ir5_cnt}')
        logging.info(f'Unique sub-topic count IR6: {unique_sub_topic_ir6_cnt}')
        logging.info(f'Unique document count IR5: {unique_document_ir5_cnt}')
        logging.info(f'Unique document count IR6: {unique_document_ir6_cnt}')
        
        end = time.time()
        logging.info(f"\n Time taken .... {end-start} secs")

    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())

    logging.info("Finished clustering pipeline testing ....")