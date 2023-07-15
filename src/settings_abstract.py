import sqlite3
import os
import pandas as pd
import numpy as np

import json
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub
from keybert import KeyBERT

import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import ndcg_score

from nltk.tokenize import sent_tokenize

import warnings
warnings.filterwarnings('ignore')

import fasttext
import fasttext.util
import logging

from sentence_transformers import SentenceTransformer
from transformers import pipeline

import spacy

DATA_PATH = '/usr/src/app/data/'
LOG_FILE = DATA_PATH + f'logs/abstract_analysis_log.log'
logging.basicConfig(handlers=[logging.FileHandler(filename=LOG_FILE, 
                                                 encoding='utf-8', mode='a+')],
                    level=logging.INFO)

tf_model = hub.load(os.path.join(DATA_PATH, 'models/USE_model'))

sqlite_db_path = DATA_PATH+'retrieval_test_dataset.db'

nlp_de = spacy.load("de_core_news_sm")
nlp_en = spacy.load("en_core_web_sm")

PRETRAINED_FASTTRACK_MODEL = os.path.join(DATA_PATH, 'models/lid.176.bin')

LANG_EN = "__label__en"
LANG_DE = "__label__de"

fasttext.FastText.eprint = lambda x: None
fasttext_model = fasttext.load_model(PRETRAINED_FASTTRACK_MODEL)

model_name = 'deutsche-telekom/mt5-small-sum-de-en-v1'
summarizer = pipeline("summarization", model=model_name, tokenizer=model_name)

df_xlm = pd.read_pickle(DATA_PATH+'final_dataframe.pkl')

arxiv_filepath = DATA_PATH+'arxiv-metadata-oai-snapshot.json'

model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
kw_model = KeyBERT(model=model)