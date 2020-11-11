#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 00:41:25 2020

@author: rashi
"""

from __future__ import division
import re
import json
import numpy as np
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
 
from nltk.tokenize import word_tokenize 
from transformers import *
from transformers import BertTokenizer, BertModel, BertForMaskedLM

import sys
sys.path.append('../')

# ------------------------------------------------------------------------
# Preprocessing
#-------------------------------------------------------------------------


# def tokenization(post):
#     wordgen.reset_stats()
#     wordgen.stream = [post]
#     for s_words, s_info in wordgen:
#         tokens = s_words
#     return tokens



def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           U"+1F923"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)





def replacements(text):
    text = re.sub(r'[@＠][a-zA-Z0-9_.!?-\\]+', "@username", text.rstrip())
    # text = re.sub(r"((www\.[^\s]+)|(https?:\/\/[^\s]+))", "url", text.rstrip())
    text = re.sub(r'https?://|www\.', "url", text.rstrip())
    text = text.replace('-', ' ')
    # text = text.replace('_', ' ')
    # text = text.replace('#', ' ')
    return text




train_df = pd.read_csv('trac2_eng_train.csv')
train_df.head(0)

sentence=[]
sentence = np.asarray(train_df["Text"])
sentence[3]

label1=[]
label1 = np.asarray(train_df["Sub-task A"])
label1[0]

label2=[]
label2 = np.asarray(train_df["Sub-task B"])
label2[0]

stop_words = set(stopwords.words('english'))

process_sen=[]
tokenized_text=[]
sentence_wo_sw=[]
for sen in sentence:
    
    prepare_emoji= remove_emoji(str(sen))
    process_after= replacements(prepare_emoji)
    
    
    # text_tokens = word_tokenize(process_after)
    # tokenized_text.append(text_tokens)
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    token_text = tokenizer.tokenize(process_after)

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # tokenized_sen = tokenizer.tokenize(text_tokens)
    # tokenized_text.append(tokenized_sen)
    

    sentence_wo_sw=[]
    for word in token_text:
        if word not in stop_words:
            sentence_wo_sw.append(word)
    
    process_sen.append(sentence_wo_sw)
    

sentence[2]   
process_sen[2]    


   
