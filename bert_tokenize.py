# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 22:52:49 2020

@author: Kapil
"""


import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
import torch.optim as optim

from transformers import BertTokenizer, BertModel, AdamW


# Load the dataset into a pandas dataframe.
train_df = pd.read_csv('trac2_eng_train.csv')
#dev_df = pd.read_csv('trac2_eng_dev.csv')

# Report the number of sentences.
print('Number of training sentences: {:,}\n'.format(train_df.shape[0]))
# Display 10 random rows from the data.
#print(train_df.sample(10))


# Get the lists of sentences and their labels.
sentences = train_df.Text.values
labels_ta = train_df.STA.values
labels_tb = train_df.STB.values

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []
count = 0;
# For every sentence...
for sent in sentences:
    # `encode` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    encoded_sent = tokenizer.encode_plus(
                        text=sent,  # the sentence to be encoded
                        add_special_tokens=True,  # Add [CLS] and [SEP]
                        max_length = 200,  # maximum length of a sentence
                        pad_to_max_length=True,  # Add [PAD]s
                        return_attention_mask = True,  # Generate the attention mask
                        #return_tensors = 'pt',  # ask the function to return PyTorch tensors
                        truncation = True
                   )
    # Add the outputs to the lists
    input_ids.append(encoded_sent.get('input_ids'))
    attention_masks.append(encoded_sent.get('attention_mask'))

# Convert lists to tensors
input_ids = torch.tensor(input_ids)
attention_masks = torch.tensor(attention_masks)


# Print sentence 0, now as a list of IDs.
print('Original: ', sentences[3])
print('Token IDs:', input_ids[3])
print('Attention Mask:', attention_masks[3])
print('Sub task A:', labels_ta[3])
print('Sub task B:', labels_tb[3])
  

