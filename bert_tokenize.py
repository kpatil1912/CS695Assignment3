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

from transformers import BertTokenizer, BertModel


# Load the dataset into a pandas dataframe.
train_df = pd.read_csv('trac2_eng_train.csv')
#test_df = pd.read_csv('trac2_eng_test.csv')

# Report the number of sentences.
print('Number of training sentences: {:,}\n'.format(train_df.shape[0]))
# Display 10 random rows from the data.
#print(train_df.sample(10))

# Get the lists of sentences and their labels.
sentences = train_df.Text.values
labels = train_df.Text.values

#print(sentences)

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
# For every sentence...
for sent in sentences:
    # `encode` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    encoded_sent = tokenizer.encode(
                        text=sent,  # the sentence to be encoded
                        add_special_tokens=True,  # Add [CLS] and [SEP]
                        max_length = 510,  # maximum length of a sentence
                        pad_to_max_length=True,  # Add [PAD]s
                        return_attention_mask = True,  # Generate the attention mask
                        return_tensors = 'pt',  # ask the function to return PyTorch tensors
                        truncation = True
                   )
    
    # Add the encoded sentence to the list.
    input_ids.append(encoded_sent)
# Print sentence 0, now as a list of IDs.
#print('Original: ', sentences[0])
#print('Token IDs:', input_ids[0])

indexed_tokens = tokenizer.convert_tokens_to_ids(input_ids)

attention_masks  = np.where(input_ids != 0, 1, 0)
print(attention_masks)

input_ids = torch.tensor([indexed_tokens])  
attention_masks = torch.tensor(attention_masks)

all_train_embedding = []

with torch.no_grad():
  for i in tqdm(range(0,len(input_ids),200)):    
    last_hidden_states = model(input_ids[i:min(i+200,len(train_df))], attention_mask = attention_masks[i:min(i+200,len(train_df))])[0][:,0,:].numpy()
    all_train_embedding.append(last_hidden_states)
    
    
class BertBinaryClassifier(nn.Module):
    def __init__(self, dropout=0.1):
        super(BertBinaryClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, tokens, masks=None):
        _, pooled_output = self.bert(tokens, attention_mask=masks, output_all_encoded_layers=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        proba = self.sigmoid(linear_output)
        return proba


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
