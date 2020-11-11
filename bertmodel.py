# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 14:39:38 2020

@author: Kapil
"""

import torch
import torch.nn as nn
from transformers import BertModel
from torch.nn import functional as F

from attention import *


# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, 2

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        self.attention = BaseAttention(768)

        # Instantiate an one-layer feed-forward classifier
        self.sequential = nn.Sequential(
            nn.Linear(768, 500),

            nn.BatchNorm1d(500),
            nn.Dropout(0.5),
            nn.Linear(500 , 100),

            nn.BatchNorm1d(100),
            nn.Dropout(0.5),
        )
        
        self.output1 = nn.Linear(100, 3) #for aggression identification
        self.output2 = nn.Linear(100, 2) #to decide whether the message is gendered or not
        
        
    def forward(self, input_ids, attention_masks, token_type_ids=None):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        print('Forward started')
        # Feed input to BERT
        hidden, _ = self.bert(input_ids)[-2:]
        sentences = hidden[-1]

        attention_applied, attention_weights = self.attention(sentences, attention_masks.float())

        x = self.sequential(attention_applied)
        out1 = F.softmax(self.output1(x), -1)
        print(out1)
        print(out1.size())
        out2 = F.softmax(self.output2(x), -1)
        print(out2)
        print(out2.size())
        print(attention_weights)

        return {
            'y_pred1': out1,
            'y_pred2': out2,
            'weights': attention_weights
        }
