# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 10:09:46 2020

@author: Rashi
"""

import torch
from tqdm.notebook import tqdm
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
from sklearn.model_selection import train_test_split


# If there's a GPU available...
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


df = pd.read_csv('trac2_iben_train.csv', encoding="utf-8")
# df_train.head()
# #print(df_train.head())

# df_test = pd.read_csv('trac2_iben_dev.csv', encoding="utf-8")
# df_test.head()

# df = df_train.append(df_test, sort = False)


df['Sub-task A'].value_counts()

possible_labels = df['Sub-task A'].unique()

label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index
label_dict
   
#print(label_dict)

df['label'] = df['Sub-task A'].replace(label_dict)

print(label_dict)


#edit 1

X_train, X_val, y_train, y_val = train_test_split(df.index.values, 
                                                  df.label.values, 
                                                  test_size=0.15, 
                                                  random_state=42, 
                                                  stratify=df.label.values)

# X_train, X_val, y_train, y_val = train_test_split(df_test.index.values, 
#                                                   df_test.label.values, 
#                                                   test_size=0.99, 
#                                                   random_state=42, 
#                                                   stratify=df_test.label.values)




df['data_type'] = ['not_set']*df.shape[0]

df.loc[X_train, 'data_type'] = 'train'
df.loc[X_val, 'data_type'] = 'val'

df.groupby(['Sub-task A', 'label', 'data_type']).count()


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', 
                                          do_lower_case=True)
                                          
encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type=='train'].Text.values, #edit2-preprocessing
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=200, 
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type=='val'].Text.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=200, 
    return_tensors='pt'
)


input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df[df.data_type=='train'].label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df[df.data_type=='val'].label.values)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)


model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

model.to(device)

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 8

dataloader_train = DataLoader(dataset_train, 
                              sampler=RandomSampler(dataset_train), 
                              batch_size=batch_size)

dataloader_validation = DataLoader(dataset_val, 
                                   sampler=SequentialSampler(dataset_val), 
                                   batch_size=batch_size)




from transformers import AdamW, get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(),
                  lr=1e-5, 
                  eps=1e-8)
                  
epochs = 5

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)




from sklearn.metrics import f1_score
import numpy as np

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()    
    print('Accuracy:', np.sum(preds_flat == labels_flat) / len(labels_flat))
        
  
import random
import matplotlib.pyplot as plt
train_loss = 0
train_loss_list = []
f1_score_list = []
val_loss_list = []


seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

def evaluate(dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals
    
for epoch in tqdm(range(1, epochs+1)):
    print(f'\nRunning epoch {epoch}/5....')
    model.train()
    
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=True)
    for batch in progress_bar:

        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       

        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        train_loss = loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        
       # progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
         
        
    torch.save(model.state_dict(), f'ben_STA_finetuned_BERT_epoch_{epoch}.model')
        
    tqdm.write(f'Epoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train)            
    tqdm.write(f'Training loss: {loss_train_avg}')
    train_loss_list.append(loss_train_avg)
    
    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_f1}')
    f1_score_list.append(val_f1)
    val_loss_list.append(val_loss)
    
    
# training loss
plt.plot(train_loss_list, label="Training Loss")
plt.legend()
plt.title("Training loss for Bengali Sub-Task A")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("Loss_ben_STA.jpg")
plt.clf()


# validation loss
plt.plot(val_loss_list, label="Validation Loss")
plt.legend()
plt.title("Validation loss for Bengali Sub-Task A")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("ValLoss_ben_STA.jpg")
plt.clf()


# plot loss
plt.plot(f1_score_list, label="F1 Score")
plt.legend()
plt.title("F1 score for Bengali Sub-Task A")
plt.xlabel("Epochs")
plt.ylabel("F1")
plt.savefig("F1_ben_STA.jpg")


# accuracy
# model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
#                                                       num_labels=len(label_dict),
#                                                       output_attentions=False,
#                                                       output_hidden_states=False)
#model.to(device)

model.load_state_dict(torch.load('ben_STA_finetuned_BERT_epoch_5.model', map_location=torch.device('cpu')))

_, predictions, true_vals = evaluate(dataloader_validation)
accuracy_per_class(predictions, true_vals)
