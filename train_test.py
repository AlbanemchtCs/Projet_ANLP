# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 19:41:54 2023

@author: rnonc
"""

import torch, os
import datasets
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer,BertForSequenceClassification,AutoModelForSequenceClassification, AdamW
from progressbar import progressbar
from tqdm import tqdm
import seaborn as sns
def encode_text_dataset(dataset, model_name, output_dir):
    # Charger le modèle mBERT pré-entraîné
    tokenizer = BertTokenizer.from_pretrained(model_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Encoder le texte dans le jeu de données
    for i in progressbar(range(len(dataset))):
        text = dataset[i]
        encoded_text = tokenizer(text, return_tensors='pt')
        output_file = os.path.join(output_dir, f"encoded_text_{i}.pt")
        torch.save(encoded_text, output_file)

    # Vider le cache de mémoire PyTorch
    torch.cuda.empty_cache()


def tok_dataset(dataset,tokenizer):
    toked = dataset.map(lambda x :  tok(x,tokenizer))
    dic = {}
    dic['input_ids']= toked['input_ids']
    dic['attention_mask']= toked['attention_mask']
    dic['token_type_ids']= toked['token_type_ids']
    dic['labels'] = [x for x in dataset['Label']]#[ [1 if j == x else 0 for j in range(num_labels)]for x in dataset['Label']]
    output  = datasets.Dataset.from_dict(dic).with_format("torch")
    return output

#Hyperparameters
MAX_LENGTH = 50
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
EPOCH = 1


def tok(examples,tokenizer):
    return tokenizer(examples['Utterance'], padding="max_length", max_length=MAX_LENGTH,truncation=True)


def exec_train(model,tokenizer,dataset,pond=False):
    #use device
    device  = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    tok_train = tok_dataset(dataset,tokenizer)
    data_train = DataLoader(tok_train,batch_size=BATCH_SIZE,shuffle=True)
    nb_classes = len(set(dataset['Label']))
    if pond:
        class_weights = np.array([0 for i in range(nb_classes)])
        for i in dataset['Label']:
            class_weights[i] += 1
        max_nb_class = class_weights.max()
        class_weights = np.power(max_nb_class/class_weights,0.25)
        class_weights =  torch.Tensor(class_weights).to(device)
        print(class_weights)
    else:
        class_weights = torch.Tensor([1. for i in range(nb_classes)]).to(device)
    
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    #Training
    for epoch in range(EPOCH):
        model.train()
        pbar = tqdm(total= len(data_train),desc=f'Training epoch {epoch+1}')
        nb_samples = 0
        loss_tot = 0
        total_correct = 0
        for batch in data_train:
            optimizer.zero_grad()
            batch['input_ids'] = batch['input_ids'].to(device)
            batch['attention_mask'] = batch['attention_mask'].to(device)
            batch['token_type_ids'] = batch['token_type_ids'].to(device)
            batch['labels'] = batch['labels'].long().to(device)
            outputs = model(**batch)
            nb_samples += BATCH_SIZE
            
            #loss
            #loss = outputs.loss
            logits = outputs['logits']
            loss = criterion(logits, batch['labels'])
            loss_tot += float(loss)
            
            #accuracy
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            total_correct += (predictions == batch['labels']).sum().item()
            
            #prompt
            pbar.set_postfix(loss=loss_tot/nb_samples,accuracy=total_correct/nb_samples)
            
            #learning
            loss.backward()
            optimizer.step()
            
            pbar.update(1)
    #return model

def exec_test(model,tokenizer,dataset,nb_label=None):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    tok_test = tok_dataset(dataset,tokenizer)
    data_test = DataLoader(tok_test,batch_size=BATCH_SIZE,shuffle=False)
    if nb_label is None:
        nb_label = np.max(dataset['Label'])+1
    acc_matrix = np.zeros((nb_label,nb_label))
    
    #validation
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        pbar = tqdm(total= len(data_test),desc='Test')
        for batch in data_test:
            batch['input_ids'] = batch['input_ids'].to(device)
            batch['attention_mask'] = batch['attention_mask'].to(device)
            batch['token_type_ids'] = batch['token_type_ids'].to(device)
            batch['labels'] = batch['labels'].long().to(device)
            
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            for i in range(len(batch['labels'])):
                acc_matrix[int(batch['labels'][i]),int(predictions[i])] += 1
            total_correct += (predictions == batch['labels']).sum().item()
            total_samples += batch['labels'].size(0)
            pbar.set_postfix(accuracy=total_correct/total_samples)
            pbar.update(1)
        val_acc = total_correct / total_samples
        print('test Acc:', val_acc)
    return acc_matrix

"""
###EXAMPLE###

#Load Dataset
dataset = datasets.load_dataset('miam.py','loria')

num_labels = len(set(dataset['train']['Label']))

#Import model et tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', return_tensors='pt')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased',num_labels=num_labels, problem_type="single_label_classification")

exec_train(model,tokenizer,datasets.concatenate_datasets([dataset['train'],dataset['validation']]),True)
M = exec_test(model,tokenizer,dataset['test'])
w = np.sum(M,axis=1)
for i in range(w.shape[0]):
    if w[i] == 0:
        w[i] = 1
sns.heatmap(M/w)
"""

