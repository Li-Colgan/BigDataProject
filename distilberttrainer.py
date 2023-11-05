#Big Data D/HD Project Script
#DistilBert Model Trainer
#Justin Li 104138316
#load libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import string
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
import time
from transformers import DistilBertForSequenceClassification, AdamW

#===============================================================================================================================================
#CLASSES
#===============================================================================================================================================
#https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class SentimentDataset(Dataset):
    def __init__(self, input_ids, attention_mask, targets):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.targets = targets

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_id = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        target = self.targets[idx]
        return {
            "input_ids": input_id,
            "attention_mask": attention_mask,
            "target": target
        }
#===============================================================================================================================================
#FUNCTIONS
#===============================================================================================================================================
#lowercases, removes punctuation and special characters, removes stop words
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    text = ' '.join(tokens)
    return text

#===============================================================================================================================================
#SETUP
#===============================================================================================================================================
#load data 
train_data = pd.read_csv("/Users/justin/Library/CloudStorage/OneDrive-SwinburneUniversity/Sem_3/BD/Sentiment Analysis/training_data.csv")
test_data = pd.read_csv("/Users/justin/Library/CloudStorage/OneDrive-SwinburneUniversity/Sem_3/BD/Sentiment Analysis/testing_data.csv")

#uses dictionaly to map negative to 0 and positive to 1 for classifier
sentiment_mapping = {'negative': 0, 'positive': 1}
#apply mappings to sentiment
train_data['sentiment'] = train_data['sentiment'].map(sentiment_mapping)
test_data['sentiment'] = test_data['sentiment'].map(sentiment_mapping)

max_length = 300  

#load model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

#tokenise
X_train = tokenizer(list(train_data['review']), truncation=True, padding='max_length', max_length=max_length, return_tensors="pt")
X_test = tokenizer(list(test_data['review']), truncation=True, padding='max_length', max_length=max_length, return_tensors="pt")

#convert to tensors. tokenizer created attention mask attribute
train_input_ids = X_train['input_ids']
train_attention_mask = X_train['attention_mask']
train_targets = torch.tensor(train_data['sentiment'])
test_input_ids = X_test['input_ids']
test_attention_mask = X_test['attention_mask']
test_targets = torch.tensor(test_data['sentiment'])

#create custom dataset
train_dataset = SentimentDataset(train_input_ids, train_attention_mask, train_targets)
test_dataset = SentimentDataset(test_input_ids, test_attention_mask, test_targets)

#define data loaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

#optimiser and loss functions
optimizer = AdamW(model.parameters(), lr=1e-5)
lossfunction = nn.CrossEntropyLoss()


#===============================================================================================================================================
#TRAINING 
#===============================================================================================================================================
#trainer
start_time = time.time()
for epoch in range(5):
    for batch in train_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        targets = batch["target"]

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = lossfunction(logits, targets)
        loss.backward()
        optimizer.step()
        print("Epoch [{}/5], Batch Loss: {:.4f}".format(epoch + 1, loss.item()))
training_time = time.time() - start_time

#===============================================================================================================================================
#EVALUATION
#===============================================================================================================================================
start_time_eval = time.time()
model.eval()
with torch.no_grad():
    test_outputs = model(test_input_ids, attention_mask=test_attention_mask)
    predicted_labels = test_outputs.logits.argmax(dim=1)

    accuracy = accuracy_score(test_data['sentiment'], predicted_labels.numpy())
    precision = precision_score(test_data['sentiment'], predicted_labels.numpy())
    sensitivity = recall_score(test_data['sentiment'], predicted_labels.numpy())
    f1 = f1_score(test_data['sentiment'], predicted_labels.numpy())
    tn = np.sum((test_data['sentiment'] == 0) & (predicted_labels.numpy() == 0))
    fp = np.sum((test_data['sentiment'] == 0) & (predicted_labels.numpy() == 1))
    specificity = tn / (tn + fp)
    evaluation_time = time.time() - start_time_eval

    print("DistilBERT Model")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    print("F1 Score:", f1)
    print("Training Time:", training_time)
    print("Evaluation Time:", evaluation_time)
    print()
    #save results to csv
    distilbert_results_df = pd.DataFrame({
        'Model' : 'DistilBERT',
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Sensitivity': [sensitivity],
        'Specificity': [specificity],
        'F1': [f1],
        'Training Time': [training_time],
        'Evaluation Time': [evaluation_time],
        'Training Data': 'DistilBERT Tensors'
        
    })

distilbert_results_df.to_csv('eval.csv', index=False, mode='a', header=False)
