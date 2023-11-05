#Big Data D/HD Project Script
#word2vec Model Trainer
#Justin Li 104138316
#https://machinelearningmastery.com/develop-word-embeddings-python-gensim/

#load libraries
import pandas as pd
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import logging
import multiprocessing
import nltk
from nltk.corpus import stopwords
import string
import time

#load dataset
data = pd.read_csv('/Users/justin/Library/CloudStorage/OneDrive-SwinburneUniversity/Sem_3/BD/Sentiment Analysis/training_data.csv')

#lowercases, removes punctuation and special characters, removes stop words
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    text = ' '.join(tokens)

    return text

#process and tokenise
data['review'] = data['review'].apply(preprocess_text)
tokens = data['review'].apply(lambda x: x.lower().split())

#logger
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

#train word2vecmodel and save for use in hybrid
start_time = time.time()
model = Word2Vec(tokens, sg=1, vector_size=100, window=5, min_count=5, workers=multiprocessing.cpu_count())
training_time = time.time() - start_time
model.save("/Users/justin/Library/CloudStorage/OneDrive-SwinburneUniversity/Sem_3/BD/Sentiment Analysis/word2vec.model")
print("Training Time:", training_time)
#save vectors as txt for classifiers
model.wv.save_word2vec_format("/Users/justin/Library/CloudStorage/OneDrive-SwinburneUniversity/Sem_3/BD/Sentiment Analysis/training_data_word2vec.txt", binary=False)

