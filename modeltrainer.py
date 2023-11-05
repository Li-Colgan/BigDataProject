#Big Data D/HD Project Script
#Training & Evaluation of Simple Classifiers with TF-IDF & Word Embeddings for Sentiment Analysis
#Justin Li 104138316

#load libraries
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gensim.models import KeyedVectors
import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import string
from sklearn.metrics import confusion_matrix
import time

#lowercases, removes punctuation and special characters, removes stop words
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    text = ' '.join(tokens)

    return text

#load data
train_data = pd.read_csv("/Users/justin/Library/CloudStorage/OneDrive-SwinburneUniversity/Sem_3/BD/Sentiment Analysis/training_data.csv")
test_data = pd.read_csv("/Users/justin/Library/CloudStorage/OneDrive-SwinburneUniversity/Sem_3/BD/Sentiment Analysis/testing_data.csv")
#process data
train_data['review'] = train_data['review'].apply(preprocess_text)
test_data['review'] = test_data['review'].apply(preprocess_text)

#list of classification models
models = {
    "Linear SVM": SVC(kernel='linear'),
    "Polynomial SVM": SVC(kernel='poly'),
    "RBF SVM": SVC(kernel='rbf'),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

#accumulates eval results
all_results = pd.DataFrame()

#loop controls----------------------------------------------------
for iteration in range(3):
    results = {} 

#===============================================================================================================================================
#TRAIN & EVALUATE MODELS USING WORD2VEC VECTORS
#===============================================================================================================================================
    #load word2vec model
    word2vec_model = Word2Vec.load("/Users/justin/Library/CloudStorage/OneDrive-SwinburneUniversity/Sem_3/BD/Sentiment Analysis/word2vec.model")

    #converts text to average embedding
    def text_to_word2vec_vector(text, model):
        words = text.split()
        vectors = [model.wv[word] if word in model.wv else np.zeros(model.vector_size) for word in words]
        return np.mean(vectors, axis=0)

    #create word2vec vectors for each review
    train_data['word2vec_vector'] = train_data['review'].apply(lambda text: text_to_word2vec_vector(text, word2vec_model))
    test_data['word2vec_vector'] = test_data['review'].apply(lambda text: text_to_word2vec_vector(text, word2vec_model))

    #fits each model to data, makes predictions, calculates metrics
    results = {}
    for model_name, model in models.items():
        start_time = time.time()
        model.fit(list(train_data['word2vec_vector']), train_data['sentiment'])
        training_time = time.time() - start_time
        start_time = time.time()
        predictions = model.predict(list(test_data['word2vec_vector']))
        evaluation_time = time.time() - start_time
        accuracy = accuracy_score(test_data['sentiment'], predictions)
        precision = precision_score(test_data['sentiment'], predictions, average='binary', pos_label='positive')
        sensitivity = recall_score(test_data['sentiment'], predictions, average='binary', pos_label='positive')
        f1 = f1_score(test_data['sentiment'], predictions, average='binary', pos_label='positive')
        tn, fp, fn, tp = confusion_matrix(test_data['sentiment'], predictions).ravel()
        specificity = tn / (tn + fp)

        results[model_name] = {
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'F1': f1,
            'Training Time': training_time,
            'Evaluation Time': evaluation_time
        }

    #print eval and save to csv
    print('-------------------------\nWORD2VEC VECTOR MODELS\n-------------------------\n')
    for model_name, result in results.items():
        print("Model:", result['Model'])
        print("Accuracy:", result['Accuracy'])
        print("Precision:", result['Precision'])
        print("Sensitivity:", result['Sensitivity'])
        print("Specificity:", result['Specificity'])
        print("F1 Score:", result['F1'])
        print("Training Time:", result['Training Time'])
        print("Evaluation Time:", result['Evaluation Time'])
        print()
    word2vec_results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Model', 'Accuracy', 'Precision', 'Sensitivity', 'F1', 'Specificity', 'Training Time', 'Evaluation Time'])
    word2vec_results_df['Training Data'] = 'Word2Vec Vectors'
    all_results = pd.concat([all_results, word2vec_results_df], axis=0)

#===============================================================================================================================================
#TRAIN & EVALUATE MODELS USING TF-IDF
#===============================================================================================================================================
    # create tf-idfs
    tfidf_vectorizer = TfidfVectorizer()
    train_tfidf = tfidf_vectorizer.fit_transform(train_data['review'])
    test_tfidf = tfidf_vectorizer.transform(test_data['review'])

    # evaluate
    results = {}
    for model_name, model in models.items():
        start_time = time.time() 
        model.fit(train_tfidf, train_data['sentiment'])
        training_time = time.time() - start_time 
        start_time = time.time() 
        predictions = model.predict(test_tfidf)
        evaluation_time = time.time() - start_time
        accuracy = accuracy_score(test_data['sentiment'], predictions)
        precision = precision_score(test_data['sentiment'], predictions, average='binary', pos_label='positive')
        sensitivity = recall_score(test_data['sentiment'], predictions, average='binary', pos_label='positive')
        f1 = f1_score(test_data['sentiment'], predictions, average='binary', pos_label='positive')
        tn, fp, fn, tp = confusion_matrix(test_data['sentiment'], predictions).ravel()
        specificity = tn / (tn + fp)
        results[model_name] = {
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'F1': f1,
            'Training Time': training_time,
            'Evaluation Time': evaluation_time
        }

    # print
    print('-------------------------\nTF-IDF MODELS\n-------------------------\n')
    for model_name, result in results.items():
        print("Model:", result['Model'])
        print("Accuracy:", result['Accuracy'])
        print("Precision:", result['Precision'])
        print("Sensitivity:", result['Sensitivity'])
        print("Specificity:", result['Specificity'])
        print("F1 Score:", result['F1'])
        print("Training Time:", result['Training Time'])
        print("Evaluation Time:", result['Evaluation Time'])
        print()

    tfidf_results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Model', 'Accuracy', 'Precision', 'Sensitivity', 'F1', 'Specificity', 'Training Time', 'Evaluation Time'])
    tfidf_results_df['Training Data'] = 'TF-IDF'
    all_results = pd.concat([all_results, tfidf_results_df], axis=0)

#===============================================================================================================================================
#TRAIN & EVALUATE MODELS WITH COMBINED TF-IDF AND WORD2VEC VECTORS
#===============================================================================================================================================

    # combine tf-idf and word2vec vectors 
    train_combined = np.hstack((train_tfidf.toarray(), np.array(list(train_data['word2vec_vector']))))
    test_combined = np.hstack((test_tfidf.toarray(), np.array(list(test_data['word2vec_vector']))))
    # evaluate 
    results = {}
    for model_name, model in models.items():
        start_time = time.time()
        model.fit(train_combined, train_data['sentiment'])
        training_time = time.time() - start_time
        start_time = time.time()
        predictions = model.predict(test_combined)
        evaluation_time = time.time() - start_time
        accuracy = accuracy_score(test_data['sentiment'], predictions)
        precision = precision_score(test_data['sentiment'], predictions, average='binary', pos_label='positive')
        sensitivity = recall_score(test_data['sentiment'], predictions, average='binary', pos_label='positive')
        f1 = f1_score(test_data['sentiment'], predictions, average='binary', pos_label='positive')
        tn, fp, fn, tp = confusion_matrix(test_data['sentiment'], predictions).ravel()
        specificity = tn / (tn + fp)
        results[model_name] = {
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'F1': f1,
            'Training Time': training_time,
            'Evaluation Time': evaluation_time
        }

    # print eval and save to csv
    print('-------------------------\nWEIGHTED TF-IDF WORD2VEC VECTOR MODELS\n-------------------------\n')
    for model_name, result in results.items():
        print("Model:", result['Model'])
        print("Accuracy:", result['Accuracy'])
        print("Precision:", result['Precision'])
        print("Sensitivity:", result['Sensitivity'])
        print("Specificity:", result['Specificity'])
        print("F1 Score:", result['F1'])
        print("Training Time:", result['Training Time'])
        print("Evaluation Time:", result['Evaluation Time'])
        print()
    weighted_results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Model', 'Accuracy', 'Precision', 'Sensitivity', 'F1', 'Specificity', 'Training Time', 'Evaluation Time'])
    weighted_results_df['Training Data'] = 'Weighted TF-IDF + Word2Vec Vectors'
    all_results = pd.concat([all_results, weighted_results_df], axis=0)
all_results.to_csv('eval.csv', index=False, header=True)