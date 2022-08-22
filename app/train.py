import pandas as pd
import numpy as np
import joblib
import json
import re

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer


def save_model(model, name):

    joblib.dump(model, name)


def save_dict(dictionary, name):
    #np.save('my_file.npy', dictionary) 
    np.save(name, dictionary) 

    # Load
    # read_dictionary = np.load('my_file.npy',allow_pickle='TRUE').item()


def feature_engineering(X:pd.DataFrame):

    labels_to_ids = {}
    ids_to_labels = {}
    for i, label in enumerate(sorted(X['label'].unique())):
        labels_to_ids[label] = i
        ids_to_labels[i] = label
    
    y = X['label'].map(labels_to_ids).values

    save_dict(labels_to_ids, "/Users/davidhajdu/Desktop/Projects/amazon pet classification/models/labelstoid.npy")
    save_dict(ids_to_labels, "/Users/davidhajdu/Desktop/Projects/amazon pet classification/models/idstolabel.npy")

    stops = stopwords.words('english')
    stemmer = PorterStemmer()
    
    # removing special characters
    X['prepared_text'] = X['text'].apply(lambda text: re.sub('[^A-Za-z]', ' ', text))
    # transform text to lowercase
    X['prepared_text'] = X['prepared_text'].str.lower()
    # tokenize the texts
    X['prepared_text'] = X['prepared_text'].apply(lambda text: word_tokenize(text))
    # removing stopwords
    X['prepared_text'] = X['prepared_text'].apply(lambda words: [word for word in words if word not in stops])
    # stemming
    X['prepared_text'] = X['prepared_text'].apply(lambda words: [stemmer.stem(word) for word in words])
    # join prepared+text to use as corpus
    X['joined_prepared_text'] = X['prepared_text'].apply(lambda words: " ".join(words))

    return X['joined_prepared_text'].values, y


def feature_transforming(X:pd.DataFrame):

    X_tr, y = feature_engineering(X)

    text_transformer = TfidfVectorizer(max_features=10000)
    print("fitting tfidf")
    transformed_text = text_transformer.fit_transform(X_tr.tolist())

    save_model(text_transformer, "/Users/davidhajdu/Desktop/Projects/amazon pet classification/models/tf_iX.pkl")
    
    return transformed_text, y


def train_model(X:np.array, y:np.array):
    logit = LogisticRegression(C=5e1, solver="saga", max_iter=400, multi_class='multinomial', random_state=17, n_jobs=-1)
    logit.fit(X, y)

    save_model(logit, "/Users/davidhajdu/Desktop/Projects/amazon pet classification/app/models/lr.pkl")


def main():
    print("Loading data...")
    train = pd.read_csv('/Users/davidhajdu/Desktop/Projects/amazon pet classification/data/train.csv', index_col='id').fillna(' ')
    valid = pd.read_csv('/Users/davidhajdu/Desktop/Projects/amazon pet classification/data/valid.csv', index_col='id').fillna(' ')

    print("Feature engineering...")
    train_val = pd.concat([train, valid])

    del train, valid
    
    tr, labels = feature_transforming(train_val)

    #print(tr[:2])

    print("Training model...")

    train_model(tr, labels)


if  __name__ == "__main__":
    main()