from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import re
import os 

class FeatureEngineering:

    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stops = stopwords.words('english')
        self.pth = Path(os.getcwd())


    def cleaning(self, X:str):
        # removing special characters
        txt = re.sub('[^A-Za-z]', ' ', X)
        txt = txt.lower()
        txt = word_tokenize(txt)
        txt_list = [word for word in txt if word not in self.stops]
        txt_list = [self.stemmer.stem(word) for word in txt_list]
        txt = " ".join(txt_list)

        return txt


    def feature_transforming(self, X:str):

        X_tr = self.cleaning(X)

        print(X_tr)

        text_transformer = joblib.load(self.pth / "app/models/tf_iX.pkl")

        transformed_text = text_transformer.transform([X_tr])

        return transformed_text


    def pipe(self, X:pd.DataFrame):

        text = self.feature_transforming(X)

        return text