from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os 

from app.fe import FeatureEngineering


pth = Path(os.getcwd())

app = FastAPI()

FE = FeatureEngineering()

clf = joblib.load(pth / "app//models/lr.pkl")
idtolabel = np.load(pth / "app//models/idstolabel.npy", allow_pickle=True)
idtolabel = idtolabel.item()



@app.post("/inference/")
def inference(X:str):

    # clean and transform
    transformed_text = FE.pipe(X)

    print(transformed_text)
    print(X)

    #predicted_class = clf.predict(transformed_text)

    return {"aa": "transformed_text"}

    pred_label = idtolabel[predicted_class]

    return {"Predicted class" : pred_label}