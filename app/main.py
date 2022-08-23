from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os 

from app.fe import FeatureEngineering


class InputM(BaseModel):

    text : str


pth = Path(os.getcwd())

print(pth)

app = FastAPI()

FE = FeatureEngineering(pth / "app/")

clf = joblib.load(pth / "app/models/lr.pkl")
idtolabel = np.load(pth / "app/models/idstolabel.npy", allow_pickle=True)
idtolabel = idtolabel.item()

@app.get("/")
def root():
    return {"root":"The app works well..."}


@app.post("/inference/")
def inference(X:InputM):

    

    # clean and transform
    transformed_text = FE.pipe(X.text)

    

    predicted_class = clf.predict(transformed_text)

    #return {"dfgfg": str(type(predicted_class[0]))}

    pred_label = idtolabel[predicted_class[0]]

    return {"Predicted class" : pred_label}