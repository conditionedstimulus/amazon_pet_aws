from fastapi import FastAPI
import pandas as pd


app  =  FastAPI()


@app.post("/inference/")
def inference(X:pd.DataFrame):

    pass