import pickle

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List


class Features(BaseModel):
    note: str
    data: List[float]
    

model_file_name = "Linear_Regression_Model.pkl"
with open(model_file_name, 'rb') as f:
    mod_LR = pickle.load(f)

app = FastAPI()
    

@app.post("/run_model")
async def run_model(features: Features):
    print( {
            'note': features.note,
            'list': features.data
        } )
    
    Y_pred = mod_LR.predict([features.data])
    print(Y_pred)

    return  {
                "note": features.note,
                "value": Y_pred[0, 0]
            }
