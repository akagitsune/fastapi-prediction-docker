from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel
from model.model import load_model, PredictItem

from typing import Dict

class ReqItem(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int 
    thalach: int 
    exang: int
    oldpeak: float
    slope: int 
    ca: int 
    thal: int 


    class Config:
        schema_extra = {
                "example": {'age': 66,
                            'sex': 1,
                            'cp': 0,
                            'trestbps': 160,
                            'chol': 228,
                            'fbs': 0,
                            'restecg': 0,
                            'thalach': 138,
                            'exang': 0,
                            'oldpeak': 2.3,
                            'slope': 2,
                            'ca': 0,
                            'thal': 1
                        } 
                }

clf = load_model('model/model.joblib')

def infer_model(item: PredictItem) -> Dict:
    print(item.to_numpy())
    y = clf.predict(item.to_numpy()).tolist()[0]
    return {'prediction': y, 'status': 0}

app = FastAPI()

@app.post("/predict")
def predict(req_item: ReqItem) -> Dict:
    try:
        item = PredictItem(**req_item.dict())
    except TypeError as err:
        print('Type error: ', err)
        return {'prediction': 0, 'status' : 1}
    except ValueError as err:
        print('Invalid value of item: ', err)
        return {'prediction': 0, 'status' : 2}
    pred = infer_model(item)
    return pred