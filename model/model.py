import pandas as pd
import numpy as np
from joblib import dump, load
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer


@dataclass
class PredictItem:
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

    def __init__(self, age: int, sex: int, cp: int, trestbps: int, 
                       chol: int, fbs: int, restecg: int, 
                       thalach: int, exang: int, oldpeak: float,
                       slope: int, ca: int, thal:int ) -> None:
        sex_valid_vals = [0, 1]
        cp_valid_vals = [0, 1, 2, 3]
        fbs_valid_vals = [0, 1]
        restecg_valid_vals = [0, 1, 2]
        exang_valid_vals = [0, 1]
        slope_valid_vals = [0, 1, 2]
        ca_valid_vals = [0, 1, 2, 3, 4]
        thal_valid_vals = [0, 1, 2, 3]

        if sex not in sex_valid_vals:
            raise ValueError('Not valid value of sex field') 
        if cp not in cp_valid_vals:
            raise ValueError('Not valid value of cp field') 
        if fbs not in fbs_valid_vals:
            raise ValueError('Not valid value of fbs field') 
        if restecg not in restecg_valid_vals:
            raise ValueError('Not valid value of restecg field') 
        if exang not in exang_valid_vals:
            raise ValueError('Not valid value of exang field') 
        if slope not in slope_valid_vals:
            raise ValueError('Not valid value of slope field') 
        if ca not in ca_valid_vals:
            raise ValueError('Not valid value of ca field') 
        if thal not in thal_valid_vals:
            raise ValueError('Not valid value of thal field')

        self.age = age
        self.sex = sex
        self.cp = cp
        self.trestbps = trestbps
        self.chol = chol
        self.fbs = fbs
        self.restecg = restecg        
        self.thalach = thalach
        self.exang = exang
        self.oldpeak = oldpeak
        self.slope = slope
        self.ca = ca
        self.thal = thal

    def to_numpy(self):
        return np.array([[self.age, self.sex, self.cp, self.trestbps, self.chol, 
        self.fbs, self.restecg, self.thalach, self.exang, self.oldpeak, self.slope,
        self.ca, self.thal]], dtype=object)

#https://www.kaggle.com/tentotheminus9/what-causes-heart-disease-explaining-the-model
def train_and_save_model(path_to_data):
    dt = pd.read_csv(path_to_data)
    dt['age'] = dt['age'].astype('int64')
    dt['sex'] = dt['sex'].astype('int64')
    dt['cp'] = dt['cp'].astype('int64')
    dt['trestbps'] = dt['trestbps'].astype('int64')
    dt['chol'] = dt['chol'].astype('int64')
    dt['fbs'] = dt['fbs'].astype('int64')
    dt['restecg'] = dt['restecg'].astype('int64')
    dt['thalach'] = dt['thalach'].astype('int64')
    dt['exang'] = dt['exang'].astype('int64')
    dt['oldpeak'] = dt['oldpeak'].astype('float64')
    dt['slope'] = dt['slope'].astype('int64')
    dt['ca'] = dt['ca'].astype('int64')
    dt['thal'] = dt['thal'].astype('int64')

    X_train = dt.drop('target', axis=1)
    y_train = dt['target']
    ct = ColumnTransformer([(('encoder'), OneHotEncoder(),[1, 2,5,
    6, 8, 10, 12])],
        remainder='passthrough')

    pipeline = Pipeline([('encoder', ct), ('classifier', RandomForestClassifier(max_depth=5))])
    pipeline.fit(X_train, y_train)
    dump(pipeline, 'model.joblib')

def load_model(path_to_model):
    pipeline = load(path_to_model)
    return pipeline
    
if __name__ == '__main__':
    train_and_save_model('heart.csv')