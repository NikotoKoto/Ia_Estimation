from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

# Charger le modèle et l'encodeur
with open('models/house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/ordinal_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

class HouseFeatures(BaseModel):
    surface_reelle_bati: float
    nombre_pieces_principales: int
    code_postal: int
    type_local: str
    surface_terrain: float

@app.post('/predict')
def predict_price(features: HouseFeatures):
    df = pd.DataFrame([features.dict()])
    df[['code_postal', 'type_local']] = encoder.transform(df[['code_postal', 'type_local']])
    prediction = model.predict(df)[0]
    return {"estimated_price": prediction}
