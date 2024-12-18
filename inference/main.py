from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = FastAPI()

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('max_sequence_length.txt', 'r') as f:
    max_sequence_length = int(f.read())

with open('name_probability_dict.pickle', 'rb') as handle:
    name_probability_dict = pickle.load(handle)

model = load_model('gender_prediction_model.h5')

def predict_gender_probability(name: str) -> float:
    if not name:
        return 0.5  
    name_lower = name.lower()
    if name_lower in name_probability_dict:
        probability = name_probability_dict[name_lower]
    else:
        sequence = tokenizer.texts_to_sequences([name])
        padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding='post')
        padded_sequence = padded_sequence.astype(np.float32)
        probability = model.predict(padded_sequence)[0][0]
    return float(probability)

class NameRequest(BaseModel):
    name: str

@app.post("/predict-gender")
async def predict_gender(request: NameRequest):
    name = request.name
    probability = predict_gender_probability(name)
    return {"name": name, "gender_probability": probability}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

