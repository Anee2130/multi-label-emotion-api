from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import pickle
import numpy as np

app = FastAPI()

# Load model and preprocessors
model = tf.keras.models.load_model("model.h5")

with open("mlb.pkl", "rb") as f:
    mlb = pickle.load(f)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

class InputText(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Emotion detection API is running!"}

@app.post("/predict")
def predict_emotion(data: InputText):
    text = data.text
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=100)
    prediction = model.predict(padded)
    labels = mlb.inverse_transform((prediction > 0.5).astype(int))
    return {"labels": labels[0] if labels else ["No label detected"]}
