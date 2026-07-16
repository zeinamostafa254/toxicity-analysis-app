import tensorflow as tf
import pickle
import os
import numpy as np

# Set the base path to the 'data' folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'data', 'lstm_toxic_model.h5')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'data', 'tokenizer.pkl')

def load_ml_artifacts():
    """Loads the LSTM model and the tokenizer."""
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

def predict_toxicity(text, model, tokenizer):
    """Processes text and predicts toxicity score."""
    sequence = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=100)
    prediction = model.predict(padded)
    return prediction[0][0] # Returns the probability score
