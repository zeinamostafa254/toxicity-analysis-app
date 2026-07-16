def load_ml_artifacts():
    import tensorflow as tf
    import pickle
    import os
    # Load your model here
    model = tf.keras.models.load_model('data/lstm_toxic_model.h5')
    with open('data/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

def predict_text(text, model, tokenizer):
    import tensorflow as tf
    # Your prediction logic here
    # ...
    return class_idx, probs
