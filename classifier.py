
def load_ml_artifacts():
    # Only import heavy libraries INSIDE the function
    import tensorflow as tf
    import pickle
    import os
    
    model = tf.keras.models.load_model('data/lstm_toxic_model.h5')
    with open('data/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

def predict_text(text, model, tokenizer):
    import tensorflow as tf
    # Your logic...
    return class_idx, probs
