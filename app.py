import streamlit as st
import pandas as pd
import sqlite3
from PIL import Image

DB_FILE = 'predictions.db'

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_input TEXT,
            classification TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def insert_record(user_input, classification):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('INSERT INTO history (user_input, classification) VALUES (?, ?)', (user_input, classification))
    conn.commit()
    conn.close()

def fetch_all_records():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM history ORDER BY timestamp DESC", conn)
    conn.close()
    return df

@st.cache_resource
def get_classifier_module():
    import classifier
    return classifier

@st.cache_resource
def get_imagecaption_module():
    import imagecaption
    return imagecaption

def main():
    st.set_page_config(page_title="Multimodal AI Suite", layout="wide")
    init_db()
    
    st.sidebar.title("Navigation")
    menu_choice = st.sidebar.radio("Select a Module:", ["Text Classification", "Image Captioning", "Database Viewing Option"])

    if menu_choice == "Text Classification":
        st.title("Toxicity Analysis Engine")
        
        # Lazy load the classifier module
        clf = get_classifier_module()
        try:
            model, tokenizer = clf.load_ml_artifacts()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()
            
        user_input = st.text_area("Input Text:", height=150)
        
        if st.button("Analyze Text"):
            if user_input.strip():
                with st.spinner("Analyzing..."):
                    class_idx, probs = clf.predict_text(user_input, model, tokenizer)
                    label = "Toxic" if class_idx == 1 else "Non-Toxic"
                    insert_record(user_input, label)
                    st.success(f"Result: {label}")
            else:
                st.warning("Please enter text.")

    elif menu_choice == "Image Captioning":
        st.title("Image Captioning")
        
        # Lazy load the imagecaption module
        img_cap = get_imagecaption_module()
        try:
            processor, caption_model = img_cap.load_caption_model()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()
            
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            if st.button("Generate Caption"):
                caption = img_cap.generate_caption(image, processor, caption_model)
                st.info(f"**Caption:** {caption}")

    elif menu_choice == "Database Viewing Option":
        st.title("Prediction Database")
        records_df = fetch_all_records()
        st.dataframe(records_df)

if __name__ == "__main__":
    main()
