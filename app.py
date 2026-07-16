
import os
import streamlit as st

# This tells the app exactly where the folder is
base_path = os.path.dirname(os.path.abspath(__file__))

# Update your file paths like this:
model_path = os.path.join(base_path, 'data', 'lstm_toxic_model.h5')
tokenizer_path = os.path.join(base_path, 'data', 'tokenizer.pkl')
data_path = os.path.join(base_path, 'data', 'cellula_toxic_data_cleaned.csv')


import streamlit as st
import pandas as pd
import sqlite3
from PIL import Image
import classifier
import imagecaption


DB_FILE = 'predictions.db'

def init_db():
    """Initializes the SQLite database and creates the history table if it doesn't exist."""
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
    """Inserts a new prediction record into the database."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO history (user_input, classification)
        VALUES (?, ?)
    ''', (user_input, classification))
    conn.commit()
    conn.close()

def fetch_all_records():
    """Retrieves all stored records from the database as a pandas DataFrame."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM history ORDER BY timestamp DESC", conn)
    conn.close()
    return df

# Main Application Logic

def main():
    st.set_page_config(page_title="Multimodal AI Suite", layout="wide")
    
    # Initialize the database on startup
    init_db()
    
    st.sidebar.title("Navigation")
    menu_choice = st.sidebar.radio(
        "Select a Module:", 
        ["Text Classification", "Image Captioning", "Database Viewing Option"]
    )
    
   
    # Text Classification
   
    if menu_choice == "Text Classification":
        st.title("Toxicity Analysis Engine")
        st.markdown("Evaluate text for toxic elements and save the results automatically.")
        
        try:
            model, tokenizer = classifier.load_ml_artifacts()
        except Exception as e:
            st.error(f"Error loading model artifacts from the 'data' folder: {e}")
            st.stop()
            
        user_input = st.text_area("Input Text:", height=150)
        
        if st.button("Analyze Text"):
            if user_input.strip():
                with st.spinner("Analyzing..."):
                    class_idx, probs = classifier.predict_text(user_input, model, tokenizer)
                    
                    # Determine classification label
                    if class_idx == 1:
                        label = "Toxic"
                        st.markdown(
                            f"<div style='padding:1rem; border-radius:0.5rem; background-color:#4a1919; color:#ffcccc;'>"
                            f"<strong>Toxic Content Detected</strong><br>"
                            f"Confidence: {probs[1]*100:.2f}%</div>", 
                            unsafe_allow_html=True
                        )
                    else:
                        label = "Non-Toxic"
                        st.markdown(
                            f"<div style='padding:1rem; border-radius:0.5rem; background-color:#194a24; color:#ccffcc;'>"
                            f"<strong>No Toxicity Detected</strong><br>"
                            f"Confidence: {probs[0]*100:.2f}%</div>", 
                            unsafe_allow_html=True
                        )
                    
                    # Save the result to the database
                    insert_record(user_input, label)
                    st.success("Record saved to database.")
            else:
                st.warning("Please enter some text to analyze.")

    #  Image Captioning
    
    elif menu_choice == "Image Captioning":
        st.title("Image Captioning (BLIP & PyTorch)")
        st.markdown("Upload an image to generate a descriptive caption.")
        
        try:
            processor, caption_model = imagecaption.load_caption_model()
        except Exception as e:
            st.error(f"Error loading BLIP model: {e}")
            st.stop()
            
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Generate Caption"):
                with st.spinner("Generating caption..."):
                    caption = imagecaption.generate_caption(image, processor, caption_model)
                    st.info(f"**Generated Caption:** {caption}")

    
    #Database Viewing Option
    
    elif menu_choice == "Database Viewing Option":
        st.title("Prediction Database")
        st.markdown("Review all past text classifications stored in the system.")
        
        records_df = fetch_all_records()
        
        if records_df.empty:
            st.info("The database is currently empty. Run some text classifications to populate it.")
        else:
            # Display metrics
            total_records = len(records_df)
            toxic_count = len(records_df[records_df['classification'] == 'Toxic'])
            safe_count = len(records_df[records_df['classification'] == 'Non-Toxic'])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Queries", total_records)
            col2.metric("Toxic Flags", toxic_count)
            col3.metric("Safe Flags", safe_count)
            
            st.markdown("### Raw Data")
            # Streamlit's native dataframe viewer allows for sorting and scrolling
            st.dataframe(
                records_df[['id', 'timestamp', 'classification', 'user_input']], 
                use_container_width=True,
                hide_index=True
            )

if __name__ == "__main__":
    main()