import os
import cv2
import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from googlesearch import search

current_directory = os.path.dirname(os.path.realpath(__file__))
model = tf.keras.models.load_model(os.path.join(current_directory, 'ResNet50_model.h5'))

def preprocess_image(image):
    image_normalized = image.astype(np.float32) / 255.0
    min_val = np.percentile(image_normalized, 5)
    max_val = np.percentile(image_normalized, 95)
    contrast_stretched = np.clip((image_normalized - min_val) * 255 / (max_val - min_val), 0, 255).astype(np.uint8)
    return contrast_stretched

def predict_single_image(image_path):
    preprocessed_image = preprocess_image(np.array(Image.open(image_path)))
    resized_image = cv2.resize(preprocessed_image, (224, 224))
    input_image = np.expand_dims(resized_image, axis=0)
    prediction = model.predict(input_image)
    predicted_class = np.argmax(prediction)
    return predicted_class, prediction

def search_solution(query):
    search_query = query + " leaf disease solution"
    solutions = []
    for j in search(search_query, num=3, stop=3, pause=2):
        solutions.append(j)
        time.sleep(2)
    return solutions

st.markdown("""
    <style>
    .container-fluid {
        width: 100%;
        padding-right: 15px;
        padding-left: 15px;
        margin-right: auto;
        margin-left: auto;
    }
    .card {
        border: 1px solid #e6e6e6;
        border-radius: 8px;
        padding: 20px;
        margin: 10px;
    }
    .card-header {
        font-weight: bold;
        margin-bottom: 10px;
    }
    .card-body {
        margin-bottom: 10px;
    }
    .upload-container {
        border: 2px dashed #ccc;
        border-radius: 5px;
        padding: 20px;
        text-align: center;
        cursor: pointer;
    }
    .text-center {
        text-align: center;
    }
    .st-emotion-cache-13ln4jf{
        max-width: 60rem!important;        
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="container-fluid">', unsafe_allow_html=True)

st.title("Tugas Besar Pengelolaan Citra Digital - Klasifikasi Penyakit Tanaman")

team_members = [
    {"name": "Abdul Wasiul Khair", "id": "1301213278", "jobdesk": "Eksplorasi Data, Preprocessing Data, Pembuatan Model, Evaluasi Model, Penyusunan Laporan"},
    {"name": "Ichwan Rizky Wahyudin", "id": "1301213434", "jobdesk": "Pembuatan Website, Eksplorasi Data, Preprocessing Data, Pembuatan Model, Evaluasi Model, Penyusunan Laporan"},
    {"name": "Wery Holanta Mangera", "id": "1301213103", "jobdesk": "Eksplorasi Data, Preprocessing Data, Pembuatan Model, Evaluasi Model, Penyusunan Laporan"}
]

cols = st.columns(3)
for col, member in zip(cols, team_members):
    with col:
        st.markdown(f"""
            <div class="card">
                <div class="card-header">{member['name']}<br>{member['id']}</div>
                <div class="card-body">
                    <p class="card-text"><b>Jobdesk: </b>{member['jobdesk']}</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

st.markdown("## Upload your image")

uploaded_file = st.file_uploader("Drag & Drop your image here or click to upload", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    classes = ['Healthy', 'Powdery','Rust']
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    predicted_class,accuracy = predict_single_image(uploaded_file)
    
    image_preprocessed = preprocess_image(np.array(Image.open(uploaded_file)))
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(np.array(Image.open(uploaded_file)), cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(image_preprocessed, cmap='gray')
    ax[1].set_title('Preprocessed Image')
    ax[1].axis('off')
    st.pyplot(fig)

    st.write(f"Jenis penyakit tanaman: {classes[predicted_class]}, Dengan akurasi: {accuracy[0][predicted_class]:.2f}")
    try:
        solutions = search_solution(classes[predicted_class])
        st.write("Solusi yang mungkin dapat membantu:")
        for solution in solutions:
            st.write(solution)
    except:
        st.write("[FAILED] Server Gagal mencari solusi")