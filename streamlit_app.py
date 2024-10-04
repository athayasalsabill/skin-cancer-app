import streamlit as st
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os
import io

# Load model YOLOv8
model = YOLO('best.pt')

# Fungsi untuk membersihkan folder static
def clear_static_folder():
    if not os.path.exists('static'):
        os.makedirs('static')
    for filename in os.listdir('static'):
        file_path = os.path.join('static', filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Fungsi utama untuk prediksi
def predict(image):
    # Simpan gambar input
    image_path = os.path.join('static', 'uploaded_image.png')
    image.save(image_path)

    # Melakukan inference menggunakan YOLOv8
    results = model.predict(source=image_path, save=False)

    # Mengecek apakah ada bounding box yang terdeteksi
    if len(results[0].boxes) > 0:
        # Dapatkan bounding box dengan confidence tertinggi
        confidences = results[0].boxes.conf.cpu().numpy()  # Dapatkan confidence setiap bounding box
        best_box_idx = confidences.argmax()  # Indeks bounding box dengan confidence tertinggi
        result_image_np = results[0].plot(boxes=[results[0].boxes[best_box_idx]])  # Plot hanya bounding box terbaik
    else:
        # Jika tidak ada bounding box, tampilkan gambar input tanpa bounding box
        result_image_np = cv2.imread(image_path)

    # Convert BGR (OpenCV format) to RGB (PIL format)
    result_image_rgb = cv2.cvtColor(result_image_np, cv2.COLOR_BGR2RGB)

    # Konversi numpy.ndarray ke PIL.Image
    result_image_pil = Image.fromarray(result_image_rgb)

    # Simpan hasil gambar ke direktori static
    static_image_path = os.path.join('static', 'result_image.png')
    result_image_pil.save(static_image_path)

    return static_image_path

# Tampilan Streamlit
st.title("Skin Cancer Detection")
st.write(
    """
    This is a deep learning model for detecting skin cancer.
    This model can detect nine types of skin lesions:
    Actinic Keratosis, Basal Cell Carcinoma, Dermatofibra, Melanoma, Melanocytic Nevus, Pigmented Benign Keratosis,
    Serborrheic Keratosis, Squamous Cell Carcinoma, and Vascular Lesion.
    """
)
st.write("Disclaimer: This model is a prototype for a college project and has not been approved by any regulatory agencies.")

# Opsi untuk upload file
uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Menggunakan PIL untuk membuka gambar
    image = Image.open(uploaded_file)

    # Tampilkan gambar yang di-upload
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Tombol untuk melakukan prediksi
    if st.button("Predict"):
        clear_static_folder()
        
        # Panggil fungsi prediksi
        result_image_path = predict(image)

        # Tampilkan hasil prediksi
        st.image(result_image_path, caption="Predicted Image", use_column_width=True)

        # Tombol untuk menyimpan gambar
        with open(result_image_path, "rb") as file:
            btn = st.download_button(
                label="Download Predicted Image",
                data=file,
                file_name="predicted_image.png",
                mime="image/png"
            )
