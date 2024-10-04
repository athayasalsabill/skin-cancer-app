import streamlit as st
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os
import io

# Load model YOLOv8
model = YOLO('best19.pt')

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
        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()  # Dapatkan confidence setiap bounding box
        best_box_idx = confidences.argmax()  # Indeks bounding box dengan confidence tertinggi
        best_box = boxes[best_box_idx]  # Ambil bounding box dengan confidence tertinggi

        # Ambil koordinat dari bounding box terbaik
        x1, y1, x2, y2 = best_box.xyxy.cpu().numpy()[0].astype(int)  # Koordinat bounding box

        # Ambil nama kelas dari prediksi (misalnya kelas 0, 1, 2, dst.)
        class_id = int(best_box.cls.cpu().numpy()[0])
        class_name = model.names[class_id]  # Ambil nama kelas dari YOLO model

        # Ambil nilai confidence dari prediksi
        confidence_score = confidences[best_box_idx]
        confidence_text = f'{confidence_score:.2f}'  # Format nilai confidence jadi dua desimal

        # Gabungkan nama kelas dan confidence
        label = f'{class_name} {confidence_text}'

        # Baca gambar input
        img = cv2.imread(image_path)

        # Gambarkan bounding box dengan confidence tertinggi di atas gambar
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Warna hijau untuk bounding box

        # Tentukan ukuran teks dan posisi
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        font_thickness = 2
        text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_width, text_height = text_size

        # Tentukan posisi kotak teks
        text_offset_x = x1
        text_offset_y = y1 - 10 if y1 - 10 > 10 else y1 + 10

        # Buat kotak latar belakang untuk teks
        box_coords = ((text_offset_x, text_offset_y - text_height - 5), (text_offset_x + text_width, text_offset_y))
        cv2.rectangle(img, box_coords[0], box_coords[1], (0, 255, 0), cv2.FILLED)

        # Tambahkan teks label (nama kelas + confidence) di dalam kotak
        cv2.putText(img, label, (text_offset_x, text_offset_y - 5), font, font_scale, (0, 0, 0), font_thickness)

        result_image_np = img
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
st.write("Disclaimer!\n This model is a prototype for a college project and has not been approved by any regulatory agencies.")

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

st.markdown("Creator: ")
st.write("Athaya Salsabil  NPM 140910200025  Universitas Padjadjaran")
