import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Load your model
model = tf.keras.models.load_model('mobilenetv2_finetuned_model.h5')
class_names = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']

def predict_disease(image):
    image = image.resize((160, 160))
    img_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)[0]
    confidence = np.max(preds)
    pred_class = class_names[np.argmax(preds)]
    if confidence < 0.80:
        return "Healthy", confidence
    return pred_class, confidence

# Streamlit UI
st.set_page_config(page_title="Pengesan Penyakit Tumbuhan", layout="centered")

st.title("🌿 Pengesan Penyakit Tumbuhan")
st.markdown("Muat naik imej tumbuhan anda untuk pengesanan penyakit tumbuhan")

upload_method = st.radio("Pilih kaedah muat naik:", ["Muat naik dari galeri", "Guna kamera"])

if upload_method == "Muat naik dari galeri":
    uploaded_file = st.file_uploader("Pilih imej daun", type=["jpg", "png", "jpeg"])
elif upload_method == "Guna kamera":
    uploaded_file = st.camera_input("Ambil gambar daun")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imej dimuat naik!", width=200)

    if st.button("🔍 Detect"):
        with st.spinner("Menganalisis Keputusan..."):
            pred_class, confidence = predict_disease(image)

        st.success("Selesai! ✅")
        if pred_class == "Healthy":
            st.markdown(
            f"<h5 style='color:green;'>Penyakit tidak dikenal pasti / Sihat</h3>"
            f"<p style='font-size:18px;'>Keyakinan: {confidence * 100:.2f}%</p>",
            unsafe_allow_html=True
        )
        else:
            st.markdown(
            f"<h5 style>Berikut adalah keputusan:</h3>"
            f"<h5 style='color:#d0021b;'>{pred_class}</h1>"
            f"<p style='font-size:18px;'>Keyakinan: {confidence:.4f}</p>",
            unsafe_allow_html=True
        )
