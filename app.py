import os
import numpy as np
import cv2
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from feature_extraction import extract_features
from utils import load_components

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Klasifikasi Jamur Beracun atau Tidak",
        page_icon="🍄",
        layout="wide"
    )
    
    # Title and description
    st.title("🍄 Klasifikasi Jamur Beracun atau Tidak")
    st.write("""
    Aplikasi ini mengklasifikasikan jamur sebagai beracun (Beracun) atau dapat dimakan (Tidak Beracun).
    Unggah gambar atau ambil foto untuk mendapatkan prediksi.
    """)
    
    # Load model components
    @st.cache_resource
    def load_model_components():
        try:
            return load_components()
        except FileNotFoundError as e:
            st.error(f"Error saat memuat komponen model: {str(e)}")
            st.stop()
    
    # Load models
    with st.spinner("Memuat model..."):
        components = load_model_components()
        model = components['model']
        scaler = components['scaler']
        pca = components['pca']
        label_encoder = components['label_encoder']
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Unggah Gambar", "Ambil Foto"])
    
    # Function to process image and make prediction
    def process_image(image):
        # Convert to OpenCV format if needed
        if isinstance(image, np.ndarray):
            img = image
        else:
            # Convert from bytes
            file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Extract features
        features = extract_features(img)
        
        # Reshape for single sample
        features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Apply PCA
        features_pca = pca.transform(features_scaled)
        
        # Predict
        pred = model.predict(features_pca)
        pred_label = label_encoder.inverse_transform(pred)
        
        # Get probability
        prob = model.predict_proba(features_pca)
        confidence = np.max(prob)
        
        return img, pred_label[0], confidence
    
    # Function to display prediction results
    def display_results(img, prediction, confidence):
        # Convert to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create columns for layout
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.image(img_rgb, caption="Gambar yang Dianalisis", use_container_width=True)
        
        with col2:
            st.subheader("Hasil Klasifikasi:")
            
            # Display prediction with appropriate styling
            if prediction == "Beracun":
                st.error(f"**Prediksi:** {prediction}")
                st.warning("⚠️ Jamur ini tampaknya BERACUN! Jangan dikonsumsi!")
            else:
                st.success(f"**Prediksi:** {prediction}")
                st.success("✅ Jamur ini tampaknya DAPAT DIMAKAN")
            
            # Display confidence
            st.metric("Tingkat Kepercayaan", f"{confidence*100:.2f}%")
            
            # Add disclaimer
            st.info("""
            **PERHATIAN:** Ini hanya prediksi AI dan tidak boleh menjadi satu-satunya dasar 
            untuk menentukan apakah jamur aman untuk dimakan. Selalu konsultasikan dengan ahli 
            jamur sebelum mengonsumsinya.
            """)
    
    # Upload image tab
    with tab1:
        uploaded_file = st.file_uploader("Pilih gambar jamur...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Make prediction
            with st.spinner("Menganalisis gambar..."):
                img, prediction, confidence = process_image(uploaded_file)
            
            # Display results
            display_results(img, prediction, confidence)
    
    # Camera tab
    with tab2:
        camera_input = st.camera_input("Ambil foto jamur")
        
        if camera_input is not None:
            # Make prediction
            with st.spinner("Menganalisis gambar..."):
                img, prediction, confidence = process_image(camera_input)
            
            # Display results
            display_results(img, prediction, confidence)

if __name__ == "__main__":
    main() 