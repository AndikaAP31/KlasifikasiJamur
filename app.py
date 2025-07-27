import os
import numpy as np
import cv2
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from feature_extraction import extract_features
from utils import load_components

def main():
    st.set_page_config(
        page_title="Klasifikasi Jamur Beracun atau Tidak",
        page_icon="🍄",
        layout="wide"
    )
    
    st.title("🍄 Klasifikasi Jamur Beracun atau Tidak")
    st.write("""
    Aplikasi ini mengklasifikasikan jamur sebagai beracun (Beracun) atau dapat dimakan (Tidak Beracun).
    Unggah gambar atau ambil foto untuk mendapatkan prediksi.
    """)
    
    @st.cache_resource
    def load_model_components():
        try:
            return load_components()
        except FileNotFoundError as e:
            st.error(f"Error saat memuat komponen model: {str(e)}")
            st.stop()

    with st.spinner("Memuat model..."):
        components = load_model_components()
        model = components['model']
        scaler = components['scaler']
        pca = components['pca']
        label_encoder = components['label_encoder']
    
    tab1, tab2, tab3 = st.tabs(["Unggah Gambar", "Ambil Foto", "Peforma Model"])
    
    def process_image(image):
        if isinstance(image, np.ndarray):
            img = image
        else:
            file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        features = extract_features(img)
        
        features = features.reshape(1, -1)
        
        features_scaled = scaler.transform(features)
        
        features_pca = pca.transform(features_scaled)
        
        pred = model.predict(features_pca)
        pred_label = label_encoder.inverse_transform(pred)
        
        prob = model.predict_proba(features_pca)
        confidence = np.max(prob)
        
        return img, pred_label[0], confidence
    
    def display_results(img, prediction, confidence):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.image(img_rgb, caption="Gambar yang Dianalisis", use_container_width=True)
        
        with col2:
            st.subheader("Hasil Klasifikasi:")
            
            if prediction == "Beracun":
                st.error(f"**Prediksi:** {prediction}")
                st.warning("⚠️ Jamur ini tampaknya BERACUN! Jangan dikonsumsi!")
            else:
                st.success(f"**Prediksi:** {prediction}")
                st.success("✅ Jamur ini tampaknya DAPAT DIMAKAN")
            
            st.metric("Tingkat Kepercayaan", f"{confidence*100:.2f}%")
            
            st.info("""
            **PERHATIAN:** Ini hanya prediksi AI dan tidak boleh menjadi satu-satunya dasar 
            untuk menentukan apakah jamur aman untuk dimakan. Selalu konsultasikan dengan ahli 
            jamur sebelum mengonsumsinya.
            """)
    
    with tab1:
        uploaded_file = st.file_uploader("Pilih gambar jamur...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            with st.spinner("Menganalisis gambar..."):
                img, prediction, confidence = process_image(uploaded_file)

            display_results(img, prediction, confidence)

    with tab2:
        camera_input = st.camera_input("Ambil foto jamur")
        
        if camera_input is not None:
            with st.spinner("Menganalisis gambar..."):
                img, prediction, confidence = process_image(camera_input)
            
            display_results(img, prediction, confidence)
    with tab3:
            st.header("Performa Model")
            st.write("Berikut adalah visualisasi performa model klasifikasi jamur:")
            
            cm_path = "confusion_matrix.png"
            cr_path = "classification_report.png"
            
            col1, col2 = st.columns(2)
            
            if os.path.exists(cm_path):
                with col1:
                    st.subheader("Confusion Matrix")
                    st.image(cm_path, caption="Confusion Matrix", use_container_width=True)
            else:
                with col1:
                    st.subheader("Confusion Matrix")
                    st.warning("File confusion matrix tidak ditemukan. Jalankan pelatihan model terlebih dahulu dengan menjalankan `python train.py`.")
            
            if os.path.exists(cr_path):
                with col2:
                    st.subheader("Laporan Klasifikasi")
                    st.image(cr_path, caption="Laporan Klasifikasi", use_container_width=True)
            else:
                with col2:
                    st.subheader("Laporan Klasifikasi")
                    st.warning("File laporan klasifikasi tidak ditemukan. Jalankan pelatihan model terlebih dahulu dengan menjalankan `python train.py`.")
            
            with st.expander("Penjelasan Metrik"):
                st.write("""
                ### Penjelasan Metrik Evaluasi
                
                **Confusion Matrix:**
                - Menunjukkan jumlah prediksi yang benar dan salah untuk setiap kelas
                - Baris menunjukkan kelas sebenarnya, kolom menunjukkan kelas yang diprediksi
                - Diagonal utama menunjukkan prediksi yang benar
                
                **Laporan Klasifikasi:**
                - **Precision:** Persentase prediksi positif yang benar (TP/(TP+FP))
                - **Recall:** Persentase kasus positif yang terdeteksi (TP/(TP+FN))
                - **F1-Score:** Rata-rata harmonik dari precision dan recall
                
                TP = True Positive, FP = False Positive, FN = False Negative
                """)

    with st.expander("Tentang Model"):
        st.write("""
        ### Informasi Model
        
        Aplikasi ini menggunakan klasifikasi Support Vector Machine (SVM) dengan fitur-fitur berikut:
        
        - **Ekstraksi Fitur:** Histogram warna, Histogram of Oriented Gradients (HOG), dan Gray-Level Co-occurrence Matrix (GLCM)
        - **Pra-pemrosesan:** StandardScaler dan PCA untuk reduksi dimensi
        - **Model:** SVM dengan kernel RBF
        - **Akurasi:** Sekitar 85% pada set pengujian
        
        ### Dataset
        
        Model dilatih pada dataset gambar jamur yang diklasifikasikan sebagai:
        - **Beracun:** Jamur beracun
        - **Tidak Beracun:** Jamur yang dapat dimakan
        """)

if __name__ == "__main__":
    main() 