import streamlit as st
import cv2
import numpy as np

# --- KONFIGURASI HALAMAN UTAMA (WAJIB DI BARIS PERTAMA) ---
st.set_page_config(
    page_title="PCA Face Recognition",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

IMG_SIZE = (100, 100)

# --- FUNGSI ALJABAR LINEAR MURNI ---
def apply_manual_svd(X_matrix, k_components=50):
    mean_face = np.mean(X_matrix, axis=0)
    X_centered = X_matrix - mean_face
    U, Sigma, V_transpose = np.linalg.svd(X_centered, full_matrices=False)
    eigenvectors_Vk = V_transpose[:k_components, :].T
    return mean_face, eigenvectors_Vk, Sigma

def project_to_eigenspace(wajah_vector, mean_face, eigenvectors_Vk):
    wajah_centered = wajah_vector - mean_face
    return np.dot(wajah_centered, eigenvectors_Vk)

def calculate_cosine_similarity(z1, z2):
    dot_product = np.dot(z1, z2)
    norm_z1 = np.linalg.norm(z1)
    norm_z2 = np.linalg.norm(z2)
    if norm_z1 == 0 or norm_z2 == 0:
        return 0.0
    return dot_product / (norm_z1 * norm_z2)

def decode_streamlit_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    normalized = resized / 255.0
    return normalized.flatten(), cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Kembalikan vektor dan gambar RGB untuk display

# --- ARSITEKTUR UI/UX ---

# 1. PANEL SAMPING (SIDEBAR): Manajemen Dataset
with st.sidebar:
    st.title("⚙️ Konfigurasi Ruang Vektor")
    st.markdown("Unggah dataset wajah untuk membangun *Eigenfaces*.")
    
    dataset_files = st.file_uploader(
        "Upload gambar dataset (Min. 2 file)", 
        accept_multiple_files=True, 
        key="dataset"
    )
    
    k_dim = st.slider("Jumlah Komponen Utama (k)", min_value=1, max_value=100, value=50, step=1)
    threshold = st.slider("Batas Threshold Kemiripan", min_value=0.0, max_value=1.0, value=0.80, step=0.01)

# 2. AREA UTAMA (MAIN CANVAS): Pengujian
st.title("🔬 Deteksi Kemiripan Wajah Berbasis Aljabar Linear")
st.markdown("Implementasi dekomposisi matriks murni tanpa modul komputasi tingkat tinggi (Non-Sklearn).")

# Logika State Management
if dataset_files and len(dataset_files) >= 2:
    # Membangun Matriks
    X_list = []
    labels = []
    for file in dataset_files:
        label = file.name.split('_')[0].lower()
        vector, _ = decode_streamlit_image(file)
        if vector is not None:
            X_list.append(vector)
            labels.append(label)
    
    X_matrix = np.array(X_list)
    
    # Komputasi SVD
    mean_face, eigenvectors_Vk, singular_values = apply_manual_svd(X_matrix, k_components=k_dim)
    
    # UI Transparansi Akademik (Expander)
    with st.expander("📊 Logika Komputasi & Dimensi Matriks (White-Box)"):
        st.markdown("**Status Memori Aljabar Linear:**")
        st.code(f"Dimensi Matriks Dataset (X): {X_matrix.shape}\n"
                f"Ukuran Ruang Komponen (V_k): {eigenvectors_Vk.shape}\n"
                f"Tiga Nilai Singular Teratas: {np.round(singular_values[:3], 2)}", language="text")
        st.caption("Nilai singular merepresentasikan seberapa besar varians yang ditangkap oleh masing-masing *eigenface*.")

    st.divider()

    # Layout Uji Kemiripan (2 Kolom Utama)
    st.header("Fase Uji Kemiripan 1:1")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Wajah A")
        img_a = st.file_uploader("Unggah gambar subjek uji pertama", key="img_a")
    
    with col2:
        st.subheader("Wajah B")
        img_b = st.file_uploader("Unggah gambar subjek uji pembanding", key="img_b")
        
    if img_a and img_b:
        vec_1, rgb_1 = decode_streamlit_image(img_a)
        vec_2, rgb_2 = decode_streamlit_image(img_b)
        
        # Proyeksi & Evaluasi
        z1 = project_to_eigenspace(vec_1, mean_face, eigenvectors_Vk)
        z2 = project_to_eigenspace(vec_2, mean_face, eigenvectors_Vk)
        score = calculate_cosine_similarity(z1, z2)
        
        st.divider()
        st.subheader("Hasil Proyeksi Ruang PCA")
        
        # Dashboard Hasil
        res_col1, res_col2, res_col3 = st.columns([1, 2, 1])
        
        with res_col1:
            st.image(rgb_1, caption="Wajah A", use_column_width=True)
            
        with res_col2:
            # Menggunakan Metric Widget untuk menegaskan angka
            st.metric(label="Cosine Similarity Score", value=f"{score:.4f}", 
                      delta="Mirip" if score >= threshold else "Berbeda", 
                      delta_color="normal" if score >= threshold else "inverse")
            
            if score >= threshold:
                st.success("✅ **KESIMPULAN:** Struktur geometri wajah memiliki korelasi tinggi.")
            else:
                st.error("❌ **KESIMPULAN:** Proyeksi matriks menolak adanya kemiripan identitas.")
                
        with res_col3:
            st.image(rgb_2, caption="Wajah B", use_column_width=True)

else:
    # State Awal jika dataset belum diunggah
    st.info("👈 **Sistem Menunggu Konfigurasi.** Buka panel di sebelah kiri dan unggah minimal 2 gambar wajah untuk membangun matriks dataset dasar.")
    st.image("https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80", caption="Menunggu inisialisasi arsitektur matriks...", use_column_width=False, width=500)