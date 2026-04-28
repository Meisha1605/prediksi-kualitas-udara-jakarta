# ============================================================
# APLIKASI PREDIKSI KUALITAS UDARA DKI JAKARTA
# ALGORITMA: RANDOM FOREST (Ensemble Learning)
# VERSI: PROFESSIONAL DASHBOARD dengan TAB METODOLOGI
# ============================================================
# 
# Nama Proyek    : Analisis Klasifikasi Kualitas Udara di DKI Jakarta
# Algoritma      : Random Forest
# Akurasi        : 99.13%
# Precision      : 99.12%
# Recall         : 99.78%
# F1-Score       : 0.99
# 
# Kelompok       : 13
# Tema           : Smart Environment
# Mata Kuliah    : Data Mining
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="AirVision DKI - Prediksi Kualitas Udara",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS (Palet Warna Modern + ISPU Color Code)
# ============================================================
st.markdown("""
<style>
    /* Font & Warna Utama - Modern Professional */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Warna Standar ISPU */
    :root {
        --ispu-baik: #00AA44;
        --ispu-sedang: #FFD700;
        --ispu-tidak-sehat: #FF6600;
        --ispu-sangat-tidak-sehat: #FF0000;
        --ispu-berbahaya: #800080;
        --primary-dark: #0B4F6C;
        --primary-teal: #1A9C7A;
        --bg-light: #F8FAFC;
        --text-dark: #1E293B;
        --text-gray: #64748B;
    }
    
    /* Header Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #0B4F6C 0%, #1A9C7A 100%);
        border-radius: 24px;
        padding: 2rem 2rem;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1);
    }
    
    .hero-section h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }
    
    .hero-section p {
        font-size: 1rem;
        opacity: 0.95;
        margin-bottom: 1rem;
        line-height: 1.5;
    }
    
    .info-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(5px);
        border-radius: 40px;
        padding: 0.4rem 1rem;
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    /* Kartu Metrik Polutan */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .metric-card {
        background: white;
        border-radius: 20px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border: 1px solid #E2E8F0;
        transition: all 0.2s ease;
        text-align: center;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 20px -12px rgba(0,0,0,0.1);
        border-color: #1A9C7A;
    }
    
    .metric-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748B;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }
    
    .metric-unit {
        font-size: 0.7rem;
        color: #94A3B8;
    }
    
    .metric-status {
        font-size: 0.65rem;
        margin-top: 0.5rem;
        padding: 2px 8px;
        border-radius: 20px;
        display: inline-block;
    }
    
    /* Warna Status ISPU */
    .status-baik { background: #DCFCE7; color: #166534; }
    .status-sedang { background: #FEF9C3; color: #854D0E; }
    .status-tidak-sehat { background: #FFEDD5; color: #9A3412; }
    .status-sangat-tidak-sehat { background: #FEE2E2; color: #991B1B; }
    .status-berbahaya { background: #F3E8FF; color: #5B21B6; }
    
    /* Container Hasil Prediksi */
    .prediction-card {
        border-radius: 24px;
        padding: 1.75rem;
        margin: 1.5rem 0;
        transition: all 0.3s ease;
        animation: fadeInUp 0.5s ease-out;
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Step Connector */
    .step-connector {
        text-align: center;
        padding: 0.5rem 0;
        color: #1A9C7A;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        margin-top: 2rem;
        border-top: 1px solid #E2E8F0;
        font-size: 0.7rem;
        color: #94A3B8;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F1F5F9;
        border-right: none;
    }
    
    .sidebar-instruction {
        background: #E2E8F0;
        padding: 0.75rem;
        border-radius: 12px;
        font-size: 0.8rem;
        color: #0F172A;
        margin-bottom: 1rem;
    }
    
    /* Tombol */
    .stButton > button {
        background: linear-gradient(135deg, #0B4F6C 0%, #1A9C7A 100%);
        color: white;
        border: none;
        border-radius: 40px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(26,156,122,0.3);
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: #F1F5F9;
        border-radius: 12px;
        padding: 0.25rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL RANDOM FOREST
# ============================================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("random_forest_model.pkl")
        le = joblib.load("label_encoder.pkl")
        features = joblib.load("feature_columns.pkl")
        return model, le, features
    except FileNotFoundError as e:
        st.error(f"❌ File model tidak ditemukan: {e}")
        st.info("Pastikan file 'random_forest_model.pkl', 'label_encoder.pkl', dan 'feature_columns.pkl' ada di folder yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model, le, feature_columns = load_model()

# ============================================================
# HERO SECTION (Header dengan Deskripsi Jelas)
# ============================================================
st.markdown("""
<div class="hero-section">
    <h1>🌿 AirVision DKI Jakarta</h1>
    <p>Sistem Prediksi Kualitas Udara Real-Time berbasis <strong>Machine Learning Random Forest</strong>.<br>
    Masukkan parameter konsentrasi polutan, bandingkan dengan standar ambang batas, dan dapatkan prediksi kategori kualitas udara beserta rekomendasi kesehatan.</p>
    <div class="info-badge">
        📊 Akurasi Model: 99.13% | 🤖 Random Forest | 🎯 Precision: 99.12%
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TABS: Dashboard | Metodologi
# ============================================================
tab1, tab2 = st.tabs(["📊 DASHBOARD PREDIKSI", "📖 METODOLOGI & INFO MODEL"])

# ============================================================
# TAB 1: DASHBOARD PREDIKSI
# ============================================================
with tab1:
    # Step 1: Input Polutan (Sidebar)
    st.markdown("### 📍 LANGKAH 1: Masukkan Konsentrasi Polutan")
    st.markdown("> *Sesuaikan slider di sidebar kiri untuk melihat dampak perubahan konsentrasi polutan.*")
    
    # SIDEBAR - Input Polutan Saja
    with st.sidebar:
        st.markdown("## 🎛️ Panel Kontrol")
        st.markdown("---")
        st.markdown("""
        <div class="sidebar-instruction">
        💡 <strong>Petunjuk:</strong><br>
        Geser slider untuk menyesuaikan konsentrasi setiap polutan. Perubahan akan langsung terlihat pada grafik perbandingan dan kartu metrik.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🌫️ Parameter Polutan")
        
        pm10 = st.slider("PM10 (µg/m³)", 0.0, 250.0, 50.0, 5.0, help="Partikulat kasar")
        pm25 = st.slider("PM2.5 (µg/m³)", 0.0, 300.0, 50.0, 5.0, help="Partikulat halus - PALING BERBAHAYA")
        so2 = st.slider("SO₂ (µg/m³)", 0.0, 100.0, 20.0, 2.0)
        co = st.slider("CO (µg/m³)", 0.0, 50.0, 10.0, 1.0)
        o3 = st.slider("O₃ (µg/m³)", 0.0, 200.0, 30.0, 5.0)
        no2 = st.slider("NO₂ (µg/m³)", 0.0, 150.0, 25.0, 5.0)
        
        st.markdown("---")
        st.caption("📌 Data referensi: Batas Standar WHO & ISPU DKI Jakarta 2023")
    
    # Step 2: Visualisasi Perbandingan
    st.markdown("---")
    st.markdown("### 📍 LANGKAH 2: Bandingkan dengan Standar Ambang Batas")
    st.markdown("> *Grafik di bawah menunjukkan perbandingan nilai polutan Anda (warna) terhadap Standar WHO (garis hitam putus-putus).*")
    
    # Data untuk grafik
    pollutants = ['PM10', 'PM2.5', 'SO₂', 'CO', 'O₃', 'NO₂']
    current_values = [pm10, pm25, so2, co, o3, no2]
    standards = [50, 25, 50, 10, 100, 40]
    
    # Fungsi untuk menentukan warna berdasarkan standar ISPU
    def get_ispu_color(polutan, value):
        if polutan == 'PM2.5':
            if value <= 50: return '#00AA44'  # Baik
            elif value <= 100: return '#FFD700'  # Sedang
            elif value <= 150: return '#FF6600'  # Tidak Sehat
            elif value <= 250: return '#FF0000'  # Sangat Tidak Sehat
            else: return '#800080'  # Berbahaya
        else:  # Polutan lain
            if value <= 50: return '#00AA44'
            elif value <= 100: return '#FFD700'
            elif value <= 200: return '#FF6600'
            elif value <= 300: return '#FF0000'
            else: return '#800080'
    
    colors = [get_ispu_color(p, v) for p, v in zip(pollutants, current_values)]
    
    # Buat grafik dengan garis threshold
    fig = go.Figure()
    
    # Bar chart untuk nilai saat ini
    fig.add_trace(go.Bar(
        name='Nilai Saat Ini', 
        x=pollutants, 
        y=current_values, 
        marker_color=colors,
        text=current_values, 
        textposition='auto',
        texttemplate='%{text:.0f} µg/m³'
    ))
    
    # Garis threshold untuk batas standar
    fig.add_trace(go.Scatter(
        name='Batas Standar WHO',
        x=pollutants,
        y=standards,
        mode='lines+markers',
        line=dict(color='#1E293B', width=2, dash='dash'),
        marker=dict(size=8, color='#1E293B'),
        text=standards,
        textposition='top center',
        texttemplate='Standar: %{text:.0f}'
    ))
    
    fig.update_layout(
        title="Perbandingan Konsentrasi Polutan dengan Batas Ambang Standar WHO",
        xaxis_title="Parameter Polutan",
        yaxis_title="Konsentrasi (µg/m³)",
        barmode='group',
        height=450,
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Kartu Metrik Polutan (6 kolom)
    st.markdown("**📋 Ringkasan Nilai Polutan:**")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        status_class = "status-baik" if pm10 <= 50 else ("status-sedang" if pm10 <= 100 else "status-tidak-sehat")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🌫️ PM10</div>
            <div class="metric-value">{pm10:.0f}<span class="metric-unit"> µg/m³</span></div>
            <div class="metric-status {status_class}">{'✓ Baik' if pm10 <= 50 else ('⚠️ Sedang' if pm10 <= 100 else '🔴 Melebihi')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        status_class = "status-baik" if pm25 <= 25 else ("status-sedang" if pm25 <= 55 else "status-tidak-sehat")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⚠️ PM2.5</div>
            <div class="metric-value">{pm25:.0f}<span class="metric-unit"> µg/m³</span></div>
            <div class="metric-status {status_class}">{'✓ Baik' if pm25 <= 25 else ('⚠️ Moderat' if pm25 <= 55 else '🔴 Berbahaya')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        status_class = "status-baik" if so2 <= 50 else "status-tidak-sehat"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🏭 SO₂</div>
            <div class="metric-value">{so2:.0f}<span class="metric-unit"> µg/m³</span></div>
            <div class="metric-status {status_class}">{'✓ Baik' if so2 <= 50 else '🔴 Melebihi'}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        status_class = "status-baik" if co <= 10 else "status-tidak-sehat"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🚗 CO</div>
            <div class="metric-value">{co:.0f}<span class="metric-unit"> µg/m³</span></div>
            <div class="metric-status {status_class}">{'✓ Baik' if co <= 10 else '🔴 Melebihi'}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        status_class = "status-baik" if o3 <= 100 else "status-tidak-sehat"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">☀️ O₃</div>
            <div class="metric-value">{o3:.0f}<span class="metric-unit"> µg/m³</span></div>
            <div class="metric-status {status_class}">{'✓ Baik' if o3 <= 100 else '🔴 Melebihi'}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        status_class = "status-baik" if no2 <= 40 else "status-tidak-sehat"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🏭 NO₂</div>
            <div class="metric-value">{no2:.0f}<span class="metric-unit"> µg/m³</span></div>
            <div class="metric-status {status_class}">{'✓ Baik' if no2 <= 40 else '🔴 Melebihi'}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Step 3: Input Waktu & Prediksi
    st.markdown("---")
    st.markdown("### 📍 LANGKAH 3: Tentukan Waktu & Dapatkan Prediksi")
    st.markdown("> *Lengkapi informasi waktu, lalu klik tombol prediksi untuk mengetahui kategori kualitas udara.*")
    
    # Input Waktu (3 kolom)
    time_col1, time_col2, time_col3, time_col4 = st.columns([2, 2, 2, 1])
    
    with time_col1:
        bulan = st.selectbox("Pilih Bulan", list(range(1,13)), format_func=lambda x: ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Ags","Sep","Okt","Nov","Des"][x-1])
    with time_col2:
        hari = st.number_input("Tanggal", 1, 31, 15)
    with time_col3:
        is_weekend = st.selectbox("Tipe Hari", [0,1], format_func=lambda x: "Hari Kerja" if x==0 else "Akhir Pekan")
    
    bulan_nama = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Ags","Sep","Okt","Nov","Des"][bulan-1]
    
    # Ringkasan Input (Expander)
    with st.expander("📋 Ringkasan Input (Periksa kembali data Anda)", expanded=False):
        st.markdown(f"""
        | Parameter | Nilai | Parameter | Nilai |
        |---|---|---|---|
        | PM10 | {pm10} µg/m³ | O₃ | {o3} µg/m³ |
        | PM2.5 | {pm25} µg/m³ | NO₂ | {no2} µg/m³ |
        | SO₂ | {so2} µg/m³ | Bulan | {bulan_nama} |
        | CO | {co} µg/m³ | Hari | {hari} |
        | **Tipe Hari** | {'Hari Kerja' if is_weekend == 0 else 'Akhir Pekan'} | | |
        """)
    
    # Tombol Prediksi
    if st.button("🔮 PREDIKSI KUALITAS UDARA", type="primary", use_container_width=True):
        # Persiapan data input
        input_data = pd.DataFrame([{
            'pm_sepuluh': pm10,
            'pm_duakomalima': pm25,
            'sulfur_dioksida': so2,
            'karbon_monoksida': co,
            'ozon': o3,
            'nitrogen_dioksida': no2,
            'bulan': bulan,
            'hari': hari,
            'is_weekend': is_weekend
        }])
        
        input_data = input_data[feature_columns]
        
        # Prediksi
        prediction = model.predict(input_data)[0]
        kategori = le.inverse_transform([prediction])[0]
        probabilities = model.predict_proba(input_data)[0]
        prob_max = probabilities[prediction] * 100
        
        # Konfigurasi berdasarkan kategori
        if kategori == "BAIK":
            bg_color = "#DCFCE7"
            border_color = "#00AA44"
            emoji = "🌿"
            rekomendasi = "Kualitas Udara BAIK"
            rekomendasi_aksi = "✅ Udara bersih dan sehat. Anda dapat beraktivitas di luar ruangan dengan nyaman tanpa risiko kesehatan."
        elif kategori == "SEDANG":
            bg_color = "#FEF9C3"
            border_color = "#FFD700"
            emoji = "⚠️"
            rekomendasi = "Kualitas Udara SEDANG"
            rekomendasi_aksi = "⚠️ Kelompok sensitif (anak-anak, lansia, penderita asma) disarankan mengurangi aktivitas luar ruangan yang lama. Gunakan masker jika perlu."
        elif kategori == "TIDAK SEHAT":
            bg_color = "#FFEDD5"
            border_color = "#FF6600"
            emoji = "😷"
            rekomendasi = "Kualitas Udara TIDAK SEHAT"
            rekomendasi_aksi = "😷 Gunakan masker N95 saat keluar rumah. Hindari aktivitas fisik berat di luar ruangan. Tutup jendela rumah."
        else:
            bg_color = "#FEE2E2"
            border_color = "#FF0000"
            emoji = "🚫"
            rekomendasi = "Kualitas Udara SANGAT TIDAK SEHAT"
            rekomendasi_aksi = "🚫 BERBAHAYA! Tetap di dalam rumah, tutup semua jendela, gunakan air purifier. Jangan keluar kecuali darurat."
        
        # Tampilkan hasil prediksi
        st.markdown(f"""
        <div class="prediction-card" style="background:{bg_color}; border-left:6px solid {border_color};">
            <div style="display:flex; align-items:center; gap:1rem;">
                <span style="font-size:3rem;">{emoji}</span>
                <div>
                    <div style="font-size:0.8rem; color:#475569;">HASIL PREDIKSI</div>
                    <div style="font-size:1.8rem; font-weight:700; color:{border_color};">{rekomendasi}</div>
                    <div style="margin-top:0.5rem;">{rekomendasi_aksi}</div>
                    <div style="margin-top:0.75rem; font-size:0.8rem;">
                        🤖 Confidence Level: <strong>{prob_max:.1f}%</strong> (Model yakin dengan prediksi ini)
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Tampilkan probabilitas per kategori
        with st.expander("📊 Detail Probabilitas per Kategori", expanded=False):
            classes = le.classes_
            for kelas, prob in zip(classes, probabilities):
                persen = prob * 100
                st.progress(persen/100, text=f"{kelas}: {persen:.2f}%")

# ============================================================
# TAB 2: METODOLOGI & INFO MODEL
# ============================================================
with tab2:
    st.markdown("## 📖 Metodologi Penelitian")
    st.markdown("---")
    
    st.markdown("""
    ### 🤖 Algoritma: Random Forest
    
    **Random Forest** adalah algoritma machine learning berbasis **ensemble learning** yang menggabungkan banyak pohon keputusan (decision trees) untuk menghasilkan prediksi yang lebih akurat dan stabil.
    
    #### Keunggulan Random Forest:
    - Mengurangi overfitting dibandingkan Decision Tree tunggal
    - Lebih stabil dan robust terhadap outlier
    - Dapat menangani data dengan dimensi tinggi
    - Memberikan feature importance untuk interpretasi model
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Performa Model")
    
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    with perf_col1:
        st.metric("Akurasi", "99.13%", delta=None)
    with perf_col2:
        st.metric("Precision (Weighted)", "99.12%", delta=None)
    with perf_col3:
        st.metric("Recall (Weighted)", "99.78%", delta=None)
    with perf_col4:
        st.metric("F1-Score (Weighted)", "0.99", delta=None)
    
    st.markdown("---")
    st.markdown("### 📂 Dataset")
    st.markdown("""
    - **Sumber Data**: Portal Satu Data Jakarta (satudata.jakarta.go.id)
    - **Periode**: Desember 2022 - November 2023
    - **Jumlah Data Awal**: 1.551 baris
    - **Jumlah Data Setelah Cleaning**: 1.461 baris
    - **Stasiun Pemantau**: 5 stasiun (DKI1-DKI5)
    - **Parameter**: PM10, PM2.5, SO₂, CO, O₃, NO₂
    """)
    
    st.markdown("---")
    st.markdown("### 🔥 Feature Importance (Polutan Paling Berpengaruh)")
    
    # Data feature importance
    imp_data = pd.DataFrame({
        'Parameter': ['PM2.5', 'PM10', 'NO₂', 'SO₂', 'O₃', 'CO', 'Bulan', 'Hari', 'Akhir Pekan'],
        'Importance': [0.32, 0.21, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02, 0.02]
    }).sort_values('Importance', ascending=True)
    
    fig_imp = px.bar(imp_data, x='Importance', y='Parameter', orientation='h',
                     title="Tingkat Pengaruh Parameter terhadap Kualitas Udara",
                     color='Importance', color_continuous_scale='Tealgrn')
    fig_imp.update_layout(height=400, plot_bgcolor='white')
    st.plotly_chart(fig_imp, use_container_width=True)
    
    st.markdown("""
    ### 📌 Kesimpulan
    Berdasarkan analisis menggunakan algoritma **Random Forest**, dapat disimpulkan bahwa:
    
    1. **PM2.5** merupakan polutan paling dominan (32%) dalam menentukan kualitas udara
    2. **PM10** menempati posisi kedua (21%)
    3. **NO₂** menjadi polutan ketiga paling berpengaruh (15%)
    4. Model memiliki akurasi **99.13%** dalam mengklasifikasikan kualitas udara
    """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    <p>🌿 AirVision DKI Jakarta © 2025 | Kelompok 13 - Data Mining | Tema: Smart Environment</p>
    <p>Model Random Forest | Data ISPU DKI Jakarta 2023</p>
</div>
""", unsafe_allow_html=True)