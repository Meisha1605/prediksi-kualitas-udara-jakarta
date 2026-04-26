# ============================================================
# APLIKASI PREDIKSI KUALITAS UDARA DKI JAKARTA
# VERSI PROFESIONAL - INPUT WAKTU DI MAIN CONTENT
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# KONFIGURASI HALAMAN (WAJIB PALING ATAS)
# ============================================================
st.set_page_config(
    page_title="AirVision DKI - Prediksi Kualitas Udara",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS (Styling Profesional)
# ============================================================
st.markdown("""
<style>
    /* Warna utama */
    :root {
        --primary: #1a237e;
        --primary-light: #283593;
        --success: #2e7d32;
        --warning: #ff8f00;
        --danger: #c62828;
        --info: #0288d1;
    }
    
    /* Judul utama */
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
        padding: 1.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        font-size: 1rem;
        opacity: 0.9;
        margin-bottom: 0;
    }
    
    /* Kartu metrik */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #5f6368;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0 0.25rem 0;
    }
    
    .metric-unit {
        font-size: 0.8rem;
        color: #9e9e9e;
    }
    
    /* Container hasil prediksi */
    .prediction-container {
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
        transition: all 0.3s ease;
    }
    
    .prediction-badge {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    .prediction-text {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .prediction-recommendation {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 2rem;
        border-top: 1px solid #e0e0e0;
        font-size: 0.8rem;
        color: #9e9e9e;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Tombol prediksi */
    .stButton > button {
        background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(26,35,126,0.3);
    }
    
    /* Expander styling di sidebar */
    .streamlit-expanderHeader {
        background-color: #f1f3f4;
        border-radius: 12px;
        font-weight: 600;
    }
    
    /* Card untuk input waktu */
    .time-card {
        background: #e8f0fe;
        border-radius: 16px;
        padding: 1rem;
        border: 1px solid #d0e0ff;
        height: 100%;
    }
    
    .time-card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #1a237e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.75rem;
    }
    
    .time-card-value {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1a237e;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("decision_tree_model.pkl")
        le = joblib.load("label_encoder.pkl")
        features = joblib.load("feature_columns.pkl")
        return model, le, features
    except FileNotFoundError:
        st.error("❌ File model tidak ditemukan. Pastikan file .pkl ada di folder yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model, le, feature_columns = load_model()

# ============================================================
# HEADER UTAMA
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🌿 AirVision DKI Jakarta</h1>
    <p>Sistem Prediksi Kualitas Udara Real-Time | Berbasis AI Decision Tree</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR - INPUT POLUTAN SAJA (TANPA WAKTU)
# ============================================================
with st.sidebar:
    st.markdown("## 🎛️ Panel Kontrol")
    st.markdown("---")
    
    # Expander untuk parameter polutan
    with st.expander("🌫️ Parameter Polutan", expanded=True):
        st.markdown("**Masukkan nilai konsentrasi polutan:**")
        st.markdown("---")
        
        pm10 = st.slider(
            "PM10 (µg/m³)", 
            min_value=0.0, 
            max_value=250.0, 
            value=50.0,
            step=5.0,
            help="Partikulat kasar (diameter < 10 µm)"
        )
        
        pm25 = st.slider(
            "PM2.5 (µg/m³)", 
            min_value=0.0, 
            max_value=300.0, 
            value=50.0,
            step=5.0,
            help="⚠️ Partikulat halus - PALING BERBAHAYA bagi kesehatan"
        )
        
        so2 = st.slider(
            "SO₂ (µg/m³)", 
            min_value=0.0, 
            max_value=100.0, 
            value=20.0,
            step=2.0,
            help="Sulfur Dioksida - berasal dari pembakaran bahan bakar"
        )
        
        co = st.slider(
            "CO (µg/m³)", 
            min_value=0.0, 
            max_value=50.0, 
            value=10.0,
            step=1.0,
            help="Karbon Monoksida - gas tidak berwarna dan beracun"
        )
        
        o3 = st.slider(
            "O₃ (µg/m³)", 
            min_value=0.0, 
            max_value=200.0, 
            value=30.0,
            step=5.0,
            help="Ozon - polutan sekunder dari reaksi kimia"
        )
        
        no2 = st.slider(
            "NO₂ (µg/m³)", 
            min_value=0.0, 
            max_value=150.0, 
            value=25.0,
            step=5.0,
            help="Nitrogen Dioksida - dari kendaraan bermotor dan industri"
        )
    
    st.markdown("---")
    st.caption("📊 Model Decision Tree • Akurasi 99.13%")
    st.caption("📌 Data ISPU DKI Jakarta 2023")

# ============================================================
# KONTEN UTAMA - METRIC CARDS (3x2 Grid)
# ============================================================
st.markdown("## 📈 Monitoring Polutan Real-Time")

# Kolom 3x2 untuk metric cards
col1, col2, col3, col4, col5, col6 = st.columns(6)

# Batas ambang standar kualitas udara
thresholds = {
    "PM10": (50, "Baik", "green"),
    "PM2.5": (25, "Sedang", "orange"),
    "SO2": (50, "Baik", "green"),
    "CO": (10, "Baik", "green"),
    "O3": (100, "Baik", "green"),
    "NO2": (40, "Baik", "green")
}

with col1:
    delta_html = f"▲ {pm10 - thresholds['PM10'][0]:.1f}" if pm10 > thresholds['PM10'][0] else f"▼ {thresholds['PM10'][0] - pm10:.1f}"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🌫️ PM10</div>
        <div class="metric-value">{pm10:.0f}</div>
        <div class="metric-unit">µg/m³</div>
        <div style="font-size:0.7rem; color:{'#c62828' if pm10 > thresholds['PM10'][0] else '#2e7d32'}">{delta_html} dari standar ({thresholds['PM10'][0]} µg/m³)</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    color = "#c62828" if pm25 > 55 else ("#ff8f00" if pm25 > 25 else "#2e7d32")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">⚠️ PM2.5</div>
        <div class="metric-value" style="color:{color}">{pm25:.0f}</div>
        <div class="metric-unit">µg/m³</div>
        <div style="font-size:0.7rem;">{'⚠️ KRITIS' if pm25 > 55 else ('⚠️ MODERAT' if pm25 > 25 else '✅ AMAN')}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🏭 SO₂</div>
        <div class="metric-value">{so2:.0f}</div>
        <div class="metric-unit">µg/m³</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🚗 CO</div>
        <div class="metric-value">{co:.0f}</div>
        <div class="metric-unit">µg/m³</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">☀️ O₃</div>
        <div class="metric-value">{o3:.0f}</div>
        <div class="metric-unit">µg/m³</div>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🏭 NO₂</div>
        <div class="metric-value">{no2:.0f}</div>
        <div class="metric-unit">µg/m³</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# GRAFIK PERBANDINGAN (BAR CHART dengan Plotly)
# ============================================================
st.markdown("## 📊 Perbandingan Polutan vs Standar Kualitas Udara")

# Data untuk grafik
pollutants = ['PM10', 'PM2.5', 'SO₂', 'CO', 'O₃', 'NO₂']
current_values = [pm10, pm25, so2, co, o3, no2]
standards = [50, 25, 50, 10, 100, 40]

# Warna bar berdasarkan perbandingan
colors = ['#c62828' if curr > std else '#2e7d32' if curr <= std/2 else '#ff8f00' 
          for curr, std in zip(current_values, standards)]

fig = go.Figure(data=[
    go.Bar(name='Nilai Saat Ini', x=pollutants, y=current_values, 
           marker_color=colors, text=current_values, textposition='auto',
           texttemplate='%{text:.0f} µg/m³', 
           hovertemplate='Polutan: %{x}<br>Nilai: %{y:.0f} µg/m³<extra></extra>'),
    go.Bar(name='Batas Standar', x=pollutants, y=standards, 
           marker_color='#9e9e9e', opacity=0.5,
           text=standards, textposition='inside',
           texttemplate='Std: %{text:.0f}', 
           hovertemplate='Polutan: %{x}<br>Batas Standar: %{y:.0f} µg/m³<extra></extra>')
])

fig.update_layout(
    title="Perbandingan Konsentrasi Polutan dengan Batas Ambang Standar",
    xaxis_title="Parameter Polutan",
    yaxis_title="Konsentrasi (µg/m³)",
    barmode='group',
    height=450,
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(family="Arial, sans-serif", size=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

fig.update_traces(
    textfont=dict(size=11, color='white'),
    marker=dict(line=dict(width=1, color='white'))
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ============================================================
# KLASIFIKASI KUALITAS UDARA (DENGAN INPUT WAKTU DI SINI)
# ============================================================
st.markdown("## 🎯 Klasifikasi Kualitas Udara")

# ============================================================
# INPUT WAKTU DI MAIN CONTENT (3 KOLOM SIMETRIS)
# ============================================================

# Inisialisasi variabel default (akan diisi ulang oleh input user)
bulan = 12  # default Desember
hari = 15   # default tanggal 15
is_weekend = 0  # default weekday

# Kolom untuk input waktu
time_col1, time_col2, time_col3 = st.columns(3)

with time_col1:
    st.markdown("""
    <div class="time-card">
        <div class="time-card-title">📅 KALENDER</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Popover untuk pengaturan waktu
    with st.popover("📆 Atur Tanggal & Bulan"):
        bulan = st.selectbox(
            "Bulan", 
            options=list(range(1, 13)),
            format_func=lambda x: ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Ags", "Sep", "Okt", "Nov", "Des"][x-1],
            index=11,
            help="Pilih bulan pengukuran"
        )
        
        hari = st.number_input(
            "Tanggal", 
            min_value=1, 
            max_value=31, 
            value=15,
            help="Masukkan tanggal (1-31)"
        )
    
    # Tampilkan ringkasan pilihan di luar popover
    bulan_nama = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Ags", "Sep", "Okt", "Nov", "Des"][bulan-1]
    st.markdown(f"""
    <div style="background:#1a237e10; border-radius:12px; padding:0.75rem; margin-top:0.5rem; text-align:center;">
        <span style="font-size:0.85rem; color:#1a237e;">📅 {bulan_nama} {hari}</span>
    </div>
    """, unsafe_allow_html=True)

with time_col2:
    st.markdown("""
    <div class="time-card">
        <div class="time-card-title">📆 TIPE HARI</div>
    </div>
    """, unsafe_allow_html=True)
    
    hari_options = {
        0: "📆 Hari Kerja (Senin-Jumat)",
        1: "🌙 Akhir Pekan (Sabtu-Minggu)"
    }
    
    is_weekend = st.selectbox(
        "Tipe Hari",
        options=[0, 1],
        format_func=lambda x: hari_options[x],
        label_visibility="collapsed",
        help="Pilih apakah hari ini adalah hari kerja atau akhir pekan"
    )
    
    # Tampilkan ikon yang dipilih
    weekend_icon = "📆" if is_weekend == 0 else "🌙"
    weekend_text = "Hari Kerja" if is_weekend == 0 else "Akhir Pekan"
    st.markdown(f"""
    <div style="background:#1a237e10; border-radius:12px; padding:0.75rem; margin-top:0.5rem; text-align:center;">
        <span style="font-size:0.85rem; color:#1a237e;">{weekend_icon} {weekend_text}</span>
    </div>
    """, unsafe_allow_html=True)

with time_col3:
    st.markdown("""
    <div class="time-card">
        <div class="time-card-title">🤖 TENTANG MODEL</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="background:#1a237e10; border-radius:12px; padding:0.75rem; margin-top:0.5rem; text-align:center;">
        <div style="font-size:0.85rem; color:#1a237e;">
            <strong>Decision Tree</strong><br>
            Akurasi: 99.13%<br>
            F1-Score: 0.99<br>
            Data: ISPU DKI 2023
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# TOMBOL PREDIKSI
# ============================================================

# Tampilan ringkasan input sebelum prediksi (opsional)
with st.container():
    st.markdown("### 📋 Ringkasan Input")
    sum_col1, sum_col2, sum_col3 = st.columns(3)
    with sum_col1:
        st.info(f"🌫️ **PM10:** {pm10} µg/m³ | **PM2.5:** {pm25} µg/m³")
    with sum_col2:
        st.info(f"🏭 **SO₂:** {so2} µg/m³ | **CO:** {co} µg/m³ | **O₃:** {o3} µg/m³ | **NO₂:** {no2} µg/m³")
    with sum_col3:
        bulan_nama = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Ags", "Sep", "Okt", "Nov", "Des"][bulan-1]
        st.info(f"📅 **Waktu:** {bulan_nama} {hari} | {'Akhir Pekan' if is_weekend == 1 else 'Hari Kerja'}")

# Tombol prediksi
if st.button("🔮 PREDIKSI SEKARANG", type="primary", use_container_width=True):
    # Buat input dataframe
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
    
    # Pastikan kolom sesuai dengan yang digunakan saat training
    input_data = input_data[feature_columns]
    
    # Prediksi
    prediction = model.predict(input_data)[0]
    kategori = le.inverse_transform([prediction])[0]
    
    # Konfigurasi warna dan rekomendasi berdasarkan hasil
    if kategori == "BAIK":
        bg_color = "#e8f5e9"
        border_color = "#2e7d32"
        emoji = "🌿"
        rekomendasi = "✨ AMAN! Udara segar dan bersih. Beraktivitas di luar ruangan dengan nyaman. ✨"
    elif kategori == "SEDANG":
        bg_color = "#fff8e1"
        border_color = "#ff8f00"
        emoji = "⚠️"
        rekomendasi = "⚠️ MODERAT: Kelompok sensitif (anak-anak, lansia, penderita asma) harap kurangi aktivitas luar ruangan yang lama."
    elif kategori == "TIDAK SEHAT":
        bg_color = "#ffebee"
        border_color = "#c62828"
        emoji = "😷"
        rekomendasi = "😷 TIDAK SEHAT: Gunakan masker N95 saat keluar rumah. Hindari aktivitas fisik berat di luar ruangan."
    elif kategori == "SANGAT TIDAK SEHAT":
        bg_color = "#ffcccc"
        border_color = "#8B0000"
        emoji = "🚫"
        rekomendasi = "🚫 BERBAHAYA! TETAP DI DALAM RUMAH. Tutup semua jendela. Gunakan air purifier. Jangan keluar kecuali darurat."
    else:
        bg_color = "#f5f5f5"
        border_color = "#9e9e9e"
        emoji = "❓"
        rekomendasi = "Tidak dapat menentukan kategori."
    
    # Tampilkan hasil prediksi
    st.markdown(f"""
    <div class="prediction-container" style="background: {bg_color}; border: 2px solid {border_color};">
        <div class="prediction-badge">{emoji}</div>
        <div class="prediction-text" style="color: {border_color};">KUALITAS UDARA: {kategori}</div>
        <div class="prediction-recommendation">{rekomendasi}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tampilkan detail probabilitas di expander
    with st.expander("📊 Detail Probabilitas Prediksi", expanded=False):
        proba = model.predict_proba(input_data)[0]
        col_prob1, col_prob2, col_prob3, col_prob4 = st.columns(4)
        classes = le.classes_
        
        # Warna berdasarkan probabilitas
        for i, (col, kelas) in enumerate(zip([col_prob1, col_prob2, col_prob3, col_prob4], classes)):
            persen = proba[i] * 100
            color_prob = "#2e7d32" if persen > 70 else ("#ff8f00" if persen > 30 else "#9e9e9e")
            with col:
                st.metric(kelas, f"{persen:.1f}%", delta=None)
                st.caption(f"Probabilitas: {persen:.1f}%")
    
    # Tampilkan info input yang digunakan
    with st.expander("📥 Data Input yang Digunakan", expanded=False):
        st.dataframe(input_data, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    <p>🌿 AirVision DKI Jakarta &copy; 2025 | Powered by Decision Tree Algorithm</p>
    <p>📊 Model dilatih dengan data ISPU DKI Jakarta 2023 | Akurasi: 99.13% | Precision: 99.12% | Recall: 99.78%</p>
    <p>📌 Data prediksi bersifat informatif. Untuk kondisi darurat, hubungi Dinas Lingkungan Hidup DKI Jakarta.</p>
</div>
""", unsafe_allow_html=True)