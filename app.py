import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Konfigurasi Halaman
st.set_page_config(
    page_title="SkyCast: Flight Booking Prediction",
    page_icon="✈️",
    layout="wide"
)

# Load Model & Components
@st.cache_resource
def load_artifacts():
    return joblib.load('flight_booking_model.joblib')

try:
    artifacts = load_artifacts()
    model = artifacts['model']
    scaler = artifacts['scaler']
    le_route = artifacts['le_route']
    le_origin = artifacts['le_origin']
    feature_columns = artifacts['feature_columns']
except FileNotFoundError:
    st.error("File 'flight_booking_model.joblib' tidak ditemukan. Pastikan Anda sudah menjalankan script training!")
    st.stop()

# --- HEADER ---
st.title("✈️ SkyCast: Prediksi Penyelesaian Booking")
st.markdown("""
Aplikasi ini menggunakan **Random Forest** untuk memprediksi apakah pelanggan akan menyelesaikan pemesanan tiket pesawat mereka atau tidak.
Silakan masukkan data pelanggan di panel sebelah kiri.
""")

# --- SIDEBAR INPUT ---
st.sidebar.header("📝 Masukkan Data Pelanggan")

def user_input_features():
    # 1. Flight Details
    st.sidebar.subheader("Detail Penerbangan")
    
    # Ambil list unik dari encoder untuk dropdown
    route_options = list(le_route.classes_)
    origin_options = list(le_origin.classes_)
    
    route = st.sidebar.selectbox("Rute Penerbangan", route_options)
    booking_origin = st.sidebar.selectbox("Asal Pemesanan (Negara)", origin_options)
    flight_duration = st.sidebar.number_input("Durasi Penerbangan (Jam)", min_value=0.0, max_value=24.0, value=5.0)
    
    # 2. Timing
    st.sidebar.subheader("Waktu Pemesanan")
    purchase_lead = st.sidebar.number_input("Jarak Pemesanan (Hari sebelum terbang)", min_value=0, value=30)
    length_of_stay = st.sidebar.number_input("Lama Tinggal (Hari)", min_value=0, value=7)
    flight_hour = st.sidebar.slider("Jam Keberangkatan (0-23)", 0, 23, 10)
    flight_day = st.sidebar.selectbox("Hari Keberangkatan", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

    # 3. Passenger Details
    st.sidebar.subheader("Detail Penumpang")
    num_passengers = st.sidebar.number_input("Jumlah Penumpang", min_value=1, max_value=10, value=1)
    sales_channel = st.sidebar.radio("Sales Channel", ["Internet", "Mobile"])
    trip_type = st.sidebar.selectbox("Tipe Perjalanan", ["RoundTrip", "OneWay", "CircleTrip"])
    
    # 4. Add-ons
    st.sidebar.subheader("Layanan Tambahan")
    wants_extra_baggage = st.sidebar.checkbox("Ingin Bagasi Tambahan?")
    wants_preferred_seat = st.sidebar.checkbox("Ingin Memilih Kursi?")
    wants_in_flight_meals = st.sidebar.checkbox("Ingin Makanan di Pesawat?")

    # Kumpulkan data dalam Dictionary
    data = {
        'num_passengers': num_passengers,
        'sales_channel': sales_channel,
        'trip_type': trip_type,
        'purchase_lead': purchase_lead,
        'length_of_stay': length_of_stay,
        'flight_hour': flight_hour,
        'flight_day': flight_day,
        'route': route,
        'booking_origin': booking_origin,
        'wants_extra_baggage': 1 if wants_extra_baggage else 0,
        'wants_preferred_seat': 1 if wants_preferred_seat else 0,
        'wants_in_flight_meals': 1 if wants_in_flight_meals else 0,
        'flight_duration': flight_duration
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# --- MAIN PANEL: PREPROCESSING & PREDICTION ---
st.subheader("📊 Review Data Input")
st.dataframe(input_df)

if st.button("🔍 Prediksi Sekarang"):
    
    # 1. Feature Engineering: is_weekend & Mapping Day
    weekend_days = ["Sat", "Sun"]
    input_df['is_weekend'] = input_df['flight_day'].apply(lambda x: 1 if x in weekend_days else 0)
    
    day_mapping = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7}
    input_df['flight_day'] = input_df['flight_day'].map(day_mapping)
    
    # 2. Label Encoding (Route & Origin)
    # Note: Kita gunakan transform. Jika ada data baru yang tidak dikenal saat training,
    # idealnya ada handle unknown, tapi untuk demo ini kita asumsikan user memilih dari dropdown yang tersedia.
    input_df['route'] = le_route.transform(input_df['route'])
    input_df['booking_origin'] = le_origin.transform(input_df['booking_origin'])
    
    # 3. One-Hot Encoding (Manual Adjustments)
    # Kita harus memastikan kolom hasil One-Hot sama persis dengan X_train
    input_df = pd.get_dummies(input_df, columns=['sales_channel', 'trip_type'], drop_first=True)
    
    # *CRITICAL STEP*: Reindex columns
    # Memastikan semua kolom yang ada di model training juga ada di data input
    # (Misal: User pilih 'Internet', maka kolom 'sales_channel_Mobile' tidak akan terbentuk otomatis, kita harus paksa buat 0)
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    
    # 4. Scaling
    input_df_scaled = scaler.transform(input_df)
    
    # 5. Prediction
    prediction = model.predict(input_df_scaled)
    prediction_proba = model.predict_proba(input_df_scaled)
    
    # --- RESULT DISPLAY ---
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hasil Prediksi")
        if prediction[0] == 1:
            st.success("✅ **Booking Complete (Berhasil)**")
            st.write("Pelanggan ini diprediksi akan menyelesaikan pembayaran.")
        else:
            st.error("❌ **Booking Incomplete (Gagal)**")
            st.write("Pelanggan ini diprediksi TIDAK akan menyelesaikan pembayaran.")

    with col2:
        st.subheader("Probabilitas")
        prob_success = prediction_proba[0][1] * 100
        st.progress(int(prob_success))
        st.write(f"Kemungkinan Booking Selesai: **{prob_success:.2f}%**")

    # Insight Tambahan
    if prediction[0] == 0:
        st.info("💡 **Saran:** Tawarkan diskon bagasi atau makanan untuk meningkatkan minat pelanggan ini.")
