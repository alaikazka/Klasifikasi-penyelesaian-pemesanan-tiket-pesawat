import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Prediksi Penyelesaian Pemesanan Tiket Pesawat",
    layout="wide"
)

# Load Model=
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

st.title("Prediksi Penyelesaian Booking")
st.markdown("""
prediksi ini menggunakan algoritma **Random Forest** untuk memprediksi apakah pelanggan akan menyelesaikan pemesanan tiket pesawat mereka atau tidak.
Silakan input data pelanggan di panel sebelah kiri.
""")


st.sidebar.header("Input Data Pelanggan")

def user_input_features():
    st.sidebar.subheader("Detail Penerbangan")
    
    route_options = list(le_route.classes_)
    origin_options = list(le_origin.classes_)
    
    route = st.sidebar.selectbox("Rute Penerbangan", route_options)
    booking_origin = st.sidebar.selectbox("Asal Pemesanan (Negara)", origin_options)
    flight_duration = st.sidebar.number_input("Durasi Penerbangan (Jam)", min_value=0.0, max_value=24.0, value=5.0)
    
    st.sidebar.subheader("Waktu Pemesanan")
    purchase_lead = st.sidebar.number_input("Jarak Pemesanan (Hari sebelum terbang)", min_value=0, value=30)
    length_of_stay = st.sidebar.number_input("Lama Tinggal (Hari)", min_value=0, value=7)
    flight_hour = st.sidebar.slider("Jam Keberangkatan (0-23)", 0, 23, 10)
    flight_day = st.sidebar.selectbox("Hari Keberangkatan", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

    st.sidebar.subheader("Detail Penumpang")
    num_passengers = st.sidebar.number_input("Jumlah Penumpang", min_value=1, max_value=10, value=1)
    sales_channel = st.sidebar.radio("Sales Channel", ["Internet", "Mobile"])
    trip_type = st.sidebar.selectbox("Tipe Perjalanan", ["RoundTrip", "OneWay", "CircleTrip"])
    
    st.sidebar.subheader("Layanan Tambahan")
    wants_extra_baggage = st.sidebar.checkbox("Ingin Bagasi Tambahan?")
    wants_preferred_seat = st.sidebar.checkbox("Ingin Memilih Kursi?")
    wants_in_flight_meals = st.sidebar.checkbox("Ingin Makanan di Pesawat?")

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

st.subheader("Review Data Input")
st.dataframe(input_df)

if st.button("🔍 Prediksi"):
    
    weekend_days = ["Sat", "Sun"]
    input_df['is_weekend'] = input_df['flight_day'].apply(lambda x: 1 if x in weekend_days else 0)
    
    day_mapping = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7}
    input_df['flight_day'] = input_df['flight_day'].map(day_mapping)
    
    input_df['route'] = le_route.transform(input_df['route'])
    input_df['booking_origin'] = le_origin.transform(input_df['booking_origin'])
    
    input_df = pd.get_dummies(input_df, columns=['sales_channel', 'trip_type'], drop_first=True)
    
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    
    input_df_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_df_scaled)
    prediction_proba = model.predict_proba(input_df_scaled)
    
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hasil Prediksi")
        
        if prob_success > 0.4:
            st.success("✅ **Booking Complete (Berhasil)**")
            st.write("Pelanggan ini diprediksi akan menyelesaikan pembayaran.")
        else:
            st.error("❌ **Booking Incomplete (Gagal)**")
            st.write("Pelanggan ini diprediksi TIDAK akan menyelesaikan pembayaran.")

    with col2:
        st.subheader("Probabilitas")
        st.metric(label="Kemungkinan Booking", value=f"{prob_success*100:.2f}%")
        st.progress(int(prob_success * 100))
        
        if prob_success > 0.4:
            st.caption("Status: High Potential")
        else:
            st.caption("Status: Low Potential")

    if prob_success <= 0.4:
        st.info("💡 **Saran:** Tawarkan diskon bagasi atau makanan untuk meningkatkan minat pelanggan ini.")
