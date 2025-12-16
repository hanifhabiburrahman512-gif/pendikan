import streamlit as st
import matplotlib.pyplot as plt
import analisis_deskriptif as ad
import analisis_prediktif as ap

st.set_page_config(
    page_title="Dashboard Analisis Pendidikan SMA",
    layout="wide"
)

st.title("📊 Dashboard Analisis Pendidikan SMA")

menu = st.sidebar.selectbox(
    "Pilih Analisis",
    ["Analisis Deskriptif", "Analisis Prediktif"]
)

df = ad.load_data()
df_cluster, summary, optimal_k, model, scaler = ad.kmeans_clustering(df)

# =====================
# DESKRIPTIF
# =====================
if menu == "Analisis Deskriptif":
    st.header("📊 Analisis Deskriptif")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Data Provinsi")
        st.dataframe(df_cluster)

    with col2:
        st.subheader("Ringkasan Cluster")
        st.dataframe(summary)

    st.subheader("Visualisasi Cluster")
    fig, ax = plt.subplots()
    ax.scatter(df_cluster['Nilai'], df_cluster['Cluster'], s=120)
    ax.set_xlabel("Tingkat Penyelesaian SMA (%)")
    ax.set_ylabel("Cluster")
    st.pyplot(fig)

    st.info(f"Jumlah cluster optimal: {optimal_k}")

# =====================
# PREDIKTIF
# =====================
if menu == "Analisis Prediktif":
    st.header("🤖 Analisis Prediktif")

    nilai = st.slider(
        "Masukkan nilai penyelesaian SMA (%)",
        min_value=30.0,
        max_value=100.0,
        value=65.0,
        step=0.1
    )

    hasil = ap.prediksi_cluster(model, scaler, nilai)

    st.success(
        f"Provinsi dengan nilai **{nilai}%** diprediksi masuk ke **Cluster {hasil}**"
    )
