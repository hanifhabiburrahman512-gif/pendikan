import pandas as pd
from io import StringIO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def load_data():
    data = """
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;70.67;ACEH
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;77.16;"SUMATERA UTARA"
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;65.96;"SUMATERA BARAT"
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;66.91;RIAU
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;65.85;JAMBI
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;67.07;"SUMATERA SELATAN"
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;64.88;BENGKULU
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;62.42;LAMPUNG
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;66.87;"KEP. BANGKA BELITUNG"
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;73.93;"KEP. RIAU"
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;87.71;"DKI JAKARTA"
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;67.05;"JAWA BARAT"
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;58.75;"JAWA TENGAH"
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;87.92;"DI YOGYAKARTA"
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;66.87;"JAWA TIMUR"
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;66.02;BANTEN
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;76.59;BALI
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;61;"NUSA TENGGARA BARAT"
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;38.47;"NUSA TENGGARA TIMUR"
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;58.4;"KALIMANTAN BARAT"
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;61.88;"KALIMANTAN TENGAH"
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;67.81;"KALIMANTAN SELATAN"
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;74;"KALIMANTAN TIMUR"
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;54.8;"KALIMANTAN UTARA"
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;66.66;"SULAWESI UTARA"
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;53.73;"SULAWESI TENGAH"
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;68.32;"SULAWESI SELATAN"
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;65.97;"SULAWESI TENGGARA"
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;45.12;GORONTALO
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;55.18;"SULAWESI BARAT"
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;72.08;MALUKU
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;67.1;"MALUKU UTARA"
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;57.07;"PAPUA BARAT"
Tingkat Penyelesaian Pendidikan Menurut Jenjang Pendidikan dan Provinsi;"SMA / Sederajat";2022;Tahun;39.01;PAPUA
"""
    df = pd.read_csv(
        StringIO(data),
        sep=';',
        names=['Indikator', 'Jenjang', 'Tahun', 'Satuan', 'Nilai', 'Provinsi']
    )

    df['Nilai'] = pd.to_numeric(df['Nilai'], errors='coerce')
    df = df.dropna(subset=['Nilai']).reset_index(drop=True)
    return df

def kmeans_clustering(df):
    X = df[['Nilai']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scores = []
    k_range = range(2, 7)

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        scores.append(silhouette_score(X_scaled, labels))

    optimal_k = list(k_range)[scores.index(max(scores))]

    model = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
    df['Cluster'] = model.fit_predict(X_scaled)

    summary = df.groupby('Cluster')['Nilai'].agg(
        Jumlah_Provinsi='count',
        Rata_rata='mean',
        Minimum='min',
        Maksimum='max'
    ).round(2)

    return df, summary, optimal_k, model, scaler
