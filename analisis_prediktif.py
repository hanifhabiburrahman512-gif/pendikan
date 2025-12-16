from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def train_model(df):
    X = df[['Nilai']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=df['Cluster'].nunique(), random_state=42)
    model.fit(X_scaled)

    return model, scaler

def prediksi_cluster(model, scaler, nilai):
    nilai_scaled = scaler.transform([[nilai]])
    return model.predict(nilai_scaled)[0]
