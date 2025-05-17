import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Klasifikasi Adaptif dengan KNN", layout="wide")
st.title("📊 Aplikasi Klasifikasi Adaptif (KNN)")

# Step 1: Upload Dataset
uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("🔍 Dataset Preview")
        st.dataframe(df.head())

        # Step 2: Pilih Label dan Deteksi Fitur
        object_cols = df.select_dtypes(include='object').columns.tolist()
        if not object_cols:
            st.warning("Tidak ada kolom kategorikal ditemukan. Harap sertakan kolom target berisi label klasifikasi.")
        else:
            label_col = st.selectbox("Pilih kolom sebagai label (target):", object_cols)
            feature_cols = df.drop(columns=[label_col]).select_dtypes(include='number').columns.tolist()

            if len(feature_cols) < 1:
                st.warning("Tidak ditemukan fitur numerik untuk klasifikasi.")
            else:
                # Preprocessing
                X = df[feature_cols]
                y = LabelEncoder().fit_transform(df[label_col])

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

                # KNN Model
                st.subheader("⚙️ Parameter KNN")
                k = st.slider("Pilih jumlah tetangga (k):", 1, 15, 5)
                model = KNeighborsClassifier(n_neighbors=k)
                model.fit(X_train, y_train)

                # Prediction & Evaluation
                y_pred = model.predict(X_test)

                st.subheader("📈 Evaluasi Model")
                st.text(f"Akurasi: {accuracy_score(y_test, y_pred):.2f}")
                st.text("Classification Report:")
                st.text(classification_report(y_test, y_pred))

                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_title("Confusion Matrix")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

                # Optional: Tambah Data Manual
                st.subheader("➕ Tambah Data Baru")
                with st.form("tambah_data"):
                    inputs = []
                    for col in feature_cols:
                        val = st.number_input(f"{col}", value=0.0, step=0.1)
                        inputs.append(val)
                    submit = st.form_submit_button("Tambahkan dan Prediksi")

                if submit:
                    try:
                        new_data = scaler.transform([inputs])
                        prediction = model.predict(new_data)
                        st.success(f"Prediksi: Kelas {prediction[0]}")
                    except Exception as e:
                        st.error(f"Gagal melakukan prediksi: {str(e)}")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat dataset: {str(e)}")
else:
    st.info("Silakan upload file CSV terlebih dahulu.")

