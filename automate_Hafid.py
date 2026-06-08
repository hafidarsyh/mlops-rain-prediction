import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Memuat dataset dari file CSV
def load_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File tidak ditemukan: {filepath}")

    df = pd.read_csv(filepath)
    print(f"[LOAD] Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


# Mengekstrak komponen tahun, bulan, dan hari dari kolom 'Date'
def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"]  = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"]   = df["Date"].dt.day
    df.drop(columns=["Date"], inplace=True)
    print("[FEATURE] Fitur tanggal (Year, Month, Day) berhasil diekstrak.")
    return df


# Mengisi missing values (numerik dengan median, kategorikal dengan modus)
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    before = df.isnull().sum().sum()

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    after = df.isnull().sum().sum()
    print(f"[MISSING] Missing values sebelum: {before} -> sesudah: {after}")
    return df


# Mengubah fitur kategorikal menjadi representasi numerik
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Binary mapping
    binary_map = {"Yes": 1, "No": 0}
    for col in ["RainToday", "RainTomorrow"]:
        if col in df.columns:
            df[col] = df[col].map(binary_map)

    # Label Encoding untuk kolom kategori lainnya
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    print(f"[ENCODE] Kolom yang di-encode: {cat_cols + ['RainToday', 'RainTomorrow']}")
    return df


# Menghapus outliers pada fitur numerik menggunakan rentang interkuartil (IQR)
def remove_outliers(df: pd.DataFrame,
                    target_col: str = "RainTomorrow") -> pd.DataFrame:
    df = df.copy()

    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != target_col
    ]

    before = len(df)
    for col in num_cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df  = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    after = len(df)
    print(f"[OUTLIER] Baris sebelum: {before} -> sesudah: {after} "
          f"(dihapus: {before - after})")
    return df.reset_index(drop=True)


# Menstandarisasi skala fitur numerik agar seragam
def scale_features(df: pd.DataFrame,
                   target_col: str = "RainTomorrow") -> pd.DataFrame:
    df = df.copy()

    feature_cols = [c for c in df.columns if c != target_col]
    scaler       = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    print(f"[SCALE] StandardScaler diterapkan pada {len(feature_cols)} fitur.")
    return df


# Membagi dataset menjadi set latih dan uji, lalu menyimpannya ke file
def split_and_save(df: pd.DataFrame,
                   output_dir: str,
                   target_col: str = "RainTomorrow",
                   test_size: float = 0.2,
                   random_state: int = 42):
    os.makedirs(output_dir, exist_ok=True)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Simpan split
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir,  "X_test.csv"),  index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir,  "y_test.csv"),  index=False)

    # Simpan dataset lengkap yang sudah diproses
    df.to_csv(
        os.path.join(output_dir, "weatherAUS_preprocessing.csv"),
        index=False
    )

    print(f"[SAVE] Dataset tersimpan di '{output_dir}'")
    print(f"X_train : {X_train.shape}")
    print(f"X_test  : {X_test.shape}")
    print(f"y_train : {y_train.shape}")
    print(f"y_test  : {y_test.shape}")

    return X_train, X_test, y_train, y_test


# Main pipeline
def preprocess(input_path: str, output_dir: str):
    print("PIPELINE PREPROCESSING - Rain in Australia")

    df = load_data(input_path)
    df = extract_date_features(df)
    df = handle_missing_values(df)
    df = encode_categorical(df)
    df = remove_outliers(df)
    df = scale_features(df)

    X_train, X_test, y_train, y_test = split_and_save(df, output_dir)

    print("PREPROCESSING SELESAI!")

    return X_train, X_test, y_train, y_test


# Entry point
if __name__ == "__main__":
    BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
    INPUT_PATH = os.path.join(BASE_DIR, "..", "dataset", "weatherAUS.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "weatherAUS_preprocessing")

    preprocess(INPUT_PATH, OUTPUT_DIR)
