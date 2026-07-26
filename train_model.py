import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors


# -------------------------
# Load Dataset
# -------------------------

df = pd.read_csv("laptops.csv")


# Remove unwanted column

if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)



# -------------------------
# Fill Missing Values
# -------------------------

df["Rating"] = df["Rating"].fillna(df["Rating"].mean())

df["Display"] = df["Display"].fillna(
    df["Display"].mode()[0]
)

df["OS"] = df["OS"].fillna(
    df["OS"].mode()[0]
)

df["Warranty"] = df["Warranty"].fillna(
    "No Warranty"
)



# -------------------------
# Clean Price
# -------------------------

df["Price"] = (
    df["Price"]
    .astype(str)
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.replace("â‚¹", "", regex=False)
)


df["Price"] = pd.to_numeric(
    df["Price"],
    errors="coerce"
)



# -------------------------
# Clean RAM
# -------------------------

df["Ram"] = (
    df["Ram"]
    .astype(str)
    .str.extract(r"(\d+)")
)


df["Ram"] = pd.to_numeric(
    df["Ram"],
    errors="coerce"
)



# -------------------------
# Encode Columns
# -------------------------

core_encoder = LabelEncoder()

gpu_encoder = LabelEncoder()

os_encoder = LabelEncoder()



df["CoreEncoded"] = core_encoder.fit_transform(
    df["Core"].astype(str)
)


df["GraphicsEncoded"] = gpu_encoder.fit_transform(
    df["Graphics"].astype(str)
)


df["OSEncoded"] = os_encoder.fit_transform(
    df["OS"].astype(str)
)



# Remove missing values

df = df.dropna()



# -------------------------
# Features
# -------------------------

X = df[
    [
        "Price",
        "Ram",
        "Rating",
        "CoreEncoded",
        "GraphicsEncoded",
        "OSEncoded"
    ]
]



# -------------------------
# SCALE FEATURES
# -------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)



# -------------------------
# Train KNN
# -------------------------

knn = NearestNeighbors(

    n_neighbors=5,

    metric="euclidean"

)


knn.fit(X_scaled)



# -------------------------
# Save Model
# -------------------------

joblib.dump(
    knn,
    "knn_model.pkl"
)



joblib.dump(

    {

        "core": core_encoder,

        "gpu": gpu_encoder,

        "os": os_encoder,

        "scaler": scaler,

        "data": df

    },

    "encoders.pkl"

)



print("Scaled KNN model trained successfully!")