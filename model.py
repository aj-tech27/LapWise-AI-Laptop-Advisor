import pandas as pd
import re

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

# -----------------------
# Load Dataset
# -----------------------

df = pd.read_csv("laptops.csv")

# Remove unwanted column
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# -----------------------
# Handle Missing Values
# -----------------------

df["Rating"] = df["Rating"].fillna(df["Rating"].mean())
df["Display"] = df["Display"].fillna(df["Display"].mode()[0])
df["OS"] = df["OS"].fillna(df["OS"].mode()[0])
df["Warranty"] = df["Warranty"].fillna("No Warranty")

# -----------------------
# Clean Price
# -----------------------

df["Price"] = (
    df["Price"]
    .astype(str)
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.extract(r"(\d+)")
)

df["Price"] = pd.to_numeric(df["Price"])

# -----------------------
# Clean RAM
# -----------------------

df["Ram"] = (
    df["Ram"]
    .astype(str)
    .str.extract(r"(\d+)")
)

df["Ram"] = pd.to_numeric(df["Ram"])

# -----------------------
# Encode Text Columns
# -----------------------

core_encoder = LabelEncoder()
graphics_encoder = LabelEncoder()

df["CoreEncoded"] = core_encoder.fit_transform(df["Core"])
df["GraphicsEncoded"] = graphics_encoder.fit_transform(df["Graphics"])

# -----------------------
# Features
# -----------------------

features = df[
    [
        "Price",
        "Ram",
        "Rating",
        "CoreEncoded",
        "GraphicsEncoded"
    ]
]

features = features.fillna(0)

# -----------------------
# Train KNN
# -----------------------

knn = NearestNeighbors(
    n_neighbors=5,
    metric="euclidean"
)

knn.fit(features)

# -----------------------
# Recommendation Function
# -----------------------

def get_recommendations(budget, ram, core, graphics):

    try:
        core = core_encoder.transform([core])[0]
    except:
        core = 0

    try:
        graphics = graphics_encoder.transform([graphics])[0]
    except:
        graphics = 0

    sample = [[
        budget,
        ram,
        4.2,
        core,
        graphics
    ]]

    distances, indices = knn.kneighbors(sample)

    result = df.iloc[indices[0]]

    return result[
        [
            "Model",
            "Price",
            "Rating",
            "Core",
            "Ram",
            "SSD",
            "Graphics",
            "OS",
            "Warranty"
        ]
    ].to_dict(orient="records")