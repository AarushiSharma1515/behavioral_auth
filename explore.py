import pandas as pd

# Load CMU dataset
df = pd.read_csv("data/DSL-StrongPasswordData.csv")

# Basic info
print("Shape:", df.shape)
print("\nFirst 3 rows:")
print(df.head(3))
print("\nUnique users:", df["subject"].nunique())
print("\nSamples per user:")
print(df["subject"].value_counts().head())

# Pick authorized user
AUTHORIZED_USER = "s002"
df["label"] = (df["subject"] == AUTHORIZED_USER).astype(int)

print("\nAuthorized samples:", df[df["label"]==1].shape[0])
print("Impostor samples:", df[df["label"]==0].shape[0])

# Compare authorized vs impostor hold time
print("\nAuthorized avg hold time:", df[df["label"]==1]["H.t"].mean().round(4))
print("Impostor avg hold time:",   df[df["label"]==0]["H.t"].mean().round(4))