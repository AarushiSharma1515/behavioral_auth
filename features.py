import pandas as pd
import numpy as np

np.random.seed(42)

# ── PART 1: Load CMU dataset ──────────────────────────────
df = pd.read_csv("data/DSL-StrongPasswordData.csv")

AUTHORIZED_USER = "s002"
df["label"] = (df["subject"] == AUTHORIZED_USER).astype(int)

# Balance CMU data
authorized = df[df["label"] == 1]
impostors  = df[df["label"] == 0].sample(n=400, random_state=42)
balanced   = pd.concat([authorized, impostors]).reset_index(drop=True)

print("CMU balanced shape:", balanced.shape)
print(balanced["label"].value_counts())

# Identify CMU timing columns
h_cols  = [col for col in balanced.columns if col.startswith("H.")]
ud_cols = [col for col in balanced.columns if col.startswith("UD.")]
dd_cols = [col for col in balanced.columns if col.startswith("DD.")]

print("\nHold columns:", len(h_cols))
print("UD columns:",   len(ud_cols))
print("DD columns:",   len(dd_cols))

# ── PART 2: Generate synthetic data ───────────────────────
N_H  = len(h_cols)
N_UD = len(ud_cols)
N_DD = len(dd_cols)

def generate_samples(n, h_mean, h_std, ud_mean, ud_std):
    rows = []
    for _ in range(n):
        h  = np.random.normal(h_mean,  h_std,  N_H).clip(0.05, 0.4)
        ud = np.random.normal(ud_mean, ud_std, N_UD).clip(0.01, 0.6)
        dd = (h[:N_DD] + ud).clip(0.05, 1.0)
        rows.append(np.concatenate([h, ud, dd]))
    return rows

# Authorized user similar to s002
syn_auth  = generate_samples(200, h_mean=0.115, h_std=0.025,
                                   ud_mean=0.09,  ud_std=0.02)
# 4 impostors overlapping with authorized
syn_imp_A = generate_samples(50,  h_mean=0.140, h_std=0.03,
                                   ud_mean=0.120, ud_std=0.025)
syn_imp_B = generate_samples(50,  h_mean=0.095, h_std=0.02,
                                   ud_mean=0.070, ud_std=0.015)
syn_imp_C = generate_samples(50,  h_mean=0.125, h_std=0.04,
                                   ud_mean=0.100, ud_std=0.035)
syn_imp_D = generate_samples(50,  h_mean=0.130, h_std=0.025,
                                   ud_mean=0.110, ud_std=0.020)

syn_all    = syn_auth + syn_imp_A + syn_imp_B + syn_imp_C + syn_imp_D
syn_labels = [1]*200 + [0]*200

all_col_names = h_cols + ud_cols + dd_cols
syn_df = pd.DataFrame(syn_all, columns=all_col_names)
syn_df["label"] = syn_labels

print("\nSynthetic data shape:", syn_df.shape)
print(syn_df["label"].value_counts())

# ── PART 3: Feature engineering function ──────────────────
def engineer_features(data, h_c, ud_c, dd_c):
    f = pd.DataFrame()

    f["h_mean"]  = data[h_c].mean(axis=1)
    f["h_std"]   = data[h_c].std(axis=1)
    f["h_min"]   = data[h_c].min(axis=1)
    f["h_max"]   = data[h_c].max(axis=1)
    f["h_range"] = f["h_max"] - f["h_min"]

    f["ud_mean"]  = data[ud_c].mean(axis=1)
    f["ud_std"]   = data[ud_c].std(axis=1)
    f["ud_min"]   = data[ud_c].min(axis=1)
    f["ud_max"]   = data[ud_c].max(axis=1)
    f["ud_range"] = f["ud_max"] - f["ud_min"]

    f["dd_mean"]  = data[dd_c].mean(axis=1)
    f["dd_std"]   = data[dd_c].std(axis=1)
    f["dd_min"]   = data[dd_c].min(axis=1)
    f["dd_max"]   = data[dd_c].max(axis=1)
    f["dd_range"] = f["dd_max"] - f["dd_min"]

    f["h_ud_ratio"] = f["h_mean"] / (f["ud_mean"] + 1e-6)
    f["h_dd_ratio"] = f["h_mean"] / (f["dd_mean"] + 1e-6)
    f["label"]      = data["label"].values
    return f

# ── PART 4: Engineer and merge ─────────────────────────────
cmu_features = engineer_features(balanced, h_cols, ud_cols, dd_cols)
syn_features = engineer_features(syn_df,   h_cols, ud_cols, dd_cols)

merged = pd.concat([cmu_features, syn_features], ignore_index=True)

print("\nCMU features shape:      ", cmu_features.shape)
print("Synthetic features shape:", syn_features.shape)
print("Merged features shape:   ", merged.shape)
print(merged["label"].value_counts())

# ── PART 5: Save ───────────────────────────────────────────
merged.to_csv("data/features.csv", index=False)
print("\nMerged features saved!")