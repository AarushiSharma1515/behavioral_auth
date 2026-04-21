import time
import numpy as np
import pandas as pd
import joblib
import statistics

PASSWORD = ".tie5Roanl"

print("=" * 45)
print("  Keystroke Dynamics Authentication Demo")
print("=" * 45)
print("\n1. Collect my typing samples")
print("2. Authenticate me")
mode = input("\nChoose mode (1 or 2): ").strip()

# ── Capture ────────────────────────────────────────────────
def capture_typing():
    print(f"\nType exactly: {PASSWORD}")
    print("Press Enter when done.\n")

    while True:
        start = time.perf_counter()
        typed = input()
        end   = time.perf_counter()
        if typed.strip():        # only return if something was actually typed
            return typed, start, end
        # silently retry if empty

# ── Timing range from personal samples ────────────────────
MIN_TIME = None
MAX_TIME = None

def load_timing_range():
    global MIN_TIME, MAX_TIME
    try:
        df         = pd.read_csv("data/my_features.csv")
        my_samples = df[df["label"] == 1]
        times      = my_samples["dd_mean"].values * 9
        mean       = times.mean()
        std        = times.std()
        MIN_TIME   = max(0.5, mean - 2.5 * std)
        MAX_TIME   = mean + 2.5 * std
    except Exception:
        MIN_TIME = 1.0
        MAX_TIME = 30.0

# ── Feature extraction ─────────────────────────────────────
def extract_features(typed, start, end):
    typed      = typed.strip()
    n          = len(typed)
    total_time = end - start

    print(f"  [debug] chars typed: {n}, time: {total_time:.3f}s")

    if n != len(PASSWORD):
        return None

    if typed != PASSWORD:
        return None

    if total_time < MIN_TIME or total_time > MAX_TIME:
        return "timing_reject"

    avg_dd = total_time / (n - 1)
    dd     = np.array([avg_dd] * 9)

    features = [
        avg_dd, 0.0, avg_dd, avg_dd, 0.0,
        avg_dd, 0.0, avg_dd, avg_dd, 0.0,
        dd.mean(), dd.std(), dd.min(), dd.max(), dd.max()-dd.min(),
        1.0,
        1.0
    ]
    return features

# ── MODE 1: Collect samples ────────────────────────────────
if mode == "1":
    load_timing_range()
    samples   = []
    n_samples = 20
    print(f"\nYou will type the password {n_samples} times.")
    print("Take a short break between each attempt.\n")

    attempt = 0
    while len(samples) < n_samples:
        attempt += 1
        print(f"Attempt {attempt} (collected {len(samples)}/{n_samples})")
        typed, start, end = capture_typing()
        features = extract_features(typed, start, end)

        if features is None or features == "timing_reject":
            print("Could not record — try again\n")
            continue

        samples.append(features)
        print(f"Recorded! Total time: {end-start:.3f}s\n")

    cols = [
        "h_mean","h_std","h_min","h_max","h_range",
        "ud_mean","ud_std","ud_min","ud_max","ud_range",
        "dd_mean","dd_std","dd_min","dd_max","dd_range",
        "h_ud_ratio","h_dd_ratio"
    ]
    my_df          = pd.DataFrame(samples, columns=cols)
    my_df["label"] = 1

    existing  = pd.read_csv("data/features.csv")
    impostors = existing[existing["label"] == 0]
    merged    = pd.concat([my_df, impostors], ignore_index=True)
    merged.to_csv("data/my_features.csv", index=False)

    print("=" * 45)
    print(f"  {len(samples)} samples collected and saved!")
    print("  Now run train_mine.py to retrain the model")
    print("=" * 45)

# ── MODE 2: Authenticate ───────────────────────────────────
elif mode == "2":
    load_timing_range()

    typed, start, end = capture_typing()
    features          = extract_features(typed, start, end)

    if features is None or features == "timing_reject":
        print("\n" + "=" * 45)
        print("  Result: Rejected ❌")
        print("  You are not the authorised user.")
        print("=" * 45)
    else:
        X          = np.array(features).reshape(1, -1)
        scaler     = joblib.load("data/my_scaler.pkl")
        models     = joblib.load("data/my_models.pkl")
        model      = models["KNN"]
        X_scaled   = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        confidence = model.predict_proba(X_scaled)[0].max()

        print("\n" + "=" * 45)
        if prediction == 1:
            print("  Result: Authenticated ✅")
        else:
            print("  Result: Rejected ❌")
            print("  You are not the authorised user.")
        print("=" * 45)