import time
import numpy as np
import pandas as pd
import joblib

PASSWORD = ".tie5Roanl"

print("=" * 45)
print("  Keystroke Dynamics Authentication Demo")
print("=" * 45)
print("\n1. Collect my typing samples")
print("2. Authenticate me")
mode = input("\nChoose mode (1 or 2): ").strip()

# ── Capture: user types password, we measure total time + char count ──
def capture_typing():
    print(f"\nType exactly: {PASSWORD}")
    print("Press Enter when done.\n")

    start = time.perf_counter()
    typed = input()
    end   = time.perf_counter()

    return typed, start, end

# ── Feature extraction from timed input ───────────────────
def extract_features(typed, start, end):
    typed = typed.strip()
    n     = len(typed)
    print(f"  [debug] chars typed: {n}, time: {end-start:.3f}s")

    if n != len(PASSWORD):
        print(f"  [!] Expected {len(PASSWORD)} chars, got {n} — type the password exactly")
        return None

    total_time = end - start
    avg_dd     = total_time / (n - 1)   # average inter-key interval

    # We only have total time, so we synthesize consistent features
    # All 9 dd slots get the average; std/range will be 0 but consistent
    dd = np.array([avg_dd] * 9)

    features = [
        avg_dd, 0.0, avg_dd, avg_dd, 0.0,   # h_ slots (hold — unavailable)
        avg_dd, 0.0, avg_dd, avg_dd, 0.0,   # ud_ slots
        dd.mean(), dd.std(), dd.min(), dd.max(), dd.max()-dd.min(),
        1.0,   # h_ud_ratio
        1.0    # h_dd_ratio
    ]
    return features

# ── MODE 1: Collect samples ────────────────────────────────
if mode == "1":
    samples   = []
    n_samples = 20
    print(f"\nYou will type the password {n_samples} times.")
    print("Try to type at your natural pace each time.\n")

    attempt = 0
    while len(samples) < n_samples:
        attempt += 1
        print(f"Attempt {attempt} (collected {len(samples)}/{n_samples})")
        typed, start, end = capture_typing()
        features = extract_features(typed, start, end)

        if features is None:
            print("Try again\n")
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
    typed, start, end = capture_typing()
    features          = extract_features(typed, start, end)

    if features is None:
        print("Could not extract features — try again")
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
        print(f"  Confidence: {confidence:.2%}")
        print("=" * 45)