# 🔐 Keystroke Dynamics Authentication

A behavioral biometric authentication system that identifies users by **how they type**, not just *what* they type. Built with Python and scikit-learn.

---

## 📌 What It Does

Instead of relying solely on a password, this system learns your unique typing rhythm — the speed and timing pattern you use when typing a specific password. Even if someone knows your password, they'll be rejected if they don't type it like you do.

---

## 🧠 How It Works

1. **Data Collection** — You type your password 20 times. The system records the timing of each keystroke.
2. **Feature Extraction** — Inter-key intervals (down-down times) are extracted and summarized into 17 statistical features per sample.
3. **Model Training** — Three classifiers (Random Forest, SVM, KNN) are trained on your samples vs. 600 impostor samples from a public dataset.
4. **Authentication** — At login, your typing is captured, features are extracted, and the KNN model decides: authorized ✅ or rejected ❌.

### Features Extracted

| Feature Group | Description |
|---|---|
| `h_*` | Hold time statistics (key press duration) |
| `ud_*` | Up-down latency between consecutive keys |
| `dd_*` | Down-down latency (most reliable feature) |
| `h_ud_ratio` | Ratio of hold time to up-down time |
| `h_dd_ratio` | Ratio of hold time to down-down time |

---

## 📁 Project Structure

```
behavioral_auth/
│
├── demo.py              # Main script — collect samples & authenticate
├── train_mine.py        # Train models on your collected data
│
└── data/
    ├── features.csv     # Public impostor dataset (600 samples)
    ├── my_features.csv  # Your samples merged with impostors (generated)
    ├── my_scaler.pkl    # Trained StandardScaler (generated)
    └── my_models.pkl    # Trained RF, SVM, KNN models (generated)
```

---

## ⚙️ Setup

### Requirements

- Python 3.8+
- Windows (uses `msvcrt` for keystroke capture)

### Install Dependencies

```bash
pip install numpy pandas scikit-learn joblib
```

### Clone the Repo

```bash
git clone https://github.com/YOUR_USERNAME/behavioral_auth.git
cd behavioral_auth
```

---

## 🚀 Usage

### Step 1 — Collect Your Typing Samples

```bash
python demo.py
```

Choose `1`. Type the password **`.tie5Roanl`** exactly 20 times when prompted. After each attempt you'll see:

```
  [debug] keystrokes captured: 10
  Recorded successfully!
```

> ⚠️ Type at your **natural, relaxed pace** every time. Consistency matters more than speed.

---

### Step 2 — Train Your Personal Model

```bash
python train_mine.py
```

Expected output:

```
Dataset shape: (620, 18)
label
0    600
1     20

Random Forest Accuracy: 0.994
SVM Accuracy: 1.000
KNN Accuracy: 0.994

Your personal models saved!
```

---

### Step 3 — Authenticate

```bash
python demo.py
```

Choose `2`. Type the password once and press Enter:

```
=============================================
  Result: Authenticated ✅
  Confidence: 100.00%
=============================================
```

---

## 🤖 Models Used

| Model | Role |
|---|---|
| **KNN** (k=3) | Primary classifier used for authentication |
| **SVM** (RBF kernel) | Trained and saved, highest accuracy |
| **Random Forest** (100 trees) | Trained and saved, most robust |

All models are trained on the same data and saved to `data/my_models.pkl`. You can swap `models["KNN"]` in `demo.py` for `models["SVM"]` or `models["Random Forest"]` to compare behavior.

---

## 📊 Dataset

The impostor samples come from the **CMU Keystroke Dynamics Benchmark Dataset** — a well-known public dataset of 51 subjects typing a fixed password 400 times each. 600 samples are used as the negative class.

Your 20 personal samples form the positive class.

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| `keystrokes captured: 0` | Make sure you're running on Windows; `msvcrt` is Windows-only |
| `Rejected` at authentication | Recollect samples and type more consistently; retrain |
| `train_mine.py` crashes with `ValueError` | Not enough personal samples collected — collect at least 10 |
| Confidence 100% but Rejected | Your auth timing was very different from training — try again at natural pace |

---

## 🛡️ Limitations

- **Windows only** — uses `msvcrt` for terminal keystroke capture
- **Single password** — the model is trained for one fixed password only
- **Timing only** — per-key hold durations are not captured in terminal mode; authentication relies on inter-key intervals
- **Small training set** — 20 samples is enough to work but a larger set improves robustness

---

## 🔮 Possible Improvements

- GUI frontend (tkinter or browser-based) for true per-key press/release timing
- Confidence threshold tuning (`authenticate only if confidence > 0.85`)
- Failed attempt logging for audit trail
- Support for multiple users and multiple passwords
- Cross-platform support using a browser-based capture approach

---

## 📄 License

MIT License. See `LICENSE` for details.

---

## 🙏 Acknowledgements

- [CMU Keystroke Dynamics Benchmark Dataset](http://www.cs.cmu.edu/~keystroke/)
- [scikit-learn](https://scikit-learn.org/)
