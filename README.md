# 🚗 Real-Time Driver Drowsiness Detection System

A real-time driver drowsiness detection system using facial landmark analysis and machine learning. The system continuously monitors the driver through a webcam, detects signs of drowsiness such as eye closure, yawning, and head tilting, and immediately alerts the driver through an audio buzzer and voice warnings.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-green)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange)
![dlib](https://img.shields.io/badge/dlib-19.22-red)
![License](https://img.shields.io/badge/license-MIT-brightgreen)

---

## 📽️ Demo

> Run `python main.py` and the system opens a live webcam window showing real-time drowsiness status, EAR, MAR, and head tilt values. When drowsiness is detected for 5 consecutive frames, a buzzer fires and a voice alert plays.

---

## 📌 Features

- 🎥 Real-time webcam face detection using dlib HOG detector
- 📍 68-point facial landmark extraction
- 👁️ Eye Aspect Ratio (EAR) - detects eye closure
- 👄 Mouth Aspect Ratio (MAR) - detects yawning
- 📐 Head Tilt Angle - detects nodding off in both directions
- 🌲 Random Forest classifier trained on the D3S dataset
- 🛡️ Rule-based override layer for improved detection accuracy
- 🔔 Audio buzzer alert via pygame
- 🗣️ Voice warning via pyttsx3 text-to-speech
- ⚡ Frame skipping for smooth real-time performance
- 🧠 Face buffer to prevent flickering No Face detection

---

## 🧠 How It Works
```
Webcam Frame
     ↓
Face Detection (dlib HOG Detector)
     ↓
68 Facial Landmark Points Extracted
     ↓
Feature Engineering
  → 136 landmark coordinates (x,y for each point)
  → Eye Aspect Ratio (EAR)
  → Mouth Aspect Ratio (MAR)
  → Head Tilt Angle
  = 139 total features
     ↓
Random Forest Classifier → "Drowsy" / "Not Drowsy"
     ↓
Rule-Based Override (catches edge cases)
     ↓
If Drowsy for 5+ consecutive frames:
  → Buzzer + Voice Alert
```

---

## 📁 Project Structure
```
Demo/
├── main.py                        # Main detection script
├── AIAgent.py                     # Optional AI voice agent (Groq/Gemini)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── assets/
    └── dlib-19.22.99-cp310-cp310-win_amd64.whl   # Prebuilt dlib wheel for Windows Python 3.10
```

> ⚠️ Model files are NOT included in this repo due to size. Download from Hugging Face — see below.

---

## ⬇️ Model Files (Hugging Face)

Download the following files from Hugging Face and place them inside the `assets/` folder:

🤗 **Model Repo:** https://huggingface.co/aryanshenoy/driver-drowsiness-detection

| File | Description | Download |
|------|-------------|----------|
| `drowsiness_rf_126.pkl` | Trained Random Forest classifier | [Download](https://huggingface.co/aryanshenoy/driver-drowsiness-detection/resolve/main/drowsiness_rf_126.pkl) |
| `scaler_126.pkl` | StandardScaler for feature normalization | [Download](https://huggingface.co/aryanshenoy/driver-drowsiness-detection/resolve/main/scaler_126.pkl) |
| `shape_predictor_68_face_landmarks.dat` | dlib 68-point landmark predictor | [Download](https://huggingface.co/aryanshenoy/driver-drowsiness-detection/resolve/main/shape_predictor_68_face_landmarks.dat) |

---

## 🛠️ Installation & Setup

### Prerequisites
- Windows 10/11
- Python 3.10 (download from https://www.python.org/downloads/)
- A working webcam

---

### Step 1 — Clone the repository
```bash
git clone https://github.com/aryanshenoy/drowsiness-detection.git
cd drowsiness-detection
```

### Step 2 — Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` appear at the start of your terminal line.

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Install dlib (prebuilt wheel for Windows)
```bash
pip install assets\dlib-19.22.99-cp310-cp310-win_amd64.whl
```

> ⚠️ This wheel is for **Python 3.10 on Windows 64-bit** only. If you're on a different version, download the appropriate wheel from https://github.com/z-mahmud22/Dlib_Windows_Python3.x/releases

### Step 5 — Download model files
Download the 3 model files from [Hugging Face](https://huggingface.co/aryanshenoy/driver-drowsiness-detection) and place them in the `assets/` folder:
```
assets/
├── drowsiness_rf_126.pkl
├── scaler_126.pkl
└── shape_predictor_68_face_landmarks.dat
```

### Step 6 — Run
```bash
python main.py
```

Press **Q** to quit the detection window.

---

## 📊 Model Training & Comparison

The model was trained on the **D3S (Driver Drowsiness Detection Dataset)**. Four models were compared:

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~85% |
| SVM (RBF Kernel) | ~88% |
| KNN (k=5) | ~84% |
| **Random Forest (n=300)** ✅ | **~95%** |

Random Forest was selected as the final model for its superior accuracy, robustness to overfitting, and ability to handle high-dimensional feature vectors (139 features).

> Update accuracy values with your actual classification report results.

---

## 🔍 Feature Engineering

### Eye Aspect Ratio (EAR)
Measures how open the eye is. Drops significantly when eyes close during drowsiness.
```
EAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)
Threshold: EAR < 0.25 → Drowsy
```

### Mouth Aspect Ratio (MAR)
Measures mouth openness. Increases significantly during yawning.
```
MAR = vertical distance / horizontal distance
Threshold: MAR > 0.65 → Drowsy (yawning)
```

### Head Tilt Angle
Measures how much the head is tilting to either side. Increases when nodding off.
```
Angle = arctan2(Δy, Δx) between left and right eye centers
Threshold: |Tilt| > 20° → Drowsy (handles both left and right tilt)
```

---

## ⚙️ Configuration

You can tune these settings in `main.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `CONSECUTIVE_ALERT_THRESHOLD` | 5 | Drowsy frames in a row before alert fires |
| `ALERT_COOLDOWN` | 6 | Seconds between repeated alerts |
| `FACE_BUFFER_SIZE` | 5 | Frames to buffer before showing No Face |
| `SKIP_FRAMES` | 2 | Run detection every N frames for performance |
| `CHECKIN_INTERVAL` | 30 | Seconds between friendly AI check-ins |

---

## 🧰 Tech Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.10 | Core language |
| dlib | 19.22 | Face detection + 68-point landmark extraction |
| OpenCV | 4.9 | Webcam capture + image processing |
| scikit-learn | 1.6.1 | Random Forest model training + inference |
| NumPy | 1.26.4 | Numerical computations |
| SciPy | latest | Euclidean distance for EAR/MAR |
| pygame | latest | Buzzer sound generation |
| pyttsx3 | latest | Text-to-speech voice alerts |
| joblib | latest | Model serialization |

---

## 🐛 Troubleshooting

| Problem | Fix |
|---------|-----|
| Webcam not opening | Change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `main.py` |
| `No module named dlib` | Run `pip install assets\dlib-19.22.99-cp310-cp310-win_amd64.whl` |
| Model file not found | Make sure `.pkl` and `.dat` files are inside `assets/` folder |
| No sound | Run terminal as Administrator |
| Laggy performance | Increase `SKIP_FRAMES` to 3 or 4 in `main.py` |
| Face not detected at angle | Move closer to camera or improve lighting |

---

## 📝 Dataset

The model was trained on the **D3S — Driver Drowsiness Detection Dataset**, consisting of video frames extracted from recordings of drivers in various states of alertness. Labels were automatically generated using rule-based EAR, MAR, and Head Tilt thresholds, producing a balanced dataset of Drowsy and Not Drowsy samples.

---

## 🔮 Future Work

- Deep learning approach using CNN/LSTM for improved accuracy
- Mobile deployment (Android/iOS)
- Integration with vehicle alert systems
- Multi-face support for co-driver monitoring
- Night vision / infrared camera support
- Cloud dashboard for fleet monitoring

---

## 👤 Author

**Aryan Shenoy**
- 🌐 Portfolio: [aryan-shenoy.vercel.app](https://aryan-shenoy.vercel.app)
- 💻 GitHub: [@aryanshenoy](https://github.com/aryanshenoy)
- 🤗 Hugging Face: [@aryanshenoy](https://huggingface.co/aryanshenoy)

---

## 📄 License

This project is licensed under the MIT License - feel free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [dlib](http://dlib.net/) by Davis King
- [D3S Dataset](https://www.kaggle.com/) for training data
- [shape_predictor_68_face_landmarks](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) by iBUG 300-W
