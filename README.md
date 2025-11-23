# 🎮 Clash Royale Emote Detector

**Real-time face emotion and hand gesture recognition with animated GIF overlays**

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![FPS](https://img.shields.io/badge/FPS-20--30-brightgreen.svg)

A real-time emotion and gesture recognition system that overlays animated GIF emotes on a live webcam feed.

---

## 🚀 Features

- 🎭 **Real-time Face Emotion Detection** - Detects happy, sad, and neutral emotions using FER
- 🎬 **Transparent GIF Overlays** - Animated emotes with alpha channel blending
- ✋ **Hand Gesture Recognition** - Triggers special overlays when hand is raised
- 🔄 **Emotion Smoothing** - 7-frame temporal filtering for stable tracking
- 📊 **Multi-layer Rendering** - Multiple overlays can appear simultaneously
- ⚡ **20-30 FPS Performance** - Real-time processing on standard CPUs

---

## 📸 Demo

### Emotion Triggers

- 😊 **Happy Face** → Laughing King GIF
- 😢 **Sad Face** → Sad King GIF
- 😐 **Neutral** → No emotion overlay

### Gesture Triggers

- ✋ **Hand Raised Above Shoulder** → 67 Meme GIF
- 👇 **Hand Lowered** → Meme disappears

### Multi-Overlay Support

- Happy + Hand Up → Both overlays visible
- Sad + Hand Up → Both overlays visible
- Neutral + Hand Up → Only gesture overlay

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Webcam
- Windows, macOS, or Linux

### Setup

```bash
# Clone the repository
git clone https://github.com/kart747/ClashRoyale.git
cd ClashRoyale

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Quick Start

```bash
python main.py
```

### Controls

| Key   | Action           |
| ----- | ---------------- |
| `Q`   | Quit application |
| `ESC` | Quit application |

### Configuration

Edit `main.py` to customize:

```python
# Detection settings
SMOOTHING_WINDOW = 7          # Frames for emotion smoothing
HAPPY_THRESHOLD = 0.6         # Happy emotion confidence threshold
SAD_THRESHOLD = 0.6           # Sad emotion confidence threshold

# Camera settings
CAMERA_INDEX = 0              # Change for different webcam
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Overlay settings
OVERLAY_SCALE = 1.3           # Scale factor for emotion overlays
```

---

## 📁 Project Structure

```
ClashRoyale/
├── main.py                      # Main application entry point
├── emotion_detector.py          # FER emotion detection module
├── emote_overlay.py             # GIF overlay rendering system
├── gif_overlay_67.py            # Hand gesture overlay handler
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .gitignore                   # Git ignore rules
├── PROJECT_PRESENTATION_REPORT.md  # Full technical documentation
├── PROJECT_PRESENTATION_REPORT.txt # Plain text report
└── assets/
    ├── emotes/                  # Emotion-triggered GIF overlays
    │   ├── laughing.gif         # Happy emotion
    │   ├── sad.gif              # Sad emotion
    │   ├── coffee.gif           # (Future: coffee drinking gesture)
    │   └── yawn.gif             # (Future: yawning detection)
    └── memes/                   # Gesture-triggered overlays
        └── 67meme.gif           # Hand-up gesture
```

---

## 🔬 Technical Stack

| Component             | Technology                       | Purpose                                   |
| --------------------- | -------------------------------- | ----------------------------------------- |
| **Emotion Detection** | FER (Facial Emotion Recognition) | Face detection and emotion classification |
| **Hand Tracking**     | MediaPipe Hands                  | Hand landmark detection                   |
| **Computer Vision**   | OpenCV                           | Video I/O, image processing               |
| **GIF Processing**    | Pillow (PIL)                     | GIF loading with alpha transparency       |
| **Deep Learning**     | TensorFlow                       | Backend for FER model                     |
| **Numerical Ops**     | NumPy                            | Array operations and blending             |

---

## ⚙️ How It Works

### 1. Emotion Detection Pipeline

```
Webcam Frame → Face Detection (Haar Cascade) →
Emotion Classification (FER) → Emotion Mapping →
Temporal Smoothing (7 frames) → Final Emotion
```

**Emotion Mapping:**

- `happy`, `surprise` → **happy**
- `sad` → **sad**
- `fear`, `neutral`, `angry`, `disgust` → **neutral**

### 2. Hand Gesture Detection

```
Webcam Frame → Hand Detection (MediaPipe) →
Wrist Landmark Extraction → Y-coordinate Comparison →
Hand Up/Down Decision
```

**Logic:** If `wrist_y < shoulder_y` → Hand is Up

### 3. Overlay Rendering

```
Base Frame → Emotion GIF Overlay (if detected) →
67 Meme Overlay (if hand up) → Alpha Blending →
Final Composite
```

**Alpha Blending Formula:**

```
result = gif_rgb × alpha + background × (1 - alpha)
```

---

## 📊 Performance

| Metric | Value |
|---------------------------|-----------||
| **FPS (No Detection)** | 30 FPS |
| **FPS (Emotion Only)** | 25-30 FPS |
| **FPS (Emotion + Hand)** | 20-25 FPS |
| **Face Detection Latency**| ~15ms |
| **Hand Detection Latency**| ~20ms |
| **Memory Usage** | ~600 MB |

---

## 🚧 Future Enhancements

- [ ] **Coffee Drinking Gesture** - Hand near mouth detection
- [ ] **Yawning Detection** - Mouth opening detection
- [ ] **Multiple Face Support** - Track multiple faces simultaneously
- [ ] **Custom Emote Creator** - Upload custom GIF overlays
- [ ] **Recording Mode** - Save demo videos with overlays

---

## 🔧 Troubleshooting

### No webcam detected

```bash
# Check available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"
```

### Low FPS

- Close other applications using the webcam
- Reduce resolution: `FRAME_WIDTH = 480`, `FRAME_HEIGHT = 360`
- Increase thresholds to reduce false detections

### MediaPipe Errors on Windows

If you see protobuf errors:

```bash
pip install protobuf==3.20.3
```

### Import errors

```bash
# Reinstall all dependencies
pip install --force-reinstall -r requirements.txt
```

---

## 📝 Adding Custom Emotes

1. Create or obtain a transparent GIF (3-5 seconds, looping)
2. Place in `assets/emotes/` or `assets/memes/`
3. Update `EMOTE_FILES` in `main.py`:

```python
EMOTE_FILES = {
    "happy": ASSETS_DIR / "laughing.gif",
    "sad": ASSETS_DIR / "sad.gif",
    "custom": ASSETS_DIR / "my_custom_emote.gif",  # Add here
}
```

4. Update emotion mapping in `emotion_detector.py` if needed

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📜 License

MIT License - feel free to use this project however you like!

---

## 🙏 Acknowledgments

- **FER Library** - Facial Emotion Recognition
- **MediaPipe** - Google's hand tracking solution
- **OpenCV** - Computer vision toolkit
- **Supercell** - Clash Royale emote inspiration

---

## 📧 Contact

Questions? Open an issue on GitHub!

---

**Enjoy detecting emotions and triggering Clash Royale emotes! 🎮**
