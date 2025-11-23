# 🎮 Clash Royale Emote Detector

**Face Emotion Detection → Clash Royale Emote Overlay with Audio**

A real-time face emotion detection system that overlays Clash Royale emote animations with synchronized audio based on detected emotions.

---

## ✨ Features

- 🎭 **Real-time Face Emotion Detection** using FER (Facial Emotion Recognition)
- 🎬 **MP4 Emote Overlay** with transparent alpha blending
- 🔊 **Synchronized Audio Playback** using pygame
- 🎯 **Smooth Emotion Tracking** with temporal smoothing
- ⚡ **20-30 FPS Performance** on modern CPUs
- 🚀 **Zero-config Setup** - clone and run!

---

## 🎮 Emotion Mapping

| Detected Emotion   | Clash Royale Emote |
| ------------------ | ------------------ |
| Happy, Surprise    | 👑 Laughing King   |
| Sad, Fear, Neutral | 😢 Sad King        |
| Angry, Disgust     | 💀 Skeleton Crying |

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Webcam
- Windows / macOS / Linux

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/clash-royale-emote-detector.git
   cd clash-royale-emote-detector
   ```

2. **Create virtual environment** (recommended)

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Add emote MP4 files**

   Place the following MP4 files in `assets/emotes/`:

   - `laughing_king.mp4`
   - `sad_king.mp4`
   - `skeleton_crying.mp4`

   These should be short looping videos with embedded audio.

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
SMOOTHING_WINDOW = 8          # Frames for emotion smoothing
CONFIDENCE_THRESHOLD = 0.42   # Minimum detection confidence
USE_MTCNN = False             # True for better accuracy (slower)

# Camera settings
CAMERA_INDEX = 0              # Change for different webcam
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30
```

---

## 📁 Project Structure

```
clash-royale-emote-detector/
├── main.py                  # Main application entry point
├── emotion_detector.py      # Emotion detection module
├── emote_player.py          # Video + audio playback system
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── assets/
    └── emotes/             # MP4 emote files (with audio)
        ├── laughing_king.mp4
        ├── sad_king.mp4
        └── skeleton_crying.mp4
```

---

## 🛠️ Technical Details

### Architecture

- **Emotion Detection**: FER library with OpenCV Haar Cascade face detection
- **Video Playback**: OpenCV VideoCapture for frame-by-frame rendering
- **Audio Playback**: pygame.mixer for MP4 audio track synchronization
- **Overlay System**: Alpha-blended compositing over face bounding boxes

### Performance

- **CPU-only**: 20-30 FPS on modern processors
- **GPU-enabled**: 30+ FPS with MTCNN face detection
- **Memory**: ~500MB with TensorFlow backend

### Dependencies

- `opencv-python` - Computer vision and video I/O
- `fer` - Facial emotion recognition
- `pygame` - Audio playback
- `tensorflow` - Deep learning backend
- `numpy` - Numerical operations

---

## 🎯 How It Works

1. **Capture Frame**: Read frame from webcam
2. **Detect Face**: Locate largest face in frame
3. **Classify Emotion**: Predict emotion from facial features
4. **Smooth Results**: Apply temporal smoothing over N frames
5. **Select Emote**: Map emotion to Clash Royale emote
6. **Render Overlay**: Composite emote video over face region
7. **Play Audio**: Loop emote audio in background
8. **Display**: Show result at 20-30 FPS

---

## 🔧 Troubleshooting

### No webcam detected

```bash
# Check available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"
```

### Low FPS

- Disable MTCNN: `USE_MTCNN = False`
- Reduce resolution: `FRAME_WIDTH = 480`
- Close other applications

### Audio not playing

- Check pygame installation: `pip install --upgrade pygame`
- Verify MP4 files have audio tracks
- Check system audio is not muted

### Import errors

```bash
# Reinstall all dependencies
pip install --force-reinstall -r requirements.txt
```

---

## 📝 Creating Custom Emotes

1. Create MP4 video (3-5 seconds, looping)
2. Ensure audio track is embedded
3. Place in `assets/emotes/`
4. Update `EMOTE_LIBRARY` in `main.py`:

```python
EMOTE_LIBRARY = {
    "happy": ASSETS_DIR / "my_happy_emote.mp4",
    "sad": ASSETS_DIR / "my_sad_emote.mp4",
    # ... more emotes
}
```

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

- **FER Library**: Facial Emotion Recognition
- **OpenCV**: Computer vision toolkit
- **Supercell**: Clash Royale emote inspiration
- **pygame**: Audio playback library

---

## 📧 Contact

Questions? Open an issue on GitHub!

---

**Enjoy detecting emotions and triggering Clash Royale emotes! 🎮**
