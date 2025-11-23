"""EmotionDetector - Clean FER-based face emotion detection."""
from typing import Optional, Tuple
from collections import deque

import numpy as np
from fer import FER


class EmotionDetector:
    """Detects facial emotions using FER. Returns: happy, sad, or neutral."""
    
    def __init__(
        self, 
        happy_threshold: float = 0.6,
        sad_threshold: float = 0.6,
        smoothing_window: int = 7
    ):
        """
        Initialize emotion detector.
        
        Args:
            happy_threshold: Minimum confidence to trigger happy (0.0-1.0)
            sad_threshold: Minimum confidence to trigger sad (0.0-1.0)
            smoothing_window: Number of frames to smooth over
        """
        self.detector = FER(mtcnn=False)  # Fast Haar Cascade detection
        self.happy_threshold = happy_threshold
        self.sad_threshold = sad_threshold
        
        # Frame-level smoothing
        self.emotion_history = deque(maxlen=smoothing_window)
        
        # Map FER emotions to our 3 emotions: happy, sad, neutral
        # NOTE: fear is now mapped to neutral (not sad) to prevent false triggers
        self.emotion_map = {
            "happy": "happy",
            "surprise": "happy",
            "sad": "sad",
            "fear": "neutral",  # Changed from "sad" to prevent false triggers
            "neutral": "neutral",
            "angry": "neutral",
            "disgust": "neutral",
        }
    
    def detect(self, frame: np.ndarray) -> Tuple[Optional[str], Optional[Tuple[int, int, int, int]]]:
        """
        Detect emotion in frame and return mapped emotion name with face location.
        
        Args:
            frame: BGR image from webcam
        
        Returns:
            (emotion_name, face_bbox) where:
                - emotion_name: "happy", "sad", "neutral", or None
                - face_bbox: (x, y, w, h) or None
        """
        try:
            results = self.detector.detect_emotions(frame)
        except Exception:
            return None, None
        
        if not results:
            return None, None
        
        # Get largest face
        largest_face = max(results, key=lambda f: f["box"][2] * f["box"][3])
        
        # Get all emotion confidences
        emotions = largest_face["emotions"]
        
        # Extract individual confidences
        happy_conf = emotions.get("happy", 0.0) + emotions.get("surprise", 0.0)
        sad_conf = emotions.get("sad", 0.0)
        neutral_conf = emotions.get("neutral", 0.0)
        
        # Determine raw emotion based on thresholds and confidence comparison
        raw_emotion = None
        
        # Happy: must exceed threshold AND beat neutral
        if happy_conf >= self.happy_threshold and happy_conf > neutral_conf:
            raw_emotion = "happy"
        # Sad: must exceed threshold AND beat neutral AND beat happy
        elif sad_conf >= self.sad_threshold and sad_conf > neutral_conf and sad_conf > happy_conf:
            raw_emotion = "sad"
        # Otherwise: neutral (explicit fallback, no default to sad)
        else:
            raw_emotion = "neutral"
        
        # Add to history for smoothing
        self.emotion_history.append(raw_emotion)
        
        # Get smoothed emotion using majority vote
        if len(self.emotion_history) > 0:
            # Count occurrences
            emotion_counts = {}
            for e in self.emotion_history:
                emotion_counts[e] = emotion_counts.get(e, 0) + 1
            
            # Get most common emotion
            final_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        else:
            final_emotion = raw_emotion
        
        # Extract bounding box
        x, y, w, h = largest_face["box"]
        
        return final_emotion, (x, y, w, h)
