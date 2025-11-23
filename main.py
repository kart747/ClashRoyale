"""
Clash Royale Emote Detector - Clean MP4 Overlay System
Face emotion → MP4 video overlay

Supports emotions and gestures:
  • happy   → laughing.gif
  • sad     → sad.gif
  • neutral → no overlay
  • hand near mouth → coffee.gif
  • hand up → 67 meme
  • yawning → yawn.gif
"""
import collections
import time
from pathlib import Path
from typing import Optional

import cv2

from emotion_detector import EmotionDetector
from emote_overlay import GifOverlay
from gif_overlay_67 import Meme67Overlay
import mediapipe as mp

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
ASSETS_DIR = Path(__file__).parent / "assets" / "emotes"

# GIF emote files for emotions
EMOTE_FILES = {
    "happy": ASSETS_DIR / "laughing.gif",
    "sad": ASSETS_DIR / "sad.gif",
}

# Coffee emote (independent overlay)
COFFEE_GIF_PATH = ASSETS_DIR / "coffee.gif"

# Yawn emote (independent overlay)
YAWN_GIF_PATH = ASSETS_DIR / "yawn.gif"

MEME_67_PATH = ASSETS_DIR.parent / "memes" / "67meme.gif"

# Detection settings
SMOOTHING_WINDOW = 7
HAPPY_THRESHOLD = 0.6
SAD_THRESHOLD = 0.5  # Lowered from 0.6 to make sad detection more sensitive
HAND_MOUTH_THRESHOLD = 0.08  # Normalized distance threshold
YAWN_THRESHOLD = 0.6  # Mouth aspect ratio threshold

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Overlay settings
OVERLAY_SCALE = 1.3


def main():
    """Main application - emotion detection with MP4 overlay."""
    print("=" * 70)
    print("🎮 CLASH ROYALE EMOTE DETECTOR")
    print("=" * 70)
    print("Emotions supported:")
    print("  • happy   → laughing.gif")
    print("  • sad     → sad.gif")
    print("  • neutral → no overlay")
    print("  • hand near mouth → coffee.gif")
    print("  • hand up → 67 meme")
    print("  • yawning → yawn.gif")
    print()
    print("Controls: Q or ESC to quit")
    print("=" * 70)
    print()
    
    # Initialize emotion detector
    print("Initializing FER emotion detector...")
    detector = EmotionDetector(
        happy_threshold=HAPPY_THRESHOLD,
        sad_threshold=SAD_THRESHOLD,
        smoothing_window=SMOOTHING_WINDOW
    )
    print("✓ Emotion detector ready")
    print()
    
    # Initialize emote overlay system
    print("Loading GIF emote animations...")
    overlay = GifOverlay(EMOTE_FILES, scale=OVERLAY_SCALE)
    
    # Initialize coffee overlay (independent)
    print("Loading coffee emote overlay...")
    coffee_overlay = GifOverlay({"coffee": COFFEE_GIF_PATH}, scale=OVERLAY_SCALE)
    
    # Initialize yawn overlay (independent)
    print("Loading yawn emote overlay...")
    yawn_overlay = GifOverlay({"yawn": YAWN_GIF_PATH}, scale=OVERLAY_SCALE)
    
    # Initialize 67 meme overlay
    print("Loading 67 meme overlay...")
    meme67 = Meme67Overlay(MEME_67_PATH)
    
    # Initialize MediaPipe Hands
    print("Initializing MediaPipe Hands...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initialize MediaPipe FaceMesh
    print("Initializing MediaPipe FaceMesh...")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    if not overlay.players:
        print("ERROR: No GIF emotes loaded!")
        return
    
    for emotion in overlay.players.keys():
        print(f"  ✓ {emotion}")
    
    print()
    
    # Open webcam
    print(f"Opening webcam (index {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    print("✓ Webcam ready")
    print()
    print("=" * 70)
    print("System active! Show emotions to trigger emotes.")
    print("=" * 70)
    print()
    
    # State variables
    fps_buffer = collections.deque(maxlen=30)
    prev_time = time.time()
    last_bbox: Optional[tuple[int, int, int, int]] = None
    smoothed_bbox: Optional[tuple[int, int, int, int]] = None
    bbox_alpha = 0.3  # Smoothing factor - LOWERED for more stability (higher = responsive, lower = smoother)
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # Mirror for natural interaction
            frame = cv2.flip(frame, 1)
            
            # Detect emotion and face (now with built-in smoothing)
            emotion, bbox = detector.detect(frame)
            
            # Detect hands and face mesh
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(frame_rgb)
            face_mesh_results = face_mesh.process(frame_rgb)
            
            # Get mouth landmarks from FaceMesh for yawn detection
            mouth_landmark = None
            yawning = False
            
            if face_mesh_results.multi_face_landmarks:
                face_landmarks = face_mesh_results.multi_face_landmarks[0]
                mouth_landmark = face_landmarks.landmark[13]  # Upper lip center (for coffee detection)
                
                # Calculate Mouth Aspect Ratio (MAR) for yawn detection
                # Landmarks: 13=upper lip, 14=lower lip, 61=left corner, 291=right corner
                upper_lip = face_landmarks.landmark[13]
                lower_lip = face_landmarks.landmark[14]
                left_corner = face_landmarks.landmark[61]
                right_corner = face_landmarks.landmark[291]
                
                # Calculate vertical distance (upper to lower lip)
                vertical_dist = ((upper_lip.x - lower_lip.x)**2 + (upper_lip.y - lower_lip.y)**2)**0.5
                
                # Calculate horizontal distance (left to right corner)
                horizontal_dist = ((left_corner.x - right_corner.x)**2 + (left_corner.y - right_corner.y)**2)**0.5
                
                # Mouth Aspect Ratio
                if horizontal_dist > 0:
                    mar = vertical_dist / horizontal_dist
                    if mar > YAWN_THRESHOLD:
                        yawning = True
            
            hand_up = False
            hand_near_mouth = False
            
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # Check if hand is up
                    # Simple heuristic: Wrist Y < Shoulder Y (estimated)
                    # We use normalized coordinates (0-1), y increases downwards
                    
                    wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
                    
                    # Estimate shoulder/face threshold
                    # If we have a face bbox, use chin level
                    threshold_y = 1.0 # Default to bottom of screen
                    
                    if bbox:
                        # bbox is (x, y, w, h) in pixels
                        # Convert to normalized
                        face_bottom_y = (bbox[1] + bbox[3]) / frame.shape[0]
                        threshold_y = face_bottom_y
                    else:
                        # Fallback: Top 1/3 of screen
                        threshold_y = 0.5
                        
                    if wrist_y < threshold_y:
                        hand_up = True
                    
                    # Check if hand is near mouth
                    if mouth_landmark:
                        # Use middle finger tip for better accuracy
                        hand_point = hand_landmarks.landmark[12]  # MIDDLE_FINGER_TIP
                        
                        # Calculate Euclidean distance in normalized coordinates
                        dx = hand_point.x - mouth_landmark.x
                        dy = hand_point.y - mouth_landmark.y
                        distance = (dx**2 + dy**2)**0.5
                        
                        if distance < HAND_MOUTH_THRESHOLD:
                            hand_near_mouth = True
            
            # Update 67 meme state
            meme67.set_active(hand_up)
            
            # Update bbox tracking with smoothing
            # Update if we have bbox from emotion detector OR if we have face from FaceMesh
            if bbox:
                last_bbox = bbox
                
                # Apply exponential moving average smoothing to reduce jitter
                if smoothed_bbox is None:
                    smoothed_bbox = bbox
                else:
                    # Smooth each coordinate
                    smoothed_bbox = (
                        int(bbox_alpha * bbox[0] + (1 - bbox_alpha) * smoothed_bbox[0]),
                        int(bbox_alpha * bbox[1] + (1 - bbox_alpha) * smoothed_bbox[1]),
                        int(bbox_alpha * bbox[2] + (1 - bbox_alpha) * smoothed_bbox[2]),
                        int(bbox_alpha * bbox[3] + (1 - bbox_alpha) * smoothed_bbox[3]),
                    )
            elif face_mesh_results.multi_face_landmarks and last_bbox is None:
                # If no emotion bbox but FaceMesh detected face, estimate bbox from landmarks
                # This ensures coffee/yawn overlays work even without emotion detection
                pass  # Keep using last_bbox if available
            
            # Use detector's smoothed emotion directly
            active_emotion = emotion
            
            # Priority logic: If yawning, suppress emotion GIFs
            # (yawning can falsely trigger happy emotion)
            if yawning:
                active_emotion = None
            
            # Set active emotion and render overlay (use smoothed bbox)
            overlay.set_active(active_emotion)
            frame = overlay.render(frame, smoothed_bbox if active_emotion else None)
            
            # Render 67 meme overlay (centered on face, use smoothed bbox)
            frame = meme67.render(frame, smoothed_bbox)
            
            # Priority logic for mouth-related emotes:
            # If yawning, don't show coffee (yawn takes precedence)
            show_coffee = hand_near_mouth and not yawning
            
            # Render coffee overlay (on top of emotion GIFs, use smoothed bbox)
            coffee_overlay.set_active("coffee" if show_coffee else None)
            frame = coffee_overlay.render(frame, smoothed_bbox if smoothed_bbox else last_bbox)
            
            # Render yawn overlay (on top of coffee, use smoothed bbox)
            yawn_overlay.set_active("yawn" if yawning else None)
            frame = yawn_overlay.render(frame, smoothed_bbox)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / max(current_time - prev_time, 1e-6)
            prev_time = current_time
            fps_buffer.append(fps)
            avg_fps = sum(fps_buffer) / len(fps_buffer)
            
            # Draw UI
            draw_ui(frame, active_emotion, avg_fps)
            
            # Display frame
            cv2.imshow("Clash Royale Emote Detector", frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # Q or ESC
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        print("\nShutting down...")
        cap.release()
        overlay.release()
        coffee_overlay.release()
        yawn_overlay.release()
        hands.close()
        face_mesh.close()
        cv2.destroyAllWindows()
        print("✓ Cleanup complete")


def draw_ui(frame, emotion: Optional[str], fps: float):
    """Draw UI overlay on frame."""
    h = frame.shape[0]
    
    # Emotion label
    if emotion:
        label = emotion.upper().replace("_", " ")
        color = (0, 255, 0)
    else:
        label = "NO EMOTION"
        color = (100, 100, 100)
    
    cv2.putText(
        frame,
        label,
        (15, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
        cv2.LINE_AA,
    )
    
    # FPS counter
    cv2.putText(
        frame,
        f"FPS: {fps:.0f}",
        (15, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    
    # Instructions
    cv2.putText(
        frame,
        "Q=Quit",
        (15, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )


if __name__ == "__main__":
    main()
