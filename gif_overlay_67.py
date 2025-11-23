"""
Overlay handler for the 67 meme.
This module handles loading, looping, and overlaying the 67 meme GIF.
"""
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageSequence

class Meme67Overlay:
    """Manages the 67 meme overlay."""

    def __init__(self, gif_path: Path, target_width: int = 300):
        """
        Initialize the overlay.
        
        Args:
            gif_path: Path to the GIF file.
            target_width: Width to resize the GIF to.
        """
        if not gif_path.exists():
            print(f"WARNING: 67 meme GIF not found at {gif_path}")
            self.frames = []
        else:
            self.frames = self._load_gif(gif_path)
            
        self.target_width = target_width
        self.current_frame_idx = 0
        self.is_active = False

    def _load_gif(self, path: Path) -> list[np.ndarray]:
        """Load GIF frames and convert to RGBA numpy arrays."""
        frames = []
        try:
            gif = Image.open(path)
            for frame in ImageSequence.Iterator(gif):
                rgba = frame.convert("RGBA")
                frames.append(np.array(rgba))
        except Exception as e:
            print(f"Error loading GIF: {e}")
        return frames

    def set_active(self, active: bool):
        """Set whether the overlay should be shown."""
        self.is_active = active

    def render(self, frame: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Render the overlay on the frame if active.
        
        Args:
            frame: The background frame (BGR).
            face_bbox: Optional face bounding box (x, y, w, h) to center the meme on.
            
        Returns:
            The frame with the overlay (if active).
        """
        if not self.is_active or not self.frames:
            return frame

        # Get current frame
        gif_frame = self.frames[self.current_frame_idx]
        self.current_frame_idx = (self.current_frame_idx + 1) % len(self.frames)

        # Resize GIF frame
        h, w = frame.shape[:2]
        aspect_ratio = gif_frame.shape[0] / gif_frame.shape[1]
        new_w = self.target_width
        new_h = int(new_w * aspect_ratio)
        
        # Resize using OpenCV
        gif_resized = cv2.resize(gif_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Position: Center on face if available, otherwise center of frame
        if face_bbox:
            # Center on face
            face_x, face_y, face_w, face_h = face_bbox
            face_center_x = face_x + face_w // 2
            face_center_y = face_y + face_h // 2
            
            x = face_center_x - new_w // 2
            y = face_center_y - new_h // 2
        else:
            # Fallback: center of frame
            x = (w - new_w) // 2
            y = (h - new_h) // 2

        # Ensure bounds
        x = max(0, min(x, w - new_w))
        y = max(0, min(y, h - new_h))
        
        # Overlay
        return self._overlay_image(frame, gif_resized, x, y)

    def _overlay_image(self, background: np.ndarray, overlay: np.ndarray, x: int, y: int) -> np.ndarray:
        """Overlay RGBA image onto BGR background."""
        h, w = overlay.shape[:2]
        
        # Crop if out of bounds
        if y + h > background.shape[0]:
            h = background.shape[0] - y
            overlay = overlay[:h, :]
        if x + w > background.shape[1]:
            w = background.shape[1] - x
            overlay = overlay[:, :w]
            
        if h <= 0 or w <= 0:
            return background

        # Extract channels
        overlay_bgr = cv2.cvtColor(overlay[:, :, :3], cv2.COLOR_RGB2BGR)
        alpha = overlay[:, :, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        
        roi = background[y:y+h, x:x+w]
        
        # Blend
        blended = (overlay_bgr * alpha + roi * (1.0 - alpha)).astype(np.uint8)
        
        background[y:y+h, x:x+w] = blended
        return background
