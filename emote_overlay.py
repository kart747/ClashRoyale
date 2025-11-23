"""GifOverlay - Transparent GIF animation overlay system.

Loads GIF emotes with transparency and overlays them on faces.
Supports: happy and sad emotes only.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageSequence


class GifPlayer:
    """Loads and loops a transparent GIF animation."""
    
    def __init__(self, gif_path: Path):
        """
        Initialize GIF player.
        
        Args:
            gif_path: Path to GIF file
        
        Raises:
            FileNotFoundError: If GIF file doesn't exist
            RuntimeError: If GIF cannot be loaded
        """
        if not gif_path.exists():
            raise FileNotFoundError(f"GIF not found: {gif_path}")
        
        self.path = gif_path
        self.frames: List[np.ndarray] = []
        self.current_frame_idx = 0
        
        # Load all GIF frames
        try:
            gif = Image.open(gif_path)
            for frame in ImageSequence.Iterator(gif):
                # Convert to RGBA to preserve transparency
                rgba_frame = frame.convert("RGBA")
                # Convert PIL image to numpy array
                np_frame = np.array(rgba_frame)
                self.frames.append(np_frame)
            
            if not self.frames:
                raise RuntimeError(f"No frames loaded from GIF: {gif_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load GIF {gif_path}: {e}")
    
    def next_frame(self) -> np.ndarray:
        """
        Get next GIF frame, loop to start if at end.
        
        Returns:
            RGBA frame as numpy array (H, W, 4)
        """
        frame = self.frames[self.current_frame_idx]
        self.current_frame_idx = (self.current_frame_idx + 1) % len(self.frames)
        return frame
    
    def release(self):
        """Release resources (cleanup)."""
        self.frames.clear()


class GifOverlay:
    """Manages GIF emote animations and renders them over face regions."""
    
    def __init__(
        self,
        emote_files: Dict[str, Path],
        scale: float = 1.0,
    ):
        """
        Initialize GIF overlay system.
        
        Args:
            emote_files: Dictionary mapping emotion names to GIF file paths
            scale: Scale factor for overlay relative to face bounding box
        """
        self.scale = scale
        self.active_emotion: Optional[str] = None
        self.players: Dict[str, GifPlayer] = {}
        
        # Load all GIF players
        for emotion, gif_path in emote_files.items():
            try:
                self.players[emotion] = GifPlayer(gif_path)
            except Exception as e:
                print(f"Warning: Failed to load {emotion}: {e}")
    
    def set_active(self, emotion: Optional[str]) -> None:
        """
        Set the currently active emotion.
        
        Args:
            emotion: Emotion name to activate, or None to clear
        """
        if emotion == self.active_emotion:
            return
        
        self.active_emotion = emotion if emotion in self.players else None
    
    def render(
        self,
        frame: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        """
        Render active GIF emote overlay on frame.
        
        Args:
            frame: Background webcam frame (BGR)
            face_bbox: Face bounding box (x, y, w, h) or None
        
        Returns:
            Frame with transparent GIF overlay applied
        """
        if not self.active_emotion or not face_bbox:
            return frame
        
        player = self.players.get(self.active_emotion)
        if not player:
            return frame
        
        gif_frame = player.next_frame()
        
        return self._overlay_gif_on_face(frame, gif_frame, face_bbox)
    
    def _overlay_gif_on_face(
        self,
        background: np.ndarray,
        gif_frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """Apply transparent GIF overlay on face region using alpha channel."""
        x, y, w, h = bbox
        
        if w <= 0 or h <= 0:
            return background
        
        # Calculate scaled GIF dimensions
        cx, cy = x + w // 2, y + h // 2
        gif_w = int(w * self.scale)
        gif_h = int(h * self.scale)
        
        # Calculate overlay position (centered on face)
        x1 = max(0, cx - gif_w // 2)
        y1 = max(0, cy - gif_h // 2)
        x2 = min(background.shape[1], x1 + gif_w)
        y2 = min(background.shape[0], y1 + gif_h)
        
        final_w = x2 - x1
        final_h = y2 - y1
        
        if final_w <= 0 or final_h <= 0:
            return background
        
        # Resize GIF frame to fit (RGBA format)
        gif_resized = cv2.resize(gif_frame, (final_w, final_h), interpolation=cv2.INTER_AREA)
        
        # Extract RGB and alpha channels
        if gif_resized.shape[2] == 4:
            gif_rgb = gif_resized[:, :, :3]  # RGB channels
            gif_alpha = gif_resized[:, :, 3:4] / 255.0  # Alpha channel (0-1)
        else:
            # Fallback if somehow not RGBA
            gif_rgb = gif_resized[:, :, :3]
            gif_alpha = np.ones((final_h, final_w, 1), dtype=np.float32)
        
        # Convert GIF RGB from RGB to BGR for OpenCV
        gif_bgr = cv2.cvtColor(gif_rgb, cv2.COLOR_RGB2BGR)
        
        # Get ROI from background
        roi = background[y1:y2, x1:x2]
        
        # Alpha blend: result = gif * alpha + background * (1 - alpha)
        blended = (gif_bgr * gif_alpha + roi * (1 - gif_alpha)).astype(np.uint8)
        
        # Apply blended region back to background
        background[y1:y2, x1:x2] = blended
        
        return background
    
    def release(self):
        """Release all GIF players."""
        for player in self.players.values():
            player.release()
        self.players.clear()
    
    def __del__(self):
        self.release()
