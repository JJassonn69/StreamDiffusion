import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
from typing import Union, Optional, List, Tuple, Dict
from .base import BasePreprocessor

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# MediaPipe to OpenPose keypoint mapping
# MediaPipe has 33 keypoints, OpenPose has 25 keypoints
# Reference: https://github.com/Atif-Anwer/Mediapipe-to-OpenPose-JSON
MEDIAPIPE_TO_OPENPOSE_MAP = {
    # OpenPose format (25 keypoints):
    # 0: Nose, 1: Neck, 2: RShoulder, 3: RElbow, 4: RWrist,
    # 5: LShoulder, 6: LElbow, 7: LWrist, 8: MidHip, 9: RHip,
    # 10: RKnee, 11: RAnkle, 12: LHip, 13: LKnee, 14: LAnkle,
    # 15: REye, 16: LEye, 17: REar, 18: LEar, 19: LBigToe,
    # 20: LSmallToe, 21: LHeel, 22: RBigToe, 23: RSmallToe, 24: RHeel
    
    0: 0,   # Nose -> Nose
    1: None, # Neck (calculated from shoulders)
    2: 12,  # RShoulder -> RightShoulder
    3: 14,  # RElbow -> RightElbow  
    4: 16,  # RWrist -> RightWrist
    5: 11,  # LShoulder -> LeftShoulder
    6: 13,  # LElbow -> LeftElbow
    7: 15,  # LWrist -> LeftWrist
    8: None, # MidHip (calculated from hips)
    9: 24,  # RHip -> RightHip
    10: 26, # RKnee -> RightKnee
    11: 28, # RAnkle -> RightAnkle
    12: 23, # LHip -> LeftHip
    13: 25, # LKnee -> LeftKnee
    14: 27, # LAnkle -> LeftAnkle
    15: 5,  # REye -> RightEye
    16: 2,  # LEye -> LeftEye
    17: 8,  # REar -> RightEar
    18: 7,  # LEar -> LeftEar
    19: 31, # LBigToe -> LeftFootIndex
    20: 31, # LSmallToe -> LeftFootIndex (approximation)
    21: 29, # LHeel -> LeftHeel
    22: 32, # RBigToe -> RightFootIndex
    23: 32, # RSmallToe -> RightFootIndex (approximation)
    24: 30  # RHeel -> RightHeel
}

# OpenPose connections for proper skeleton rendering
OPENPOSE_LIMB_SEQUENCE = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
    [1, 8], [8, 9], [9, 10], [10, 11], [8, 12], [12, 13], 
    [13, 14], [1, 0], [0, 15], [15, 17], [0, 16], [16, 18],
    [14, 19], [19, 20], [14, 21], [11, 22], [22, 23], [11, 24]
]

# Standard OpenPose colors (BGR format) - matching actual OpenPose output
OPENPOSE_COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], 
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], 
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], 
    [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 0], [255, 85, 0],
    [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0]
]

# OpenPose Face connections (70 keypoints from diagram)
OPENPOSE_FACE_CONNECTIONS = [
    # Jawline (0-16)
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
    (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
    # Left Eyebrow (17-21)
    (17, 18), (18, 19), (19, 20), (20, 21),
    # Right Eyebrow (22-26)
    (22, 23), (23, 24), (24, 25), (25, 26),
    # Nose Bridge (27-30)
    (27, 28), (28, 29), (29, 30),
    # Nose Lower (31-35)
    (31, 32), (32, 33), (33, 34), (34, 35),
    # Left Eye (36-41)
    (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),
    # Right Eye (42-47)
    (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),
    # Left Pupil 
    (68, 68),
    # Right Pupil
    (69, 69),
    # Outer Lips (48-59)
    (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54),
    (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48),
    # Inner Lips (60-67)
    (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 60),
    # Pupils (68-69)
    (68, 68), (69, 69)
]

# Color mapping for face parts (BGR)
# A simple color is assigned to each connection based on its group
FACE_COLORS = list(
    # Jawline (16 connections)
    [(255, 255, 255)] * 16 +
    # Right Eyebrow (4 connections)
    [(0, 255, 0)] * 4 +
    # Left Eyebrow (4 connections)
    [(0, 255, 0)] * 4 +
    # Nose Bridge (3 connections)
    [(255, 0, 255)] * 3 +
    # Nose Lower (4 connections)
    [(255, 0, 255)] * 4 +
    # Right Eye (6 connections)
    [(0, 0, 255)] * 6 +
    # Left Eye (6 connections)
    [(0, 0, 255)] * 6 +
    # Outer Lips (12 connections)
    [(255, 0, 0)] * 12 +
    # Inner Lips (8 connections)
    [(255, 0, 0)] * 8 +
    # Pupils (2 connections)
    [(255, 0, 0)] * 2
    
)


# A mapping from MediaPipe's 468 face landmarks to OpenPose's 70 face keypoints.
# This mapping has been manually created by referencing the official MediaPipe
# 468 landmark diagram and the OpenPose 70 keypoint standard.
MEDIAPIPE_TO_OPENPOSE_FACE_MAP = {
    # Jawline (OpenPose 0-16) - Mapped to follow the outer contour from MediaPipe diagram
    0: 127,  # Subject's Right Jaw - upper part, near ear/cheek connection
    1: 234,   # Moving down along the right jaw
    2: 93,
    3: 132,
    4: 58,
    5: 172,
    6: 136,
    7: 150,   # Subject's Right Jaw - point closest to chin tip
    8: 152,   # Chin Tip
    9: 400,   # Subject's Left Jaw - point closest to chin tip
    10: 365,  # Moving up along the left jaw
    11: 397,
    12: 435,
    13: 401,
    14: 323,
    15: 454,
    16: 356    # Subject's Left Jaw - upper part, near ear/cheek connection
,
    # Left Eyebrow (OpenPose 17-21)
    17: 55, 18: 65, 19: 52, 20: 53, 21: 46,
    # Right Eyebrow (OpenPose 22-26)
    22: 285, 23: 295, 24: 282, 25: 283, 26: 276,
    # Nose Bridge (OpenPose 27-30)
    27: 168, 28: 197, 29: 5, 30: 4,
    # Nose Lower (OpenPose 31-35)
    31: 166, 32: 44, 33: 19, 34: 457, 35: 455,
    # Left Eye (OpenPose 36-41)
    36: 33, 37: 160, 38: 158, 39: 155, 40: 145, 41: 163,
    # Right Eye (OpenPose 42-47)
    42: 463, 43: 385, 44: 388, 45: 263, 46: 373, 47: 381,
    # Outer Lips (OpenPose 48-59)
    48: 185, 49: 39, 50: 37, 51: 0, 52: 267, 53: 270, 54: 409, 55: 321, 56: 314, 57: 17, 58: 181, 59: 146,
    # Inner Lips (OpenPose 60-67)
    60: 78, 61: 81, 62: 13, 63: 311, 64: 409, 65: 402, 66: 14, 67: 178,
    # Pupils (OpenPose 68-69) - Approximated from nearby landmarks as pupils are not in the 468 set
    68: 468, # Approximation for Left Pupil (subject's left)
    69: 473, # Approximation for Right Pupil (subject's right)
}


class MediaPipePosePreprocessor(BasePreprocessor):
    """
    MediaPipe-based pose preprocessor for ControlNet that outputs OpenPose-style annotations
    
    Converts MediaPipe's 33 keypoints to OpenPose's 25 keypoints format and renders
    them in the standard OpenPose style for ControlNet compatibility.
    
    Improvements inspired by TouchDesigner MediaPipe plugin:
    - Better confidence filtering
    - Temporal smoothing for jitter reduction
    - Improved multi-pose support preparation
    """
    
    def __init__(self,
                 detect_resolution: int = 512,
                 image_resolution: int = 512,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 model_complexity: int = 1,
                 static_image_mode: bool = True,
                 draw_hands: bool = True,
                 draw_face: bool = False,  # Simplified - disable face by default
                 line_thickness: int = 2,
                 circle_radius: int = 4,
                 confidence_threshold: float = 0.3,  # TouchDesigner-style confidence filtering
                 enable_smoothing: bool = True,  # TouchDesigner-inspired smoothing
                 smoothing_factor: float = 0.7,  # Smoothing strength
                 **kwargs):
        """
        Initialize MediaPipe pose preprocessor with TouchDesigner-inspired improvements
        
        Args:
            detect_resolution: Resolution for pose detection
            image_resolution: Output image resolution
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            model_complexity: MediaPipe model complexity (0, 1, or 2)
            static_image_mode: Treat each image independently
            draw_hands: Whether to draw hand poses
            draw_face: Whether to draw face landmarks
            line_thickness: Thickness of skeleton lines
            circle_radius: Radius of joint circles
            confidence_threshold: Minimum confidence for rendering keypoints
            enable_smoothing: Enable temporal smoothing
            smoothing_factor: Smoothing strength (0-1, higher = more smoothing)
            **kwargs: Additional parameters
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "MediaPipe is required for MediaPipe pose preprocessing. "
                "Install it with: pip install mediapipe"
            )
        
        super().__init__(
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
            static_image_mode=static_image_mode,
            draw_hands=draw_hands,
            draw_face=draw_face,
            line_thickness=line_thickness,
            circle_radius=circle_radius,
            confidence_threshold=confidence_threshold,
            enable_smoothing=enable_smoothing,
            smoothing_factor=smoothing_factor,
            **kwargs
        )
        
        self._detector = None
        self._current_options = None
        # TouchDesigner-style smoothing buffers
        self._smoothing_buffers = {}
    
    @property
    def detector(self):
        """Lazy loading of the MediaPipe Holistic detector"""
        new_options = {
            'min_detection_confidence': self.params.get('min_detection_confidence', 0.5),
            'min_tracking_confidence': self.params.get('min_tracking_confidence', 0.5),
            'model_complexity': self.params.get('model_complexity', 1),
            'static_image_mode': self.params.get('static_image_mode', True),
        }
        
        # Initialize or update detector if needed
        if self._detector is None or self._current_options != new_options:
            if self._detector is not None:
                self._detector.close()
                
            print(f"MediaPipePosePreprocessor.detector: Initializing MediaPipe Holistic detector")
            self._detector = mp.solutions.holistic.Holistic(
                static_image_mode=new_options['static_image_mode'],
                model_complexity=new_options['model_complexity'],
                enable_segmentation=False,
                refine_face_landmarks=False,  # Keep simple
                min_detection_confidence=new_options['min_detection_confidence'],
                min_tracking_confidence=new_options['min_tracking_confidence'],
            )
            self._current_options = new_options
            
        return self._detector
    
    def _apply_smoothing(self, keypoints: List[List[float]], pose_id: str = "default") -> List[List[float]]:
        """
        Apply TouchDesigner-inspired temporal smoothing
        
        Args:
            keypoints: Current frame keypoints
            pose_id: Unique identifier for this pose
            
        Returns:
            Smoothed keypoints
        """
        if not self.params.get('enable_smoothing', True) or not keypoints:
            return keypoints
            
        smoothing_factor = self.params.get('smoothing_factor', 0.7)
        
        # Initialize buffer for this pose if needed
        if pose_id not in self._smoothing_buffers:
            self._smoothing_buffers[pose_id] = keypoints.copy()
            return keypoints
            
        # Apply exponential smoothing (simplified 1-euro filter style)
        smoothed = []
        previous = self._smoothing_buffers[pose_id]
        
        for i, (current_point, prev_point) in enumerate(zip(keypoints, previous)):
            if current_point[2] > 0.1:  # Only smooth if confidence is good
                smoothed_x = prev_point[0] * smoothing_factor + current_point[0] * (1 - smoothing_factor)
                smoothed_y = prev_point[1] * smoothing_factor + current_point[1] * (1 - smoothing_factor)
                smoothed_conf = current_point[2]  # Keep current confidence
                smoothed.append([smoothed_x, smoothed_y, smoothed_conf])
            else:
                smoothed.append(current_point)
        
        # Update buffer
        self._smoothing_buffers[pose_id] = smoothed
        return smoothed
    
    def _mediapipe_to_openpose(self, mediapipe_landmarks: List, image_width: int, image_height: int) -> List[List[float]]:
        """
        Convert MediaPipe landmarks to OpenPose format
        
        Args:
            mediapipe_landmarks: MediaPipe pose landmarks
            image_width: Image width
            image_height: Image height
            
        Returns:
            OpenPose keypoints in [x, y, confidence] format
        """
        if not mediapipe_landmarks:
            return []
        
        # Initialize OpenPose keypoints array (25 points x 3 values)
        openpose_keypoints = [[0.0, 0.0, 0.0] for _ in range(25)]
        
        # Convert MediaPipe landmarks to pixel coordinates
        mp_points = []
        for landmark in mediapipe_landmarks:
            x = landmark.x * image_width
            y = landmark.y * image_height
            confidence = landmark.visibility if hasattr(landmark, 'visibility') else 1.0
            mp_points.append([x, y, confidence])
        
        # Map MediaPipe points to OpenPose format
        for openpose_idx, mediapipe_idx in MEDIAPIPE_TO_OPENPOSE_MAP.items():
            if mediapipe_idx is not None and mediapipe_idx < len(mp_points):
                openpose_keypoints[openpose_idx] = mp_points[mediapipe_idx]
        
        # Calculate derived points
        confidence_threshold = self.params.get('confidence_threshold', 0.3)
        
        # Neck (1): midpoint between shoulders
        if (len(mp_points) > 12 and mp_points[11][2] > confidence_threshold and 
            mp_points[12][2] > confidence_threshold):
            neck_x = (mp_points[11][0] + mp_points[12][0]) / 2
            neck_y = (mp_points[11][1] + mp_points[12][1]) / 2
            neck_conf = min(mp_points[11][2], mp_points[12][2])
            openpose_keypoints[1] = [neck_x, neck_y, neck_conf]
        
        # MidHip (8): midpoint between hips
        if (len(mp_points) > 24 and mp_points[23][2] > confidence_threshold and 
            mp_points[24][2] > confidence_threshold):
            midhip_x = (mp_points[23][0] + mp_points[24][0]) / 2
            midhip_y = (mp_points[23][1] + mp_points[24][1]) / 2
            midhip_conf = min(mp_points[23][2], mp_points[24][2])
            openpose_keypoints[8] = [midhip_x, midhip_y, midhip_conf]
        
        return openpose_keypoints
    
    def _draw_openpose_skeleton(self, image: np.ndarray, keypoints: List[List[float]]) -> np.ndarray:
        """
        Draw OpenPose-style skeleton on image
        
        Args:
            image: Input image
            keypoints: OpenPose keypoints
            
        Returns:
            Image with skeleton drawn
        """
        if not keypoints or len(keypoints) != 25:
            return image
        
        h, w = image.shape[:2]
        line_thickness = self.params.get('line_thickness', 2)
        circle_radius = self.params.get('circle_radius', 4)
        confidence_threshold = self.params.get('confidence_threshold', 0.3)
        
        # Draw limbs
        for i, (start_idx, end_idx) in enumerate(OPENPOSE_LIMB_SEQUENCE):
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                keypoints[start_idx][2] > confidence_threshold and keypoints[end_idx][2] > confidence_threshold):
                
                start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                
                # Use standard OpenPose colors
                color = OPENPOSE_COLORS[i % len(OPENPOSE_COLORS)]
                
                cv2.line(image, start_point, end_point, color, line_thickness)
        
        # Draw keypoints
        for i, keypoint in enumerate(keypoints):
            if keypoint[2] > confidence_threshold:
                center = (int(keypoint[0]), int(keypoint[1]))
                color = OPENPOSE_COLORS[i % len(OPENPOSE_COLORS)]
                cv2.circle(image, center, circle_radius, color, -1)
        
        return image
    
    def _draw_hand_keypoints(self, image: np.ndarray, hand_landmarks: List, is_left_hand: bool = True) -> np.ndarray:
        """
        Draw hand keypoints in OpenPose style - FIXED coordinate mapping
        
        Args:
            image: Input image
            hand_landmarks: MediaPipe hand landmarks
            is_left_hand: Whether this is the left hand
            
        Returns:
            Image with hand keypoints drawn
        """
        if not hand_landmarks:
            return image
        
        h, w = image.shape[:2]
        confidence_threshold = self.params.get('confidence_threshold', 0.3)
        
        # Standard hand connections (21 landmarks per hand)
        hand_connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger  
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm connections
            (5, 9), (9, 13), (13, 17),
        ]
        
        # Convert to pixel coordinates - FIXED
        hand_points = []
        for landmark in hand_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            hand_points.append((x, y))
        
        # Standard hand colors
        hand_color = [255, 128, 0] if is_left_hand else [0, 255, 255]  # Orange for left, cyan for right
        
        # Draw connections
        for start_idx, end_idx in hand_connections:
            if start_idx < len(hand_points) and end_idx < len(hand_points):
                cv2.line(image, hand_points[start_idx], hand_points[end_idx], hand_color, 2)
        
        # Draw keypoints
        for point in hand_points:
            cv2.circle(image, point, 3, hand_color, -1)
        
        return image
    
    def _draw_face_keypoints(self, image: np.ndarray, face_landmarks: List) -> np.ndarray:
        """
        Draw face landmarks in OpenPose style.

        Args:
            image: Input image canvas.
            face_landmarks: MediaPipe face landmarks (468 points).

        Returns:
            Image with face skeleton drawn.
        """
        if not face_landmarks:
            return image

        h, w = image.shape[:2]
        line_thickness = self.params.get('line_thickness', 2)
        confidence_threshold = self.params.get('confidence_threshold', 0.3)

        # Convert MediaPipe landmarks to a list of (x, y, conf) tuples
        mp_points = []
        for landmark in face_landmarks:
            x = landmark.x * w
            y = landmark.y * h
            # Face landmarks don't have visibility/confidence, so we assume 1.0
            confidence = 1.0
            mp_points.append([x, y, confidence])

        # Map the 468 MediaPipe points to the 70 OpenPose points
        openpose_face_keypoints = [[0.0, 0.0, 0.0] for _ in range(70)]
        for openpose_idx, mediapipe_idx in MEDIAPIPE_TO_OPENPOSE_FACE_MAP.items():
            if mediapipe_idx < len(mp_points):
                openpose_face_keypoints[openpose_idx] = mp_points[mediapipe_idx]

        # Draw connections
        for i, (start_idx, end_idx) in enumerate(OPENPOSE_FACE_CONNECTIONS):
            if (start_idx < len(openpose_face_keypoints) and end_idx < len(openpose_face_keypoints) and
                openpose_face_keypoints[start_idx][2] >= confidence_threshold and
                openpose_face_keypoints[end_idx][2] >= confidence_threshold):

                start_point = (int(openpose_face_keypoints[start_idx][0]), int(openpose_face_keypoints[start_idx][1]))
                end_point = (int(openpose_face_keypoints[end_idx][0]), int(openpose_face_keypoints[end_idx][1]))
                color = FACE_COLORS[i % len(FACE_COLORS)]
                cv2.line(image, start_point, end_point, color, line_thickness)

        return image
    
    def process(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Apply MediaPipe pose detection and create OpenPose-style annotation
        
        Args:
            image: Input image
            
        Returns:
            PIL Image with OpenPose-style pose skeleton on black background
        """
        # Convert to PIL Image if needed
        image = self.validate_input(image)
        
        # Resize for detection
        detect_resolution = self.params.get('detect_resolution', 512)
        image_resized = image.resize((detect_resolution, detect_resolution), Image.LANCZOS)
        
        # Convert to RGB numpy array for MediaPipe
        rgb_image = cv2.cvtColor(np.array(image_resized), cv2.COLOR_BGR2RGB)
        
        # Run MediaPipe detection
        results = self.detector.process(rgb_image)
        
        # Create black background for pose annotation
        pose_image = np.zeros((detect_resolution, detect_resolution, 3), dtype=np.uint8)
        
        # Draw pose skeleton if detected
        if results.pose_landmarks:
            # Convert MediaPipe to OpenPose format
            openpose_keypoints = self._mediapipe_to_openpose(
                results.pose_landmarks.landmark, 
                detect_resolution, 
                detect_resolution
            )
            
            # Apply TouchDesigner-style smoothing
            openpose_keypoints = self._apply_smoothing(openpose_keypoints, "main_pose")
            
            # Draw OpenPose-style skeleton
            pose_image = self._draw_openpose_skeleton(pose_image, openpose_keypoints)
        
        # Draw hands if enabled
        draw_hands = self.params.get('draw_hands', True)
        if draw_hands:
            if results.left_hand_landmarks:
                pose_image = self._draw_hand_keypoints(
                    pose_image, results.left_hand_landmarks.landmark, is_left_hand=True
                )
            
            if results.right_hand_landmarks:
                pose_image = self._draw_hand_keypoints(
                    pose_image, results.right_hand_landmarks.landmark, is_left_hand=False
                )
        
        # Draw face if enabled
        draw_face = self.params.get('draw_face', False)
        if draw_face and results.face_landmarks:
            pose_image = self._draw_face_keypoints(
                pose_image, results.face_landmarks.landmark
            )
        
        # Convert back to PIL
        pose_pil = Image.fromarray(cv2.cvtColor(pose_image, cv2.COLOR_BGR2RGB))
        
        # Resize to target resolution
        image_resolution = self.params.get('image_resolution', 512)
        if pose_pil.size != (image_resolution, image_resolution):
            pose_pil = pose_pil.resize((image_resolution, image_resolution), Image.LANCZOS)
        
        return pose_pil
    
    def process_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Process tensor directly on GPU to avoid unnecessary CPU transfers
        
        Args:
            image_tensor: Input image tensor on GPU
            
        Returns:
            Processed pose tensor on GPU
        """
        # For MediaPipe, we need to go through CPU anyway, so use standard process
        pil_image = self.tensor_to_pil(image_tensor)
        processed_pil = self.process(pil_image)
        return self.pil_to_tensor(processed_pil)
    
    def reset_smoothing_buffers(self):
        """Reset smoothing buffers (useful for new sequences)"""
        print("MediaPipePosePreprocessor.reset_smoothing_buffers: Clearing smoothing buffers")
        self._smoothing_buffers.clear()
    
    def __del__(self):
        """Cleanup MediaPipe detector"""
        if hasattr(self, '_detector') and self._detector is not None:
            self._detector.close() 