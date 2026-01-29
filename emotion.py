"""
Real-Time Facial Emotion Recognition using MediaPipe
Run locally for continuous webcam emotion detection
"""

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import argparse
import time
import os
from collections import deque

# Constants
IMG_SIZE = 96
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_COLORS = {
    'angry': (68, 68, 255),      # Red
    'disgust': (179, 39, 156),   # Purple
    'fear': (0, 153, 255),       # Orange
    'happy': (76, 175, 80),      # Green
    'neutral': (139, 125, 96),   # Gray
    'sad': (244, 150, 33),       # Blue
    'surprise': (60, 235, 255)   # Yellow
}

class EmotionDetector:
    def __init__(self, model_path, use_mediapipe=True):
        """Initialize emotion detector"""
        print("="*60)
        print("FACIAL EMOTION RECOGNITION - INITIALIZING")
        print("="*60)
        
        # Load emotion recognition model
        print(f"\n📁 Loading model: {model_path}")
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("✅ Model loaded successfully!")
            print(f"   Input shape: {self.model.input_shape}")
            print(f"   Output shape: {self.model.output_shape}")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            exit(1)
        
        self.use_mediapipe = use_mediapipe
        
        if use_mediapipe:
            # Initialize MediaPipe Face Detection
            print("\n🔧 Initializing MediaPipe Face Detection...")
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,  # 1 for full range, 0 for short range
                min_detection_confidence=0.5
            )
            print("✅ MediaPipe initialized!")
        else:
            # Initialize Haar Cascade as fallback
            print("\n🔧 Initializing Haar Cascade Face Detection...")
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            print("✅ Haar Cascade initialized!")
        
        # For FPS calculation
        self.fps_deque = deque(maxlen=30)
        self.prev_time = time.time()
        
        print("\n✅ Emotion Detector Ready!")
        print("="*60 + "\n")
    
    def preprocess_face(self, face_img):
        """Preprocess face image for emotion prediction"""
        # Convert to grayscale
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
        
        # Resize to model input size
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        
        # Normalize
        normalized = resized / 255.0
        
        # Reshape for model
        preprocessed = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        
        return preprocessed
    
    def predict_emotion(self, face_img):
        """Predict emotion from face image"""
        preprocessed = self.preprocess_face(face_img)
        predictions = self.model.predict(preprocessed, verbose=0)[0]
        
        # Get top emotion
        emotion_idx = np.argmax(predictions)
        emotion = EMOTION_LABELS[emotion_idx]
        confidence = predictions[emotion_idx]
        
        return emotion, confidence, predictions
    
    def detect_faces_mediapipe(self, frame):
        """Detect faces using MediaPipe"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            h, w, _ = frame.shape
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Ensure coordinates are within frame
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                faces.append((x, y, width, height))
        
        return faces
    
    def detect_faces_haar(self, frame):
        """Detect faces using Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def calculate_fps(self):
        """Calculate current FPS"""
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        self.fps_deque.append(fps)
        return np.mean(self.fps_deque)
    
    def draw_emotion_box(self, frame, x, y, w, h, emotion, confidence, predictions):
        """Draw bounding box and emotion information"""
        # Get color for emotion
        color = EMOTION_COLORS.get(emotion, (255, 255, 255))
        
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        # Prepare main label
        label = f"{emotion.upper()}: {confidence*100:.1f}%"
        
        # Calculate label background size
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
        )
        
        # Draw label background
        cv2.rectangle(
            frame,
            (x, y - label_h - 20),
            (x + label_w + 10, y),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label,
            (x + 5, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )
        
        # Draw emotion probability bars (top 3 emotions)
        top_emotions = np.argsort(predictions)[-3:][::-1]
        bar_y = y + h + 30
        
        for idx in top_emotions:
            emo = EMOTION_LABELS[idx]
            prob = predictions[idx]
            
            # Draw bar background
            bar_width = int(200 * prob)
            cv2.rectangle(
                frame,
                (x, bar_y),
                (x + 200, bar_y + 20),
                (50, 50, 50),
                -1
            )
            
            # Draw bar fill
            cv2.rectangle(
                frame,
                (x, bar_y),
                (x + bar_width, bar_y + 20),
                EMOTION_COLORS[emo],
                -1
            )
            
            # Draw text
            text = f"{emo}: {prob*100:.0f}%"
            cv2.putText(
                frame,
                text,
                (x + 5, bar_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            bar_y += 25
        
        return frame
    
    def process_frame(self, frame):
        """Process a single frame"""
        # Detect faces
        if self.use_mediapipe:
            faces = self.detect_faces_mediapipe(frame)
        else:
            faces = self.detect_faces_haar(frame)
        
        # Process each face
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Skip if face is too small
            if face_roi.size == 0:
                continue
            
            # Predict emotion
            emotion, confidence, predictions = self.predict_emotion(face_roi)
            
            # Draw results
            frame = self.draw_emotion_box(frame, x, y, w, h, emotion, confidence, predictions)
        
        # Calculate and display FPS
        fps = self.calculate_fps()
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Display face count
        cv2.putText(
            frame,
            f"Faces: {len(faces)}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Display detection method
        method = "MediaPipe" if self.use_mediapipe else "Haar Cascade"
        cv2.putText(
            frame,
            f"Method: {method}",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        
        return frame
    
    def run(self, camera_id=0, width=1280, height=720):
        """Run real-time emotion detection"""
        # Open webcam
        print(f"📹 Opening camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Error: Cannot access camera!")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Get actual resolution
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✅ Camera opened successfully!")
        print(f"   Resolution: {actual_width}x{actual_height}")
        print("\n" + "="*60)
        print("CONTROLS:")
        print("="*60)
        print("  Q          - Quit application")
        print("  S          - Save screenshot")
        print("  SPACE      - Pause/Resume")
        print("  H          - Toggle help text")
        print("="*60 + "\n")
        print("🎬 Starting emotion detection...\n")
        
        paused = False
        show_help = True
        screenshot_count = 0
        
        while True:
            if not paused:
                ret, frame = cap.read()
                
                if not ret:
                    print("❌ Error: Failed to capture frame!")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                current_frame = processed_frame
            
            # Draw help text
            if show_help:
                help_y = frame.shape[0] - 100
                cv2.putText(current_frame, "Press 'H' to hide help", (10, help_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(current_frame, "Q: Quit | S: Screenshot | SPACE: Pause", (10, help_y + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Real-Time Emotion Recognition', current_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("\n🛑 Quitting...")
                break
            elif key == ord('s') or key == ord('S'):
                filename = f"emotion_screenshot_{screenshot_count}.jpg"
                cv2.imwrite(filename, current_frame)
                print(f"📸 Screenshot saved: {filename}")
                screenshot_count += 1
            elif key == ord(' '):
                paused = not paused
                status = "⏸️  Paused" if paused else "▶️  Resumed"
                print(status)
            elif key == ord('h') or key == ord('H'):
                show_help = not show_help
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        if self.use_mediapipe:
            self.face_detection.close()
        
        print("\n✅ Emotion detection stopped.")
        print(f"📊 Total screenshots saved: {screenshot_count}")
        print("="*60)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Real-Time Facial Emotion Recognition with MediaPipe',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python emotion.py
  python emotion.py --model best_model.keras
  python emotion.py --model facial.h5 --width 1920 --height 1080
  python emotion.py --no-mediapipe --camera 1
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='best_model.keras',
        help='Path to trained emotion recognition model (default: best_model.keras)'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera ID (default: 0)'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=1280,
        help='Camera width (default: 1280)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=720,
        help='Camera height (default: 720)'
    )
    parser.add_argument(
        '--no-mediapipe',
        action='store_true',
        help='Use Haar Cascade instead of MediaPipe (faster but less accurate)'
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        print("\nTrying to find alternative models...")
        
        # Try to find alternative models
        alternatives = ['best_model.keras', 'facial.h5', 'best_model.h5']
        found = False
        
        for alt in alternatives:
            if os.path.exists(alt):
                print(f"✅ Found: {alt}")
                args.model = alt
                found = True
                break
        
        if not found:
            print("\n❌ No model found. Please specify a valid model path.")
            print("Usage: python emotion.py --model your_model.h5")
            return
    
    # Create detector
    detector = EmotionDetector(
        model_path=args.model,
        use_mediapipe=not args.no_mediapipe
    )
    
    # Run detection
    try:
        detector.run(
            camera_id=args.camera,
            width=args.width,
            height=args.height
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()