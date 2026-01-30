import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import argparse
import time
import os
from collections import deque

# ===================== CONSTANTS =====================
IMG_SIZE = 96
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

EMOTION_COLORS = {
    'angry': (68, 68, 255),
    'disgust': (179, 39, 156),
    'fear': (0, 153, 255),
    'happy': (76, 175, 80),
    'neutral': (139, 125, 96),
    'sad': (244, 150, 33),
    'surprise': (60, 235, 255)
}

# ===================== EMOTION DETECTOR =====================
class EmotionDetector:
    def __init__(self, model_path):
        print("=" * 60)
        print("FACIAL EMOTION RECOGNITION - INITIALIZING")
        print("=" * 60)
        self.model = tf.keras.models.load_model(model_path)
        print("✅ Emotion model loaded")
        print("🔧 Initializing MediaPipe Face Detector (Tasks API)")
        base_options = python.BaseOptions(
            model_asset_path=self._download_face_model()
        )
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.5
        )
        self.face_detector = vision.FaceDetector.create_from_options(options)
        print("✅ MediaPipe initialized")

        self.prev_time = time.time()
        self.fps_deque = deque(maxlen=30)

    # ---------------- FACE MODEL ----------------
    def _download_face_model(self):
        model_path = "face_detector.tflite"
        if not os.path.exists(model_path):
            import urllib.request
            print("⬇️ Downloading MediaPipe face model...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
                model_path
            )
        return model_path

    # ---------------- PREPROCESS ----------------
    def preprocess_face(self, face):
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        normalized = resized / 255.0
        return normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    # ---------------- PREDICT ----------------
    def predict_emotion(self, face):
        preds = self.model.predict(self.preprocess_face(face), verbose=0)[0]
        idx = np.argmax(preds)
        return EMOTION_LABELS[idx], preds[idx], preds

    # ---------------- FACE DETECTION ----------------
    def detect_faces(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = self.face_detector.detect(mp_image)

        faces = []
        h, w, _ = frame.shape

        for det in result.detections:
            box = det.bounding_box
            x, y, bw, bh = box.origin_x, box.origin_y, box.width, box.height

            x = max(0, x)
            y = max(0, y)
            bw = min(bw, w - x)
            bh = min(bh, h - y)

            faces.append((x, y, bw, bh))

        return faces

    # ---------------- FPS ----------------
    def fps(self):
        now = time.time()
        fps = 1 / (now - self.prev_time)
        self.prev_time = now
        self.fps_deque.append(fps)
        return int(np.mean(self.fps_deque))

    # ---------------- RUN ----------------
    def run(self, cam=0):
        cap = cv2.VideoCapture(cam)
        print("🎬 Starting webcam emotion detection")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = self.detect_faces(frame)

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                if face.size == 0:
                    continue

                emo, conf, preds = self.predict_emotion(face)
                color = EMOTION_COLORS[emo]

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(
                    frame,
                    f"{emo.upper()} {conf*100:.1f}%",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

            cv2.putText(frame, f"FPS: {self.fps()}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Emotion Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("✅ Stopped")

# ===================== MAIN =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='best_model.keras')
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print("❌ Model not found")
        return

    EmotionDetector(args.model).run(args.camera)

if __name__ == "__main__":
    main()
