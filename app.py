import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .emotion-box {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 10px 0;
    }
    .confidence-bar {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
IMG_SIZE = 96
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_COLORS = {
    'angry': '#ff4444',
    'disgust': '#9c27b0',
    'fear': '#ff9800',
    'happy': '#4caf50',
    'neutral': '#607d8b',
    'sad': '#2196f3',
    'surprise': '#ffeb3b'
}

# Load model with caching
@st.cache_resource
def load_model(model_path):
    """Load the trained emotion recognition model"""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image, img_size=IMG_SIZE):
    """Preprocess image for model prediction"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Resize to model input size
    resized = cv2.resize(gray, (img_size, img_size))
    
    # Normalize
    normalized = resized / 255.0
    
    # Reshape for model input
    preprocessed = normalized.reshape(1, img_size, img_size, 1)
    
    return preprocessed

def detect_face(image):
    """Detect face in image using Haar Cascade"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    return faces

def predict_emotion(model, face_image):
    """Predict emotion from face image"""
    preprocessed = preprocess_image(face_image)
    predictions = model.predict(preprocessed, verbose=0)
    
    # Get prediction probabilities
    emotion_probs = {EMOTION_LABELS[i]: float(predictions[0][i]) for i in range(len(EMOTION_LABELS))}
    
    # Get top emotion
    top_emotion = EMOTION_LABELS[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    
    return top_emotion, confidence, emotion_probs

def draw_results(image, faces, emotions_data):
    """Draw bounding boxes and emotion labels on image"""
    output_image = image.copy()
    
    for (x, y, w, h), (emotion, confidence) in zip(faces, emotions_data):
        # Draw rectangle around face
        color = tuple(int(EMOTION_COLORS[emotion].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        cv2.rectangle(output_image, (x, y), (x+w, y+h), color[::-1], 3)
        
        # Prepare label
        label = f"{emotion.upper()}: {confidence*100:.1f}%"
        
        # Calculate label size and position
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        
        # Draw label background
        cv2.rectangle(output_image, (x, y-label_h-15), (x+label_w+10, y), color[::-1], -1)
        
        # Draw label text
        cv2.putText(output_image, label, (x+5, y-8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return output_image

# Main App
def main():
    # Header
    st.markdown('<p class="main-header">😊 Real-Time Facial Emotion Recognition 😊</p>', unsafe_allow_html=True)
    
    model_path = "best_model.keras"
    model = None
    if os.path.exists(model_path):
        model = load_model(model_path)
    
    # Mode selection
    st.sidebar.markdown("---")
    mode = st.sidebar.radio("Select Mode:", ["🖼️ Test Images", "📹 Webcam"])
    
    # Main content
    if model is None:
        st.warning("⚠️ Please ensure best_model.keras is in the same folder as app.py")
        st.info("📁 Expected file: **best_model.keras**")
        
        # Display some info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 📊 Supported Emotions")
            for emotion in EMOTION_LABELS:
                st.write(f"• {emotion.capitalize()}")
        
        with col2:
            st.markdown("### 🎯 Features")
            st.write("• Multi-face detection")
            st.write("• Real-time analysis")
            st.write("• Confidence scores")
            st.write("• Visual probability bars")
        
        with col3:
            st.markdown("### 📝 Model Info")
            st.write("• Input: 96x96 grayscale")
            st.write("• Classes: 7 emotions")
            st.write("• Format: .keras")
            st.write("• Dataset: FER2013")
        
        return
    
    if mode == "🖼️ Test Images":
        image_mode(model)
    else:
        webcam_mode(model)

def image_mode(model):
    """Handle test image upload and prediction"""
    st.subheader("🖼️ Test Images - Emotion Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            if len(image_np.shape) == 3:
                image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                image_cv = image_np
            
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔍 Detect Emotions", type="primary"):
                with st.spinner("Analyzing emotions..."):
                    # Detect faces
                    faces = detect_face(image_cv)
                    
                    if len(faces) == 0:
                        st.warning("⚠️ No faces detected in the image. Please try another image.")
                    else:
                        st.success(f"✅ Detected  face(s)!")
                        
                        # Predict emotions for each face
                        emotions_data = []
                        all_predictions = []
                        
                        for (x, y, w, h) in faces:
                            face_roi = image_cv[y:y+h, x:x+w]
                            emotion, confidence, emotion_probs = predict_emotion(model, face_roi)
                            emotions_data.append((emotion, confidence))
                            all_predictions.append(emotion_probs)
                        
                        # Draw results
                        result_image = draw_results(image_cv, faces, emotions_data)
                        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                        
                        with col2:
                            st.image(result_image_rgb, caption="Detection Results", use_container_width=True)
                        
                        # Display detailed results
                        st.subheader("📊 Detailed Results")
                        
                        for i, (emotion_probs, (emotion, confidence)) in enumerate(zip(all_predictions, emotions_data)):
                            with st.expander(f"Face #{i+1} - {emotion.upper()} ({confidence*100:.1f}%)", expanded=True):
                                # Main emotion display
                                st.markdown(f"""
                                    <div class="emotion-box" style="background: {EMOTION_COLORS[emotion]};">
                                        {emotion.upper()}
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                # Confidence bars
                                st.write("**All Emotion Probabilities:**")
                                sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
                                
                                for emo, prob in sorted_emotions:
                                    st.progress(prob, text=f"{emo.capitalize()}: {prob*100:.2f}%")

def webcam_mode(model):
    """Handle webcam emotion detection"""
    st.subheader("📹 Webcam - Emotion Detection")
    
    st.info("📌 **Two Options Available:**")
    
    tab1, tab2 = st.tabs(["📸 Single Frame Capture", "🎬 Real-Time Detection"])
    
    with tab1:
        st.markdown("### Capture Single Frame from Webcam")
        st.write("Click below to capture one image from your webcam for emotion analysis.")

        # ✅ MUST BE OUTSIDE BUTTON
        camera_image = st.camera_input("📸 Capture Image from Webcam")

        if camera_image is not None:
            ret = True
            image = Image.open(camera_image)
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ret = False

        if ret:
            faces = detect_face(frame)

            col1, col2 = st.columns(2)

            with col1:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption="Captured Frame", use_container_width=True)

            if len(faces) == 0:
                st.warning("⚠️ No faces detected. Please adjust your position and try again.")
            else:
                st.success(f"✅ Detected {len(faces)} face(s)!")

                emotions_data = []
                all_predictions = []

                for (x, y, w, h) in faces:
                    face_roi = frame[y:y+h, x:x+w]
                    emotion, confidence, emotion_probs = predict_emotion(model, face_roi)
                    emotions_data.append((emotion, confidence))
                    all_predictions.append(emotion_probs)

                result_image = draw_results(frame, faces, emotions_data)
                result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

                with col2:
                    st.image(result_image_rgb, caption="Detection Results", use_container_width=True)

                st.subheader("📊 Detection Results")

                for i, (emotion_probs, (emotion, confidence)) in enumerate(zip(all_predictions, emotions_data)):
                    with st.expander(
                        f"Face #{i+1} - {emotion.upper()} ({confidence*100:.1f}%)",
                        expanded=True
                    ):
                        st.markdown(f"""
                            <div class="emotion-box" style="background: {EMOTION_COLORS[emotion]};">
                                {emotion.upper()}
                            </div>
                        """, unsafe_allow_html=True)

                        sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
                        for emo, prob in sorted_emotions:
                            st.progress(prob, text=f"{emo.capitalize()}: {prob*100:.2f}%")

    
    with tab2:
        st.markdown("### Continuous Real-Time Detection")
        st.write("For better performance with continuous video stream, use the standalone script:")
        
        st.code("python emotion.py", language="bash")
        
        st.markdown("""
        **Features of emotion.py:**
        - ⚡ Real-time face detection with MediaPipe
        - 📊 Live emotion recognition
        - 🎯 FPS counter
        - 💾 Screenshot capability (press 'S')
        - ⏸️ Pause/Resume (press SPACE)
        - 🚪 Quit (press 'Q')
        
        **Usage:**
        ```bash
        # Default usage (automatically finds best_model.keras)
        python emotion.py
        
        # Specify model
        python emotion.py --model best_model.keras
        
        # High resolution
        python emotion.py --width 1920 --height 1080
        
        # Use Haar Cascade (faster)
        python emotion.py --no-mediapipe
        ```
        """)

if __name__ == "__main__":
    main()