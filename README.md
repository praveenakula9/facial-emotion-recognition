# 😊 Facial Emotion Recognition – Streamlit App

This project is a **Facial Emotion Recognition web application** built using **Streamlit** and **Deep Learning**.  
It allows users to detect emotions from **uploaded images** or **webcam captures** using a trained CNN model.

---

## ✨ Features
- Emotion detection from **uploaded images**
- Emotion detection using **webcam capture**
- Multi-face detection using **OpenCV Haar Cascades**
- CNN-based emotion classification
- Confidence scores for each emotion
- Clean and interactive **Streamlit UI**
- Cached model loading for faster performance

---

## 🧠 Emotions Detected
- Angry  
- Disgust  
- Fear  
- Happy  
- Neutral  
- Sad  
- Surprise  

---

## 🛠 Tech Stack
- Python
- Streamlit
- TensorFlow / Keras
- OpenCV
- NumPy
- Pillow (PIL)

---

## 📁 Project Structure

```text
.
├── app.py                # Main real-time emotion detection script
├── best_model.keras      # Trained CNN model (Git LFS)
├── requirements.txt      # Python dependencies
├── .gitattributes        # Git LFS configuration
└── README.md
```

---

## 🧠 How It Works
1. User uploads an image or captures one using webcam  
2. Faces are detected using Haar Cascade classifier  
3. Face regions are preprocessed (grayscale, resize, normalize)  
4. CNN model predicts emotion probabilities  
5. Emotion label and confidence are displayed on the UI  

---

## 🚀 Quick Start (Local Execution)

Follow these steps to run the app on your local machine:

```bash
git clone https://github.com/praveenakula9/facial-emotion-recognition.git
cd facial-emotion-recognition
python -m venv mp_env
mp_env\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
---
## 🌐 Online Demo

You can try the Streamlit demo online here:

👉 **https://facial-emotion-recognition-380.streamlit.app/**


