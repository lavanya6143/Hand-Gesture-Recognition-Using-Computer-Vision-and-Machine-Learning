# Hand-Gesture-Recognition-Using-Computer-Vision-and-Machine-Learning
Recognises hand gestures from live webcam feed and converts them into system commands.

# ✋ Hand Gesture Recognition Using CNN & OpenCV  
### Final Year B.Tech Project  
_A real-time hand gesture recognition system using Computer Vision, Deep Learning, NLP autocorrect, and Text-to-Speech._

---

## 📌 Project Overview

This project implements a **real-time Hand Gesture Recognition System** that detects hand gestures from a webcam, classifies them using a Convolutional Neural Network (CNN), and converts them into meaningful text and speech.  
The system is built with:

- **Python**
- **OpenCV**
- **TensorFlow/Keras CNN model**
- **NLP-based word segmentation & autocorrect**
- **Text-to-Speech (TTS)**

It is capable of understanding gestures for alphabet characters and constructing full words or sentences from them.

---

## 👥 Who Can Use This Project?

### 🔹 People with Speech or Hearing Impairments  
- Communicate naturally using hand gestures  
- System converts gestures → text → speech  
- Very useful for assistive communication  

### 🔹 Students & Researchers  
Great for learning:
- Computer Vision  
- CNN implementation  
- Gesture recognition  
- Real-time image processing  
- Human–Computer Interaction  

### 🔹 Developers & Engineers  
Can extend this system to build:
- Gesture-controlled smart home devices  
- Gesture-based computer control  
- Robotics gesture navigation  
- Virtual sign language translators  

### 🔹 Rehabilitation Centers & Special Education  
Helps individuals practice sign gestures and communicate more effectively.

---

## 🎯 How This Project Is Useful

### ✨ Enhances Accessibility  
Provides a communication bridge for individuals who cannot speak, allowing them to communicate through gestures.

### ✨ Real-Time Recognition  
Works directly from a **webcam**, no gloves or sensors needed.

### ✨ Custom Dataset Support  
You can create your own dataset using `generate_data.py`.

### ✨ Practical Learning Tool  
Combines:
- CNN training  
- Image preprocessing  
- Real-time inference  
- NLP autocorrect  
- Speech synthesis  

### ✨ Extendable System  
Useful for:
- ASL recognition  
- Smart automation  
- Robotics  
- AI-driven assistive devices  

---

## 🗂 Project Structure

Hand-Gesture-Recognition/
│
├── main.py # Full real-time gesture recognition system
├── main45.py # Simplified recognition version
├── cnn.py # CNN training script
├── check_data.py # Dataset inspection script
├── generate_data.py # Script to capture dataset images
├── trained.h5 # Trained CNN model
├── label_encoded.csv # Label mapping
├── Hand Gesture Recognition.bat
├── Dataset.zip # Zipped dataset containing A, B, C... folders

---

## 📦 Dataset Information

Dataset/
├── A/Original/.jpg
├── B/Original/.jpg
├── C/Original/*.jpg
└── ...

---

## 🧠 CNN Model Summary

The CNN architecture includes:

- 3 Convolution Blocks  
  - Conv2D → Conv2D → MaxPool → Dropout  
- Flatten layer  
- Dense (128 neurons)  
- Output layer (Softmax for classification)

The model is trained for:

- **200 epochs**
- **Batch size: 32**
- **Loss: Categorical Crossentropy**
- **Optimizer: Adam**

Saved model: **trained.h5**

---

## ▶️ How to Run the Application

### 1️⃣ Install Dependencies

```bash
pip install opencv-python numpy pandas tensorflow keras pyttsx3 wordsegment pyenchant autocomplete

# 2️⃣ Run the Main System

python main.py

🎥 Real-Time Output Includes:

Predicted alphabet for each gesture

Constructed text

Top 10 NLP autocorrect suggestions

Gesture boundary visualization

Spoken output

# 🚀 Future Enhancements

Support full ASL alphabet (A–Z)

Add dynamic gestures using LSTM (Hello, Yes, No, Thank You, etc.)

Deploy model on mobile using TensorFlow Lite

Improve gesture segmentation using background subtraction

Build a full Sign Language Translator

# 👨‍💻 Author

Lavanya Malavagoppa Gangadhara
Bachelor of Engineering — Final Year Project
Hand Gesture Recognition System using CNN & OpenCV

# 📝 Notes

The full dataset is uploaded as Dataset.zip to avoid GitHub folder upload limitations.

Extract after cloning to use in training or testing.

All core Python scripts are directly visible on GitHub for review.
