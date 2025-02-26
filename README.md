# 🚀 Enhanced Vision Assistant  

A **state-of-the-art** computer vision application designed to assist **visually impaired individuals** with real-time navigation and situational awareness. The system leverages cutting-edge AI technologies (**Vertex AI, Gemini Pro, Google Cloud Vision API, and Google Text-to-Speech API**) to **identify objects, evaluate risks, and deliver smart audio directions** through voice commands and natural language processing.

## 🎥 Demo Video

https://github.com/user-attachments/assets/fe943124-3d3b-4ba1-ab87-324b21469421

## ✨ Features  

- **Real-time object detection** and **depth estimation**  
- **Intelligent scene analysis** and **risk assessment**  
- **Priority-based audio guidance system**  
- **Context-aware navigation assistance**  
- **Dynamic hazard detection** and **avoidance**  
- **Advanced motion tracking** and **trajectory analysis**  
- **Voice-activated commands and responses**  
- **Natural language scene description**  
- **Spatial awareness and proximity alerts**  
- **Debug visualization for development purposes**  

## 🛠️ Technologies Used  

- **Computer Vision**: OpenCV, Google Cloud Vision API  
- **AI/ML**: Google Vertex AI (**Gemini Pro**)  
- **Speech Synthesis**: Google Cloud Text-to-Speech  
- **Audio Processing**: Pygame  
- **Additional Libraries**: NumPy, SciPy  

## 📋 Requirements  

- 🐍 **Python 3.7+**  
- ☁️ **Google Cloud Platform account** with the following APIs enabled:  
  - Cloud Vision API  
  - Text-to-Speech API  
  - Vertex AI API  
- 📷 **Webcam** or compatible camera device  
- 🎧 **Audio output device**  

## 🚀 Installation  

1️⃣ **Clone the repository:**  
```bash
git clone https://github.com/yourusername/enhanced-vision-assistant.git
cd enhanced-vision-assistant
```  

2️⃣ **Install required packages:**  
```bash
pip install -r requirements.txt  
```  

3️⃣ **Set up Google Cloud credentials:**  
   - **Create a service account** and download the **JSON key file**  
   - **Set the path** to your credentials in the `CREDENTIALS_PATH` variable  
   - **Configure your** Google Cloud **Project ID** in the `PROJECT_ID` variable  

## ⚙️ Configuration  

Update the following variables in the `EnhancedVisionAssistant` class:  

```python
self.PROJECT_ID = 'your-project-id'
self.REGION = 'your-region'
self.CREDENTIALS_PATH = 'path/to/your/credentials.json'
```

## 🎯 How It Works  

1️⃣ **Initialize** the camera and audio systems  
2️⃣ **Detect** objects in real time  
3️⃣ **Analyze** the environment and provide **smart audio guidance**  
4️⃣ **Display** a debug window showing detected objects and their priorities  

**Press 'q' to quit the application.**  
