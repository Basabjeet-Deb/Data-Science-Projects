# Drowsiness Detector 😴🚗

A computer vision and deep learning project designed to prevent accidents by detecting driver drowsiness in real-time. This system monitors the user's eyes and triggers an alert if signs of drowsiness (closed eyes) are detected for a specific duration.

## 📂 Repository Structure

Based on the files in this repository:

* **`haar cascade files/`**: Contains the XML files required by OpenCV for face and eye detection.
* **`models/`**: Stores the pre-trained Convolutional Neural Network (CNN) model used for classifying the state of the eyes (Open/Closed).
* **`Drowsiness Detector main`**: The Kaggle Notebook containing the source code for data preprocessing and model training.
* **`Normal Main Code`**: The main Python script to launch the real-time detection application.

## 🛠️ Tech Stack

* **Language:** Python
* **Computer Vision:** OpenCV (cv2)
* **Deep Learning:** Keras / TensorFlow (CNN)
* **Other Libraries:** NumPy, Pygame (for audio alerts)

## 🚀 How to Run

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Basabjeet-Deb/Drowsiness-Detector.git](https://github.com/Basabjeet-Deb/Drowsiness-Detector.git)
    cd Drowsiness-Detector
    ```

2.  **Install Dependencies:**
    Ensure you have the necessary libraries installed:
    ```bash
    pip install opencv-python tensorflow numpy pygame
    ```

3.  **Run the Application:**
    Execute the main script to start the camera and detection system:
    ```bash
    python "Normal Main Code"
    ```

## ⚙️ How It Works

1.  **Face & Eye Detection:** The system uses Haar Cascade classifiers to locate the face and eyes from the webcam feed.
2.  **State Classification:** The detected eye region is fed into a trained CNN model to predict if the eye is "Open" or "Closed".
3.  **Score Calculation:** A "Score" increases for every frame the eyes are closed.
4.  **Alert:** If the score exceeds a defined threshold (indicating prolonged closure), an alarm is triggered to wake the driver.

## 👨‍💻 Author

**Basabjeet Deb**
* [LinkedIn](https://www.linkedin.com/in/basabjeet-deb)
* [GitHub](https://github.com/Basabjeet-Deb)

---
*This project was developed as part of my portfolio in Data Science and Machine Learning.*
