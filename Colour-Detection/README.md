# Colour-Detection
🎨 Color Detection Using OpenCV & Gradio
An interactive color detection project using OpenCV, Pandas, and Gradio. Detects the color of any pixel in an uploaded image by clicking on it, displaying the color name, RGB values, and a color preview.

🚀 Features
✅ Detects color at any clicked position in an image
✅ Uses a color dataset (colors.csv) to match the closest color
✅ Supports interactive Gradio UI
✅ Displays RGB values and color patch
✅ Works with any uploaded image

📂 Installation
Clone the repository:
git clone https://github.com/your-username/color-detection-opencv.git
cd color-detection-opencv

Install dependencies:
pip install opencv-python pandas numpy gradio

▶️ Run the Project:
python color_detection.py

It will launch a Gradio web interface where you can upload an image and click to detect colors.

🖼️ Usage
Upload an image (or use the default one).
Click on the image to detect the color at that position.
See the detected color name, RGB values, and preview.
📜 Dataset (colors.csv)
The project uses a CSV dataset of 865 colors, including their RGB values and names.

🛠️ Technologies Used
Python
OpenCV (for image processing)
Pandas (for handling color dataset)
NumPy (for array manipulation)
Gradio (for interactive UI)
🤝 Contributing
Pull requests are welcome! Feel free to submit issues or feature requests.

📜 License
MIT License
