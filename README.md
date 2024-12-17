# **Hand Gesture Recognition: Data Collection and Classification**

## **Overview**
This project includes two key components:
1. **Data Collection Tool**: Collect hand gesture images using a webcam to train a machine learning model.
2. **Hand Gesture Classifier**: Classifies hand gestures in real-time using a trained model and displays Arabic labels with proper **right-to-left (RTL)** text rendering.

You can use **[Teachable Machine by Google](https://teachablemachine.withgoogle.com/)** to train a hand gesture classification model using the data collected from this tool.

---

## **Features**

### Data Collection Tool (`datacollection.py`)
- Real-time hand detection using the webcam.
- Crops and preprocesses hand images into a consistent size (300x300 pixels).
- Saves images to organized folders for model training.
- Interactive controls:
   - Press **`s`** to save an image.
   - Press **`q`** to quit the program.

### Hand Gesture Classifier (`main.py`)
- Classifies hand gestures using a pre-trained model.
- Displays classification results as Arabic text with proper RTL support.
- Renders text using **PIL** and `arabic-reshaper` for correct Arabic display.
- Interactive controls:
   - Press **`q`** to quit.

---

## **Folder Structure**

Your project directory should look like this:

```plaintext
Hand-Gesture-Recognition/
│
├── Data/                  # Directory to store collected images
│   ├── Gesture_1/         # Folder for 'Gesture 1' data
│   ├── Gesture_2/         # Folder for 'Gesture 2' data
│   └── ...                # Add folders for additional gestures
│
├── Fonts/                 # Arabic font files
│   └── arial.ttf          # Example Arabic font
│
├── Model/                 # Pre-trained model and labels
│   ├── keras_model.h5     # Keras-trained model file
│   └── labels.txt         # Labels for the hand gestures
│
├── datacollection.py      # Script for hand gesture data collection
├── main.py                # Main script for real-time classification
├── requirements.txt       # List of required libraries
└── README.md              # Project documentation
```

---

## **Setup Instructions**

### **1. Install Dependencies**

Ensure you have Python 3 installed, then install the required libraries:

```bash
pip install -r requirements.txt
```

---

### **2. Run the Data Collection Tool**

1. Update the `folder` variable in `datacollection.py` to specify where the images will be saved:
   ```python
   folder = "Data/Thanks"
   ```
   Replace "Thanks" with the gesture name you are collecting.

2. Run the script:
   ```bash
   python datacollection.py
   ```

3. Controls:
   - **`s`**: Save the current hand image.
   - **`q`**: Quit the program.

4. Collected images will be saved in the specified folder.

---

### **3. Train a Model Using Teachable Machine**

[Teachable Machine by Google](https://teachablemachine.withgoogle.com/) is a free and user-friendly tool to train image classification models without writing code.

#### Steps to Train the Model:
1. Visit [Teachable Machine](https://teachablemachine.withgoogle.com/).
2. Choose **"Image Project"**.
3. Upload the collected gesture images into separate classes.
4. Train the model.
5. Export the trained model in **Keras format** (TensorFlow/Keras `.h5`).
6. Save the exported model file (`keras_model.h5`) and the labels file (`labels.txt`) into the `Model/` folder of this project.

---

### **4. Run the Hand Gesture Classifier**

1. Ensure your trained model (`keras_model.h5`) and labels file (`labels.txt`) are in the `Model` folder.

2. Run the `main.py` script:
   ```bash
   python main.py
   ```

3. Controls:
   - **`q`**: Quit the program.

4. Real-time predictions will appear above the detected hand with Arabic text displayed correctly from **right-to-left**.

---

## **Code Breakdown**

### **Data Collection (`datacollection.py`)**
- Detects hands using `cvzone.HandTrackingModule`.
- Crops the hand region and resizes it to a white background of size 300x300 pixels.
- Saves preprocessed images with unique filenames using timestamps.

### **Hand Gesture Classifier (`main.py`)**
- Detects the hand and predicts the gesture using a trained model.
- Displays the result as Arabic text above the hand using:
   - **`arabic-reshaper`**: Ensures proper Arabic letter connections.
   - **`bidi.algorithm`**: Enforces right-to-left text alignment.
   - **PIL**: Renders text cleanly on the OpenCV window.

---

## **Requirements**

The project requires the following libraries:

```
opencv-python
cvzone
numpy
pillow
arabic-reshaper
python-bidi
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## **Future Improvements**
- Integrate automatic training pipelines after data collection.
- Add gesture-controlled actions (e.g., controlling devices).
- Use more advanced gesture models for higher accuracy.

---
# Hand-Gesture-Recognition
