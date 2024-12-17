import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from PIL import ImageFont, ImageDraw, Image
import arabic_reshaper
from bidi.algorithm import get_display
import logging

# ------------------------- Setup Logging -------------------------
logging.basicConfig(level=logging.INFO)

# ------------------------- Function Definitions -------------------------
def initialize_resources():
    """Initialize hand detector, classifier, and font."""
    detector = HandDetector(maxHands=1)
    classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
    fontpath = "Fonts/arial.ttf"

    try:
        font = ImageFont.truetype(fontpath, 32)
    except Exception as e:
        logging.error(f"Error loading font: {e}")
        exit()
    
    return detector, classifier, font


def preprocess_hand_image(img, hand, imgSize=300, offset=25):
    """Preprocess the cropped hand image for classification."""
    x, y, w, h = hand['bbox']
    imgCrop = img[max(0, y-offset):y+h+offset, max(0, x-offset):x+w+offset]
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

    aspectRatio = h / w
    if aspectRatio > 1:
        k = imgSize / h
        wCal = math.ceil(k * w)
        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
        wGap = math.ceil((imgSize - wCal) / 2)
        imgWhite[:, wGap:wGap + wCal] = imgResize
    else:
        k = imgSize / w
        hCal = math.ceil(k * h)
        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
        hGap = math.ceil((imgSize - hCal) / 2)
        imgWhite[hGap:hGap + hCal, :] = imgResize

    return imgWhite


def render_arabic_text(imgOutput, text, position, font):
    """Render Arabic text on an OpenCV image."""
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)

    # Convert image to PIL for Arabic rendering
    img_pil = Image.fromarray(cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, bidi_text, font=font, fill=(255, 255, 255))

    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def main():
    """Main loop for hand gesture classification and display."""
    cap = cv2.VideoCapture(0)
    detector, classifier, font = initialize_resources()

    # Labels and parameters
    labels = ["مرحبا", "بحبك", "شكرا"]
    offset = 25
    imgSize = 300

    while True:
        success, img = cap.read()
        if not success:
            logging.error("Error: Cannot read video feed.")
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']  # Proper extraction of bbox values

            # Process and classify the hand image
            imgWhite = preprocess_hand_image(img, hand)
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

            # Render Arabic text
            imgOutput = render_arabic_text(imgOutput, labels[index], (x, y - offset - 40), font)

            # Draw bounding box
            cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (255, 0, 255), 4)
            #cv2.imshow('Image White', imgWhite)

        # Show final output
        cv2.imshow('Image', imgOutput)

        # Exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



# ------------------------- Run the Main Function -------------------------
if __name__ == "__main__":
    main()
