import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# ------------------------ Function Definitions ------------------------

def initialize_resources(folder_path, img_size=300, offset=25):
    """
    Initializes video capture, hand detector, and settings.
    Ensures the save folder exists.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Error: Cannot access the webcam.")

    detector = HandDetector(maxHands=1)

    # Create save folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created at: {folder_path}")

    print("Resources initialized. Press 's' to save, 'q' to quit.")
    return cap, detector, img_size, offset


def preprocess_hand_image(img, hand, img_size=300, offset=25):
    """
    Crops and preprocesses the hand image to a fixed size.
    Returns the processed white image.
    """
    x, y, w, h = hand['bbox']
    imgCrop = img[max(0, y-offset):y+h+offset, max(0, x-offset):x+w+offset]
    imgWhite = np.ones((img_size, img_size, 3), np.uint8) * 255

    aspectRatio = h / w
    if aspectRatio > 1:
        k = img_size / h
        wCal = math.ceil(k * w)
        imgResize = cv2.resize(imgCrop, (wCal, img_size))
        wGap = math.ceil((img_size - wCal) / 2)
        imgWhite[:, wGap:wGap + wCal] = imgResize
    else:
        k = img_size / w
        hCal = math.ceil(k * h)
        imgResize = cv2.resize(imgCrop, (img_size, hCal))
        hGap = math.ceil((img_size - hCal) / 2)
        imgWhite[hGap:hGap + hCal, :] = imgResize

    return imgWhite, imgCrop


def save_image(folder, image, counter):
    """
    Saves the processed image to the specified folder with a timestamp.
    """
    filename = f"{folder}/Image_{time.time()}.jpg"
    cv2.imwrite(filename, image)
    print(f"Image {counter} saved: {filename}")


# ------------------------ Main Function ------------------------

def main():
    # Parameters
    # Name of the folder/The word 
    folder = "Test/Thanks"
    counter = 0
    cap, detector, img_size, offset = initialize_resources(folder)

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Cannot read the video feed.")
            break

        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            imgWhite, imgCrop = preprocess_hand_image(img, hand, img_size, offset)

            # Display the processed images
            cv2.imshow('Processed Image', imgWhite)
            cv2.imshow('Cropped Image', imgCrop)

        # Show original feed
        cv2.imshow('Webcam Feed', img)

        # Keyboard controls
        key = cv2.waitKey(1)
        if key == ord('s'):  # Save image
            counter += 1
            save_image(folder, imgWhite, counter)
        elif key == ord('q'):  # Quit program
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()


# ------------------------ Run the Script ------------------------

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")

