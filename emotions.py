import tkinter as tk
from PIL import ImageTk, Image
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from threading import Thread
from time import sleep
import os


# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# variable contains mood
emotion = "Sad"

def detector():
    # emotions will be displayed on your face from the webcam feed
    model.load_weights('model.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # start the webcam feed
    cap = cv2.VideoCapture(0)
    max_freq = []
    while True:
        flag = True
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # judege which expression is the most frequent in last 15 iterations
            max_freq.append(maxindex)
            if len(max_freq) == 150:
                largest = max(set(max_freq), key=max_freq.count)
                emotion = emotion_dict[largest]
                max_freq.clear()
                flag = False
        cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
        if flag is False or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def gui():
    # create gui
    # Main window
    root = tk.Tk()
    root.title("EmoExpressor")
    root.geometry("700x500+300+150")
    root.resizable(width=True, height=True)

    # Logo
    img = Image.open("logo.png")
    img = img.resize((350, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = tk.Label(root, image = img)
    panel.pack(side = "top", fill = "both", expand = 0)

    # Feedback
    message = tk.Label(root, text="You looks {0} do you want to try these?".format(emotion), width=200, font=("Arial", 25), anchor="w", height = 2)
    message.pack()

    # Emoji
    T = tk.Text(root, height=7, width=100)
    T.pack()
    T.insert(tk.END, 'Your emoji goes here...')

    root.mainloop()

def main():
    t1 = Thread(target=gui)
    t2 = Thread(target=detector)
    t1.start()
    t2.start()


if __name__ == '__main__':
    main()