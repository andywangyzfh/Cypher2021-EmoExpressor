import tkinter as tk
import re
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
import curses

def update_text(screen, emotion):
   if emotion == 'Angry':
       screen.clear()
       screen.addstr(1, 1, "You looks angry. Do you want to try these?\n")
       screen.addstr(3, 1, "😡\tಠ_ಠ\t⋋_⋌\t(｀Д´)\t(▽д▽)\t-`д´-\n")
       screen.refresh()
   elif emotion == 'Happy':
       screen.clear()
       screen.addstr(1, 1, "You looks happy. Do you want to try these?\n")
       screen.addstr(3, 1, "😁\t(•‿•)\t(≧▽≦)\t⊙▽⊙\t｡^‿^｡\t\^o^/\n")
       screen.refresh()
   elif emotion == 'Neutral':
       screen.clear()
       screen.addstr(1, 1, "You looks calm. Do you want to try these?\n")
       screen.addstr(3, 1, "😐\n(•‿•)\t(--_--)\t(￣ヘ￣)\t( -_・)\t(^_-)\n")
       screen.refresh()
   elif emotion == 'Sad':
       screen.clear()
       screen.addstr(1, 1, "You looks sad. Do you want to try these?\n")
       screen.addstr(3, 1, "😢\t(｡•́︿•̀｡)\t(｡╯︵╰｡)\t(╯_╰)\t(T_T)\t(>_<)\n")
       screen.refresh()
   elif emotion == 'Surprise':
       screen.clear()
       screen.addstr(1, 1, "You looks sad. Do you want to try these?\n")
       screen.addstr(3, 1, "😲\t(⊙_⊙)\t(O.O)\t(°ロ°) !\t(・□・;)\t(・о・)\n")
       screen.refresh()


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

screen = curses.initscr()
screen.addstr(0,0,"welcome to EmoExpressor!\n")
screen.refresh()
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
        if len(max_freq) == 50:
            largest = max(set(max_freq), key=max_freq.count)
            emotion = emotion_dict[largest]
            update_text(screen, emotion)
            max_freq.clear()
            flag = False
    cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
    if flag is False or cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
curses.endwin()

def draw_tui(stdscr): 
    # create gui
    # Clear and refresh the screen for a blank canvas
    stdscr.clear()
    stdscr.refresh()


def main():
    curses.wrapper(draw_tui)

