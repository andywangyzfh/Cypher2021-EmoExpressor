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

_nonbmp = re.compile(r'[\U00010000-\U0010FFFF]')
def _surrogatepair(match):#convert emoji to unicode chars that compiler recognizes
    char = match.group()
    assert ord(char) > 0xffff
    encoded = char.encode('utf-16-le')
    return (
        chr(int.from_bytes(encoded[:2], 'little')) + 
        chr(int.from_bytes(encoded[2:], 'little')))
def with_surrogates(text):
    return _nonbmp.sub(_surrogatepair, text)

 

class MainFrame(tk.Frame):
    def __init__(self,master):
        self.master = master
        self.frame = tk.Frame(self.master)
        self.frame.pack()
        master.title("EmoExpressor")
        master.geometry("700x500+300+150")
        master.resizable(width=True, height=True)
        # Logo
        img = Image.open("logo.png")
        img = img.resize((350, 250), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        self.panel = tk.Label(master, image = img)
        self.panel.pack(side = "top", fill = "both", expand = 0)
        # Feedback
        self.message = tk.Label(master, text="You looks {0} do you want to try these?".format(emotion), width=200, font=("Arial", 25), anchor="w", height = 2)
        self.message.pack()
        # Emoji
        self.T = tk.Text(master, height=7, width=100)
        self.T.pack()
        self.displayemote = tk.Label(master, text ='Your emote', font = "50") 
        self.displayemote.pack()
        #w.config(text= with_surrogates('😐'))

    def display_emote(self):
        #update the emtion based on emote returned from detector
        if emotion == 'Angry':
            self.displayemote.config(text= with_surrogates('😡'))
            self.T.config(text="ಠ_ಠ\t⋋_⋌\t(｀Д´)\t(▽д▽)\t-`д´-\n")
        elif emotion == 'Happy':
            self.displayemote.config(text= with_surrogates('😁'))
            self.T.config(text="(•‿•)\t(≧▽≦)\t⊙▽⊙\t｡^‿^｡\t\^o^/\n")
        elif emotion == 'Neutral':
            self.displayemote.config(text= with_surrogates('😐'))
            self.T.config(text="(•‿•)\t(--_--)\t(￣ヘ￣)\t( -_・)\t(^_-)\n")
        elif emotion == 'Sad':
            self.displayemote.config(text= with_surrogates('😢'))
            self.T.config(text="(｡•́︿•̀｡)\t(｡╯︵╰｡)\t(╯_╰)\t(T_T)\t(>_<)\n")
        elif emotion == 'Surprise':
            self.displayemote.config(text= with_surrogates('😲'))
            self.T.config(text="(⊙_⊙)\t(O.O)\t(°ロ°) !\t(・□・;)\t(・о・)\n")
        self.master.after(1000, self.display_emote)

def gui(): 
    # create gui
    root = tk.Tk()
    app = MainFrame(root)
    root.after(1000, app.display_emote())
    root.mainloop()

def main():
    t1 = Thread(target=gui)
    t2 = Thread(target=detector)
    t1.start()
    t2.start()


if __name__ == '__main__':
    main()
