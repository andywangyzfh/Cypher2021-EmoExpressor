from tkinter import * 
from tkinter.ttk import *
root = Tk()
root.title("EmoExpresser")
label = 'Angry'
w = Label(root, text ='Your emote', font = "50") 
w.pack()
def display_emote():
    label = label #update the label based on emote returned from detector
    if label == 'Angry':
        w.config(text="Angry")
    elif label == 'Happy':
        w.config(text="Happy")
    elif label == 'Neutral':
        w.config(text="Neutral")
    elif label == 'Sad':
        w.config(text="Sad")
    elif label == 'Surprise':
        w.config(text="Surprise")
    root.after(1000, display_emote)    

root.after(1000, display_emote)
root.mainloop()

