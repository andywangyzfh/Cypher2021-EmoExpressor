<<<<<<< HEAD
from tkinter import * 
from tkinter.ttk import *
import re
root = Tk()
root.title("EmoExpresser")
label = 'Angry'
w = Label(root, text ='Your emote', font = "50") 
w.pack()
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
def display_emote():
    #label = label update the label based on emote returned from detector
    if label == 'Angry':
        w.config(text= with_surrogates('😡'))
    elif label == 'Happy':
        w.config(text= with_surrogates('😁'))
    elif label == 'Neutral':
        w.config(text= with_surrogates('😐'))
    elif label == 'Sad':
        w.config(text= with_surrogates('😢'))
    elif label == 'Surprise':
        w.config(text= with_surrogates('😲'))
    root.after(1000, display_emote)    

root.after(1000, display_emote)
root.mainloop()

