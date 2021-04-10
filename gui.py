import tkinter as tk
from PIL import ImageTk, Image

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
message = tk.Label(root, text="You looks ..., do you want to try these?", width=200, font=("Arial", 25), anchor="w", height = 2)
message.pack()

# Emoji
T = tk.Text(root, height=7, width=100)
T.pack()
T.insert(tk.END, 'Your emoji goes here...')

root.mainloop()
