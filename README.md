# Cypher 2021 - EmoExpressor

An emoji recommender based on real-time emotion detection.  

### What it does?  
- Capture your face, analyze your emotion, and recommend some emoji and emoticons that suit your mood.  

### Inspiration  
- We often use emojis and emoticons to express our feelings when texting or using social media. However, it's not easy to select and paste emojis on computers.
- People often don't know which emoji to choose. AI can decide it for you.  
- College students face a lot of pressure. So why not have some fun with an AI?  

### How we build it
- Use OpenCV to analyze the real-time video captured from the front camera
- Use Tensorflow and Keras to build the emotion recognition models
- Implement the web-based front end via Flask

### Challenges we face  
- We are all new to deep learning. It took us effort to understand the logic behind emotion detection.  
- We are also new to front-end developing. We spent lots of time making the UI responds to the outputs of the deep learning model.  

### Accomplishments that we're proud of
- Achieved a working emotion detection function
- Produced a nice web UI with no prior experience
- Learned lots of new techniques within days

### What we learned?  
- Implementation of CNN in python (not the whole theory, but how to use it)  
- Using OpenCV package to implement emotion recognition  
- Developing a web-based front end for a python program with flask  
- Develop user interface with Tkinter and curses (though they are not used in the final version)  
- Project management for team work

### What's next for EmoExpressor
- A better user interface and a desktop client
- More available emotions and better emotion recognition precision
- Better user experience design, e.g., click to copy, camera switch, etc.

### Reference
- Thanks to [atulapra](https://github.com/atulapra/Emotion-detection) for his models
- Data set used in this project: [FER-2013](https://www.kaggle.com/deadskull7/fer2013)


**Requirement:** Python3, OpenCV, Keras, Tensorflow, Flask
