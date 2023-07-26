
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from imutils import paths
import numpy as np
from collections import defaultdict
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
from keras_preprocessing.image import img_to_array
from keras.models import load_model
import imutils
import cv2
import numpy as np
import sys
from tkinter import ttk
import tkinter as tk
import os
from playsound import playsound



main = tk.Tk()
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
main.destroy()

# Calculate size and position of background label
scale_x = 1.5
scale_y = 1.2
width = int(screen_width * scale_x)
height = int(screen_height * scale_y)

main = tk.Tk()
main.title("EMOTION BASED MUSIC RECOMMENDATION SYSTEM")
main.geometry('{}x{}'.format(width, height))

bg_image = tk.PhotoImage(file="C:/Users/Administrator/Desktop/dj.png")

bg_label = tk.Label(main, image=bg_image)
bg_label.place(x=0, y=0, width=screen_width, height=screen_height)

global value
global filename
global faces
global frame
detection_model_path = 'haarcascade_frontalface_default.xml'
emotion_model_path = '_mini_XCEPTION.106-0.65.hdf5'
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]

def upload():
    global filename
    global value
    global frame
    filename = askopenfilename(initialdir = "images")
    pathlabel.config(text=filename)
    frame = cv2.imread(filename)

def start_capture():
    capture_frame()

def capture_frame():
    global frame
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow("Live Camera", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    preprocess()

        

def preprocess():
    global frame
    global faces
    text.delete('1.0', END)
    orig_frame = frame.copy()
    orig_frame = cv2.resize(orig_frame, (48, 48))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    text.insert(END, "Total number of faces detected : " + str(len(faces)))
    
def detectEmotion():
    global faces
    if len(faces) > 0:
        faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        roi = frame[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        messagebox.showinfo("Emotion Prediction Screen", "Emotion Detected As : " + label)
        value.clear()
        path = 'songs'
        for r, d, f in os.walk(path):
            for file in f:
                if file.find(label) != -1:
                    value.append(file)
        songslist.delete(0, END)
        for val in value:
            songslist.insert(END, val)
    else:
        messagebox.showinfo("Emotion Prediction Screen", "No face detected in the live camera feed")


def playSong():
    name = songslist.get()
    playsound('songs/'+name)
    

font = ('times', int(12*scale_y), 'bold')
title = Label(main, text='EMOTION BASED MUSIC RECOMMENDATION SYSTEM')
title.config(bg='#000000', fg='white')  
title.config(font=font)           
title.config(height=int(3*scale_y), width=int(90*scale_x))       
title.place(x=int(12*scale_x), y=int(5*scale_y))


font1 = ('times', int(14*scale_y), 'bold')
upload = Button(main, text="Upload Image With Face", command=upload)
upload.place(x=int(50*scale_x), y=int(100*scale_y))
upload.config(font=font1, bg='#008CBA', fg='white', relief='raised', padx=10, pady=5, bd=2) 


pathlabel = Label(main)
pathlabel.config(bg='#004e92', fg='yellow')  
pathlabel.config(font=font1)           
pathlabel.place(x=int(300*scale_x), y=int(100*scale_y))

capture = Button(main, text="Start Live Video", command=start_capture)
capture.place(x=int(50*scale_x), y=int(300*scale_y))
capture.config(font=font1, bg='#008CBA', fg='white', relief='raised', padx=10, pady=5, bd=2) 

preprocessbutton = Button(main, text="Preprocess & Detect Face in Image", command=preprocess, fg="white", bg="#393E46", bd=0, highlightthickness=0, activebackground="#00ADB5", font=font1)
preprocessbutton.place(x=int(50*scale_x), y=int(150*scale_y))

emotion = Button(main, text="Detect Emotion", command=detectEmotion, bg="#ff5733", fg="white", activebackground="#e8613a", activeforeground="white", relief="flat", borderwidth=0, padx=int(10*scale_x), pady=int(5*scale_y), font=font1)
emotion.place(x=int(50*scale_x), y=int(200*scale_y))

emotionlabel = Label(main)
emotionlabel.config(bg='#000000', fg='white')  
emotionlabel.config(font=font1)           
emotionlabel.place(x=int(610*scale_x), y=int(200*scale_y))
emotionlabel.config(text="Predicted Song")

value = ["Song List"]
songslist = ttk.Combobox(main,values=value,postcommand=lambda: songslist.configure(values=value)) 
songslist.place(x=int(760*scale_x), y=int(210*scale_y))
songslist.current(0)
songslist.config(font=font1)  

playsong = Button(main, text="Play Song", command=playSong, bg="#2ecc71", fg="white", bd=0, padx=10, pady=5, activebackground="#27ae60", activeforeground="white", font=font1)
playsong.place(x=int(50*scale_x), y=int(250*scale_y))


font1 = ('times', int(12*scale_y), 'bold')
text = Text(main, height=int(3*scale_y), width=int(20*scale_x), bg='#000000' , fg= 'white')
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=int(400*scale_x), y=int(500*scale_y))
text.config(font=font1)

main.config(bg='#000000')
main.mainloop()