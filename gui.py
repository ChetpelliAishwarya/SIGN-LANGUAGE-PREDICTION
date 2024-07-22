import cv2
import numpy as np
import tkinter as tk
import time

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.utils import to_categorical
import tkinter as tk
import time

model = load_model(r'C:\Users\chais\Desktop\SIGN DETECTION\signlanguagedetectionmodel.h5')


image_height, image_width = 64, 64
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                'U', 'V', 'W', 'X', 'Y', 'Z', 'nothing', 'space', 'del']

phrase_map = {
    ('H', 'O', 'W'): 'how',
    ('A', 'R', 'E'): 'are',
    ('Y', 'O', 'U'): 'you'
}


def preprocess_frame(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    frame_resized = cv2.resize(frame_gray, (image_height, image_width))
    frame_resized = np.expand_dims(frame_resized, axis=0)
    frame_resized = np.expand_dims(frame_resized, axis=-1)  
    frame_resized = frame_resized.astype('float32') / 255.0  
    return frame_resized

def predict(frame):
    preprocessed_frame = preprocess_frame(frame)
    prediction = model.predict(preprocessed_frame)
    predicted_class = np.argmax(prediction)
    return class_labels[predicted_class]


cap = cv2.VideoCapture(0)  


root = tk.Tk()
root.title('ASL Alphabet Prediction')
prediction_label = tk.Label(root, text='Predicted Sentence:')
prediction_label.pack()

sentence_label = tk.Label(root, text='')
sentence_label.pack()

#
def update_sentence():
    global predicted_sentence
    sentence_label.config(text=''.join(predicted_sentence))
    root.after(100, update_sentence)  


predicted_sentence = []

def video_loop():
    global predicted_sentence

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1) 

    
    cv2.imshow('ASL Alphabet Prediction', frame)

   
    predicted_sign = predict(frame)

    
    if predicted_sign == 'space':
        predicted_sentence.append(' ')
    elif predicted_sign == 'del':
        if predicted_sentence:
            predicted_sentence.pop()
    else:
        predicted_sentence.append(predicted_sign)

    
    sentence_label.config(text=''.join(predicted_sentence))

   
    time.sleep(3)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        root.destroy()
    else:
        root.after(100, video_loop)  


video_loop()


root.mainloop()