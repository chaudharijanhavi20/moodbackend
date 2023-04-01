from deepface import DeepFace
import cv2
import numpy as np
import json 
from flask import Flask, request

app = Flask(__name__)



@app.route('/hello', methods=['GET', 'POST'])
def hello():
    if request.method == 'GET':
        
        cap = cv2.VideoCapture(0)
        listmood=[]
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Draw the bounding box on the frame
                listmood.append(frame)
                # Display the resulting frame
                cv2.imshow('frame', frame)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
            else:
                break
        obj = DeepFace.analyze(img_path = listmood[-4], actions = ['gender', 'emotion'])
        return json.dumps(obj)
    elif request.method == 'POST':
        name = request.form['name']
        return f'Hello, {name}!'

if __name__ == '__main__':
    app.run()