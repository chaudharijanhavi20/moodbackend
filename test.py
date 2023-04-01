from deepface import DeepFace
import cv2
import json
from flask import Flask, request
from flask_cors import CORS, cross_origin
import cv2
import dlib
import imutils
from imutils import face_utils
from imutils.video import VideoStream
from imutils.face_utils import FaceAligner
import numpy as np

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/hello', methods=['GET', 'POST'])
def hello():
    if request.method == 'GET':
        vs = VideoStream(src=0).start()
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(
            'shape_predictor_68_face_landmarks.dat')
        fa = FaceAligner(predictor, desiredFaceWidth=96)
        listmood = []
        while True:
            test_img = vs.read()
            test_img = imutils.resize(test_img, width=800)
            gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 0)
            for face in faces:
                print("inside for loop")
                (x, y, w, h) = face_utils.rect_to_bb(face)

                face_aligned = fa.align(test_img, gray, face)
                # Saving the image dataset, but only the face part, cropping the rest

                if face is None:
                    print("face is none")
                    continue
                listmood.append(test_img)
                face_aligned = imutils.resize(face_aligned, width=400)
                cv2.rectangle(test_img, (x, y), (x+w, y+h), (0, 255, 0), 1)

                cv2.imshow("Add Images", test_img)
                cv2.waitKey(1)
            if 0xFF == ord('q') or len(listmood) > 10:
                break
        vs.stop()
        cv2.destroyAllWindows()
        obj = DeepFace.analyze(img_path=listmood[3], actions=[
                               'emotion'], enforce_detection=False)
        # destroying all the windows
        return json.dumps(obj)
    
    
    elif request.method == 'POST':
        name = request.form['name']
        return f'Hello, {name}!'


if __name__ == '__main__':
    app.run()
