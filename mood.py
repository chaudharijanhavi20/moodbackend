import cv2
import numpy as np
from keras.models import load_model
new_model=load_model('C:/Users/AYUSH SHUKLA/Downloads/archive/MyModelFaceRecogD5.h5')
# test_img=cv2.imread("C:/Users/AYUSH SHUKLA/Downloads/archive/images/validation/happy/80.jpg")
# test_img.shape
# plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
# faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
 
# gray=cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

# faces=faceCascade.detectMultiScale(gray,1.1,4)

# for x,y,w,h in faces:
#     roi_gray=gray[y:y+h, x:x+w]
#     roi_color=test_img[y:y+h, x:x+w]
#     cv2.rectangle(test_img, (x,y), (x+w,y+h), (255,0,0),2)
#     facess=faceCascade.detectMultiScale(roi_gray)
    
#     if len(facess) == 0:
#         print("Face not detected")
#     else:
#         for (ex,ey,ew,eh) in facess:
#             face_roi=roi_color[ey: ey+eh, ex: ex+ew]

# face_roi=cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

# plt.imshow(face_roi)
# finalimage=cv2.resize(face_roi , (48,48))
# finalimage1=np.expand_dims(finalimage, axis=0)
# finalimage=finalimage1/255.0


# prediction=new_model.predict(finalimage)
# prediction
# path="haarcascade_frontalface_default.xml"
# font_scale=1.6
# font=cv2.FONT_ITALIC

# rectangle_bgr=(255,255,255)
# img=np.zeros((500,500))

# text="some text in a boxl"

# (text_width, text_height)=cv2.getTextSize(text,font, fontScale=font_scale, thickness=1)[0]
# text_offset_x=10
# text_offset_y=img.shape[0]-25

# box_coords=((text_offset_x,text_offset_y),(text_offset_x+text_width+2,text_offset_y-text_height-2))

# cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)

# cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0,0,115), thickness=1)

cap=cv2.VideoCapture(0)
if not cap.isOpened():
    cap=cv2.VideoCapture(0)
    print('lets go')
if not cap.isOpened():
    raise IOError("Cannot Open Camera")

while True:
    ret,test_img=cap.read()

    faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    gray=cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.1, 4)


    for x,y,w,h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color=test_img[y:y+h, x:x+w]
        cv2.rectangle(test_img, (x,y), (x+w,y+h), (255,0,0),2)
        facess=faceCascade.detectMultiScale(roi_gray)
        
        if len(facess) == 0:
            print("Face not detected")
        else:
            for (ex,ey,ew,eh) in facess:
                face_roi=roi_color[ey: ey+eh, ex: ex+ew]


            face_roi=cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            finalimage=cv2.resize(face_roi, (48,48))
            finalimage=np.expand_dims(finalimage, axis=0)
            finalimage=finalimage/255.0

            finalimage=finalimage.reshape(1,48,48,3)

            font=cv2.FONT_HERSHEY_SIMPLEX

            prediction=new_model.predict(finalimage)
            print(prediction)

            font_scale=1.5
            font=cv2.FONT_HERSHEY_PLAIN
            if (np.argmax(prediction)==0):
                status= "surprise"

                x1,y1,w1,h1=0,0,175,75
                cv2.rectangle(test_img, (x1,x1),(x1+w1, y1+h1),(0,0,0), -1)
                cv2.putText(test_img, status, (x1+int(w1/10),y1+int(h1/2)), cv2.FONT_HERSHEY_PLAIN, 0.3, (0,0,255),2)
                cv2.putText(test_img, status,(100,150), font , 3, (0,0,255),2, cv2.LINE_4)
                cv2.rectangle(test_img, (x,y), (x+w, y+h), (112,0,25))


            elif (np.argmax(prediction)==1):
                status= "sad"

                x1,y1,w1,h1=0,0,175,75
                cv2.rectangle(test_img, (x1,x1),(x1+w1, y1+h1),(0,0,0), -1)
                cv2.putText(test_img, status, (x1+int(w1/10),y1+int(h1/2)), cv2.FONT_HERSHEY_PLAIN, 0.3, (0,0,255),2)
                cv2.putText(test_img, status,(100,150), font , 3, (0,0,255),2, cv2.LINE_4)
                cv2.rectangle(test_img, (x,y), (x+w, y+h), (112,0, 25))

            elif (np.argmax(prediction)==2):
                status= "angry"

                x1,y1,w1,h1=0,0,175,75
                cv2.rectangle(test_img, (x1,x1),(x1+w1, y1+h1),(0,0,0), -1)
                cv2.putText(test_img, status, (x1+int(w1/10),y1+int(h1/2)), cv2.FONT_HERSHEY_PLAIN, 0.3, (0,0,255),2)
                cv2.putText(test_img, status,(100,150), font , 3, (0,0,255),2, cv2.LINE_4)
                cv2.rectangle(test_img, (x,y), (x+w, y+h), (112,0, 25))

            elif (np.argmax(prediction)==3):
                status= "neutral"

                x1,y1,w1,h1=0,0,175,75
                cv2.rectangle(test_img, (x1,x1),(x1+w1, y1+h1),(0,0,0), -1)
                cv2.putText(test_img, status, (x1+int(w1/10),y1+int(h1/2)), cv2.FONT_HERSHEY_PLAIN, 0.3, (0,0,255),2)
                cv2.putText(test_img, status,(100,150), font , 3, (0,0,255),2, cv2.LINE_4)
                cv2.rectangle(test_img, (x,y), (x+w, y+h), (112,0, 25))

            elif (np.argmax(prediction)==4):
                status= "happy"

                x1,y1,w1,h1=0,0,175,75
                cv2.rectangle(test_img, (x1,x1),(x1+w1, y1+h1),(0,0,0), -1)
                cv2.putText(test_img, status, (x1+int(w1/10),y1+int(h1/2)), cv2.FONT_HERSHEY_PLAIN, 0.3, (0,0,255),2)
                cv2.putText(test_img, status,(100,150), font , 3, (0,0,255),2, cv2.LINE_4)
                cv2.rectangle(test_img, (x,y), (x+w, y+h), (112,0, 25))

            elif (np.argmax(prediction)==5):
                status= "fear"

                x1,y1,w1,h1=0,0,175,75
                cv2.rectangle(test_img, (x1,x1),(x1+w1, y1+h1),(0,0,0), -1)
                cv2.putText(test_img, status, (x1+int(w1/10),y1+int(h1/2)), cv2.FONT_HERSHEY_PLAIN, 0.3, (0,0,255),2)
                cv2.putText(test_img, status,(100,150), font , 3, (0,0,255),2, cv2.LINE_4)
                cv2.rectangle(test_img, (x,y), (x+w, y+h), (112,0, 25))

            else:
                status= "disgust"

                x1,y1,w1,h1=0,0,175,75
                cv2.rectangle(test_img, (x1,x1),(x1+w1, y1+h1),(0,0,0), -1)
                cv2.putText(test_img, status, (x1+int(w1/10),y1+int(h1/2)), cv2.FONT_HERSHEY_PLAIN, 0.3, (0,255,0),2)
                cv2.putText(test_img, status,(100,150), font , 3, (0,0,255),2, cv2.LINE_4)
                cv2.rectangle(test_img, (x,y), (x+w, y+h), (112,0,25))


            cv2.imshow('Face Emotion Recognition', test_img)

    if cv2.waitKey(2) & 0xFF==ord('q'):
        break

cap.release
cv2.destroyAllWindows()