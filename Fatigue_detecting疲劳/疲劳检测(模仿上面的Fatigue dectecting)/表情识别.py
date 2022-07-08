#coding=utf-8
#表情识别

import cv2
from keras.models import load_model
import numpy as np
import datetime

startTime = datetime.datetime.now()
emotion_classifier = load_model('simple_CNN.530-0.65.hdf5')
endTime = datetime.datetime.now()
print(endTime - startTime)

emotion_labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'calm'
}

img = cv2.imread("1.jpg")
face_classifier = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(
    gray, scaleFactor=1.2, minNeighbors=3, minSize=(40, 40))
color = (255, 0, 0)
font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
for (x, y, w, h) in faces: # 矩形框位置
    gray_face = gray[(y):(y + h), (x):(x + w)]
    gray_face = cv2.resize(gray_face, (48, 48))
    gray_face = gray_face / 255.0
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
    emotion = emotion_labels[emotion_label_arg]
    print(emotion)
    cv2.rectangle(img, (x + 10, y + 10), (x + h - 10, y + w - 10),
                  color, 2)
    img = img.copy()  # 备份操作
    img2 = cv2.putText(img, emotion, (int(x + h * 0.3), int(y)), font, 1, color, 2)
                  # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度

cv2.imshow("Image", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
