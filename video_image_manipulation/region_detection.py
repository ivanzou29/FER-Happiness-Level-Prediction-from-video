import cv2
import os

xml_path_name = 'cascades'

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade.load(os.path.join(xml_path_name, 'haarcascade_frontalface_default.xml'))

lefteye_cascade = cv2.CascadeClassifier('haarcascade_mcs_lefteye.xml')
lefteye_cascade.load(os.path.join(xml_path_name, 'haarcascade_mcs_lefteye.xml'))

righteye_cascade = cv2.CascadeClassifier('haarcascade_mcs_righteye.xml')
righteye_cascade.load(os.path.join(xml_path_name, 'haarcascade_mcs_righteye.xml'))

mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
mouth_cascade.load(os.path.join(xml_path_name, 'haarcascade_mcs_mouth.xml'))

nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
nose_cascade.load(os.path.join(xml_path_name, 'haarcascade_mcs_nose.xml'))

smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
smile_cascade.load(os.path.join(xml_path_name, 'haarcascade_smile.xml'))

img = cv2.imread('output6.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.2, 3)
for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]

    left_eye = lefteye_cascade.detectMultiScale(roi_gray, 1.2, 3)
    for (ex, ey, ew, eh) in left_eye:
        print("left eye detected!")
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        break

    mouth = mouth_cascade.detectMultiScale(roi_gray, 1.5, 5)
    for (mx, my, mw, mh) in mouth:
        print("mouth detected!")
        cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)

    nose = nose_cascade.detectMultiScale(roi_gray, 1.2, 5)
    for (nx, ny, nw, nh) in nose:
        print("nose detected!")
        cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 255), 2)


cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()