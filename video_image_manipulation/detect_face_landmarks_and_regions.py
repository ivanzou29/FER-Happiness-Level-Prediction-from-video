import cv2
import dlib
import numpy as np
from imutils import face_utils
import imutils


def detect_landmark_points(src, img):
    points_keys = []
    PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    rects = detector(img, 1)
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x,p.y] for p in predictor(img,rects[i]).parts()])
        img = img.copy()
        for idx, point in enumerate(landmarks):
            pos = (point[0,0], point[0,1])
            points_keys.append([point[0,0], point[0,1]])
            cv2.circle(src, pos, 2, (255, 0, 0), -1)
        print(points_keys)
        return src, points_keys

def draw_landmarks(img_path, output_path):
    frame = cv2.imread(img_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_key, points_keys = detect_landmark_points(frame, gray)
    cv2.imwrite(output_path, face_key)
    cv2.waitKey(0)

def detect_face_regions(img_path):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(img_path)
    image = imutils.resize(image, height=500, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # loop over the face parts individually
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            # clone the original image so we can draw on it, then
            # display the name of the face part on the image
            clone = image.copy()
            cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

            # loop over the subset of facial landmarks, drawing the
            # specific face part
            for (x, y) in shape[i:j]:
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

            # extract the ROI of the face region as a separate image
            (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
            roi = image[y:y + h, x:x + w]
            roi = imutils.resize(roi, height=200, width=200, inter=cv2.INTER_CUBIC)

            # show the particular face part
            cv2.imwrite("ROI_" + name + '.png', roi)
            cv2.imwrite("Image_" + name + '.png', clone)
            cv2.waitKey(0)

if __name__ == '__main__':
    #process_image('output1.png', 'output1_landmarks.png')
    detect_face_regions('output13.png')