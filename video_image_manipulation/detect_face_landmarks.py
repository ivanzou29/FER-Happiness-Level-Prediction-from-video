import cv2
import dlib
import numpy as np

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
        return src

def process_image(img_path, output_path):
    frame = cv2.imread(img_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_key = detect_landmark_points(frame, gray)
    cv2.imwrite(output_path, face_key)
    cv2.waitKey(0)

if __name__ == '__main__':
    process_image('output1.png', 'output1_landmarks.png')