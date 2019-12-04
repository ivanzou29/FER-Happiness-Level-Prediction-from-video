import cv2
import dlib
import numpy as np
from imutils import face_utils
import imutils

def detect_face_and_regions(img_path):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    print(rects)

    # loop over the face detections
    for (i, rect) in enumerate(rects):

        face_img = image[rect.top():rect.bottom(), rect.left():rect.right()]
        face_img = imutils.resize(face_img, width=224, height=224)
        cv2.imwrite('face_' + img_path, face_img)
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # loop over the face parts individually
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            # clone the original image so we can draw on it, then
            # display the name of the face part on the image
            print(name)
            if (name == 'mouth'):
                (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))

                # crop the mouth to be square-shaped
                if (w % 2 == 0):
                    roi = image[y+h//2-w//2:y+h//2+w//2, x:x+w]
                else:
                    roi = image[y+h//2-w//2:y+h//2+w//2+1, x:x+w]
                # resize the ROI
                roi = imutils.resize(roi, width=224, height=224)
                cv2.imwrite("ROI_" + name + '.png', roi)
                cv2.waitKey(0)

            elif (name == 'nose'):
                (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))

                # crop the nose to be square-shaped
                if (h % 2 == 0):
                    roi = image[y:y+h, x+w//2-h//2:x+w//2+h//2]
                else:
                    roi = image[y:y+h, x+w//2-h//2:x+w//2+h//2+1]
                # resize the ROI
                roi = imutils.resize(roi, width=224, height=224)
                cv2.imwrite("ROI_" + name + '.png', roi)
                cv2.waitKey(0)

            elif (name == 'left_eyebrow' or name == 'right_eyebrow'):
                (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                roi = image[y:y+w, x:x+w]
                # resize the ROI
                roi = imutils.resize(roi, width=224, height=224)
                cv2.imwrite("ROI_" + name[:-4] + '.png', roi)
                cv2.waitKey(0)

        break

if __name__ == '__main__':
    detect_face_and_regions('output1.png')