import os

import face_recognition
from PIL import Image


def detect_face_and_resize(sequence_path, filename):
    img = face_recognition.load_image_file(os.path.join(sequence_path, filename))
    print("Performing face detection...")
    face_locations = face_recognition.face_locations(img, model='cnn')
    print("Found {} face(s) in this photograph.".format(len(face_locations)))
    if (len(face_locations) > 0):
        face_location = face_locations[0]
        # Print the location of each face in this image
        top, right, bottom, left = face_location
        print(
            "A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom,
                                                                                                  right))

        # You can access the actual face itself like this:
        face_image = img[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.resize((224, 224))
        pil_image.save(os.path.join(sequence_path, 'face_' + filename))

#dir = '0.9_00583436'

if __name__ == '__main__':
#   for sequence_path in os.listdir('frames'):
#        for frame in os.listdir(os.path.join('frames', sequence_path)):
#            detect_face(os.path.join('frames', sequence_path), frame)

#    for frame in os.listdir(dir):
#        detect_face_and_resize(dir, frame)
    detect_face_and_resize("", "output6.png")