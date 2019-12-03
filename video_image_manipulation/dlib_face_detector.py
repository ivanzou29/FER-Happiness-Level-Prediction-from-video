import dlib
from skimage import io


detector = dlib.get_frontal_face_detector()

img = io.imread('output26.png')

win = dlib.image_window()
win.set_image(img)

faces = detector(img, 1)

print("num/ faces totally: ", len(faces))
for i, d in enumerate(faces):
    print(i+1, ": ", "left:", d.left(), '\t', "right:", d.right(), '\t', "top:", d.top(),'\t',  "bottom:", d.bottom())
    win.add_overlay(faces)
    face = img[d.top():d.bottom, d.left():d.right()]
    face.save('face.png')
    dlib.hit_enter_to_continue()
