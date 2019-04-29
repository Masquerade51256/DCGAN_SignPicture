import cv2
import sys
import os.path
from glob import glob

def detect(filename,cascade_file="lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(
        gray,
        # detector options
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (48,48)
    )

    for i,(x,y,w,h) in enumerate(faces):
        face = image[y: y+h, x:x+w, :]
        face = cv2.resize(face,(96,96))
        save_filename = '%s.jpg' % (os.path.basename(filename).split('.')[0])
        cv2.imwrite("faces/"+save_filename,face)

if __name__ == '__main__':
    if os.path.exists('faces') is False:
        os.makedirs('faces')
    file_list = glob('imgs/*.png')
    for filename in file_list:
        detect(filename)
