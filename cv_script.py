from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

# constrcut argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
#maps the arg(shape-preidctor) to user input
args = vars(ap.parse_args())

print(args)


#initailizes dlib's face detector to locate the face then
#initialize the predictor to locate facial landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

#preprocessing
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#converts image to greyscale
#creates a bounding box of faces for images (seems like it's a list)
rects = detector(gray, 1)

#loop over face detections
for (index, rect) in enumerate(rects):
    #determine the facial landmarks for the face region, then convert the
    #facial landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    for(name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
        print(name)
        print((i, j))

    #convert dlib's rectangle to a OpenCV-style bounding box
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x,y), (x + w, y + h), (0, 255, 0), 2)

    #show face number
    cv2.putText(image, "Face #{}".format(index+1), (x-10, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #loop over the (x,y)-coordinates for the facial landmarks
    #and draw them on the images
    for(x, y) in shape:
        cv2.circle(image, (x,y), 1, (0, 0, 255), -1)

#show the output image with the face detections + facial landmarks

image = face_utils.visualize_facial_landmarks(image, shape)

cv2.imshow("Output", image)
cv2.waitKey(0)
