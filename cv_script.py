from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

def average(matrix):
    counter = 0
    sum_x = 0
    sum_y = 0
    for points in matrix:
        print("x: ", points[0, 0], "y: ", points[0, 1])
        sum_x += points[0, 0]
        sum_y += points[0, 1]
        counter += 1
    return(int(sum_x / counter), int(sum_y / counter))

# constrcut argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
#maps the arg(shape-preidctor) to user input
args = vars(ap.parse_args())

#initailizes dlib's face detector to locate the face then
#initialize the predictor to locate facial landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

#preprocessing
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
height, width, channel = image.shape
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#converts image to greyscale
#creates a bounding box of faces for images (seems like it's a list)
rects = detector(gray, 1)

dict_points = {}

#loop over face detections
for (index, rect) in enumerate(rects):
    #determine the facial landmarks for the face region, then convert the
    #facial landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, rect)
    counter = 0
    landmarks = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()])
    for(name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
        #center_point_x = sum(landmarks[i:j]) / len(landmarks[i:j])
        print(name)
        center_point_x, center_point_y = average(landmarks[i:j])
        print("center point x: ", center_point_x, "center point y: ", center_point_y)
        #args img, center, radius, color
        cv2.circle(image, (center_point_x, center_point_y), 5, (0,0,0), thickness=-1)
        dict_points[name] = (center_point_x, center_point_y)
    shape = face_utils.shape_to_np(shape)


    #convert dlib's rectangle to a OpenCV-style bounding box
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x,y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(image, (int((x + w / 2)), int((y + h / 2))), 10, (255, 0 , 0), thickness=-1)

    #show face number
    cv2.putText(image, "Face #{}".format(index+1), (x-10, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #loop over the (x,y)-coordinates for the facial landmarks
    #and draw them on the images
    for(x, y) in shape:
        cv2.circle(image, (x,y), 1, (0, 0, 255), -1)

#show the output image with the face detections + facial landmarks
print(dict_points)
eyebrow_dist = int((dict_points['left_eyebrow'][0] + dict_points['right_eyebrow'][0])/2)
eye_dist = int((dict_points['left_eye'][0] + dict_points['right_eye'][0])/2)
nose_y_coord = dict_points['nose'][1]
center_point = (int(width / 2), int(height / 2))
print("eyebrow_dist: ", eyebrow_dist)
print("eye_dist: ", eye_dist)
print("nose: ", nose_y_coord)
print("image height: ", height, "image width: ", width)
print("box width: ", w / 4, "box height: ", h)
print("init box x: ", x, "init box y: ", y)
for i in range(1, 4):
    cv2.line(image, (int(x + i * (w / 4)), y), (int(x + i * (w / 4)), y + h), (0, 255, 0), thickness=2)
cv2.circle(image, (eyebrow_dist, nose_y_coord), 5, (0,0,0), thickness=-1)
cv2.circle(image, (eye_dist, nose_y_coord), 5, (0,0,0), thickness=-1)
cv2.circle(image, center_point, 10, (0, 0, 255), thickness=-1)

#image = face_utils.visualize_facial_landmarks(image, shape)

cv2.imshow("Output", image)
cv2.waitKey(0)
