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

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape-predictor"])
print(args)
