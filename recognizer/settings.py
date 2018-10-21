import cv2
import dlib
import logging


recognizer = cv2.face.LBPHFaceRecognizer_create(threshold=95)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_cascade = cv2.CascadeClassifier('opencv_files/lbpcascade_frontalface.xml')
logger = logging.getLogger('Debug logger')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
# ch.setLevel(logging.INFO)
logger.addHandler(ch)
labels_dict = {}