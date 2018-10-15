import sys
import os
import cv2
import numpy as np
from collections import OrderedDict
import logging

# logging.basicConfig(level=logging.DEBUG)


face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# face_recognizer = cv2.face.FisherFaceRecognizer_create()


labels_dict = {}


def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    label_numbers = [i for i in range(len(dirs))]
    labels_dict.update(zip(dirs, label_numbers))
    logger.debug('labels dict: {}'.format(labels_dict))
    for dir_name in dirs:

        images = os.listdir(os.path.join(data_folder_path, dir_name))
        label = labels_dict[dir_name]
        for image_path in images:
            if os.path.join(data_folder_path, dir_name, image_path).endswith('jpg'):
                image = cv2.imread(os.path.join(data_folder_path, dir_name, image_path))

                # display an image window to show the image
                # cv2.imshow("Training on image...", image)
                # cv2.waitKey(100)

                # detect face
                faces_folder = os.path.join('faces', dir_name, image_path)
                face, rect = detect_and_save_face(image, faces_folder)

                if face is not None:
                    # add face to list of faces
                    faces.append(face)
                    labels.append(label)
                    logger.debug('Detecting {}, {}'.format(os.path.join(data_folder_path, dir_name, image_path), label))

                # cv2.destroyAllWindows()
                # cv2.waitKey(1)
                # cv2.destroyAllWindows()

    return faces, labels


def detect_and_save_face(img, path):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.array(gray_image, 'uint8')
    face_cascade = cv2.CascadeClassifier('opencv_files/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if (len(faces) == 0):
        return None, None

    (x, y, w, h) = faces[0]
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    faces_folder = 'faces'
    cv2.imwrite(path, img)
    # return only the face part of the image
    logger.debug('1st parameter: {},\n 2nd parameter: {}'.format(gray[y:y + w, x:x + h], faces[0]))
    return gray[y:y + w, x:x + h], faces[0]


def predict(test_img):
    img = test_img.copy()
    # detect face from the image
    face, rect = detect_and_save_face(img, os.path.join('results', 'test.jpg'))

    # predict the image using our face recognizer
    label, confidence = face_recognizer.predict(face)
    print('Label: {}'.format(label))
    print('Confidence: {}'.format(confidence))
    print(labels_dict)
    for k, v in labels_dict.items():
        if v == label:
            print('NAME:', k)

    # # draw a rectangle around face detected
    # draw_rectangle(img, rect)
    # # draw name of predicted person
    # draw_text(img, label_text, rect[0], rect[1] - 5)

    return img


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


if __name__ == '__main__':
    logger = logging.getLogger('Debug logger')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    print("Preparing data...")
    faces, labels = prepare_training_data("samples/small_sample")
    print("Data prepared")

    # print total faces and labels
    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))
    face_recognizer.train(faces, np.array(labels))
    face_recognizer.save('trainner.yml')

    print("Predicting images...")

    # load test image
    test_img1 = cv2.imread("test_data/al-gore.jpg")

    # perform a prediction
    predicted_img1 = predict(test_img1)
    print("Prediction complete")
