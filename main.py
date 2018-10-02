import sys
import os
import cv2
import numpy as np
from collections import OrderedDict

subjects = ["", "liv tyler", "adrien brody"]
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# face_recognizer = cv2.face.FisherFaceRecognizer_create()


def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    # print(dirs)
    faces = []
    labels = []
    label_numbers = [i for i in range(len(dirs))]
    # print(label_numbers)
    labels_dict = OrderedDict(zip(dirs, label_numbers))
    print(labels_dict)
    for dir in dirs:

        images = os.listdir(os.path.join(data_folder_path, dir))
        label = labels_dict[dir]
        for image_path in images:
            if os.path.join(data_folder_path, dir, image_path).endswith('jpg'):
                image = cv2.imread(os.path.join(data_folder_path, dir, image_path))
                # print(os.path.join(data_folder_path, dir, image_path))

                # display an image window to show the image
                # cv2.imshow("Training on image...", image)
                # cv2.waitKey(100)

                # detect face
                face, rect = detect_face(image)

                if face is not None:
                    # add face to list of faces
                    faces.append(face)
                    labels.append(label)
                    print(os.path.join(data_folder_path, dir, image_path), label)

                cv2.destroyAllWindows()
                cv2.waitKey(1)
                cv2.destroyAllWindows()

    return faces, labels


def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if (len(faces) == 0):
        return None, None

    (x, y, w, h) = faces[0]

    # return only the face part of the image
    return gray[y:y + w, x:x + h], faces[0]


def predict(test_img):
    img = test_img.copy()
    # detect face from the image
    face, rect = detect_face(img)
    print(face)

    # predict the image using our face recognizer
    label = face_recognizer.predict(face)
    print(label)
    # get name of respective label returned by face recognizer
    label_text = label[0]
    print(label_text)

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
    print("Preparing data...")
    faces, labels = prepare_training_data("samples/thumbnails_features_deduped_sample")
    print("Data prepared")

    # print total faces and labels
    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))
    face_recognizer.train(faces, np.array(labels))


    print("Predicting images...")

    # load test images
    test_img1 = cv2.imread("test_data/brody.jpg")
    # test_img2 = cv2.imread("test_data/test2.jpg")

    # perform a prediction
    predicted_img1 = predict(test_img1)
    # predicted_img2 = predict(test_img2)
    print("Prediction complete")

    # display both images
    # cv2.imshow(subjects[1], predicted_img1)
    # cv2.imshow(subjects[2], predicted_img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


