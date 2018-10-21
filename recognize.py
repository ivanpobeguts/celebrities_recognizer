import sys
import os
import cv2
import numpy as np
import dlib
import logging
import math
import time

recognizer = cv2.face.LBPHFaceRecognizer_create(threshold=95)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_cascade = cv2.CascadeClassifier('opencv_files/lbpcascade_frontalface.xml')
save = []
labels_dict = {}


def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    for dir_name in dirs:

        images = os.listdir(os.path.join(data_folder_path, dir_name))
        for image_path in images:
            if os.path.join(data_folder_path, dir_name, image_path).endswith('jpg'):
                image = cv2.imread(os.path.join(data_folder_path, dir_name, image_path))
                faces_folder = os.path.join('faces', dir_name, image_path)
                face = detect_and_save_face(image, faces_folder)

                if face is not None:
                    logger.debug('Detecting {}'.format(os.path.join(data_folder_path, dir_name, image_path)))


def train():
    dirs = os.listdir('faces')
    faces = []
    labels = []
    label_numbers = [i for i in range(len(dirs))]
    labels_dict.update(zip(dirs, label_numbers))
    logger.info('labels dict: {}'.format(labels_dict))
    for dir_name in dirs:
        images = os.listdir(os.path.join('faces', dir_name))
        label = labels_dict[dir_name]
        for image_path in images:
            face_store = []
            image = cv2.imread(os.path.join('faces', dir_name, image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (100, 100))
            face_store.append(image)
            output = align(face_store)
            if output[0] == 'y':
                out = output[1][0]
                faces.append(out)
                labels.append(label)
                logger.info('{}: {}'.format(os.path.join('faces', dir_name, image_path), label))
    img_array = np.asarray(faces)
    label_array = np.asarray(labels)
    logger.debug(label_array)
    recognizer.train(img_array, label_array)


def align(face_store):
    output = []
    flag = 0
    for ix in range(0, len(face_store)):
        flag = 0
        detections = detector(face_store[ix], 2)
        for k, d in enumerate(detections):
            shape = predictor(face_store[ix], d)
            p1 = [(shape.part(45).x, shape.part(45).y), (shape.part(36).x, shape.part(36).y)]
            p2 = [((int(0.7 * 100), 33)), (int(0.3 * 100), 33)]
            s60 = math.sin(60 * math.pi / 180)
            c60 = math.cos(60 * math.pi / 180)
            inPts = np.copy(p1).tolist()
            outPts = np.copy(p2).tolist()
            xin = c60 * (inPts[0][0] - inPts[1][0]) - s60 * (inPts[0][1] - inPts[1][1]) + inPts[1][0]
            yin = s60 * (inPts[0][0] - inPts[1][0]) + c60 * (inPts[0][1] - inPts[1][1]) + inPts[1][1]
            inPts.append([np.int(xin), np.int(yin)])
            xout = c60 * (outPts[0][0] - outPts[1][0]) - s60 * (outPts[0][1] - outPts[1][1]) + outPts[1][0]
            yout = s60 * (outPts[0][0] - outPts[1][0]) + c60 * (outPts[0][1] - outPts[1][1]) + outPts[1][1]
            outPts.append([np.int(xout), np.int(yout)])
            tform = cv2.estimateRigidTransform(np.array([inPts]), np.array([outPts]), False)
            img2 = cv2.warpAffine(face_store[ix], tform, (100, 100))

            detections = detector(img2, 3)
            for k, d in enumerate(detections):
                flag = 1
                face = [[abs(d.left()), abs(d.right())], [abs(d.top()), abs(d.bottom())]]
                shape = predictor(img2, d)
                l_eye = np.asarray([(shape.part(36).x, shape.part(36).y), (shape.part(37).x, shape.part(37).y),
                                    (shape.part(38).x, shape.part(38).y), (shape.part(39).x, shape.part(39).y),
                                    (shape.part(40).x, shape.part(40).y), (shape.part(41).x, shape.part(41).y)])
                r_eye = np.asarray([(shape.part(42).x, shape.part(42).y), (shape.part(43).x, shape.part(43).y),
                                    (shape.part(44).x, shape.part(44).y), (shape.part(45).x, shape.part(45).y),
                                    (shape.part(46).x, shape.part(46).y), (shape.part(47).x, shape.part(47).y)])
                eye_left = np.mean(l_eye, axis=0)
                eye_right = np.mean(r_eye, axis=0)
                face[0][0] = int((eye_left[0] + face[0][0]) / 2.0)
                face[0][1] = int((eye_right[0] + face[0][1]) / 2.0)
                face[1][1] = int((shape.part(10).y + shape.part(57).y) / 2.0)
                img2_cropped = img2[face[1][0]:face[1][1], face[0][0]:face[0][1]]
                img2_cropped = cv2.resize(img2_cropped, (100, 100))
                output.append(img2_cropped)
            # if flag == 0:
            #     del (save[ix])
    if len(output) == 0:
        return ('n', 1)
    return ('y', output)


def detect_and_save_face(img, path):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_store = []
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=6)
    if (len(faces) == 0):
        return None, None

    # (x, y, w, h) = faces[0]
    for (x, y, w, h) in faces:
        save.append((x, y, w, h))
        if y - 10 >= 0 and x - 10 >= 0:
            f = gray[y - 10:(y + h + 10), x - 10:(x + w + 10)]
            f = cv2.resize(f, (100, 100))
            cv2.imwrite(path, f)
            face_store.append(f)
        else:
            f = gray[y:(y + h + 10), x:(x + w + 10)]
            f = cv2.resize(f, (100, 100))
            cv2.imwrite(path, f)
            face_store.append(f)
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # cv2.imwrite(path, img)
    # return only the face part of the image
    logger.debug('1st parameter: {},\n 2nd parameter: {}'.format(gray[y:y + w, x:x + h], faces[0]))
    # return gray[y:y + w, x:x + h], faces[0]
    return face_store


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


if __name__ == '__main__':
    logger = logging.getLogger('Debug logger')
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    # ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    # print("Preparing data...")
    # faces = prepare_training_data("samples/small_sample")
    # print("Data prepared")

    start_time = time.time()
    train()
    logger.info(f'Train seconds {time.time() - start_time}')
    face_store = []
    images = cv2.imread('test_data/brody.jpg')
    copy_image = images
    # images_new = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    face_st = detect_and_save_face(images, 'output.jpg')
    output = align(face_st)

    if output[0] == 'y':
        for ix in range(0, len(output[1])):
            out = output[1][ix]
            lab = recognizer.predict(out)
            logger.debug(lab)
            for k, v in labels_dict.items():
                if v == lab[0]:
                    logger.info('NAME: {}\nProbability: {} '.format(k, lab[1]))
