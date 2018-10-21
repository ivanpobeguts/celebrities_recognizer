from recognizer.settings import labels_dict, logger, recognizer
from recognizer.utils import align
import os
import cv2
import numpy as np


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
