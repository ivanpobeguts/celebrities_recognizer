import os
import cv2
from recognizer.utils import detect_and_save_face
from recognizer.settings import logger


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
