from recognizer.utils import detect_and_save_face, align, logger
from recognizer.settings import recognizer, labels_dict
import cv2


def recognize():
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
