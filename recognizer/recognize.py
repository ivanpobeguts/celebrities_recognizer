from recognizer.utils import detect_and_save_face, align, logger
from recognizer.settings import recognizer, labels_dict
from recognizer.train import train
import cv2


def recognize(npimg):
    face_store = []
    # images = cv2.imread('test_data/brody.jpg')
    images = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    copy_image = images
    # train()
    # images_new = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    # logger.info(images)
    face_st, rect = detect_and_save_face(images, 'output.jpg')
    output = align(face_st)

    if output[0] == 'y':
        for ix in range(0, len(output[1])):
            x, y, w, h = rect[ix][0], rect[ix][1], rect[ix][2], rect[ix][3]
            cv2.rectangle(copy_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            out = output[1][ix]
            lab = recognizer.predict(out)
            logger.debug(lab)
            for k, v in labels_dict.items():
                if v == lab[0]:
                    logger.info('NAME: {}\nProbability: {} '.format(k, lab[1]))
                    result = k
            # if lab[0] == 1:
            cv2.putText(copy_image, result, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # else:
            #     cv2.putText(copy_image, 'Unknown', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite('static/people/out.jpg', copy_image)
        return copy_image
