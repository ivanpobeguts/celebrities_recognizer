import cv2
from recognizer.settings import *
import numpy as np
import math


def detect_and_save_face(img, path):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_store = []
    rectangle = []
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=6)
    if (len(faces) == 0):
        return None, None

    # (x, y, w, h) = faces[0]
    for (x, y, w, h) in faces:
        rectangle.append((x, y, w, h))
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
    return face_store, rectangle


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
