from django.conf import settings
import numpy as np
import cv2
import sys
import os
import dlib
import glob
import math


def opencv_dface(path):
    img = cv2.imread(path, 1)

    if (type(img) is np.ndarray):
        print(img.shape)

        factor = 1
        if img.shape[1] > 640:
            factor = 640.0 / img.shape[1]
        elif img.shape[0] > 480:
            factor = 480.0 / img.shape[0]

        if factor != 1:
            w = img.shape[1] * factor
            h = img.shape[0] * factor
            img = cv2.resize(img, (int(w), int(h)))

        size = img.shape

        baseUrl = settings.MEDIA_ROOT_URL + settings.MEDIA_URL
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(baseUrl + 'shape_predictor_68_face_landmarks.dat')

        dlibimg = dlib.load_rgb_image(path)
        dets = detector(dlibimg, 1)
        for k, d in enumerate(dets):
            # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            #     k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = predictor(dlibimg, d)
            print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                      shape.part(1)))

            landmarks = list()
            for i in range(0, 68):
                landmarks.append([int(shape.part(i).x * factor), int(shape.part(i).y * factor)])
                # print(landmarks[i])
                cv2.circle(img, (landmarks[i][0], landmarks[i][1]), 2, (0, 0, 255), -1)


        #####Orientation
        image_points = np.array([
            (landmarks[30][0], landmarks[30][1]),  # Nose tip
            (landmarks[8][0], landmarks[8][1]),  # Chin
            (landmarks[45][0], landmarks[45][1]),  # Left eye left corner
            (landmarks[36][0], landmarks[36][1]),  # Right eye right corne
            (landmarks[54][0], landmarks[54][1]),  # Left Mouth corner
            (landmarks[48][0], landmarks[48][1])  # Right mouth corner
        ], dtype="double")

        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corne
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])

        # Camera internals

        center = (size[1] / 2, size[0] / 2)
        focal_length = center[0] / np.tan(60 / 2 * np.pi / 180)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        axis = np.float32([[500, 0, 0],
                           [0, 500, 0],
                           [0, 0, 500]])

        imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix,
                                           dist_coeffs)
        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

        proj_matrix = np.hstack((rvec_matrix, translation_vector))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

        pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))

        rotate_degree = (str(int(roll)), str(int(pitch)), str(int(yaw)))
        nose = (landmarks[30][0], landmarks[30][1])

        cv2.line(img, nose, tuple(imgpts[1].ravel()), (0, 255, 0), 3)  # GREEN
        cv2.line(img, nose, tuple(imgpts[0].ravel()), (255, 0,), 3)  # BLUE
        cv2.line(img, nose, tuple(imgpts[2].ravel()), (0, 0, 255), 3)  # RED

        # for j in range(len(rotate_degree)):
        #     cv2.putText(img, ('{:05.2f}').format(float(rotate_degree[j])), (10, 30 + (50 * j)),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)

        if roll == 0:
            roll_comment = str(int(roll))
        elif roll > 0:
            roll_comment = 'Rt tilt ' + str(int(roll))
        elif roll < 0:
            roll_comment = 'Lt tilt ' + str(int(roll * -1))

        if pitch == 0:
            pitch_comment = str(int(pitch))
        elif pitch > 0:
            pitch_comment = 'Up ' + str(int(pitch))
        elif pitch < 0:
            pitch_comment = 'Down ' + str(int(pitch * -1))

        if yaw == 0:
            yaw_comment = str(int(yaw))
        elif yaw > 0:
            yaw_comment = 'Rt yaw ' + str(int(yaw))
        elif yaw < 0:
            yaw_comment = 'Lt yaw ' + str(int(yaw * -1))


        result = np.zeros((img.shape[0] + 300,img.shape[1],3))
        result[:img.shape[0], :img.shape[1]] = img

        cv2.putText(result, roll_comment, (10, img.shape[0] + (50 * 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)

        cv2.putText(result, pitch_comment, (10, img.shape[0] + (50 * 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)

        cv2.putText(result, yaw_comment, (10, img.shape[0] + (50 * 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)



        for k in range(len(image_points)):
            # print(image_points[k].ravel()[0])
            cv2.circle(img, (int(image_points[k].ravel()[0]), int(image_points[k].ravel()[1])), 5, (240, 255, 10), -1)

        cv2.imwrite(path, result)

    else:
        print('someting error')
        print(path)