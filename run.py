import argparse
import json

import cv2
import numpy as np
import tensorflow as tf


def drawKeypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 0, 0), 6)


def drawConnections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge in edges:
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 4)

def calculateAngle(landmark1, landmark2, landmark3):
    a = np.array(landmark1)  # First
    b = np.array(landmark2)  # Mid
    c = np.array(landmark3)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def displayAngle(image, angle, value, width, height):
    return cv2.putText(
        image,
        str(int(angle)),
        tuple(np.multiply(value, [width, height]).astype(int)),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (128, 255, 64),
        3,
        cv2.LINE_AA,
    )

def getMidpoint(p1, p2):
    return list(((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[1] + p2[1]) / 2))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model_path", help="Movenetmodel path")
    parser.add_argument("-f", "--map_file", help = "Path to the keypoints mapping json file")
    args = parser.parse_args()

    with open(args.map_file, "r") as f:
        myfile = json.load(f)

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        _, frame = cap.read()

        # Reshape image
        img = frame.copy()
        height, width, _ = img.shape
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256, 256)
        input_image = tf.cast(img, dtype=tf.float32)
        interpreter = tf.lite.Interpreter(model_path=args.model_path)
        interpreter.allocate_tensors()

        # Setup input and output
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Make predictions
        interpreter.set_tensor(input_details[0]["index"], np.array(input_image))
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]["index"])
        # print(keypoints_with_scores)
        landmarks = keypoints_with_scores[0][0]
        # print(landmarks)

        left_shoulder = landmarks[5][:2]
        right_shoulder = landmarks[6][:2]
        left_elbow = landmarks[7][:2]
        right_elbow = landmarks[8][:2]
        left_wrist = landmarks[9][:2]
        right_wrist = landmarks[10][:2]

        left_hip = landmarks[11][:2]
        right_hip = landmarks[12][:2]

        # remove connection of mediapipe. shoulder and hip connections.
        # add connection beterrn central hip and central shoulder
        # 17 and 18 index are appended to the existing list

        # print(left_shoulder)
        neckpoint = getMidpoint(landmarks[5], landmarks[6])
        # print(neckpoint)
        hippoint = getMidpoint(landmarks[11], landmarks[12])
        kps = landmarks.tolist()
        kps.append(neckpoint)
        kps.append(hippoint)
        # print(neckpoint)
        # print(len(kps))
        # landmarks.append(neckpoint.tolist())

        # print(kps)

        # right shoulder angle
        rightshoulderAngle = calculateAngle(left_shoulder, right_shoulder, right_elbow)
        displayAngle(frame, rightshoulderAngle, right_shoulder, width, height)

        # left shoulder angle
        leftshoulderAngle = calculateAngle(right_shoulder, left_shoulder, left_elbow)
        displayAngle(frame, leftshoulderAngle, left_shoulder, width, height)

        # left elbow angle
        leftelbowAngle = calculateAngle(left_shoulder, left_elbow, left_wrist)
        displayAngle(frame, leftelbowAngle, left_elbow, width, height)

        # right elbow angle
        rightelbowAngle = calculateAngle(right_shoulder, right_elbow, right_wrist)
        displayAngle(frame, rightelbowAngle, right_elbow, width, height)

        # print(rightshoulderAngle)

        # Rendering
        drawConnections(frame, kps, myfile["edge_mapping"], 0.4)
        drawKeypoints(frame, kps, 0.4)

        cv2.imshow("MoveNet Lightning", frame)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
