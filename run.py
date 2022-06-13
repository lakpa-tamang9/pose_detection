# %%
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2
import json

# %%
interpreter = tf.lite.Interpreter(
    model_path="./lite-model_movenet_singlepose_thunder_3.tflite"
)
interpreter.allocate_tensors()


with open("./mapping.json", "r") as f:
    myfile = json.load(f)

# %%
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 0, 0), 6)


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge in edges:
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 4)


# %%
def calculate_angle(landmark1, landmark2, landmark3):
    """
    landmark1 : First point
    landmark2 : mid point i.e., the vertex
    landmark3 : last point
    """
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


# %%
def display_angle(image, angle, value, width, height):
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


# %%
def midpoint(p1, p2):
    return list(((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[1] + p2[1]) / 2))


def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        _, frame = cap.read()

        # Reshape image
        img = frame.copy()
        height, width, _ = img.shape
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256, 256)
        input_image = tf.cast(img, dtype=tf.float32)

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
        neckpoint = midpoint(landmarks[5], landmarks[6])
        # print(neckpoint)
        hippoint = midpoint(landmarks[11], landmarks[12])
        kps = landmarks.tolist()
        kps.append(neckpoint)
        kps.append(hippoint)
        # print(neckpoint)
        print(len(kps))
        # landmarks.append(neckpoint.tolist())

        # print(kps)

        # right shoulder angle
        rightshoulderAngle = calculate_angle(left_shoulder, right_shoulder, right_elbow)
        display_angle(frame, rightshoulderAngle, right_shoulder, width, height)

        # left shoulder angle
        leftshoulderAngle = calculate_angle(right_shoulder, left_shoulder, left_elbow)
        display_angle(frame, leftshoulderAngle, left_shoulder, width, height)

        # left elbow angle
        leftelbowAngle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        display_angle(frame, leftelbowAngle, left_elbow, width, height)

        # right elbow angle
        rightelbowAngle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        display_angle(frame, rightelbowAngle, right_elbow, width, height)

        # print(rightshoulderAngle)

        # Rendering
        draw_connections(frame, kps, myfile["edge_mapping"], 0.4)
        draw_keypoints(frame, kps, 0.4)

        cv2.imshow("MoveNet Lightning", frame)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
