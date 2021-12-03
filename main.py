import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model, config_file)

class_labels = []
file_name = 'class-names.txt'
with open(file_name, 'rt') as fpt:
    class_labels = fpt.read().rstrip('\n').split('\n')

print('class_labels: ', class_labels)

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

cap = cv2.VideoCapture('My video')

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError

font_scale = 1
font = cv2.FONT_HERSHEY_PLAIN

while True:
    start_time = time.time()
    ret, frame = cap.read()

    class_index, confidence, bbox = model.detect(frame, confThreshold=0.55)
    print('class_index: ', class_index)
    if len(class_index) != 0:
        for class_ind, conf, boxes in zip(class_index.flatten(), confidence.flatten(), bbox):
            if class_ind == 1:
                print('boxes: ', boxes)
                # boxes content: boxes[0] -> x1, boxes[1] -> y1, boxes[2] -> width, boxes[3] -> height
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    class_labels[class_ind - 1],
                    (boxes[0] + 10, boxes[1] + 40),
                    font,
                    fontScale=font_scale,
                    color=(0, 255, 0)
                )

                human_frame = frame[
                    boxes[1]: boxes[1] + boxes[3],
                    boxes[0]: boxes[0] + boxes[2]
                ]
                (4, 5, 100, 200)
                with mp_pose.Pose(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as pose:
                    # To improve performance, optionally mark the image as not writeable to
                    # pass by reference.
                    human_frame.flags.writeable = False
                    human_frame = cv2.cvtColor(human_frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(human_frame)
                    if results.pose_landmarks:
                        print('pose landmarks type: ', type(results.pose_landmarks))
                        print('pose landmarks: ', results.pose_landmarks.landmark)


                    # Draw the pose annotation on the image.
                    human_frame.flags.writeable = True
                    human_frame = cv2.cvtColor(human_frame, cv2.COLOR_RGB2BGR)
                    mp_drawing.draw_landmarks(
                        human_frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )

                frame[
                    boxes[1]: boxes[1] + boxes[3],
                    boxes[0]: boxes[0] + boxes[2]
                ] = human_frame

    cv2.imshow('Object detection', cv2.flip(frame, 1))

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

    end_time = time.time()
    print('FPS: ',  1 / (end_time - start_time))

cap.release()
cv2.destroyAllWindows()


# https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
# Concept: https://www.youtube.com/watch?v=RFqvTmEFtOE