# Import the libraries
import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pywavefront import Wavefront

# Load OBJ model
mesh = Wavefront(r'C:\Users\Rahul Malawat\OneDrive\Desktop\untitled.obj')

# Getting 2 mediapipe components
mp_pose = mp.solutions.pose  # for detecting body pose
mp_hands = mp.solutions.hands  # for detecting hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()
face_cascade = cv2.CascadeClassifier(r"C:\Users\Rahul Malawat\OneDrive\Desktop\haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cv2.namedWindow("AR/VR Avatar", cv2.WINDOW_NORMAL)
window_width = 1080
window_height = 720
cv2.resizeWindow("AR/VR Avatar", window_width, window_height)

ret, frame = cap.read()

fig = plt.figure(dpi=180)
ax = fig.add_subplot(111, projection='3d')
ax.set_title("AR/VR Avatar")

hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=6)

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)


while True:
    ret, frame = cap.read()

    gray_scale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_scale_image, 1.1, 4)

    for (x, y, w, h) in faces:
        center_x = x + w // 2
        center_y = y + h // 2
        radius = min(w, h) // 2 + 10
        cv2.circle(frame, (center_x, center_y), radius - 4, (102, 204, 0), 4)
        cv2.circle(frame, (center_x, center_y), radius, (153, 0, 76), 4)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)
    image.flags.writeable = False

    hand_results = hands.process(image)
    entire_body_pose_results = pose.process(image)

    ax.clear()
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([-0.5, 3])
    ax.set_zlim([1, -1.5])
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    face_results = face_detection.process(image)

    


    if hand_results.multi_hand_landmarks:
        for num, hand in enumerate(hand_results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(image,
                                      hand,
                                      mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(153, 0, 76), thickness=-1, circle_radius=4),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(102, 204, 0), thickness=2, circle_radius=2)
                                      )

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            x_hand_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_hand_coords = [landmark.y for landmark in hand_landmarks.landmark]
            z_hand_coords = [landmark.z for landmark in hand_landmarks.landmark]

        ax.scatter3D(x_hand_coords, y_hand_coords, z_hand_coords, color='purple', s=10)

        for hand_connection in mp_hands.HAND_CONNECTIONS:
            start = hand_connection[0]
            end = hand_connection[1]
            ax.plot([x_hand_coords[start], x_hand_coords[end]],
                    [y_hand_coords[start], y_hand_coords[end]],
                    [z_hand_coords[start], z_hand_coords[end]], color='green', linewidth=2)

    mp_drawing.draw_landmarks(image, entire_body_pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing.DrawingSpec(color=(153, 0, 76), thickness=-1,
                                                                            circle_radius=4),
                              connection_drawing_spec=mp_drawing.DrawingSpec(color=(102, 204, 0), thickness=2,
                                                                             circle_radius=2))

    if entire_body_pose_results.pose_landmarks:
        x_coords = [landmark.x for landmark in entire_body_pose_results.pose_landmarks.landmark]
        y_coords = [landmark.y for landmark in entire_body_pose_results.pose_landmarks.landmark]
        z_coords = [landmark.z for landmark in entire_body_pose_results.pose_landmarks.landmark]

        ax.scatter3D(x_coords, y_coords, z_coords, color='purple', s=10)

        for connection in mp_pose.POSE_CONNECTIONS:
            start = connection[0]
            end = connection[1]
            ax.plot([x_coords[start], x_coords[end]],
                    [y_coords[start], y_coords[end]],
                    [z_coords[start], z_coords[end]], color='green', linewidth=2)

    plt.show(block=False)
    plt.pause(0.001)
    cv2.imshow("AR/VR Avatar", image)

    if cv2.waitKey(1) == 27:
        break