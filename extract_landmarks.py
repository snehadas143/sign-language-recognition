import os
import cv2
import mediapipe as mp
import pandas as pd

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

data = []
labels = []

dataset_path = "data"

for label in os.listdir(dataset_path):

    label_path = os.path.join(dataset_path, label)

    if not os.path.isdir(label_path):
        continue

    for image_name in os.listdir(label_path):

        image_path = os.path.join(label_path, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:

            hand_landmarks = results.multi_hand_landmarks[0]

            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            data.append(landmarks)
            labels.append(label)

df = pd.DataFrame(data)

df["label"] = labels

df.to_csv("landmarks.csv", index=False)

print("Landmarks saved to landmarks.csv")