import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Bosluk'}

while True:
    data_aux_1 = []
    x_1 = []
    y_1 = []

    data_aux_2 = []
    x_2 = []
    y_2 = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for j in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[j].x
                y = hand_landmarks.landmark[j].y

                if i == 0:
                    x_1.append(x)
                    y_1.append(y)
                elif i == 1:
                    x_2.append(x)
                    y_2.append(y)

            for j in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[j].x
                y = hand_landmarks.landmark[j].y

                if i == 0:
                    data_aux_1.append(x - min(x_1))
                    data_aux_1.append(y - min(y_1))
                elif i == 1:
                    data_aux_2.append(x - min(x_2))
                    data_aux_2.append(y - min(y_2))

    # Tahminler
    if data_aux_1:
        x1_1 = int(min(x_1) * W) - 10
        y1_1 = int(min(y_1) * H) - 10
        x2_1 = int(max(x_1) * W) - 10
        y2_1 = int(max(y_1) * H) - 10

        prediction_1 = model.predict([np.asarray(data_aux_1)])
        predicted_character_1 = labels_dict[int(prediction_1[0])]
        cv2.rectangle(frame, (x1_1, y1_1), (x2_1, y2_1), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character_1, (x1_1, y1_1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    if data_aux_2:
        x1_2 = int(min(x_2) * W) - 10
        y1_2 = int(min(y_2) * H) - 10
        x2_2 = int(max(x_2) * W) - 10
        y2_2 = int(max(y_2) * H) - 10

        # Sağ elin özellik vektörlerini aynala
        mirrored_x_2 = [W - x_val for x_val in x_2]
        mirrored_data_aux_2 = mirrored_x_2 + y_2

        prediction_2 = model.predict([np.asarray(mirrored_data_aux_2)])
        predicted_character_2 = labels_dict[int(prediction_2[0])]
        cv2.rectangle(frame, (x1_2, y1_2), (x2_2, y2_2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character_2, (x1_2, y1_2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    # 'x' tuşuna basıldığında döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
