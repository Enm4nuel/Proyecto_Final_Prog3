import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if ret == False:
            break

        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        print("Handedness: ", results.multi_handedness)

        cv2.putText(frame, "hola", (100, 100 - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

        # handlandmarks
        if results.multi_hand_landmarks is not None:

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                x1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * width)
                y1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * height)

                cv2.circle(frame, (x1, y1), 3, (255, 0, 0), 3)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()


