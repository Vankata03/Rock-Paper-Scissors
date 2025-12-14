import cv2
import mediapipe as mp
import csv
import os

# Настройки
FILE_NAME = 'hand_data.csv'

# Инициализация на MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

# Създаваме CSV файла и пишем заглавния ред, ако файлът не съществува
if not os.path.exists(FILE_NAME):
    with open(FILE_NAME, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Създаваме заглавия: label, x0, y0, z0, x1, y1, z1 ...
        header = ['label']
        for i in range(21):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        writer.writerow(header)

print("Скриптът работи!")
print("Покажи жест и задръж съответния клавиш, за да записваш данни:")
print("Натисни '0' за КАМЪК")
print("Натисни '1' за НОЖИЦА")
print("Натисни '2' за ХАРТИЯ")
print("Натисни 'q' за ИЗХОД")

with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Обработка
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Ако има ръка
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Рисуваме скелета за визуализация
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Подготвяме данните (координатите)
                row = []
                for landmark in hand_landmarks.landmark:
                    row.extend([landmark.x, landmark.y, landmark.z])

                # Проверка за натиснат клавиш и запис
                k = cv2.waitKey(1)
                
                if k == ord('0'): # КАМЪК
                    with open(FILE_NAME, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([0] + row)
                    cv2.putText(image, "Recording ROCK (0)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                elif k == ord('1'): # НОЖИЦА
                    with open(FILE_NAME, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([1] + row)
                    cv2.putText(image, "Recording SCISSORS (1)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                elif k == ord('2'): # ХАРТИЯ
                    with open(FILE_NAME, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([2] + row)
                    cv2.putText(image, "Recording PAPER (2)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                elif k == ord('q'): # ИЗХОД
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

        cv2.imshow('Data Collection', image)
        # Малко забавяне за стабилност
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()