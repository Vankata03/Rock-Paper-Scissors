import cv2
import mediapipe as mp
import pickle
import numpy as np
import random
import time

# 1. Зареждане на обучения модел
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict

# Настройки на MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

# Речник за превод от числа към думи
labels_dict = {0: 'Камък', 1: 'Ножица', 2: 'Хартия'}

# Променливи за играта
computer_move = "???"
game_result = "Press SPACE to Play"
last_played_time = 0

def get_winner(player, computer):
    if player == computer:
        return "Равно!"
    elif (player == 'Камък' and computer == 'Ножица') or \
         (player == 'Ножица' and computer == 'Хартия') or \
         (player == 'Хартия' and computer == 'Камък'):
        return "Ти победи!"
    else:
        return "Компютърът победи!"

with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        h, w, c = image.shape
        
        # Обръщаме огледално за по-естествено усещане
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = hands.process(image_rgb)
        
        current_gesture = "None"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Рисуване на скелета
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Подготовка на данните за предсказание
                data_aux = []
                for landmark in hand_landmarks.landmark:
                    data_aux.extend([landmark.x, landmark.y, landmark.z])

                # ПРЕДСКАЗАНИЕ
                prediction = model.predict([data_aux])
                predicted_character = int(prediction[0])
                current_gesture = labels_dict[predicted_character]

                # Визуализация на това, което вижда AI
                cv2.rectangle(image, (0, 0), (300, 60), (0, 0, 0), -1)
                cv2.putText(image, f"TI: {current_gesture}", (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)

        # ЛОГИКА НА ИГРАТА
        # Натисни SPACE, за да играеш срещу компютъра
        key = cv2.waitKey(1)
        if key == 32: # 32 е кодът за Space
            if current_gesture != "None":
                # Компютърът избира случайно
                moves = ['Камък', 'Ножица', 'Хартия']
                computer_move = random.choice(moves)
                game_result = get_winner(current_gesture, computer_move)
            else:
                game_result = "Няма ръка!"

        # Показване на резултата на екрана
        # Бял фон за текста долу
        cv2.rectangle(image, (0, h-80), (w, h), (255, 255, 255), -1)
        
        color_res = (0, 0, 0)
        if "Ти победи!" in game_result: color_res = (0, 200, 0)
        elif "COMPUTER" in game_result: color_res = (0, 0, 200)

        cv2.putText(image, f"CPU: {computer_move}", (10, h-45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.putText(image, game_result, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_res, 3)

        cv2.imshow('Rock Paper Scissors AI', image)
        
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()