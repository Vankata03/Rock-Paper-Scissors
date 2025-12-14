import cv2
import mediapipe as mp
import pickle
import numpy as np
import random
import os
from PIL import Image, ImageDraw, ImageFont

# --- НАСТРОЙКИ ---
CAMERA_ID = 0  # Смени на 1, ако не тръгва

def get_mac_font_path():
    possible_paths = [
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Tahoma.ttf"
    ]
    for path in possible_paths:
        if os.path.exists(path): return path
    return None

FONT_PATH = get_mac_font_path()

# --- ЦВЕТОВЕ (R, G, B за PIL) ---
COLOR_WIN   = (0, 255, 0)       # Зелено
COLOR_LOSE  = (255, 0, 0)       # Червено
COLOR_DRAW  = (255, 200, 0)     # Оранжево
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GRAY  = (50, 50, 50)

# --- ЗАРЕЖДАНЕ НА МОДЕЛА ---
if not os.path.exists('./model.p'):
    print("ГРЕШКА: Файлът 'model.p' липсва!")
    exit()

model = pickle.load(open('./model.p', 'rb'))

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(CAMERA_ID)

labels_dict = {0: 'КАМЪК', 1: 'НОЖИЦА', 2: 'ХАРТИЯ'}

# Състояние на играта
computer_move = ""
game_result_text = ""       
game_result_color = None    
info_text = "Натисни SPACE за игра" 
show_result_timer = 0

# --- ФУНКЦИЯ ЗА ТЕКСТ (Поправена ширина) ---
def draw_ui_text(image, text, x, y, color, size_percent=0.05, align="left"):
    h, w, c = image.shape
    font_size = int(h * size_percent) 
    
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    
    try:
        font = ImageFont.truetype(FONT_PATH, font_size) if FONT_PATH else ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # ПОПРАВКА: Увеличаваме множителя на 0.6, за да не се реже текста
    text_w = len(text) * (font_size * 0.65) 
    
    final_x = x
    if align == "center":
        final_x = int(x - text_w / 2)
    elif align == "right":
        final_x = int(x - text_w)
        
    draw.text((final_x, y), text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# --- ЛОГИКА ---
def get_winner(player, computer):
    if player == computer: 
        return "РАВЕНСТВО!", COLOR_DRAW
    if (player == 'КАМЪК' and computer == 'НОЖИЦА') or \
       (player == 'НОЖИЦА' and computer == 'ХАРТИЯ') or \
       (player == 'ХАРТИЯ' and computer == 'КАМЪК'):
        return "ПОБЕДА!", COLOR_WIN
    return "ЗАГУБА!", COLOR_LOSE

# --- ГЛАВЕН ЦИКЪЛ ---
with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success: break

        image = cv2.flip(image, 1)
        h, w, c = image.shape
        
        # 1. MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        current_gesture = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                data_aux = []
                for landmark in hand_landmarks.landmark:
                    data_aux.extend([landmark.x, landmark.y, landmark.z])
                try:
                    prediction = model.predict([data_aux])
                    current_gesture = labels_dict[int(prediction[0])]
                except: pass

        # 2. Логика
        key = cv2.waitKey(1)
        if key == 32: # SPACE
            if current_gesture:
                moves = ['КАМЪК', 'НОЖИЦА', 'ХАРТИЯ']
                computer_move = random.choice(moves)
                game_result_text, game_result_color = get_winner(current_gesture, computer_move)
                
                # ТУК Е ПРОМЯНАТА НА ТЕКСТА
                info_text = f"Компютър: {computer_move}"
                show_result_timer = 60 
            else:
                info_text = "Не виждам ръка!"
                game_result_text = ""

        # --- 3. UI ОТРИСУВАНЕ ---
        
        # Ленти
        overlay = image.copy()
        top_bar_h = int(h * 0.12)
        bottom_bar_h = int(h * 0.12)
        
        cv2.rectangle(overlay, (0, 0), (w, top_bar_h), (0, 0, 0), -1)
        
        bottom_color = (0, 0, 0)
        
        if show_result_timer > 0:
            r, g, b = game_result_color
            bottom_color = (b, g, r) 
            # Рамка
            cv2.rectangle(image, (0, 0), (w, h), bottom_color, thickness=20)
            show_result_timer -= 1
        else:
            game_result_text = ""
            info_text = "Натисни SPACE за игра"
        
        cv2.rectangle(overlay, (0, h - bottom_bar_h), (w, h), bottom_color, -1)
        image = cv2.addWeighted(overlay, 0.8, image, 0.2, 0)

        # Текстове
        display_gesture = current_gesture if current_gesture else "..."
        image = draw_ui_text(image, f"ТИ: {display_gesture}", 20, 20, COLOR_WHITE, size_percent=0.06)

        # ДОЛНА ЧАСТ
        if game_result_text:
            # Намалих размера на шрифта леко (0.04), за да се събере дългият текст "Компютър"
            image = draw_ui_text(image, game_result_text, 30, h - bottom_bar_h + 15, COLOR_WHITE, size_percent=0.07)
            # Слагаме малко повече отстъп отдясно (w - 40)
            image = draw_ui_text(image, info_text, w - 40, h - bottom_bar_h + 20, COLOR_WHITE, size_percent=0.045, align="right")
        else:
            image = draw_ui_text(image, info_text, w // 2, h - bottom_bar_h + 20, COLOR_WHITE, size_percent=0.05, align="center")

        cv2.imshow('Rock Paper Scissors PRO', image)
        if key == ord('q'): break

cap.release()
cv2.destroyAllWindows()