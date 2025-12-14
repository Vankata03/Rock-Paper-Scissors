import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Зареждане на данните
try:
    data = pd.read_csv('hand_data.csv')
except FileNotFoundError:
    print("Грешка: Файлът hand_data.csv липсва! Първо събери данни.")
    exit()

# Разделяме входа (координатите) от изхода (етикета: 0, 1 или 2)
X = data.drop('label', axis=1) # Всичко без етикета
y = data['label']              # Само етикета

# 2. Разделяне на данни за обучение и за тест (20% за тест)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

# 3. Създаване и обучение на класификатора (Random Forest)
model = RandomForestClassifier()
print("Обучение на модела...")
model.fit(x_train, y_train)

# 4. Проверка колко е точен
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)
print(f'Готово! Точност на модела: {score * 100:.2f}%')

# 5. Запазване на модела във файл
with open('model.p', 'wb') as f:
    pickle.dump(model, f)

print("Моделът е запазен успешно във файл 'model.p'")