import cv2
import os
from ultralytics import YOLO
from collections import Counter

model = YOLO('yolov8n.pt')

# Список изображений
image_paths = [os.path.join("C:\\Users\\Aigul\\PycharmProjects\\pythonProject\\С#", f"image{i}.jpg") for i in range(1, 21)]

# Переменные для хранения точности и средней точности
accuracies = []
total_accuracy = 0

# Проверка на 20 изображениях и оценка точности
for image_path in image_paths:
    results = model(image_path)

    for result in results:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 640, 480)  # Изменение размера окна

        prediction = result.boxes.cls.numpy()
        for i, box in enumerate(result.boxes.xyxy):
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(img, model.names[prediction[i]], (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0), 2)

        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Вычисление точности для текущего изображения
        prediction = result.boxes.cls.numpy()
        real_labels = os.path.basename(image_path).split('_')[1:]
        if len(real_labels) > 0:
            real_labels = [int(label.split('.')[0]) for label in real_labels]
            intersection = Counter(real_labels) & Counter(prediction)
            accuracy = int((sum(intersection.values()) / len(real_labels) * 100))
            accuracies.append(accuracy)
            total_accuracy += accuracy
            print(f'Точность предсказания для {os.path.basename(image_path)}: {accuracy}%')

# Расчет средней точности
average_accuracy = total_accuracy / len(accuracies) if accuracies else 0
print(f'Средняя точность: {average_accuracy}%')

# Распознавание с веб-камеры
cap = cv2.VideoCapture(0)

# Распознавание из видеофайла
# cap = cv2.VideoCapture("gerda.mp4")

confidence_threshold = 0.7

while True:
    # Считывание кадра
    ret, frame = cap.read()

    # Применение модели к кадру
    results = model(frame)[0]

    # Отображение обнаруженных объектов на кадре
    for data in results.boxes.data.tolist():
        x1, y1, x2, y2, confidence, class_id = data
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_name = results.names[int(class_id)]

        if confidence > confidence_threshold:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Отображение кадра с обнаруженными объектами
    cv2.imshow('YOLO Object Detection', frame)

    # Нажатие кнопки 'q' для выхода из цикла
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение захвата
cap.release()
cv2.destroyAllWindows()