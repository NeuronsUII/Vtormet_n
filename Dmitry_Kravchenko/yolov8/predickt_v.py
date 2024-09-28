import cv2
import os
from ultralytics import YOLO
from collections import defaultdict

# Загрузка предобученной модели YOLOv8 с лучшими весами
model = YOLO('C:/Dima/Projects/Cuda/LOM/runs/detect/train36/weights/best.pt')

# Путь к тестовому видео
video_path = "C:/Dima/Projects/Cuda/LOM/predickt/TestV.mp4"
output_video_path = "C:/Dima/Projects/Cuda/LOM/results/output_TestV.mp4"
output_frame_path = "C:/Dima/Projects/Cuda/LOM/results/frames"  # Папка для сохранения кадров с обнаружениями
results_file = "C:/Dima/Projects/Cuda/LOM/results/detection_results.txt"  # Файл для записи результатов

# Открываем видео для обработки
cap = cv2.VideoCapture(video_path)

# Получаем характеристики исходного видео
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Определяем видео writer для сохранения результата
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Папка для сохранения кадров с обнаружениями
if not os.path.exists(output_frame_path):
    os.makedirs(output_frame_path)

# Словарь для подсчета обнаруженных объектов и запоминания классов
detected_classes = defaultdict(int)
saved_frames = set()  # Запоминаем классы, для которых уже сохранены кадры

# Открываем файл для записи результатов
with open(results_file, 'w') as f:
    f.write("Результаты распознавания:\n\n")

# Проходим по каждому кадру видео
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразуем кадр в формат, который понимает модель YOLOv8
    results = model(frame)

    # Рисуем результаты предсказаний на кадре
    annotated_frame = results[0].plot()

    # Сохраняем аннотированный кадр в выходной видеофайл
    out.write(annotated_frame)

    # Обрабатываем каждый объект на кадре
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Получаем ID класса
            confidence = float(box.conf[0])  # Уровень уверенности
            
            detected_classes[class_id] += 1  # Увеличиваем счетчик для класса

            # Сохраняем только один кадр для каждого обнаруженного класса с BB и уверенностью
            if class_id not in saved_frames and confidence > 0.15:
                time_detected = frame_idx / fps  # Время обнаружения в секундах
                save_frame_path = os.path.join(output_frame_path, f"class_{class_id}_frame_{frame_idx}.jpg")
                
                # Отрисовываем bounding boxes на изображении
                annotated_frame_with_bb = results[0].plot()  # Важно: используем тот же кадр с результатом
                
                # Сохраняем изображение с bounding box и уровнем уверенности
                cv2.imwrite(save_frame_path, annotated_frame_with_bb)
                saved_frames.add(class_id)  # Добавляем класс в сохраненные

                # Записываем результат в файл
                with open(results_file, 'a') as f:
                    f.write(f"Класс {class_id}: уверенность {confidence:.2f}, время обнаружения: {time_detected:.2f} сек\n")
                
                print(f"Обнаружен класс {class_id} с уверенностью {confidence:.2f} на {time_detected:.2f} секундах. Кадр сохранен: {save_frame_path}")

    frame_idx += 1
    print(f"Обработано {frame_idx}/{frame_count} кадров")

# Вывод итоговой статистики и запись в файл
with open(results_file, 'a') as f:
    f.write("\nОбщая статистика:\n")
    for class_id, count in detected_classes.items():
        f.write(f"Класс {class_id}: обнаружено {count} раз\n")
        print(f"Класс {class_id}: обнаружено {count} раз")

# Освобождаем ресурсы
cap.release()
out.release()

print(f"\nРезультаты сохранены в {output_video_path} и {results_file}")
