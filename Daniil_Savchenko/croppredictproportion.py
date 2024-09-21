import cv2
import numpy as np
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Путь к модели 'D:/Dow/best180940+60.pt'
yolov8_model_path = 'D:/Dow/best 44.pt'

# Инициализация модели YOLOv8
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=yolov8_model_path,
    confidence_threshold=0.2,
    device="cuda:0",
    tracker='botsort.yaml'
)

# Путь к видеофайлу
video_path = 'F:/УЧЕБА/WhatsApp Video 2024-06-15 at 08.54.44.mp4'

# Открытие видеофайла
cap = cv2.VideoCapture(video_path)

# Переменная для хранения предыдущих боксов
prev_boxes = []

# Переменная для контроля частоты обработки кадров
frame_rate = 10  # Каждый 10-й кадр
current_frame = 0

# Настройки разрезания кадра (1/6 по горизонтали, 1/9 по вертикали)
slices_x = 4  # Количество частей по горизонтали для 6 на 6 4  для полноразмерных 4
slices_y = 7  # Количество частей по вертикали              7                     2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width, _ = frame.shape

    # Вычисляем размеры частей
    slice_height = frame_height // slices_y
    slice_width = frame_width // slices_x

    # Рисование вертикальных линий (по горизонтальным частям)
    for i in range(1, slices_x):
        x_pos = i * slice_width
        cv2.line(frame, (x_pos, 0), (x_pos, frame_height), (0, 0, 255), 2)

    # Рисование горизонтальных линий (по вертикальным частям)
    for j in range(1, slices_y):
        y_pos = j * slice_height
        cv2.line(frame, (0, y_pos), (frame_width, y_pos), (0, 0, 255), 2)

    # Выполнение детекции только на каждом 10-м кадре
    if current_frame % frame_rate == 0:
        # Переменная для хранения всех боксов на кадре
        prev_boxes = []

        # Проходим по каждой области кадра
        for i in range(slices_x):
            for j in range(slices_y):
                # Вычисляем координаты текущего среза
                x1 = i * slice_width
                y1 = j * slice_height
                x2 = x1 + slice_width
                y2 = y1 + slice_height

                # Извлекаем срез кадра
                frame_slice = frame[y1:y2, x1:x2]

                # Выполнение предсказания на срезе
                result = get_sliced_prediction(
                    frame_slice,
                    detection_model,
                    slice_height=slice_height,
                    slice_width=slice_width,
                    overlap_height_ratio=0,
                    overlap_width_ratio=0
                )

                # Конвертирование PIL Image в numpy массив, если нужно
                if isinstance(result.image, Image.Image):
                    result_image = np.array(result.image)
                else:
                    result_image = np.array(result.image)

                # Сохраняем боксы с преобразованием к координатам исходного кадра
                for obj in result.object_prediction_list:
                    if obj.score.value >= 0.2:
                        bbox = obj.bbox
                        # Преобразование координат среза к исходным координатам кадра
                        new_x1 = x1 + bbox.minx
                        new_y1 = y1 + bbox.miny
                        new_x2 = x1 + bbox.maxx
                        new_y2 = y1 + bbox.maxy
                        obj.bbox.minx, obj.bbox.miny = new_x1, new_y1
                        obj.bbox.maxx, obj.bbox.maxy = new_x2, new_y2
                        prev_boxes.append(obj)

    # Используем сохранённые боксы (либо от текущего, либо от предыдущего кадра)
    for obj in prev_boxes:
        bbox = obj.bbox
        x1, y1, x2, y2 = map(int, [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy])

        # Попробуем использовать 'category', если 'category_name' недоступен
        label = f"{obj.category.name} {obj.score.value:.2f}"

        # Рисуем прямоугольник (bbox) на кадре
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Добавляем метку класса над боксом
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Отображение кадра
    cv2.imshow('YOLOv8 Tracking', frame)

    # Переход к следующему кадру
    current_frame += 1

    # Выход при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
