import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import queue
import torch
# Загрузка модели YOLOv8
model = YOLO('bestvop.pt')  # Укажите свой путь к модели
# Проверяем доступность CUDA и выбираем устройство
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Глобальные переменные для параметров
grid_size = (3, 3)
source_type = 'camera'
video_path = None

# Глобальные переменные для хранения информации о детектируемых объектах
detected_objects = {}
timeout = 150  # Количество кадров, на которые будем задерживать боксы

# Очередь для кадров
frame_queue = queue.Queue()

# Флаг для остановки потоков
is_running = False

# Функция для выбора источника (веб-камера или видеофайл)
def choose_source():
    global source_type, video_path
    if source_type == 'video' and video_path:
        cap = cv2.VideoCapture(video_path)  # Захват видеофайла
    else:
        cap = cv2.VideoCapture(0)  # Захват с веб-камеры (0 - первая веб-камера)

    if not cap.isOpened():
        print(f"Не удалось открыть {source_type}")
        return None
    return cap

# Функция для захвата видео/веб-камеры в отдельном потоке
def capture_video_stream(cap):
    global is_running
    while is_running:
        ret, frame = cap.read()
        if not ret:
            break
        # Помещаем кадр в очередь
        frame_queue.put(frame)

    cap.release()

# Функция для обработки кадров в отдельном потоке
def capture_video_stream(cap):
    global is_running
    while is_running:
        ret, frame = cap.read()
        if not ret:
            break
        # Помещаем кадр в очередь
        frame_queue.put(frame)

    cap.release()

# Функция для обработки кадров в отдельном потоке
max_missed_frames = 90  # Количество кадров, на которых будем задерживать боксы (примерно 3 секунды при 30 fps)

# Модифицируем функцию process_video_stream
# Добавляем переменную для отслеживания текущего кадра
frame_counter = 0
frame_interval = 5  # Пропускать 2 кадра, обрабатывать 1

def process_video_stream():
    global grid_size, detected_objects, timeout, is_running, frame_counter, frame_interval

    while is_running:
        if not frame_queue.empty():
            frame = frame_queue.get()  # Получаем кадр из очереди

            # Увеличиваем счетчик кадров
            frame_counter += 1

            # Пропускаем обработку, если текущий кадр не соответствует заданному интервалу
            if frame_counter % frame_interval != 0:
                continue  # Пропустить обработку текущего кадра

            # Чтение параметров видео
            frame_width = frame.shape[1]
            frame_height = frame.shape[0]

            # Размеры каждой части кадра
            part_width = frame_width // grid_size[1]
            part_height = frame_height // grid_size[0]

            # Список для хранения обработанных частей
            processed_parts = []
            current_detections = []  # Сохранение текущих обнаружений для последующей обработки

            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    part_frame = frame[i * part_height: (i + 1) * part_height, j * part_width: (j + 1) * part_width]

                    # Применение YOLO трекинга
                    results = model.track(source=part_frame, show=False, save=False, tracker='botsort.yaml', conf=0.3, iou=0.475, agnostic_nms=False, max_det=100)

                    # Получение предсказанного кадра
                    processed_part = results[0].plot()  # Получение обработанного изображения
                    processed_parts.append(processed_part)

                    # Сохранение текущих обнаружений
                    for result in results:
                        for box in result.boxes:
                            current_detections.append(box)

            # Обработка временных рамок
            for obj_id in list(detected_objects.keys()):
                detected_objects[obj_id]['missed'] += 1  # Увеличиваем счетчик пропущенных кадров

            # Сохранение информации о текущих детекциях
            for box in current_detections:
                obj_id = box.id
                if obj_id in detected_objects:
                    detected_objects[obj_id]['box'] = box.xyxy.cpu().numpy()  # Обновляем позицию бокса
                    detected_objects[obj_id]['missed'] = 0  # Сбрасываем счетчик пропущенных кадров
                else:
                    detected_objects[obj_id] = {'box': box.xyxy.cpu().numpy(), 'missed': 0}

            # Отрисовка боксов на кадре
            for obj_id, info in list(detected_objects.items()):
                if info['missed'] < max_missed_frames:
                    # Отрисовка бокса на кадре
                    x1, y1, x2, y2 = info['box'][0]  # Получение координат бокса
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                elif info['missed'] >= max_missed_frames:
                    # Если объект не обнаруживается дольше max_missed_frames, удаляем его
                    del detected_objects[obj_id]

            # Объединение частей обратно в один кадр
            rows = []
            for i in range(grid_size[0]):
                row = np.hstack(processed_parts[i * grid_size[1]: (i + 1) * grid_size[1]])
                rows.append(row)

            final_frame = np.vstack(rows)

            # Отображение итогового кадра
            cv2.imshow('ВОП детектор', final_frame)

            # Прерывание программы по нажатию клавиши 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_tracking()
                break

    cv2.destroyAllWindows()
# Функция для выбора файла
def choose_file():
    global video_path
    video_path = filedialog.askopenfilename(title="Выберите видеофайл")
    video_path_entry.delete(0, tk.END)
    video_path_entry.insert(0, video_path)

# Функция запуска обработки
def start_tracking():
    global source_type, grid_size, is_running

    # Получаем размеры сетки из полей ввода
    try:
        grid_x = int(grid_x_entry.get())
        grid_y = int(grid_y_entry.get())
        grid_size = (grid_x, grid_y)
    except ValueError:
        grid_size = (2, 2)  # Значения по умолчанию, если не удалось прочитать

    cap = choose_source()
    if cap is not None:
        is_running = True

        # Запускаем потоки
        capture_thread = threading.Thread(target=capture_video_stream, args=(cap,))
        process_thread = threading.Thread(target=process_video_stream)

        capture_thread.start()
        process_thread.start()

# Функция остановки трекинга
def stop_tracking():
    global is_running
    is_running = False

# Функция изменения источника на основе выбранного переключателя
def update_source_type():
    global source_type
    source_type = source_var.get()
    if source_type == 'video':
        video_path_entry.config(state=tk.NORMAL)
        choose_file_button.config(state=tk.NORMAL)
    else:
        video_path_entry.config(state=tk.DISABLED)
        choose_file_button.config(state=tk.DISABLED)

# Интерфейс Tkinter
root = tk.Tk()
root.title("ВОП детектор")

# Поле для ввода пути к видеофайлу
video_path_label = tk.Label(root, text="Путь к видеофайлу:")
video_path_label.pack()

video_path_entry = tk.Entry(root, width=50, state=tk.DISABLED)
video_path_entry.pack()

choose_file_button = tk.Button(root, text="Выбрать видеофайл", command=choose_file, state=tk.DISABLED)
choose_file_button.pack()

# Переключатели для выбора источника
source_var = tk.StringVar(value='camera')  # По умолчанию выбрана камера

video_radio = tk.Radiobutton(root, text="Видео", variable=source_var, value='video', command=update_source_type)
video_radio.pack()

camera_radio = tk.Radiobutton(root, text="Камера", variable=source_var, value='camera', command=update_source_type)
camera_radio.pack()

# Поля для ввода размерности сетки
grid_label = tk.Label(root, text="Укажите размер сетки (x на y):")
grid_label.pack()

grid_x_entry = tk.Entry(root)
grid_x_entry.pack()
grid_x_entry.insert(0, '3')  # По умолчанию

grid_y_entry = tk.Entry(root)
grid_y_entry.pack()
grid_y_entry.insert(0, '3')  # По умолчанию

# Кнопка для старта трекинга
start_button = tk.Button(root, text="Начать трекинг", command=start_tracking)
start_button.pack()

# Кнопка для остановки трекинга
stop_button = tk.Button(root, text="Остановить трекинг", command=stop_tracking)
stop_button.pack()

# Запуск Tkinter
root.mainloop()
