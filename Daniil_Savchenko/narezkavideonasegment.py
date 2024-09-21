import cv2
import os


def process_video(video_path, output_folder, n_frame=20, seg_rows=3, seg_cols=2, target_size=(640, 640)):
    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка при открытии видео.")
        return

    frame_count = 0
    saved_image_count = 0

    # Создаем папку для сохранения изображений
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    while True:
        ret, frame = cap.read()

        # Если кадр не считан, то конец видео
        if not ret:
            break

        # Обрабатываем каждый n-ый кадр
        if frame_count % n_frame == 0:
            h, w, _ = frame.shape  # Получаем размеры кадра

            # Размеры каждого сегмента
            seg_h, seg_w = h // seg_rows, w // seg_cols

            # Разделяем кадр на сегменты и сохраняем их
            for row in range(seg_rows):
                for col in range(seg_cols):
                    # Координаты сегмента
                    x_start, x_end = col * seg_w, (col + 1) * seg_w
                    y_start, y_end = row * seg_h, (row + 1) * seg_h

                    # Вырезаем сегмент
                    segment = frame[y_start:y_end, x_start:x_end]

                    # Масштабируем сегмент до 640x640
                    segment_resized = cv2.resize(segment, target_size)

                    # Сохраняем сегмент как изображение
                    save_path = os.path.join(output_folder, f"frame_{frame_count}_seg_{row}_{col}.jpg")
                    cv2.imwrite(save_path, segment_resized)
                    saved_image_count += 1

        frame_count += 1

    # Освобождаем видео
    cap.release()
    print(f"Обработка завершена. Сохранено {saved_image_count} изображений.")


# Пример использования
video_path = 'D:/Dow/Воп.mp4'  # Замените на путь к вашему видео
output_folder = r'D:/anaconda3/envs/test/output_folder'   # Папка для сохранения изображений
process_video(video_path, output_folder, n_frame=20)
