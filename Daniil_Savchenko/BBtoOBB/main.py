from ultralytics import SAM
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
# from IPython.display import display, Image
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageChops, ImageTk, ImageFont
import matplotlib.patches as patches
import os
import threading
from matplotlib.widgets import Slider  # test delet
from ultralytics.models.sam import Predictor as SAMPredictor
from matplotlib.widgets import Button
from viewer import main_viewer

# Словарь для цветов классов
CLASS_COLORS = {
    0: 'red',
    1: 'blue',
    2: 'green',
    3: 'orange',
    4: 'purple',
    5: 'brown',
    6: 'pink',
    7: 'gray',
    8: 'cyan',
    9: 'magenta',
}


def load_model(model_path):
    return SAM(model_path)


def read_annotations_from_file(file_path):
    annotations = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                parts = line.split()
                print(f"Processing line: {line.strip()}")
                if len(parts) == 5:
                    try:
                        annotations.append([float(part) for part in parts])
                    except ValueError:
                        print(f"Skipping invalid line: {line.strip()}")
                else:
                    print(f"Invalid format: {line.strip()}")
    return annotations


def read_annotations_from_file_OBB(file_path):
    annotations_obb = []
    type_ann = None  # Инициализация переменной type_ann значением по умолчанию

    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    parts = line.split()
                    print(f"Processing line: {line}")  # Отладочный вывод

                    if len(parts) == 9:
                        try:
                            annotations_obb.append([float(part) for part in parts])
                            type_ann = 1  # Установить тип аннотации как OBB
                        except ValueError:
                            print(f"Skipping invalid line (OBB): {line}")
                    elif len(parts) == 5:
                        try:
                            annotations_obb.append([float(part) for part in parts])
                            type_ann = 0  # Установить тип аннотации как BB
                        except ValueError:
                            print(f"Skipping invalid line (BB): {line}")
                    else:
                        print(f"Invalid format: {line}")

    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return [], None

    if not annotations_obb:
        print("Файл аннотаций пуст или имеет неправильный формат аннотаций.")
        return [], None

    if type_ann is None:
        print(f"File content: {annotations_obb}")  # Отладочный вывод
        raise ValueError(
            "Не удалось определить тип аннотации. Файл может содержать аннотации в неверном формате.")

    return annotations_obb, type_ann

def denormalize_and_convert(cx, cy, rw, rh, img_width, img_height):
    x_center = cx * img_width
    y_center = cy * img_height
    box_width = rw * img_width
    box_height = rh * img_height

    x_min = int(x_center - box_width / 2)
    y_min = int(y_center - box_height / 2)
    x_max = int(x_center + box_width / 2)
    y_max = int(y_center + box_height / 2)
    print(f'denormalize_and_convert:{[x_min, y_min, x_max, y_max]}')
    return [x_min, y_min, x_max, y_max]

def denormalize_coordinates(coords, img_width, img_height):
     pixel_coords =[(x * img_width, y * img_height) for x, y in zip(coords[0::2], coords[1::2])]
     print(f"Координаты в пикселях denormalize_coordinates: {pixel_coords}")
     return
def expand_bbox(img_width, img_height, bbox, scale_factor):
    center_x = img_width / 2
    center_y = img_height / 2

    x_min, y_min, x_max, y_max = bbox

    # Уменьшаем координаты по коэффициенту
    width = x_max - x_min
    height = y_max - y_min

    new_width = width / scale_factor
    new_height = height / scale_factor

    # Пересчитываем координаты
    new_x_min = center_x - (center_x - x_min) / scale_factor
    new_y_min = center_y - (center_y - y_min) / scale_factor
    new_x_max = center_x + (x_max - center_x) / scale_factor
    new_y_max = center_y + (y_max - center_y) / scale_factor

    # Проверяем, чтобы координаты не выходили за границы изображения
    x_min = max(0, new_x_min)
    y_min = max(0, new_y_min)
    x_max = min(img_width, new_x_max)
    y_max = min(img_height, new_y_max)



    # Печатаем новые координаты для отладки
    print(f'shrink_bbox:{[x_min, y_min, x_max, y_max]}')

    return [x_min, y_min, x_max, y_max]

def get_mask_outline(mask_array):
    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(mask_array, kernel, iterations=1)
    outline = dilated_mask - mask_array
    outline_color = 255
    highlighted_mask = np.zeros_like(mask_array)
    highlighted_mask[mask_array > 0] = outline_color
    fatmask = cv2.add(highlighted_mask, outline)
    return fatmask


def get_bbox_from_mask(fatmask):
    contours, _ = cv2.findContours(fatmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("Контуры не найдены.")
        return None

    all_points = np.concatenate(contours)
    rect = cv2.minAreaRect(all_points)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    return box


def normalize_coordinates(coords, img_width, img_height):
    return [(x / img_width, y / img_height) for x, y in zip(coords[0::2], coords[1::2])]


def process_annotations(model, image_path, annotations, img_width, img_height, scale_factor):
    final_annotations = []

    for a in annotations:
        clas, cx, cy, rw, rh = a
        clas = int(clas)
        print(clas)

        bbox_pixel = denormalize_and_convert(cx, cy, rw, rh, img_width, img_height)
        expanded_bbox = expand_bbox(img_width, img_height, bbox_pixel, scale_factor)
        results = model(image_path, bboxes=[expanded_bbox])

        mask = results[0].masks.data[0]
        mask_array = mask.cpu().numpy().astype(np.uint8) * 255
        fatmask = get_mask_outline(mask_array)
        box = get_bbox_from_mask(fatmask)

        if box is None:
            print("Контур не найден, используем BB для OBB.")
            x1, y1 = bbox_pixel[0], bbox_pixel[1]
            x2, y2 = bbox_pixel[0] + bbox_pixel[2], bbox_pixel[1]
            x3, y3 = bbox_pixel[0] + bbox_pixel[2], bbox_pixel[1] + bbox_pixel[3]
            x4, y4 = bbox_pixel[0], bbox_pixel[1] + bbox_pixel[3]
        else:
            x1, y1 = box[0]
            x2, y2 = box[1]
            x3, y3 = box[2]
            x4, y4 = box[3]

        coordinates = f"{x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}"
        print("Координаты углов прямоугольника:", coordinates)

        annotations_list = [float(x) for x in coordinates.split()]
        normalized_coords = normalize_coordinates(annotations_list, img_width, img_height)

        flattened_coords = [coord for pair in normalized_coords for coord in pair]
        final_list = [clas] + flattened_coords
        final_annotations.append(final_list)

    return final_annotations


def save_annotations_to_file(annotations, output_path):
    with open(output_path, 'w') as file:
        for annotation in annotations:
            annotation_str = ' '.join(map(str, annotation))
            file.write(annotation_str + '\n')


def draw_bbox_on_image(image, annotations, img_height, img_width):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for annotation in annotations:
        _, x1, y1, x2, y2, x3, y3, x4, y4 = annotation
        coords = [x1, y1, x2, y2, x3, y3, x4, y4]
        print(f"Normalized coordinates: {coords}")

        denormalized_coords = [
            (x * img_width, y * img_height)
            for x, y in zip(coords[0::2], coords[1::2])
        ]
        print(f"Denormalized coordinates: {denormalized_coords}")

        polygon = patches.Polygon(denormalized_coords, closed=True, edgecolor='green', linewidth=3, fill=False)
        ax.add_patch(polygon)

    plt.show()


class ImageViewer:
    def __init__(self, root, image_paths, annotations_list, img_heights, img_widths):
        self.root = root
        self.image_paths = image_paths
        self.annotations_list = annotations_list
        self.img_heights = img_heights
        self.img_widths = img_widths
        self.idx = 0

        # Set up the canvas and frames
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, width=int(self.img_widths[0]), height=int(self.img_heights[0]))
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.info_frame = tk.Frame(root)
        self.info_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Buttons for navigation
        self.btn_prev = tk.Button(self.info_frame, text="Назад", command=self.prev_image)
        self.btn_prev.pack(side=tk.LEFT, padx=10, pady=10)

        self.btn_next = tk.Button(self.info_frame, text="Вперед", command=self.next_image)
        self.btn_next.pack(side=tk.RIGHT, padx=10, pady=10)

        # Slider for image navigation
        self.slider = tk.Scale(self.info_frame, from_=0, to=len(image_paths) - 1, orient=tk.HORIZONTAL, command=self.update_image_from_slider)
        self.slider.pack(fill=tk.X, padx=10, pady=5)

        # Label for image information
        self.info_label = tk.Label(self.info_frame, text=f'Изображение {self.idx + 1}/{len(self.image_paths)} - {os.path.basename(self.image_paths[self.idx])}', font=('Arial', 14))
        self.info_label.pack(fill=tk.X, padx=10, pady=5)

        # Frame for legend
        self.legend_frame = tk.Frame(self.info_frame)
        self.legend_frame.pack(fill=tk.X, padx=10, pady=5)

        # Create the legend
        self.create_legend()

        # Display the first image
        self.display_image()

    def display_image(self):
        # Load image
        img = Image.open(self.image_paths[self.idx])

        # Draw annotations
        draw = ImageDraw.Draw(img)
        annotations, type_ann = self.annotations_list[self.idx]
        self.draw_annotations_obb(draw, annotations, self.img_widths[self.idx], self.img_heights[self.idx], type_ann)

        # Update canvas with new image
        self.img_tk = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)
        self.update_info()

    def draw_annotations_obb(self, draw, annotations, img_width, img_height, type_ann):
        # Загрузка шрифта для текста
        try:
            font = ImageFont.truetype("arial.ttf", 14)  # Используйте подходящий шрифт и размер
        except IOError:
            font = ImageFont.load_default()

        for annotation in annotations:
            clas = int(annotation[0])
            coords = annotation[1:]
            color = CLASS_COLORS.get(clas, 'black')  # Цвет для аннотаций
            text_color = 'white'  # Цвет текста
            box_color = color  # Цвет фона текста

            if type_ann == 0:  # BB
                cx, cy, rw, rh = coords
                x_center = cx * img_width
                y_center = cy * img_height
                box_width = rw * img_width
                box_height = rh * img_height
                x_min = int(x_center - box_width / 2)
                y_min = int(y_center - box_height / 2)
                x_max = int(x_center + box_width / 2)
                y_max = int(y_center + box_height / 2)

                # Рисуем прямоугольник аннотации
                draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=5)

                # Текст аннотации
                text = f'Class {clas}'
                text_bbox = draw.textbbox((x_min, y_min - 20), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                # Прямоугольник для текста
                draw.rectangle([x_min, y_min - text_height - 4,
                                x_min + text_width + 4, y_min], fill=box_color)
                draw.text((x_min + 2, y_min - text_height - 2), text, font=font, fill=text_color)

            elif type_ann == 1:  # OBB
                x1, y1, x2, y2, x3, y3, x4, y4 = coords
                denormalized_coords = [
                    (x * img_width, y * img_height)
                    for x, y in zip([x1, x2, x3, x4], [y1, y2, y3, y4])
                ]

                # Рисуем аннотацию с помощью многоугольника
                draw.polygon(denormalized_coords, outline=color, width=5)

                # Текст аннотации
                text = f'Class {clas}'
                text_bbox = draw.textbbox(denormalized_coords[0], text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                # Прямоугольник для текста
                draw.rectangle([denormalized_coords[0][0], denormalized_coords[0][1] - text_height - 4,
                                denormalized_coords[0][0] + text_width + 4, denormalized_coords[0][1]], fill=box_color)
                draw.text((denormalized_coords[0][0] + 2, denormalized_coords[0][1] - text_height - 2), text, font=font,
                          fill=text_color)

    def create_legend(self):
        # Create legend in two rows
        legend_items = list(CLASS_COLORS.items())
        num_items = len(legend_items)
        num_cols = 5
        num_rows = (num_items + num_cols - 1) // num_cols  # Calculate number of rows

        # Create frames for rows
        row_frames = [tk.Frame(self.legend_frame) for _ in range(num_rows)]
        for frame in row_frames:
            frame.pack(fill=tk.X)

        # Add legend items to frames
        for i, (clas, color) in enumerate(legend_items):
            row = i // num_cols
            col = i % num_cols
            color_frame = tk.Frame(row_frames[row], width=10, height=10, bg=color, borderwidth=2, relief=tk.SOLID)
            color_frame.pack(side=tk.LEFT, padx=5, pady=5)

            label = tk.Label(row_frames[row], text=f'Class {clas}', fg='white', bg=color, padx=10, pady=5, font=('Arial', 12))
            label.pack(side=tk.LEFT)

    def update_info(self):
        self.info_label.config(text=f'Изображение {self.idx + 1}/{len(self.image_paths)} - {os.path.basename(self.image_paths[self.idx])}', font=('Arial', 14))
        self.slider.set(self.idx)

    def next_image(self):
        self.idx = (self.idx + 1) % len(self.image_paths)
        self.display_image()

    def prev_image(self):
        self.idx = (self.idx - 1) % len(self.image_paths)
        self.display_image()

    def update_image_from_slider(self, value):
        self.idx = int(value)
        self.display_image()

# Функция для отображения изображений с аннотациями OBB
def display_images_with_annotations_OBB(image_paths, annotations_list, img_heights, img_widths, parent_root):
    viewer_window = tk.Toplevel(parent_root)
    viewer_window.title("Просмотр аннотаций")
    viewer = ImageViewer(viewer_window, image_paths, annotations_list, img_heights, img_widths)

    # Start the event loop for the new window
    viewer_window.mainloop()
def load_model_with_progress(model_path, progress_bar, status_label):
    status_label.config(text="Загрузка модели...")
    progress_bar.start(10)  # Начинаем анимацию прогресс-бара
    model = SAM(model_path)
    progress_bar.stop()  # Останавливаем анимацию прогресс-бара
    status_label.config(text="Модель загружена.")
    return model


def main(image_folder, annotation_folder, progress_bar, status_label, scale_factor):
    model_path = "sam2_b.pt"

    # Создание выходной папки на основе имени папки с аннотациями
    annotation_folder_name = os.path.basename(annotation_folder.rstrip("/"))
    output_folder = os.path.join(os.path.dirname(annotation_folder), annotation_folder_name + '_OBB')
    os.makedirs(output_folder, exist_ok=True)

    # Загрузка модели с прогресс-баром
    model = load_model_with_progress(model_path, progress_bar, status_label)

    image_paths = []
    annotations_list = []
    img_heights = []
    img_widths = []

    for file_name in os.listdir(image_folder):
        if file_name.lower().endswith('.jpg'):  # Обработка только .jpg файлов
            image_path = os.path.join(image_folder, file_name)
            annotation_file_name = file_name.replace('.jpg', '.txt')
            annotations_path = os.path.join(annotation_folder, annotation_file_name)

            if not os.path.exists(annotations_path):
                print(f"Файл аннотаций {annotations_path} не существует, пропускаем.")
                continue

            annotations = read_annotations_from_file(annotations_path)
            print(f"Аннотации для {file_name}: {annotations}")

            img = cv2.imread(image_path)
            img_height, img_width = img.shape[:2]

            final_annotations = process_annotations(model, image_path, annotations, img_width, img_height, scale_factor)
            print(f'Список OBB для {file_name}:', final_annotations)

            output_annotations_path = os.path.join(output_folder, annotation_file_name)
            save_annotations_to_file(final_annotations, output_annotations_path)
            print(f'Нормализованные аннотации для {file_name} сохранены в файл: {output_annotations_path}')

            image_paths.append(image_path)
            annotations_list.append(final_annotations)
            img_heights.append(img_height)
            img_widths.append(img_width)

    status_label.config(text="Завершено.")


def select_folder(entry):
    folder_selected = filedialog.askdirectory()
    entry.delete(0, tk.END)
    entry.insert(0, folder_selected)

def check_annotation_type(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
        parts = first_line.split()
        if len(parts) == 9:
            return 'OBB'
        elif len(parts) == 5:
            return 'BB'
        else:
            return None
def process_images_and_annotations(model, image_folder, annotation_folder, progress_bar, status_label, scale_factor):
    # Создание выходной папки на основе имени папки с аннотациями
    annotation_folder_name = os.path.basename(annotation_folder.rstrip("/"))
    output_folder = os.path.join(os.path.dirname(annotation_folder), annotation_folder_name + '_OBB')
    os.makedirs(output_folder, exist_ok=True)

    image_paths = []
    annotations_list = []
    img_heights = []
    img_widths = []

    # Получаем список всех изображений для оценки количества итераций
    images = [file_name for file_name in os.listdir(image_folder) if file_name.endswith('.jpg')]
    total_images = len(images)
    progress_bar['maximum'] = total_images

    # Проверяем тип аннотаций из первого файла
    annotation_file = [file_name for file_name in os.listdir(annotation_folder) if file_name.endswith('.txt')]
    if annotation_file:
        annotation_type = check_annotation_type(os.path.join(annotation_folder, annotation_file[0]))
        if annotation_type == 'OBB':
            status_label.config(text="Выбраны OBB аннотации!")

    for idx, file_name in enumerate(images):
        image_path = os.path.join(image_folder, file_name)
        annotation_file_name = file_name.replace('.jpg', '.txt')
        annotations_path = os.path.join(annotation_folder, annotation_file_name)

        if not os.path.exists(annotations_path):
            print(f"Файл аннотаций {annotations_path} не существует, пропускаем.")
            continue

        annotations = read_annotations_from_file_OBB(annotations_path)[0]
        print(f"Аннотации для {file_name}: {annotations}")

        img = cv2.imread(image_path)
        img_height, img_width = img.shape[:2]

        final_annotations = process_annotations(model, image_path, annotations, img_width, img_height, scale_factor)
        print(f'Список OBB для {file_name}:', final_annotations)

        output_annotations_path = os.path.join(output_folder, annotation_file_name)
        save_annotations_to_file(final_annotations, output_annotations_path)
        print(f'Нормализованные аннотации для {file_name} сохранены в файл: {output_annotations_path}')

        image_paths.append(image_path)
        annotations_list.append(final_annotations)
        img_heights.append(img_height)
        img_widths.append(img_width)

        # Обновление прогресс-бара и статуса
        progress_bar.step(1)
        status_label.config(text=f"Обработка {idx + 1}/{total_images}...")

    status_label.config(text="Обработка завершена.")
    preview_obb(image_folder, f'{annotation_folder}_OBB')



        # Создание основного окна
    vind = tk.Tk()
    vind.withdraw()  # Скрыть основное окно, так как нам нужно только окно сообщения

        # Показ окна сообщения
    messagebox.showinfo(f'Готово!', f"Аннотаци в папке {annotation_folder}_OBB \n Изображения в папке {image_folder}_OBB")
        # Закрытие основного окна после показа сообщения
    vind.destroy()
    # Вызов функции для отображения сообщения
    show_message()


def preview_obb(image_folder, annotation_folder):
    image_paths = []
    annotations_list = []
    img_heights = []
    img_widths = []

    for file_name in os.listdir(image_folder):
        if file_name.endswith('.jpg'):
            image_path = os.path.join(image_folder, file_name)
            annotation_file_name = file_name.replace('.jpg', '.txt')
            annotations_path = os.path.join(annotation_folder, annotation_file_name)

            if not os.path.exists(annotations_path):
                print(f"Файл аннотаций {annotations_path} не существует, пропускаем.")
                continue

            try:
                annotations, type_ann = read_annotations_from_file_OBB(annotations_path)
                if type_ann is None:
                    print(f"Не удалось определить тип аннотации для файла {annotations_path}. Пропускаем.")
                    continue

                print(f"Аннотации для {file_name}: {annotations}")

                img = cv2.imread(image_path)
                if img is None:
                    print(f"Не удалось прочитать изображение {image_path}. Пропускаем.")
                    continue

                img_height, img_width = img.shape[:2]

                image_paths.append(image_path)
                annotations_list.append((annotations, type_ann))  # Изменено
                img_heights.append(img_height)
                img_widths.append(img_width)
            except ValueError as e:
                print(f"Ошибка чтения аннотаций из файла {annotations_path}: {e}")

    print("Список изображений:", image_paths)
    print("Список аннотаций:", annotations_list)
    try:
        display_images_with_annotations_OBB(image_paths, annotations_list, img_heights, img_widths, root)
    except Exception as e:
        # Игнорируем ошибку и продолжаем обработку следующего изображения
        print(f"Игнорирование ошибки потоков, пока не решено")


#def resize_and_pad_image(image, scale_factor):
#    """
#    Увеличивает изображение на заданный коэффициент, добавляя пустое пространство (черные поля).
#    """
#    img_height, img_width, _ = image.shape
#    new_width = int(img_width * scale_factor)
#    new_height = int(img_height * scale_factor)
#
#    # Создаем пустое изображение (черные поля) с новыми размерами
#    padded_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
#
#    # Вычисляем смещение для размещения оригинального изображения в центре нового изображения
#    x_offset = (new_width - img_width) // 2
#    y_offset = (new_height - img_height) // 2
#
#    # Вставляем оригинальное изображение в центр нового изображения
#    padded_image[y_offset:y_offset + img_height, x_offset:x_offset + img_width] = image
#
#    return padded_image

def resize_and_pad_image(image, scale_factor,fill_var):

    """
    Увеличивает изображение на заданный коэффициент, добавляя границы.
    """
    img_height, img_width, _ = image.shape
    new_width = int(img_width * scale_factor)
    new_height = int(img_height * scale_factor)

    # Создаем пустое изображение с новыми размерами
    if fill_var.get():
        padded_image = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)

        # Вычисляем смещение для размещения оригинального изображения в центре нового изображения
        x_offset = (new_width - img_width) // 2
        y_offset = (new_height - img_height) // 2

        # Заполняем края и углы нового изображения значениями крайних пикселей
        # Верхняя граница
        padded_image[:y_offset, x_offset:x_offset + img_width] = image[0, :, :]
        # Нижняя граница
        padded_image[-y_offset:, x_offset:x_offset + img_width] = image[-1, :, :]
        # Левая граница
        padded_image[y_offset:y_offset + img_height, :x_offset] = image[:, 0, :][:, np.newaxis, :]
        # Правая граница
        padded_image[y_offset:y_offset + img_height, -x_offset:] = image[:, -1, :][:, np.newaxis, :]

        # Углы
        padded_image[:y_offset, :x_offset] = image[0, 0, :]
        padded_image[:y_offset, -x_offset:] = image[0, -1, :]
        padded_image[-y_offset:, :x_offset] = image[-1, 0, :]
        padded_image[-y_offset:, -x_offset:] = image[-1, -1, :]

        # Вставляем оригинальное изображение в центр нового изображения
        padded_image[y_offset:y_offset + img_height, x_offset:x_offset + img_width] = image

    else:
        # Создаем пустое изображение (черные поля) с новыми размерами
        padded_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        x_offset = (new_width - img_width) // 2
        y_offset = (new_height - img_height) // 2
        padded_image[y_offset:y_offset + img_height, x_offset:x_offset + img_width] = image

    return padded_image

def resize_image_to_original(image, original_size):
    """
    Уменьшает изображение до оригинального размера.
    """
    img_height, img_width = original_size
    resized_image = cv2.resize(image, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
    return resized_image

def process_images_in_folder(input_folder, output_folder, scale_factor, fill_var):
    """
    Обрабатывает все изображения в указанной папке, увеличивает их с пустыми полями, а затем уменьшает до исходного размера.
    """

    # Создаем папку для сохранения изображений, если ее нет
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Проходим по всем файлам в папке
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            original_size = (image.shape[1], image.shape[0])

            # Увеличиваем изображение и добавляем пустое пространство
            padded_image = resize_and_pad_image(image, scale_factor, fill_var)

            # Уменьшаем изображение до оригинального размера
            resized_image = resize_image_to_original(padded_image, original_size)

            # Формируем путь для сохранения измененного изображения
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, resized_image)
            print(f"Изображение сохранено: {save_path}")

def run_script(image_folder_entry, annotation_folder_entry, progress_bar, status_label, scale_factor_entry, fill_var):

    image_folder = image_folder_entry.get()
    annotation_folder = annotation_folder_entry.get()
    scale_factor = scale_factor_entry.get()
    fill_value = fill_var.get()  # Получаем значение fill_var
    # Проверяем, что все поля заполнены
    if not image_folder or not annotation_folder or not scale_factor:
        messagebox.showwarning("Требуется ввод", "Пожалуйста, выберите обе папки перед запуском скрипта.")
        return

    try:
        scale_factor = float(scale_factor)  # Преобразуем фактор заполнения в число с плавающей запятой
    except ValueError:
        messagebox.showerror("Ошибка", "Фактор заполнения должен быть числом.")
        return
    def thread_target():
        status_label.config(text="Запуск...")
        model = load_model_with_progress("sam2_b.pt", progress_bar, status_label)
        process_images_in_folder(image_folder, f'{image_folder}_OBB', scale_factor, fill_var)  #,
        process_images_and_annotations(model, f'{image_folder}_OBB', annotation_folder, progress_bar, status_label, scale_factor)

    # Создание и запуск потока для выполнения основной задачи
    processing_thread = threading.Thread(target=thread_target)
    processing_thread.start()


def run_viewer():
    # Создание потока для запуска функции main_viewer
    viewer_thread = threading.Thread(target=main_viewer)

    # Запуск потока
    viewer_thread.start()
def create_gui():
    global root
    root = tk.Tk()
    root.title("BB to OBB")

    # Выбор папки с изображениями
    tk.Label(root, text="Выберите папку с изображениями:").grid(row=0, column=0, padx=10, pady=10)
    image_folder_entry = tk.Entry(root, width=50)
    image_folder_entry.grid(row=0, column=1, padx=10, pady=10)
    tk.Button(root, text="Обзор", command=lambda: select_folder(image_folder_entry)).grid(row=0, column=2, padx=10,
                                                                                          pady=10)

    # Выбор папки с аннотациями
    tk.Label(root, text="Выберите папку с аннотациями:").grid(row=1, column=0, padx=10, pady=10)
    annotation_folder_entry = tk.Entry(root, width=50)
    annotation_folder_entry.grid(row=1, column=1, padx=10, pady=10)
    tk.Button(root, text="Обзор", command=lambda: select_folder(annotation_folder_entry)).grid(row=1, column=2, padx=10,
                                                                                               pady=10)
    # Фактор заполнения
    tk.Label(root, text="Фактор заполнения:").grid(row=2, column=2, padx=10, pady=0, sticky='e')
    scale_factor_entry = tk.Entry(root, width=5)
    scale_factor_entry.grid(row=3, column=2, padx=10, pady=0)
    scale_factor_entry.insert(0, "1.5")  # Устанавливаем значение по умолчанию

    # Интерфейс для выбора метода заполнения
    fill_var = tk.BooleanVar(root)
    fill_check = tk.Checkbutton(root, text="Заполняющая заливка", variable=fill_var)
    fill_check.grid(row=2, column=0, padx=10, pady=0)

    # Кнопка запуска
    run_button = tk.Button(root, text="Запуск конвертации",
                           command=lambda:  run_script(image_folder_entry, annotation_folder_entry, progress_bar,
                                                      status_label, scale_factor_entry, fill_var))
    run_button.grid(row=2, columnspan=3, pady=10)
    # Кнопка просмотра OBB
    preview_button = tk.Button(root, text="Просмотр BB и OBB",
                               command=lambda: preview_obb(image_folder_entry.get(), annotation_folder_entry.get()))
    preview_button.grid(row=3, columnspan=3, pady=10)

    # Кнопка сравнения
    run_button = tk.Button(root, text="Сравнение аннотаций",
                           command=lambda: run_viewer())
    run_button.grid(row=4, columnspan=3, pady=10)

    # Прогресс-бар
    progress_bar = ttk.Progressbar(root, orient="horizontal", mode="indeterminate", length=400)
    progress_bar.grid(row=5, columnspan=3, pady=10)

    # Метка статуса
    status_label = tk.Label(root, text="Готово к работе.")
    status_label.grid(row=6, columnspan=3, pady=10)

    root.mainloop()


if __name__ == "__main__":

    create_gui()
