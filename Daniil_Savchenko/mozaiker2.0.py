from PIL import Image, ImageOps
import numpy as np
import os
import random

# Функция для чтения аннотаций из файла YOLO
def read_yolo_annotations(ann_path):
    if not os.path.exists(ann_path):
        return []  # Возвращаем пустой список, если файл не найден

    annotations = []
    with open(ann_path, 'r') as f:
        for line in f:
            try:
                cls, x_center, y_center, width, height = map(float, line.strip().split())
                cls = int(cls)  # Приводим класс к целому числу
                annotations.append([cls, x_center, y_center, width, height])
            except ValueError as e:
                print(f"Ошибка при разборе строки: {line.strip()}\nОшибка: {e}")
    return annotations

# Функция для пересчета координат аннотаций для мозаики с учетом зеркалирования
def adjust_bbox(bbox, img_index, img_size, mosaic_size, grid_size, flip_horizontal=False, flip_vertical=False):
    cls, x_center, y_center, width, height = bbox
    row, col = divmod(img_index, grid_size)  # Определяем строку и столбец в мозаике

    img_w, img_h = img_size
    mosaic_w, mosaic_h = mosaic_size

    # Если изображение зеркалится по горизонтали
    if flip_horizontal:
        x_center = 1.0 - x_center

    # Если изображение зеркалится по вертикали
    if flip_vertical:
        y_center = 1.0 - y_center

    # Координаты с учетом смещения для каждого сегмента мозаики
    # Умножаем на размер изображения для перехода от относительных координат
    # к абсолютным, затем добавляем смещение и делим на размер мозаики
    x_center = (x_center * img_w + col * img_w) / mosaic_w
    y_center = (y_center * img_h + row * img_h) / mosaic_h
    width = (width * img_w) / mosaic_w
    height = (height * img_h) / mosaic_h

    return [cls, x_center, y_center, width, height]

# Функция для создания мозаики изображений и аннотаций с зеркалированием
# Функция для создания мозаики изображений и аннотаций с зеркалированием
# Функция для создания мозаики изображений и аннотаций с зеркалированием
def create_mosaic(image_paths, annotation_paths, output_img, output_ann, grid_size=3, augment=False):
    # Получаем размеры первого изображения
    img = Image.open(image_paths[0])
    w, h = img.size

    mosaic_size = (w * grid_size, h * grid_size)  # Размер мозаики
    mosaic_img = Image.new('RGB', mosaic_size)  # Создаем пустое изображение для мозаики
    all_annotations = []

    # Сортировка изображений случайным образом
    img_ann_pairs = list(zip(image_paths, annotation_paths))
    random.shuffle(img_ann_pairs)  # Рандомная сортировка
    image_paths, annotation_paths = zip(*img_ann_pairs)

    num_images = len(image_paths)

    for i in range(grid_size ** 2):
        # Обрабатываем все изображения, и если их меньше чем нужно - повторяем последние
        if i < num_images:
            img_path = image_paths[i]
            ann_path = annotation_paths[i]
        else:
            # Если мы превысили количество изображений, используем одно из предыдущих
            img_path = image_paths[i % num_images]  # Повторяем изображение
            ann_path = annotation_paths[i % num_images]  # Повторяем аннотацию

        img = Image.open(img_path)
        img = img.resize((w, h))  # Изменяем размер изображения

        # Применение аугментации: зеркалирование изображений
        flip_horizontal = False
        flip_vertical = False
        if augment:
            flip_horizontal = random.choice([True, False])  # Случайное горизонтальное зеркалирование
            flip_vertical = random.choice([True, False])  # Случайное вертикальное зеркалирование

            if flip_horizontal:
                img = ImageOps.mirror(img)  # Зеркалим изображение по горизонтали
            if flip_vertical:
                img = ImageOps.flip(img)  # Зеркалим изображение по вертикали

        row, col = divmod(i, grid_size)
        mosaic_img.paste(img, (col * w, row * h))

        # Чтение и пересчет аннотаций с учетом зеркалирования
        if os.path.exists(ann_path):
            annotations = read_yolo_annotations(ann_path)
            adjusted_annotations = [adjust_bbox(bbox, i, (w, h), mosaic_size, grid_size, flip_horizontal, flip_vertical)
                                    for bbox in annotations]
            all_annotations.extend(adjusted_annotations)
        else:
            print(f"Аннотации не найдены для {img_path}, пропуск.")

    # Уменьшение размера мозаики до 640x640 пикселей
    resized_mosaic = mosaic_img.resize((640, 640), Image.Resampling.LANCZOS)

    # Сохраняем итоговое изображение и аннотации
    resized_mosaic.save(output_img)
    with open(output_ann, 'w') as f:
        for ann in all_annotations:
            f.write(' '.join(map(str, ann)) + '\n')


# Функция для стратифицированного выбора изображений равномерно по классам
def stratified_image_selection(images_by_class, num_images_per_class):
    selected_images = []
    for images in images_by_class:
        selected = random.sample(images, min(len(images), num_images_per_class))
        selected_images.extend(selected)
    random.shuffle(selected_images)  # Перемешиваем выбранные изображения
    return selected_images

# Основная логика создания мозаики с несколькими пересборками
def main(image_folder, annotation_folder, output_folder, grid_size=3, num_repeats=1):
    os.makedirs(output_folder, exist_ok=True)

    # Получаем все изображения и аннотации
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])
    annotation_files = [f.replace('.jpg', '.txt') for f in image_files]

    # Классифицируем изображения по классам
    images_by_class = {i: [] for i in range(4)}  # Предполагается, что классов 4
    for img_file, ann_file in zip(image_files, annotation_files):
        ann_path = os.path.join(annotation_folder, ann_file)
        annotations = read_yolo_annotations(ann_path)
        if annotations:
            cls = annotations[0][0]  # Берем класс первого объекта
            images_by_class[cls].append(os.path.join(image_folder, img_file))

    # Убедимся, что для каждого класса есть хотя бы одно изображение
    min_class_count = min(len(images) for images in images_by_class.values())

    # Создаем мозаики с разной перетасовкой и зеркалированием
    for repeat in range(num_repeats):
        print(f"Пересборка мозаики: {repeat + 1}/{num_repeats}")
        for i in range(0, len(image_files), grid_size ** 2):
            # Выбираем изображения и аннотации равномерно по классам
            image_paths = stratified_image_selection(list(images_by_class.values()), grid_size ** 2 // 4)
            annotation_paths = [os.path.join(annotation_folder, os.path.basename(img).replace('.jpg', '.txt')) for img in image_paths]

            # Генерация имен файлов для вывода
            output_img = os.path.join(output_folder, f'mosaic_{repeat + 1}_{i // (grid_size ** 2)}.jpg')
            output_ann = os.path.join(output_folder, f'mosaic_{repeat + 1}_{i // (grid_size ** 2)}.txt')

            # Создаем мозаику с зеркалированием и рандомным распределением классов
            create_mosaic(image_paths, annotation_paths, output_img, output_ann, grid_size, augment=True)

# Пример использования
image_folder = r"F:/УЧЕБА/совмес/vtormetnew2009/train/images"
annotation_folder = r"F:/УЧЕБА/совмес/vtormetnew2009/train/labels"
output_folder = "F:/УЧЕБА/moz/output_mosaicstrainTV5n5"

main(image_folder, annotation_folder, output_folder, grid_size=5, num_repeats=10)
