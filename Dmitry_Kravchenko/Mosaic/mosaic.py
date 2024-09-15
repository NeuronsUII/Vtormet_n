import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from collections import Counter

# Функция для загрузки изображения и его аннотаций
def load_image_and_label(image_path, label_path):
    image = cv2.imread(image_path)
    label = []
    if not os.path.exists(label_path):
        print(f"Аннотация для {image_path} не найдена.")
        return image, label
    with open(label_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    label.append(parts)
                except ValueError:
                    print(f"Ошибка в аннотации {label_path}: {line}")
    return image, label

# Функция для сохранения изображения и аннотаций
def save_image_and_label(output_image, output_label, output_image_path, output_label_path):
    cv2.imwrite(output_image_path, output_image)
    with open(output_label_path, 'w') as file:
        for item in output_label:
            file.write(' '.join(item) + '\n')

# Функция для визуализации нескольких мозаичных изображений
def visualize_mosaic_images(output_image_dir, num_images=3):
    if not os.path.exists(output_image_dir):
        print(f"Путь {output_image_dir} не существует. Пропуск визуализации.")
        return

    images = [img for img in os.listdir(output_image_dir) if img.endswith('.jpg')]

    if len(images) == 0:
        print(f"В папке {output_image_dir} нет изображений для визуализации.")
        return

    plt.figure(figsize=(15, 5))
    for i in range(min(num_images, len(images))):
        img_path = os.path.join(output_image_dir, images[i])
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(1, num_images, i + 1)
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(f"Мозаика {i + 1}")
    plt.tight_layout()
    plt.show()

# Функция для визуализации аннотаций на мозаичных изображениях
def visualize_annotations(image_dir, label_dir, num_images=1):
    images = [img for img in os.listdir(image_dir) if img.endswith('.jpg')]
    labels = [lbl for lbl in os.listdir(label_dir) if lbl.endswith('.txt')]

    if len(images) == 0 or len(labels) == 0:
        print("Нет изображений или аннотаций для визуализации.")
        return

    plt.figure(figsize=(15, 5 * num_images))
    for i in range(min(num_images, len(images))):
        img_path = os.path.join(image_dir, images[i])
        label_path = os.path.join(label_dir, images[i].replace('.jpg', '.txt'))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width, _ = image.shape

        if not os.path.exists(label_path):
            print(f"Аннотация для {images[i]} не найдена.")
            continue

        with open(label_path, 'r') as file:
            annotations = file.readlines()

        for ann in annotations:
            class_id, x_center, y_center, bbox_width, bbox_height = map(float, ann.strip().split())
            x_center *= width
            y_center *= height
            bbox_width *= width
            bbox_height *= height

            x_min = int(x_center - bbox_width / 2)
            y_min = int(y_center - bbox_height / 2)
            x_max = int(x_center + bbox_width / 2)
            y_max = int(y_center + bbox_height / 2)

            # Отрисовка прямоугольника и метки класса
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
            cv2.putText(image, str(int(class_id)), (x_min, max(y_min - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        plt.subplot(num_images, 1, i + 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Изображение: {images[i]} с аннотациями")

    plt.tight_layout()
    plt.show()

# Функция для создания графиков распределения классов и статистики
def plot_class_distribution_with_stats(class_counter_before, class_counter_after, total_instances_before, total_instances_after, output_file_path):
    # Объединяем все классы
    all_classes = sorted(set(class_counter_before.keys()).union(set(class_counter_after.keys())))
    
    counts_before = [class_counter_before.get(cls, 0) for cls in all_classes]
    counts_after = [class_counter_after.get(cls, 0) for cls in all_classes]
    counts_total = [counts_before[i] + counts_after[i] for i in range(len(all_classes))]
    
    # Построение графиков
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # График "до обработки"
    axs[0].bar(all_classes, counts_before, color='skyblue')
    axs[0].set_title('Распределение классов до обработки')
    axs[0].set_xlabel('Класс')
    axs[0].set_ylabel('Количество экземпляров')
    
    # График "после обработки"
    axs[1].bar(all_classes, counts_after, color='lightgreen')
    axs[1].set_title('Распределение классов после обработки')
    axs[1].set_xlabel('Класс')
    axs[1].set_ylabel('Количество экземпляров')
    
    # График "общее распределение"
    axs[2].bar(all_classes, counts_total, color='salmon')
    axs[2].set_title('Общее распределение классов')
    axs[2].set_xlabel('Класс')
    axs[2].set_ylabel('Количество экземпляров')
    
    # Текстовая информация под графиками
    text_before = (f"До обработки:\n"
                   f"Всего экземпляров: {total_instances_before}\n" +
                   ''.join([f" - Класс {cls}: {count} шт\n" for cls, count in zip(all_classes, counts_before)]))
    
    text_after = (f"После обработки:\n"
                  f"Всего экземпляров: {total_instances_after}\n" +
                  ''.join([f" - Класс {cls}: {count} шт\n" for cls, count in zip(all_classes, counts_after)]))
    
    text_total = (f"Общее распределение:\n"
                  f"Всего экземпляров: {total_instances_before + total_instances_after}\n" +
                  ''.join([f" - Класс {cls}: {count} шт\n" for cls, count in zip(all_classes, counts_total)]))
    
    # Добавляем текст под каждым графиком
    axs[0].text(0.5, -0.35, text_before, transform=axs[0].transAxes, fontsize=10, va='top', ha='center')
    axs[1].text(0.5, -0.35, text_after, transform=axs[1].transAxes, fontsize=10, va='top', ha='center')
    axs[2].text(0.5, -0.35, text_total, transform=axs[2].transAxes, fontsize=10, va='top', ha='center')
    
    # Настройка отступов
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.4)
    
    # Сохранение графика
    plt.savefig(output_file_path, bbox_inches='tight')
    plt.show()
    
    print(f"Графики и статистика сохранены по пути: {output_file_path}")

# Функция для фильтрации изображений по количеству экземпляров классов
def filter_images_by_class_instances(label_paths, max_instances_per_class=2):
    filtered_indices = []
    for idx, label_path in enumerate(label_paths):
        class_counts = Counter()
        with open(label_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
        # Проверяем, не превышает ли количество экземпляров любого класса заданный порог
        if all(count <= max_instances_per_class for count in class_counts.values()):
            filtered_indices.append(idx)
    return filtered_indices

# Функция для создания мозаики из изображений с учетом недостающих классов
def mosaic_augmentation(image_paths, label_paths, output_image_dir, output_label_dir, mosaic_size=8, mosaic_image_size=640, target_class_instances=500):
    print("Начало мозаичной обработки...")

    # Проверка и создание директорий для изображений и аннотаций
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)

    # Подсчет начальной статистики (по количеству экземпляров классов)
    class_counter_before = Counter()
    print("Подсчет начальной статистики...")

    for label_path in label_paths:
        with open(label_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        class_id = int(parts[0])
                        class_counter_before[class_id] += 1  # Считаем каждый экземпляр класса
                    except ValueError:
                        print(f"Ошибка в аннотации {label_path}: {line}")

    total_instances_before = sum(class_counter_before.values())
    total_images_before = len(image_paths)

    # Определяем сколько нужно добавить экземпляров для каждого класса
    class_deficit = {class_id: max(0, target_class_instances - count) for class_id, count in class_counter_before.items()}

    # Логирование дефицита классов
    print("\nДефицит экземпляров по классам:")
    for class_id, deficit in class_deficit.items():
        print(f" - {class_id} класс: нехватает {deficit} экземпляров")

    # Создаем словарь, где для каждого класса есть список индексов изображений, содержащих этот класс
    class_to_image_indices = {}
    for class_id in class_deficit.keys():
        indices = []
        for idx, label_path in enumerate(label_paths):
            with open(label_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) >= 5 and int(parts[0]) == class_id:
                        indices.append(idx)
                        break  # Достаточно знать, что класс присутствует в этом изображении
        class_to_image_indices[class_id] = indices

    # Инициализируем счетчик добавленных экземпляров по классам
    total_instances_added = Counter()

    # Генерация мозаичных изображений
    mosaic_count = 0
    while any(deficit > 0 for deficit in class_deficit.values()):
        print(f"\nСоздание мозаичного изображения {mosaic_count + 1}")

        selected_indices = set()

        # Собираем необходимые экземпляры классов
        deficit_classes = [class_id for class_id, deficit in class_deficit.items() if deficit > 0]
        if not deficit_classes:
            print("Все дефициты покрыты. Обработка завершена.")
            break

        # Перемешиваем классы для случайного выбора
        random.shuffle(deficit_classes)

        # Выбираем изображения, содержащие недостающие классы
        while len(selected_indices) < mosaic_size ** 2 and deficit_classes:
            class_id = deficit_classes.pop(0)
            class_images = [idx for idx in class_to_image_indices.get(class_id, []) if idx not in selected_indices]
            if not class_images:
                continue
            idx = random.choice(class_images)
            selected_indices.add(idx)

            # Обновляем дефицит класса
            with open(label_paths[idx], 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        if cls_id in class_deficit and class_deficit[cls_id] > 0:
                            total_instances_added[cls_id] += 1
                            class_deficit[cls_id] -= 1

            # Если дефицит класса все еще есть, добавляем класс обратно в список
            if class_deficit[class_id] > 0:
                deficit_classes.append(class_id)

        # Заполняем оставшиеся места случайными изображениями, исключая те, которые содержат избыточные классы
        if len(selected_indices) < mosaic_size ** 2:
            remaining_indices = list(set(range(len(image_paths))) - selected_indices)
            num_additional = mosaic_size ** 2 - len(selected_indices)
            random.shuffle(remaining_indices)
            for idx in remaining_indices:
                # Проверяем, не содержит ли изображение классы, достигшие целевого количества
                with open(label_paths[idx], 'r') as file:
                    contains_excess_class = False
                    for line in file:
                        cls_id = int(line.strip().split()[0])
                        if total_instances_added[cls_id] >= target_class_instances:
                            contains_excess_class = True
                            break
                    if not contains_excess_class:
                        selected_indices.add(idx)
                if len(selected_indices) >= mosaic_size ** 2:
                    break

        selected_indices = list(selected_indices)[:mosaic_size ** 2]

        images = []
        labels = []
        for idx in selected_indices:
            img, lbl = load_image_and_label(image_paths[idx], label_paths[idx])
            images.append(img)
            labels.append(lbl)

        # Создание пустого изображения для мозаики
        tile_h = mosaic_image_size // mosaic_size
        tile_w = mosaic_image_size // mosaic_size
        mosaic_image = np.zeros((mosaic_image_size, mosaic_image_size, 3), dtype=np.uint8)

        new_labels = []

        # Заполнение мозаики изображениями и корректировка аннотаций
        for idx in range(mosaic_size ** 2):
            if idx >= len(images):
                break
            image = images[idx]
            label_list = labels[idx]

            row = idx // mosaic_size
            col = idx % mosaic_size
            start_y = row * tile_h
            start_x = col * tile_w

            resized_image = cv2.resize(image, (tile_w, tile_h))
            mosaic_image[start_y:start_y + tile_h, start_x:start_x + tile_w] = resized_image

            for label in label_list:
                class_id = int(label[0])

                # Проверяем, не достигли ли мы целевого количества экземпляров для класса
                if total_instances_added[class_id] >= target_class_instances:
                    continue

                x_center = (float(label[1]) * tile_w + start_x) / mosaic_image_size
                y_center = (float(label[2]) * tile_h + start_y) / mosaic_image_size
                width = float(label[3]) * tile_w / mosaic_image_size
                height = float(label[4]) * tile_h / mosaic_image_size

                # Проверяем корректность координат
                if width <= 0 or height <= 0 or x_center <= 0 or y_center <= 0 or x_center >= 1 or y_center >= 1:
                    continue

                new_label = [str(class_id), str(x_center), str(y_center), str(width), str(height)]
                new_labels.append(new_label)

                # Обновляем счетчики после добавления аннотации
                total_instances_added[class_id] += 1
                class_deficit[class_id] = max(0, class_deficit[class_id] - 1)

        # Сохранение мозаичного изображения и аннотаций
        output_image_path = os.path.join(output_image_dir, f'mosaic_{mosaic_count}.jpg')
        output_label_path = os.path.join(output_label_dir, f'mosaic_{mosaic_count}.txt')
        save_image_and_label(mosaic_image, new_labels, output_image_path, output_label_path)

        mosaic_count += 1

    # Подсчет статистики после обработки
    class_counter_after = Counter()
    print("\nПодсчет статистики для мозаичных изображений...")

    for label_file in os.listdir(output_label_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(output_label_dir, label_file), 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(parts[0])
                            class_counter_after[class_id] += 1
                        except ValueError:
                            print(f"Ошибка в аннотации {label_file}: {line}")

    total_instances_after = sum(class_counter_after.values())
    total_images_after = len(os.listdir(output_image_dir))

    # Вывод статистики
    print("\nДо обработки:")
    print(f"Всего изображений: {total_images_before}")
    print(f"Всего экземпляров: {total_instances_before}")
    print(f"Классы: {len(class_counter_before)}")
    for class_id, count in sorted(class_counter_before.items()):
        print(f" - {class_id} класс - {count} экземпляров")

    print("\nПосле обработки:")
    print(f"Всего изображений: {total_images_after}")
    print(f"Всего экземпляров: {total_instances_after}")
    print(f"Классы: {len(class_counter_after)}")
    for class_id, count in sorted(class_counter_after.items()):
        print(f" - {class_id} класс - {count} экземпляров")

    # Создание и сохранение графиков
    output_graph_path = os.path.join(output_image_dir, 'class_distribution.png')
    plot_class_distribution_with_stats(class_counter_before, class_counter_after, total_instances_before, total_instances_after, output_graph_path)

    # Визуализация нескольких мозаичных изображений
    visualize_mosaic_images(output_image_dir)

    # Визуализация аннотаций на мозаичных изображениях
    visualize_annotations(output_image_dir, output_label_dir, num_images=1)

    print("Обработка завершена, ошибок нет.")

# Пример использования
if __name__ == "__main__":
    image_dir = r'C:\Dima\Projects\Cuda\LOM\mosaic\images'
    label_dir = r'C:\Dima\Projects\Cuda\LOM\mosaic\labels'
    output_image_dir = r'C:\Dima\Projects\Cuda\LOM\mosaic\mosaic_output\images'
    output_label_dir = r'C:\Dima\Projects\Cuda\LOM\mosaic\mosaic_output\labels'

    # Получаем полные списки путей к изображениям и аннотациям
    image_paths_all = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
    label_paths_all = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.txt')]

    # Фильтруем изображения
    max_instances_per_class = 2  # Задайте желаемое максимальное количество экземпляров класса в изображении
    filtered_indices = filter_images_by_class_instances(label_paths_all, max_instances_per_class=max_instances_per_class)

    # Обновляем списки путей к изображениям и аннотациям
    image_paths = [image_paths_all[idx] for idx in filtered_indices]
    label_paths = [label_paths_all[idx] for idx in filtered_indices]

    # Вызов функции с обновленными списками
    mosaic_augmentation(
        image_paths,
        label_paths,
        output_image_dir,
        output_label_dir,
        mosaic_size=8,
        mosaic_image_size=640,
        target_class_instances=500  # Задайте желаемое количество экземпляров каждого класса
    )
