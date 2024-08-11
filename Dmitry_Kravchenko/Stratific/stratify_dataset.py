import os
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

def create_stratified_datasets(images_dir, labels_dir, base_dir, val_ratio=0.1, test_ratio=0.05):
    # Сначала соберем все пути к изображениям и меткам
    images = sorted([os.path.join(images_dir, img) for img in os.listdir(images_dir)])
    labels = sorted([os.path.join(labels_dir, lbl) for lbl in os.listdir(labels_dir)])
    
    # Создаем DataFrame для удобной работы
    data = pd.DataFrame({
        'image_path': images,
        'label_path': labels
    })
    
    # Пример того, как можно извлечь класс из файла аннотации. Предполагаем, что в аннотациях есть класс
    # Если структура меток другая, этот код нужно будет адаптировать
    data['class'] = data['label_path'].apply(lambda x: extract_class(x))
    
    # Разделяем данные на train, valid и test
    train_data, temp_data = train_test_split(data, test_size=(val_ratio + test_ratio), stratify=data['class'])
    val_data, test_data = train_test_split(temp_data, test_size=test_ratio/(test_ratio + val_ratio), stratify=temp_data['class'])
    
    # Функция для копирования файлов в новые директории
    def copy_files(file_paths, dest_dir):
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        for file_path in file_paths:
            shutil.copy(file_path, dest_dir)
    
    # Копируем файлы в соответствующие директории в проекте
    copy_files(train_data['image_path'], os.path.join(base_dir, 'train', 'images'))
    copy_files(train_data['label_path'], os.path.join(base_dir, 'train', 'labels'))
    
    copy_files(val_data['image_path'], os.path.join(base_dir, 'valid', 'images'))
    copy_files(val_data['label_path'], os.path.join(base_dir, 'valid', 'labels'))
    
    copy_files(test_data['image_path'], os.path.join(base_dir, 'test', 'images'))
    copy_files(test_data['label_path'], os.path.join(base_dir, 'test', 'labels'))

    return train_data, val_data, test_data

def extract_class(label_path):
    # Пример функции для извлечения класса из метки.
    # Добавляем проверку на пустой файл.
    with open(label_path, 'r') as f:
        lines = f.readlines()
        if not lines:  # если файл пустой
            return 'unknown'  # или любое другое значение по умолчанию
        # Предполагается, что класс - это первая цифра в первой строке
        class_id = lines[0].split()[0] if lines[0].strip() else 'unknown'
    return class_id

# Укажи директории для изображений и меток
images_dir = 'C:/Dima/Projects/LOM/Stratific/images'
labels_dir = 'C:/Dima/Projects/LOM/Stratific/labels'

# Указываем базовую директорию проекта
base_dir = 'C:/Dima/Projects/LOM/Stratific'

# Указываем директорию для сохранения результатов (выводов и графиков)
output_dir = os.path.join(base_dir, 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Выполнение стратификации
train_data, val_data, test_data = create_stratified_datasets(images_dir, labels_dir, base_dir)

# Открываем файл для записи информации
with open(os.path.join(output_dir, 'dataset_summary.txt'), 'w') as f:
    # Записываем количество файлов в каждой выборке
    f.write(f"Train set contains {len(train_data)} files.\n")
    f.write(f"Validation set contains {len(val_data)} files.\n")
    f.write(f"Test set contains {len(test_data)} files.\n\n")

    # Подсчет количества каждого класса в каждой выборке
    train_class_counts = train_data['class'].value_counts()
    val_class_counts = val_data['class'].value_counts()
    test_class_counts = test_data['class'].value_counts()

    f.write("Class distribution in Train set:\n")
    f.write(train_class_counts.to_string())
    f.write("\n\n")

    f.write("Class distribution in Validation set:\n")
    f.write(val_class_counts.to_string())
    f.write("\n\n")

    f.write("Class distribution in Test set:\n")
    f.write(test_class_counts.to_string())
    f.write("\n")

# Визуализация распределения классов и сохранение графиков
train_class_counts.plot(kind='bar', title='Class Distribution in Train Set')
plt.xlabel('Class')
plt.ylabel('Number of Instances')
plt.savefig(os.path.join(output_dir, 'train_class_distribution.png'))
plt.show()

val_class_counts.plot(kind='bar', title='Class Distribution in Validation Set')
plt.xlabel('Class')
plt.ylabel('Number of Instances')
plt.savefig(os.path.join(output_dir, 'val_class_distribution.png'))
plt.show()

test_class_counts.plot(kind='bar', title='Class Distribution in Test Set')
plt.xlabel('Class')
plt.ylabel('Number of Instances')
plt.savefig(os.path.join(output_dir, 'test_class_distribution.png'))
plt.show()
