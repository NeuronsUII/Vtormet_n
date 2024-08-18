import os
import matplotlib.pyplot as plt

def count_files_in_dir(directory):
    """Функция для подсчета количества файлов в папке."""
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

def plot_comparison(original_counts, augmented_counts, labels, title, save_path):
    """Функция для построения графика сравнения с текстовым выводом количества файлов и сохранением в файл."""
    x = range(len(labels))
    width = 0.4

    fig, ax = plt.subplots()
    ax.bar(x, original_counts, width, label='До аугментации', color='blue')
    ax.bar([p + width for p in x], augmented_counts, width, label='После аугментации', color='orange')

    ax.set_xlabel('Выборка')
    ax.set_ylabel('Количество файлов')
    ax.set_title(title)
    ax.set_xticks([p + width/2 for p in x])
    ax.set_xticklabels(labels)
    ax.legend()

    for i in x:
        ax.text(i, original_counts[i] + 5, f'{original_counts[i]}', ha='center', color='blue')
        ax.text(i + width, augmented_counts[i] + 5, f'{augmented_counts[i]}', ha='center', color='orange')

    # Сохранение графика
    plt.savefig(save_path)
    plt.show()

# Пути к папкам для train и val выборок
train_image_dir = r'C:\Dima\Projects\Cuda\Albumentations\dataset\images\train'
train_label_dir = r'C:\Dima\Projects\Cuda\Albumentations\dataset\labels\train'
val_image_dir = r'C:\Dima\Projects\Cuda\Albumentations\dataset\images\val'
val_label_dir = r'C:\Dima\Projects\Cuda\Albumentations\dataset\labels\val'

# Пути для аугментированных данных
output_train_image_dir = r'C:\Dima\Projects\Cuda\Albumentations\augmented_images_train'
output_train_label_dir = r'C:\Dima\Projects\Cuda\Albumentations\augmented_labels_train'
output_val_image_dir = r'C:\Dima\Projects\Cuda\Albumentations\augmented_images_val'
output_val_label_dir = r'C:\Dima\Projects\Cuda\Albumentations\augmented_labels_val'

# Подсчет количества изображений и аннотаций до и после аугментации
original_image_counts = [
    count_files_in_dir(train_image_dir),
    count_files_in_dir(val_image_dir)
]

augmented_image_counts = [
    count_files_in_dir(output_train_image_dir),
    count_files_in_dir(output_val_image_dir)
]

original_label_counts = [
    count_files_in_dir(train_label_dir),
    count_files_in_dir(val_label_dir)
]

augmented_label_counts = [
    count_files_in_dir(output_train_label_dir),
    count_files_in_dir(output_val_label_dir)
]

# Визуализация для изображений
plot_comparison(
    original_image_counts,
    augmented_image_counts,
    labels=['Тренировочная', 'Валидационная'],
    title='Сравнение количества изображений до и после аугментации',
    save_path='comparison_images.png'
)

# Визуализация для аннотаций
plot_comparison(
    original_label_counts,
    augmented_label_counts,
    labels=['Тренировочная', 'Валидационная'],
    title='Сравнение количества аннотаций до и после аугментации',
    save_path='comparison_labels.png'
)
