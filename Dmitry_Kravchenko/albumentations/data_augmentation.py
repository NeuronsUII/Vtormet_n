import cv2
import albumentations as A
import os

# определяем аугментации с учетом параметров боксов
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(p=0.2),
    A.Blur(p=0.2),
    A.CLAHE(p=0.2),
    A.RGBShift(p=0.2),
    A.ToGray(p=0.2),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

def clip_bbox(bbox, width, height):
    """Ограничивает координаты бокса значениями от 0 до ширины/высоты изображения."""
    x_min, y_min, x_max, y_max = bbox

    x_min = max(0, min(x_min, width))
    y_min = max(0, min(y_min, height))
    x_max = max(0, min(x_max, width))
    y_max = max(0, min(y_max, height))

    return [x_min, y_min, x_max, y_max]

def augment_and_save(input_image_dir, input_label_dir, output_image_dir, output_label_dir, augmentations_per_image=5):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    for img_name in os.listdir(input_image_dir):
        img_path = os.path.join(input_image_dir, img_name)
        img = cv2.imread(img_path)
        height, width, _ = img.shape

        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(input_label_dir, label_name)

        with open(label_path, 'r') as file:
            annotations = file.readlines()

        bboxes = []
        class_labels = []

        for annotation in annotations:
            class_id, x_center, y_center, bbox_width, bbox_height = map(float, annotation.split())

            x_min = (x_center - bbox_width / 2) * width
            y_min = (y_center - bbox_height / 2) * height
            x_max = (x_center + bbox_width / 2) * width
            y_max = (y_center + bbox_height / 2) * height

            # Обрезка координат бокса
            clipped_bbox = clip_bbox([x_min, y_min, x_max, y_max], width, height)
            bboxes.append(clipped_bbox)
            class_labels.append(class_id)

        # Создаем несколько аугментированных версий каждого изображения
        for i in range(augmentations_per_image):
            augmented = augmentations(image=img, bboxes=bboxes, class_labels=class_labels)
            aug_img = augmented['image']
            aug_bboxes = augmented['bboxes']

            aug_img_name = f'aug_{i}_{img_name}'
            cv2.imwrite(os.path.join(output_image_dir, aug_img_name), aug_img)

            with open(os.path.join(output_label_dir, f'aug_{i}_{label_name}'), 'w') as aug_file:
                for bbox, class_id in zip(aug_bboxes, class_labels):
                    x_min, y_min, x_max, y_max = bbox
                    new_x_center = (x_min + x_max) / 2 / width
                    new_y_center = (y_min + y_max) / 2 / height
                    new_bbox_width = (x_max - x_min) / width
                    new_bbox_height = (y_max - y_min) / height
                    aug_file.write(f"{class_id} {new_x_center} {new_y_center} {new_bbox_width} {new_bbox_height}\n")

# Пути к папкам для train и val выборок
train_image_dir = r'C:\Dima\Projects\Cuda\Albumentations\dataset\images\train'
train_label_dir = r'C:\Dima\Projects\Cuda\Albumentations\dataset\labels\train'
val_image_dir = r'C:\Dima\Projects\Cuda\Albumentations\dataset\images\val'
val_label_dir = r'C:\Dima\Projects\Cuda\Albumentations\dataset\labels\val'

# Пути для сохранения аугментированных данных
output_train_image_dir = r'C:\Dima\Projects\Cuda\Albumentations\augmented_images_train'
output_train_label_dir = r'C:\Dima\Projects\Cuda\Albumentations\augmented_labels_train'
output_val_image_dir = r'C:\Dima\Projects\Cuda\Albumentations\augmented_images_val'
output_val_label_dir = r'C:\Dima\Projects\Cuda\Albumentations\augmented_labels_val'

# Аугментация данных для train выборки
augment_and_save(train_image_dir, train_label_dir, output_train_image_dir, output_train_label_dir, augmentations_per_image=5)

# Аугментация данных для val выборки
augment_and_save(val_image_dir, val_label_dir, output_val_image_dir, output_val_label_dir, augmentations_per_image=5)
