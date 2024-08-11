# renaming.py

import os
from utils import read_numbering, save_numbering

def rename_files(data_path, numbering):
    for dataset in ['train', 'valid', 'test']:
        images_path = os.path.join(data_path, dataset, 'images')
        labels_path = os.path.join(data_path, dataset, 'labels')

        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            print(f"Directory {dataset} does not exist, skipping.")
            continue

        images = sorted(os.listdir(images_path))
        labels = sorted(os.listdir(labels_path))

        for img, lbl in zip(images, labels):
            img_extension = os.path.splitext(img)[1]
            lbl_extension = os.path.splitext(lbl)[1]

            new_name = f'{numbering:06d}'
            os.rename(os.path.join(images_path, img), os.path.join(images_path, new_name + img_extension))
            os.rename(os.path.join(labels_path, lbl), os.path.join(labels_path, new_name + lbl_extension))

            numbering += 1

    return numbering

def rename_dataset(data_path):
    numbering = read_numbering(data_path)
    numbering = rename_files(data_path, numbering)
    save_numbering(data_path, numbering)
