# config.py

import os

# Общие переменные
project_path = 'C:/Dima/Projects/LOM/Stratific'
path_to_dataset = project_path
dataset_yaml_path = os.path.join(project_path, 'data.yaml')

path_to_train = os.path.join(path_to_dataset, "train/")
path_to_train_images = os.path.join(path_to_train, 'images/')
path_to_train_labels = os.path.join(path_to_train, 'labels/')

path_to_valid = os.path.join(path_to_dataset, "valid/")
path_to_valid_images = os.path.join(path_to_valid, 'images/')
path_to_valid_labels = os.path.join(path_to_valid, 'labels/')

path_to_test = os.path.join(path_to_dataset, "test/")
path_to_test_images = os.path.join(path_to_test, 'images/')
path_to_test_labels = os.path.join(path_to_test, 'labels/')

# переменные для renaming
# переменные для create_strat_dataset
val_ratio = 0.25
test_ratio = 0.15

# переменные для custom_dataset_analysis
path_to_train_labels_cus_dataset = os.path.join(path_to_dataset, "train/labels")
path_to_train_images_cus_dataset = os.path.join(path_to_dataset, "train/images")
path_to_valid_labels_cus_dataset = os.path.join(path_to_dataset, "valid/labels")
path_to_valid_images_cus_dataset = os.path.join(path_to_dataset, "valid/images")
path_to_test_labels_cus_dataset = os.path.join(path_to_dataset, "test/labels")
path_to_test_images_cus_dataset = os.path.join(path_to_dataset, "test/images")
