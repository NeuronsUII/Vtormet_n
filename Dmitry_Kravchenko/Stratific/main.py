import argparse
import os
from renaming import rename_files, read_numbering, save_numbering
from stratify import create_stratified_datasets
from visualize import visualize_train_valid_distribution

# Общие переменные
project_path = 'C:/Dima/Projects/LOM/Stratific'
path_to_dataset = project_path

path_to_train = os.path.join(path_to_dataset, "train")
path_to_train_images = os.path.join(path_to_train, 'images')
path_to_train_labels = os.path.join(path_to_train, 'labels')

path_to_valid = os.path.join(path_to_dataset, "valid")
path_to_valid_images = os.path.join(path_to_valid, 'images')
path_to_valid_labels = os.path.join(path_to_valid, 'labels')

path_to_test = os.path.join(path_to_dataset, "test")
path_to_test_images = os.path.join(path_to_test, 'images')
path_to_test_labels = os.path.join(path_to_test, 'labels')

# Параметры для create_strat_dataset
val_ratio = 0.1
test_ratio = 0.05

def main(task_mode):
    if task_mode == "renaming":
        numbering = read_numbering(path_to_dataset)
        numbering = rename_files(path_to_dataset, numbering)
        save_numbering(path_to_dataset, numbering)
    elif task_mode == "create_strat_dataset":
        source_data, train_data, validation_data, test_data, class_list = create_stratified_datasets(
            path_to_train_labels, path_to_train_images, path_to_valid, path_to_test, val_ratio, test_ratio)
        
        class_list = source_data.columns[2:].tolist()
        visualize_train_valid_distribution(source_data, train_data, validation_data, test_data, path_to_dataset, class_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Stratification")
    parser.add_argument("--task_mode", type=str, required=True, choices=["renaming", "create_strat_dataset"],
                        help="Task mode: renaming или create_strat_dataset")
    args = parser.parse_args()
    main(args.task_mode)
