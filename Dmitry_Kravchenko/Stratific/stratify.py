import os
import shutil
import pandas as pd
import random
from collections import Counter

def create_subset(data, class_list, subset_ratio, ratio_type):
    subset_indices = set()
    subset_data = pd.DataFrame(columns=data.columns)
    
    for c in sorted(class_list, key=lambda x: data[x].sum()):
        required_samples = int(data[c].sum() * subset_ratio)
        class_total = 0

        for index, row in data[data[c] > 0].iterrows():
            if class_total + row[c] > required_samples:
                break
            subset_indices.add(index)
            class_total += row[c]

    subset_indices_list = list(subset_indices)
    subset_data = data.loc[subset_indices_list]
    subset_data = subset_data.drop_duplicates().sort_values(by='random_id')

    updated_train_data = data.drop(subset_indices)

    return subset_data, updated_train_data

def ensure_class_representation(train_df, valid_df, test_df, class_list):
    def move_samples(source_df, target_df, class_id, num_samples):
        # Фильтруем элементы, соответствующие class_id
        samples = source_df[source_df[class_id] > 0]
    
        # Проверяем, есть ли доступные элементы для выборки
        if len(samples) > 0:
            # Если доступных элементов меньше, чем нужно, уменьшаем количество
            num_samples = min(num_samples, len(samples))
    
            # Выполняем выборку
            sampled_data = samples.sample(num_samples, random_state=42)
            source_df = source_df.drop(sampled_data.index)
            target_df = pd.concat([target_df, sampled_data])
    
        return source_df, target_df

    for class_id in class_list:
        if valid_df[class_id].sum() == 0:
            train_df, valid_df = move_samples(train_df, valid_df, class_id, 1)
        if test_df[class_id].sum() == 0:
            train_df, test_df = move_samples(train_df, test_df, class_id, 1)

    return train_df, valid_df, test_df

def create_stratified_datasets(path_to_train_labels, path_to_images, path_to_valid, path_to_test, validation_ratio, test_ratio):
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    class_list = set()
    file_data = []

    for label_file in os.listdir(path_to_train_labels):
        file_path = os.path.join(path_to_train_labels, label_file)
        base_name = label_file.replace('.txt', '')

        image_file_name = next((base_name + '.' + ext for ext in image_extensions if os.path.exists(os.path.join(path_to_images, base_name + '.' + ext))), None)
        if image_file_name is None:
            continue

        class_counts = {}
        with open(file_path, 'r') as file:
            for line in file:
                class_id = line.split()[0]
                class_list.add(class_id)
                class_counts[class_id] = class_counts.get(class_id, 0) + 1

        random_id = random.randint(0, 1000000)
        file_data.append([label_file, random_id] + [class_counts.get(c, 0) for c in class_list])

    columns = ['label_file', 'random_id'] + list(class_list)
    source_data = pd.DataFrame(file_data, columns=columns)
    source_data.sort_values(by='random_id', inplace=True)

    valid_data, remaining_data = create_subset(source_data, class_list, validation_ratio, 'validation')
    adjusted_test_ratio = test_ratio / (1 - validation_ratio)
    test_data, train_data = create_subset(remaining_data, class_list, adjusted_test_ratio, 'test')

    train_data, valid_data, test_data = ensure_class_representation(train_data, valid_data, test_data, class_list)

    for dataset, path_to_dataset_dir in zip([valid_data, test_data], [path_to_valid, path_to_test]):
        for index, row in dataset.iterrows():
            label_file = row['label_file']
            base_name = label_file.replace('.txt', '')

            shutil.move(os.path.join(path_to_train_labels, label_file), os.path.join(path_to_dataset_dir, 'labels', label_file))

            for ext in image_extensions:
                image_file = base_name + '.' + ext
                if os.path.exists(os.path.join(path_to_images, image_file)):
                    shutil.move(os.path.join(path_to_images, image_file), os.path.join(path_to_dataset_dir, 'images', image_file))
                    break

    return source_data, train_data, valid_data, test_data, class_list