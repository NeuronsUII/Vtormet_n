import matplotlib.pyplot as plt
import pandas as pd
import os

def visualize_train_valid_distribution(source_data, train_data, validation_data, test_data, path_to_dataset, class_list):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

    # Точечная диаграмма для распределения файлов
    axes[0, 0].scatter(range(len(train_data)), train_data['random_id'], alpha=0.5, label='Train')
    axes[0, 0].scatter(range(len(validation_data)), validation_data['random_id'], alpha=0.5, label='Valid')
    axes[0, 0].scatter(range(len(test_data)), test_data['random_id'], alpha=0.5, label='Test')
    axes[0, 0].set_title('Точечная диаграмма номеров файлов')
    axes[0, 0].set_xlabel('Индекс файла')
    axes[0, 0].set_ylabel('Номер файла')
    axes[0, 0].legend(loc='lower right', bbox_to_anchor=(1, -0.26), fancybox=True, shadow=True, ncol=1)

    source_class_counts = source_data[class_list].sum()
    train_class_counts = train_data[class_list].sum()
    vaild_class_counts = validation_data[class_list].sum()
    test_class_counts = test_data[class_list].sum()
    labels = class_list
    train_counts = train_class_counts.values
    vaild_counts = vaild_class_counts.values
    test_counts = test_class_counts.values
    x = range(len(labels))

    offset = 0.2
    x_test = [xi + offset for xi in x]
    axes[1, 0].bar(x, train_counts, width=0.4, label='Train', align='center')
    axes[1, 0].bar(x, vaild_counts, width=0.4, label='Valid', align='edge')
    axes[1, 0].bar(x_test, test_counts, width=0.4, label='Test', align='edge')

    axes[1, 0].set_title('Распределение вхождений в классы на наборах данных')
    axes[1, 0].set_xlabel('Классы')
    axes[1, 0].set_ylabel('Количество вхождений')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation='vertical')
    axes[1, 0].legend(loc='lower right', bbox_to_anchor=(1, -0.26), fancybox=True, shadow=True, ncol=1)

    def plot_class_distribution(ax, data, title):
        class_counts = data[class_list].sum()
        ax.barh(class_counts.index, class_counts.values, align='center')
        ax.set_yticks(class_counts.index)
        ax.set_yticklabels(class_counts.index)
        ax.invert_yaxis()
        ax.set_xlabel('Количество вхождений')
        ax.set_title(title)

    plot_class_distribution(axes[0, 1], validation_data, 'Количество вхождений классов в валидационных данных')
    plot_class_distribution(axes[1, 1], test_data, 'Количество вхождений классов в тестовых данных')

    fig.suptitle('Сводная визуализация датасета - Распределение данных по выборкам', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    graph_path = os.path.join(path_to_dataset, 'combined_distribution_plots.png')
    plt.savefig(graph_path, dpi=300)
    plt.show()
    plt.close()

# Пример вызова функции
# class_list = source_data.columns[2:].tolist()  # Пример получения списка классов
# visualize_train_valid_distribution(source_data, train_data, validation_data, test_data, path_to_dataset, class_list)
