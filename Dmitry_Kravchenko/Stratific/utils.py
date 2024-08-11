# utils.py

import os
import shutil

def clear_directory(path):
    """ Удаление всех файлов в директории. """
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path):
            os.unlink(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

def read_numbering(data_path):
    numbering_path = os.path.join(data_path, 'numbering.txt')
    if os.path.exists(numbering_path):
        with open(numbering_path, 'r') as file:
            return int(file.read().strip())
    return 1

def save_numbering(data_path, numbering):
    with open(os.path.join(data_path, 'numbering.txt'), 'w') as file:
        file.write(str(numbering))
