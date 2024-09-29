import tkinter as tk
from tkinter import filedialog, messagebox

import config
import asyncio

class ImageDownloaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Yandex Image Downloader")

        # Путь к папке с изображениями
        self.image_path = tk.StringVar(value=config.IMAGE_PATH)
        self.max_count = tk.IntVar(value=config.MAX_COUNT)
        self.desired_size = tk.IntVar(value=config.DESIRED_SIZE)
        self.resize_option = tk.IntVar(value=config.resize_option)
        # Создание интерфейса
        self.create_widgets()

    def create_widgets(self):
        # Путь к папке с изображениями
        tk.Label(self.root, text="Путь к папке с изображениями:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
        tk.Entry(self.root, textvariable=self.image_path, width=50).grid(row=0, column=1, padx=10, pady=5)
        tk.Button(self.root, text="Выбрать папку", command=self.select_image_path).grid(row=0, column=2, padx=10,
                                                                                        pady=5)

        # Максимальное количество изображений
        tk.Label(self.root, text="Максимальное количество изображений:").grid(row=1, column=0, padx=10, pady=5,
                                                                              sticky="e")
        tk.Entry(self.root, textvariable=self.max_count, width=10).grid(row=1, column=1, padx=10, pady=5, sticky="w")

        # Размер выходных изображений
        tk.Label(self.root, text="Размер выходных изображений:").grid(row=2, column=0, padx=10, pady=5, sticky="e")
        tk.Entry(self.root, textvariable=self.desired_size, width=10).grid(row=2, column=1, padx=10, pady=5, sticky="w")

        # Переключатель режима обработки изображений
        tk.Label(self.root, text="Режим обработки изображений:").grid(row=3, column=0, padx=10, pady=5, sticky="e")
        tk.Radiobutton(self.root, text="Дорисовывать края", variable=self.resize_option, value=0).grid(row=3, column=1,
                                                                                                       padx=10, pady=5,
                                                                                                       sticky="w")
        tk.Radiobutton(self.root, text="Растягивать изображение", variable=self.resize_option, value=1).grid(row=3,
                                                                                                             column=1,
                                                                                                             padx=10,
                                                                                                             pady=5,
                                                                                                             sticky="e")

        # Кнопка запуска
        tk.Button(self.root, text="Начать загрузку", command=self.start_download).grid(row=4, column=0, columnspan=3,
                                                                                       padx=10, pady=20)

        # Кнопка закрытия
        tk.Button(self.root, text="Закрыть", command=self.root.quit).grid(row=5, column=0, columnspan=3, padx=10,
                                                                          pady=10)

    def select_image_path(self):
        path = filedialog.askdirectory(title="Выберите папку с изображениями")
        if path:
            self.image_path.set(path)
        return path
    def start_download(self):
        # Здесь будет ваша логика загрузки изображений
        image_path  = self.image_path.get()
        config.IMAGE_PATH = image_path
        max_count  = self.max_count.get()
        config.MAX_COUNT = max_count
        desired_size  = self.desired_size.get()
        config.DESIRED_SIZE = desired_size
        config.resize_option = self.resize_option.get()
        import parser
        asyncio.run(parser.process_images(parser.image_dict, parser.TARGET_FOLDER))


  #      def count_files_in_directory(directory_path):
  #          file_count = 0
  #          for root, dirs, files in os.walk(directory_path):
  #              file_count += len(files)
  #          return file_count   Кол-во файлов: {count_files_in_directory(parser.TARGET_FOLDER)}
        # Пример сообщения
        messagebox.showinfo("Информация", f"Загрузка завершена\nПуть: {parser.TARGET_FOLDER}\n")

# Основная функция запуска Tkinter
def main():
    root = tk.Tk()
    app = ImageDownloaderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
