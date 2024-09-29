import os
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox


def denormalize_bbox(cx, cy, rw, rh, img_width, img_height):
    """Денормализует координаты Bounding Box (BB)."""
    x_center = cx * img_width
    y_center = cy * img_height
    box_width = rw * img_width
    box_height = rh * img_height

    x_min = int(x_center - box_width / 2)
    y_min = int(y_center - box_height / 2)
    x_max = int(x_center + box_width / 2)
    y_max = int(y_center + box_height / 2)
    return [x_min, y_min, x_max, y_max]


def denormalize_obb(coords, img_width, img_height):
    """Денормализует координаты Oriented Bounding Box (OBB)."""
    return [c * img_width if i % 2 == 0 else c * img_height for i, c in enumerate(coords)]


def draw_annotations(draw, annotations, color, width, height):
    """Отрисовывает аннотации на изображении."""
    for annotation in annotations:
        if len(annotation) == 0:  # Игнорируем пустые аннотации
            continue
        class_id = int(annotation[0])
        if len(annotation) == 5:  # Bounding Box (BB)
            x_min, y_min, x_max, y_max = denormalize_bbox(annotation[1], annotation[2], annotation[3], annotation[4],
                                                          width, height)
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)
            draw.text((x_min, y_min - -30), f'Class {class_id} BB', fill=color, font=None, font_size=14)
        elif len(annotation) == 9:  # Oriented Bounding Box (OBB)
            points = np.array(denormalize_obb(annotation[1:], width, height), dtype=np.int32).reshape((-1, 2))
            draw.polygon([tuple(p) for p in points], outline=color, width=2)
            draw.text((points[0, 0], points[0, 1] - -10), f'Class {class_id} OBB', fill=color, font=None, font_size=14)


def select_folder(prompt):
    """Выбирает одну папку."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    folder_path = filedialog.askdirectory(title=prompt)
    root.destroy()
    return folder_path


def select_multiple_folders(prompt):
    """Выбирает несколько папок."""
    folders = []
    while True:
        folder = select_folder(prompt)
        if folder:
            folders.append(folder)
            add_more = messagebox.askyesno("Добавить еще?", "Добавить следующую папку аннотаций для сравнения?")
            if not add_more:
                break
        else:
            break
    return folders


def parse_annotation(file_path):
    """Читает аннотации из файла и возвращает их в виде списка."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    annotations = []
    for line in lines:
        coords = list(map(float, line.strip().split()))
        annotations.append(coords)
    return annotations


def compare_annotations(ann1, ann2):
    """Сравнивает две аннотации."""
    if len(ann1) != len(ann2):
        return False
    for a1, a2 in zip(ann1, ann2):
        if len(a1) != len(a2) or a1 != a2:
            return False
    return True


class AnnotationViewer:
    def __init__(self, master, image_folder, annotation_folders):
        self.master = master
        self.image_folder = image_folder
        self.annotation_folders = annotation_folders
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith((".jpg", ".png"))]
        self.index = 0
        self.current_image = None
        self.annotations_dict = {}
        self.valid_images = []

        # Create UI elements
        self.canvas = tk.Canvas(master, width=800, height=600)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.title_label = tk.Label(master, text="", font=("Arial", 14))
        self.title_label.pack()

        self.info_label = tk.Label(master, text="Выберите, какую аннотацию сохранить", font=("Arial", 14))
        self.info_label.pack(side=tk.TOP)

        self.legend_frame = tk.Frame(master)
        self.legend_frame.pack(side=tk.TOP, fill=tk.X)



        self.image_counter_label = tk.Label(master, text="1 / 1", font=("Arial", 14))
        self.image_counter_label.pack(side=tk.TOP)

        self.prev_button = tk.Button(master, text="Назад", command=self.previous_image)
        self.prev_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.next_button = tk.Button(master, text="Далее", command=self.next_image)
        self.next_button.pack(side=tk.RIGHT, padx=5, pady=5)

        self.load_image_and_annotations()

    def update_legend(self):
        """Обновляет легенду с цветами аннотаций."""
        for widget in self.legend_frame.winfo_children():
            widget.destroy()  # Clear previous legend items

        for folder, (_, color) in self.annotations_dict.items():
            color_hex = '#%02x%02x%02x' % color
            label = tk.Label(self.legend_frame, text=os.path.basename(folder), bg=color_hex, fg='white', padx=10,
                             pady=5, font=("Arial", 14))
            label.bind("<Button-1>", lambda e, f=folder: self.save_annotation(f))
            label.pack(side=tk.LEFT, padx=5)

    def save_annotation(self, folder):
        """Сохраняет аннотацию в выбранную папку."""
        save_folder = os.path.join(os.path.dirname(self.image_folder),
                                   os.path.basename(self.image_folder) + "_SELECTANNOTATION")
        os.makedirs(save_folder, exist_ok=True)
        image_file = self.valid_images[self.index][0]
        annotation_file = image_file.replace(".jpg", ".txt").replace(".png", ".txt")
        annotations = self.annotations_dict.get(folder, ([], (255, 255, 255)))[0]

        if annotations:
            save_path = os.path.join(save_folder, annotation_file)
            with open(save_path, 'w') as file:
                for ann in annotations:
                    file.write(' '.join(map(str, ann)) + '\n')
            messagebox.showinfo("Сохранено", f"Аннотации сохранены в {save_path}")


        self.next_image()

    def load_image_and_annotations(self):
        """Загружает изображения и аннотации, если аннотации различаются."""
        self.valid_images = []

        for image_file in self.image_files:
            annotation_dict = {}
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

            for i, folder in enumerate(self.annotation_folders):
                annotation_path = os.path.join(folder, image_file.replace(".jpg", ".txt").replace(".png", ".txt"))
                if os.path.exists(annotation_path):
                    annotations = parse_annotation(annotation_path)
                    if annotations:  # Проверяем, что аннотации не пустые
                        annotation_dict[folder] = (annotations, colors[i])

            if len(annotation_dict) > 1:
                all_annotations = list(annotation_dict.values())
                base_annotations = all_annotations[0][0]
                annotations_different = False

                for annotations, _ in all_annotations[1:]:
                    if not compare_annotations(base_annotations, annotations):
                        annotations_different = True
                        break

                if annotations_different:
                    self.valid_images.append((image_file, annotation_dict))

        if self.valid_images:
            self.index = 0
            self.display_image()
        else:
            self.title_label.config(text="Нет изображений с различиями в аннотациях")
            self.canvas.delete("all")
            self.image_counter_label.config(text="0 / 0")

    def display_image(self):
        """Отображает текущее изображение и аннотации."""
        if not self.valid_images:
            return

        image_file, annotation_dict = self.valid_images[self.index]
        image_path = os.path.join(self.image_folder, image_file)
        image = Image.open(image_path)
        width, height = image.size
        self.current_image = ImageTk.PhotoImage(image)

        # Clear previous content
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image)

        # Draw annotations
        draw = ImageDraw.Draw(image)
        for folder, (annotations, color) in annotation_dict.items():
            draw_annotations(draw, annotations, color, width, height)

        # Display the updated image with annotations
        self.current_image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image)

        # Update title, legend, and image counter
        self.annotations_dict = annotation_dict  # Update the annotations dictionary
        self.title_label.config(text=os.path.basename(image_path))
        self.update_legend()
        self.image_counter_label.config(text=f"{self.index + 1} / {len(self.valid_images)}")

    def previous_image(self):
        """Переходит к предыдущему изображению."""
        if self.index > 0:
            self.index -= 1
            self.display_image()

    def next_image(self):
        """Переходит к следующему изображению."""
        if self.index < len(self.valid_images) - 1:
            self.index += 1
            self.display_image()


def main_viewer():
    # Вместо создания нового окна, используйте существующий root, если он передан
    root = tk.Toplevel()  # Создаем новое окно Toplevel внутри существующего Tkinter приложения
    root.title("Сравнение аннотаций")

    # Здесь вызываем всю вашу логику, которая отображает Viewer в этом новом окне.
    image_folder = select_folder("Выберите папку с изображениями")
    annotation_folders = select_multiple_folders("Выберите папки с аннотациями")

    AnnotationViewer(root, image_folder, annotation_folders)

