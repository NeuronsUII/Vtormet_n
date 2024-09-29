import os
import cv2
import numpy as np
import asyncio
from playwright.async_api import async_playwright
import requests
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse, urljoin
import itertools
import config

# Устанавливаем nest_asyncio для разрешения вложенных циклов событий
import nest_asyncio
#nest_asyncio.apply()

def get_file_extension(content_type):
    """Определяет расширение файла по MIME типу."""
    content_type_to_ext = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/svg+xml": ".svg",
        "image/x-icon": ".ico"
    }
    return content_type_to_ext.get(content_type, ".jpg")

def enhance_image_quality(image):
    # Увеличение резкости
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

    # Уменьшение шума
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # Улучшение контрастности
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return image

def filepath_fix_existing(directory_path: str, name: str, extension: str) -> str:
    """Добавляет числовой суффикс к имени файла, если файл с таким именем уже существует."""
    base = os.path.join(directory_path, name + extension)
    if os.path.exists(base):
        for i in itertools.count(start=1):
            new_name = f'{name} ({i}){extension}'
            new_filepath = os.path.join(directory_path, new_name)
            if not os.path.exists(new_filepath):
                return new_filepath
    return base

def download_single_image(img_url: str, output_directory: str):
    try:
        if 'preview' in img_url or 'ocr' in img_url:
            print(f"Пропущено (неподходящий URL): {img_url}")
            return
        # Создаем директорию, если она не существует
        os.makedirs(output_directory, exist_ok=True)

        # Скачиваем изображение
        response = requests.get(img_url, timeout=200)
        response.raise_for_status()  # Генерирует исключение для HTTP ошибок

        # Определение типа изображения и его расширения
        content_type = response.headers.get("Content-Type", "image/jpeg")
        file_extension = get_file_extension(content_type)

        # Формируем имя файла
        img_name = urlparse(img_url).path.split('/')[-1] or 'image'
        img_path = filepath_fix_existing(output_directory, img_name, file_extension)

        # Открываем изображение с помощью PIL
        image = Image.open(BytesIO(response.content))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        old_size = image.shape[:2]  # старый размер [высота, ширина]
        desired_size = config.DESIRED_SIZE
        if config.resize_option == 0:
            ratio = float(desired_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])

            # Изменяем размер изображения
            image = cv2.resize(image, (new_size[1], new_size[0]))

            # Создаем новое изображение и заполняем его
            delta_w = desired_size - new_size[1]
            delta_h = desired_size - new_size[0]
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)

            color = [0, 0, 0]
            new_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REPLICATE, value=color)
        elif config.resize_option == 1:
            new_image = cv2.resize(image, (desired_size, desired_size))
        else: print('Значение resize_option в файле config может быть 0 или 1')

        # Повышаем качество изображения
        new_image = enhance_image_quality(new_image)

        # Сохраняем изображение с помощью PIL
        final_image = Image.fromarray(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
        final_image.save(img_path)

        print(f"Сохранено: {img_path}")

    except Exception as e:
        print(f"Ошибка при скачивании изображения {img_url}: {e}")

async def download_images(url, output_directory, max_count):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        # Открываем URL
        await page.goto(url)

        # Ждём загрузки страницы и кнопку "Похожие"
        await page.wait_for_selector('#cbir-similar-title > a')
        # Нажимаем на вкладку "Похожие"
        await page.click('#cbir-similar-title > a')
        # Ожидаем загрузки результатов
        await page.wait_for_load_state('networkidle')

        images_collected = 0

        while images_collected < max_count:
            # Прокручиваем страницу
            await page.evaluate("window.scrollBy(0, document.body.scrollHeight);")

            # Ждём, пока контент подгрузится
            await page.wait_for_timeout(3000)

            # Собираем все изображения
            images = await page.query_selector_all('img')

            # Проходимся по каждому изображению
            for image in images:
                if images_collected >= max_count:
                    break

                src = await image.get_attribute('src')
                if not src:
                    continue

                # Преобразование относительных URL в абсолютные
                image_url = urljoin(url, src)
                if image_url is None:
                    continue

                download_single_image(image_url, output_directory)

                images_collected += 1

        await browser.close()