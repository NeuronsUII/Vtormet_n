from geturlimg import *
import config
async def process_images(image_dict, target_folder):
    for linkpath, linkurl in image_dict.items():
        url = f"https://yandex.ru/images/search?source=collections&rpt=imageview&url={linkurl}"
        relative_path = os.path.relpath(linkpath, config.IMAGE_PATH)
        directory_name, file_name = os.path.split(relative_path)
        without_extension = os.path.splitext(file_name)[0]

        # Создание пути
        _path = os.path.join(target_folder, directory_name, without_extension)
        if not os.path.exists(_path):
            os.makedirs(_path)
        await download_images(url, _path, config.MAX_COUNT)
        print(url)

# ИСПОЛНИТЕЛЬНАЯ ФУНКЦИЯ
#asyncio.run(process_images(image_dict, TARGET_FOLDER))
