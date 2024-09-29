import aiohttp
from main import *
import config
from imagekitio import ImageKit
import random
from urllib.parse import urlparse

# Client ID от Imgur
#CLIENT_ID = 'c610f40746a3ef8'
last_folder_name = os.path.basename(os.path.normpath(config.IMAGE_PATH))
parent_directory = os.path.dirname(os.path.normpath(config.IMAGE_PATH))


# Определение списка имен классов
CLASS_LIST = sorted(os.listdir(config.IMAGE_PATH))

# Определение количества классов
CLASS_COUNT = len(CLASS_LIST)

imagekit = ImageKit(
    private_key=config.private_key,
    public_key=config.public_key,  # Add this if you need the public key as well
    url_endpoint='https://ik.imagekit.io/your_imagekit_id/'
)


async def upload_image_to_imgur(session, image_path):
    headers = {
        'Authorization': f'Client-ID {config.CLIENT_ID}',
    }

    try:
        with open(image_path, 'rb') as image_file:
            files = {
                'image': image_file.read(),
            }

            async with session.post('https://api.imgur.com/3/upload', headers=headers, data=files) as response:
                if response.status == 200:
                    json_response = await response.json()
                    image_url = json_response['data']['link']
                    return image_url
                else:
                    print(f"Failed to upload image: {response.status}, {await response.text()}")
                    return None
    except Exception as e:
        print(f"Error occurred while uploading {image_path}: {e}")
        return None
async def upload_image_to_imagekit(session, image_path):
    try:
        result = imagekit.upload_file(
            file=open(image_path, 'rb'),
            file_name=os.path.basename(image_path)
        )
        image_url = result.url if result else None
        if image_url:
            return image_url
        else:
            print(f"ImageKit upload failed: {result}")
            return None
    except Exception as e:
        print(f"Error occurred while uploading {image_path} to ImageKit: {e}")
        return None

async def main(image_paths):

    async with aiohttp.ClientSession() as session:
        image_urls = []
        for image_path in image_paths:
            try:
                imgur_url = await upload_image_to_imgur(session, image_path)
                if imgur_url:
                    image_urls.append(imgur_url)
                else:
                    raise Exception("Simulating Imgur failure for testing")
            except Exception as e:
                print(f"Imgur upload failed for {image_path}. Trying ImageKit...")
                imagekit_url = await upload_image_to_imagekit(session, image_path)
                if imagekit_url:
                    image_urls.append(imagekit_url)
                else:
                    print(f"Failed to upload {image_path} to both Imgur and ImageKit.")
                    image_urls.append(None)
        return image_urls

def create_directory(path):
    if os.path.exists(path):
        suffix = 1
        new_directory_name = f"{last_folder_name}_{suffix}"
        new_directory_path = os.path.join(parent_directory, new_directory_name)

        while os.path.exists(new_directory_path):
            suffix += 1
            new_directory_name = f"{last_folder_name}_{suffix}"
            new_directory_path = os.path.join(parent_directory, new_directory_name)

        os.makedirs(new_directory_path)
        print(f"Directory '{new_directory_path}' created.")
    else:
        os.makedirs(path)
        print(f"Directory '{path}' created.")
    return new_directory_path

TARGET_FOLDER = create_directory(os.path.join(parent_directory, last_folder_name))
config.TARGET_FOLDER = TARGET_FOLDER

def copy_directory_structure(source_dir, target_dir):
    for item in os.listdir(source_dir):
        source_item_path = os.path.join(source_dir, item)

        if os.path.isdir(source_item_path):
            target_item_path = os.path.join(target_dir, item)
            if not os.path.exists(target_item_path):
                os.makedirs(target_item_path)
                print(f"Directory '{target_item_path}' created.")
            else:
                print(f"Directory '{target_item_path}' already exists.")

copy_directory_structure(config.IMAGE_PATH, TARGET_FOLDER)

# Example usage
image_paths = []
image_urls = []

for cls in CLASS_LIST:
    class_path = os.path.join(config.IMAGE_PATH, cls)
    if os.path.exists(class_path) and os.path.isdir(class_path):
        for file_name in os.listdir(class_path):
            if file_name.lower().endswith('.jpg'):
                file_path = os.path.join(class_path, file_name)
                image_paths.append(file_path)

# Run the asynchronous image upload method
image_urls = asyncio.run(main(image_paths))
image_dict = dict(zip(image_paths, image_urls))

# Output URLs of the images
for image_path, image_url in image_dict.items():
    if image_url:
        print(f"{image_path}: {image_url}")
    else:
        print(f"Failed to upload {image_path}.")
#async def process_images(image_dict, target_folder):
#    for linkpath, linkurl in image_dict.items():
#        url = f"https://yandex.ru/images/search?source=collections&rpt=imageview&url={linkurl}"
#        relative_path = os.path.relpath(linkpath, config.IMAGE_PATH)
#        directory_name, file_name = os.path.split(relative_path)
#        without_extension = os.path.splitext(file_name)[0]
#
#        # Создание пути
#        _path = os.path.join(target_folder, directory_name, without_extension)
#        if not os.path.exists(_path):
#            os.makedirs(_path)
#
#        await download_images(url, _path, config.MAX_COUNT)
#        #print(url)