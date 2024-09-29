import os
import shutil
from cx_Freeze import setup, Executable

# Установка base для скрытия командной строки (только для Windows)
base = 'Win32GUI' if os.name == 'nt' else None

# Настройка файлов браузера Playwright
def copy_playwright_browsers():
    try:
        # Определение пути к кэшу Playwright
        src = os.path.join(os.environ['USERPROFILE'], '.cache', 'ms-playwright')
        dest = os.path.join('build', 'exe.win-amd64-3.12', 'lib', 'playwright', 'driver', 'package')

        # Проверка наличия исходного каталога
        if not os.path.exists(src):
            print(f"Source path {src} does not exist. Please install Playwright browsers using 'playwright install'.")
            return

        # Создание конечного каталога, если он не существует
        os.makedirs(dest, exist_ok=True)

        # Копирование браузеров Playwright
        shutil.copytree(src, dest, dirs_exist_ok=True)
        print(f"Copied Playwright browsers from {src} to {dest}.")
    except Exception as e:
        print(f"An error occurred while copying Playwright browsers: {e}")

copy_playwright_browsers()

# Объект Executable
target = Executable(
    script="Parserwindow.py",
    base=base
)

# Конфигурация setup
setup(
    name="YourAppName",
    version="1.0",
    description="Your app description",
    executables=[target]
)