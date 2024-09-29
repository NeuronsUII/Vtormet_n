Проект: Yandex Image Downloader
Описание
Собирает картинки похожие на те что вы указали в выбаном каталоге. Нужно указать папку проекта с подпапками в которых находятся изображения в формате JPG (программа работает только с ними). Программа сделает парсинг по "Яндекс.картинки" и соберет похожие изображения.

Установка
Требования
Python 3.12.4 или выше
Установленные зависимости в requirements.txt

Anaconda:

https://github.com/Dezmoond/Image_Parser_Yandex.git

cd Image_Parser_Yandex

conda env create -f environment.yml

conda activate parser

python Parserwindow.py


Установка без использования Anaconda:

https://github.com/Dezmoond/Image_Parser_Yandex.git

cd Image_Parser_Yandex

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

python Parserwindow.py
