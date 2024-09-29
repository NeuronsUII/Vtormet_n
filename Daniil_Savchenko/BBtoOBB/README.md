Проект: BBtoOBB_Converter Описание: переводит координаты из формата Bounding Box в Oriented Bounding Box путем сегментации изображения. Боксы поворачиваются по направлению обьекта
Установка Требования Python 3.12.4 или выше Установленные зависимости в requirements.txt

Anaconda:

https://github.com/Dezmoond/BBtoOBB_CONVERTER.git

cd OBBCONVERER
conda env create -f environment.yml

conda activate OBBCONVERER

python main.py

Установка без использования Anaconda:

https://github.com/Dezmoond/BBtoOBB_CONVERTER.git

cd OBBCONVERER

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

python main.py
