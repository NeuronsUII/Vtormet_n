Проект: VOP-tracing предназначается для детекции взрывоопасных предметов на вторичной переработке металлов

Примечание: Если у вас не используется поддержка GPU подключите CUDA 11.2 или 11.8 с пакетом cudnn.

установка Anaconda:

git clone https://github.com/Dezmoond/Vop-tracking.git

cd VOP conda env create -f environment.yml

conda activate VOP

python main.py

Установка без использования Anaconda:

git clone https://github.com/Dezmoond/BBtoOBB_CONVERTER.git

cd VOP

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

python main.py
