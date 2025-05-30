# Testcase generator

Приложение для генерации шагов тест-кейса по краткому текстовому описанию его сути. 

## Структура проекта

 - [dataset_preprocessing.ipynb](./dataset_preprocessing.ipynb) - предобработка датасета
 - [gpt2_train.ipynb](./gpt2_train.ipynb) - дообучение GPT-2
 - [t5_train.ipynb](./t5_train.ipynb) - дообучение T5
 - [bart_train.ipynb](./bart_train.ipynb) - дообучение BART
 - [llama3.2_train.ipynb](./llama3.2_train.ipynb) - дообучение Llama 3.2
 - [metrics.ipynb](./metrics.ipynb) - расчет метрик
 - [app.py](./app.py) - приложение с веб-интерфейсом для генерации шагов тест-кейса

Ссылка на предобработанный датасет:
https://drive.google.com/file/d/14cNUtYwYOkg6RJY4CgrhQWBINYezOpHW/view?usp=sharing  
Содержимое архива необходимо извлечь в корневую папку проекта

Ссылка на независимый датасет:  
https://drive.google.com/file/d/17gSt2wuDgTiH46n0DQDym-Z9Cuix4OCO/view?usp=sharing  
Содержимое архива необходимо извлечь в корневую папку проекта

Ссылка на дообученную модель GPT-2:  
https://drive.google.com/file/d/1K4i2zcaN92NyuH7r_FNHr60iBS1bw-7B/view?usp=sharing  
Содержимое архива необходимо извлечь в папку models

Ссылка на дообученную модель T5:  
https://drive.google.com/file/d/19ZlfJKqzj1lA6RdFut7sooTa4_yACWPO/view?usp=sharing  
Содержимое архива необходимо извлечь в папку models

Ссылка на дообученную модель BART:  
https://drive.google.com/file/d/1GxmEvIsfL6nXtQXPbpb-YM24au7ZMwf1/view?usp=sharing  
Содержимое архива необходимо извлечь в папку models

Ссылка на дообученную модель Llama 3.2:  
https://drive.google.com/file/d/1Jxcms0fUlkm6Cl5v1jjmriirS279GoQt/view?usp=sharing  
Содержимое архива необходимо извлечь в папку models


## Запуск приложения:
![Screen-374](https://github.com/user-attachments/assets/0984ee84-4efb-4dc9-954b-01029c5a57eb)

После загрузки исходных файлов в директории проекта:
1) Скачиваем итоговую модель Llama 3.2  
2) Распаковываем в каталог ./models  
Итоговый путь к файлам модели должен выглядеть ./models/llama3.2-testcase
3) Подготавливаем среду 
````bash
# Создание виртуального окружения
python -m venv env

# Активация виртуального окружения в Linux
source env/bin/activate

# Активация виртуального окружения в Windows
env\Scripts\activate.bat

# Устанавливаем библиотеки
pip install -r requirements.txt
````

4) Устанавливаем PyTorch.

Комманду для установки на Вашу ОС и аппаратную конфигурацию можно уточнить на сайте PyTorch
https://pytorch.org/get-started/locally/

5) Переходим в каталог приложения

6) Запускаем streamlit сервер реализующий web-интерфейс 

````bash
streamlit run app.py
````
