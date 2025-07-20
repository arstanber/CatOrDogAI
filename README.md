# 🐶🐱 CatOrDogAI

CatOrDogAI — это простое Python-приложение, которое использует камеру и модель компьютерного зрения ResNet18 из PyTorch для определения, кто перед вами: **кошка** или **собака**.

## 🚀 Возможности

- 📷 Использование веб-камеры
- 🧠 Распознавание изображений с помощью ResNet18
- 🔊 Озвучивание результатов голосом
- ⚡ Поддержка macOS и Windows (через OpenCV и pyttsx3)

## 🧠 Как работает

1. Камера захватывает изображение.
2. Модель ResNet18 (предобученная на ImageNet) делает предсказание.
3. Если предсказание относится к кошке или собаке — результат озвучивается.

## 📦 Установка

### 1. Клонируй проект:

```bash
git clone https://github.com/arstanber/CatOrDodAI.git
cd CatOrDogAI

2. Создай и активируй виртуальное окружение:

python3 -m venv venv
source venv/bin/activate  # macOS / Linux
# .\venv\Scripts\activate  # Windows

3. Установи зависимости:

pip install -r requirements.txt

Если файла requirements.txt нет, можешь установить вручную:

pip install torch torchvision opencv-python pyttsx3 pillow

4. Убедись, что у тебя есть файл imagenet_classes.txt

Если его нет — создай файл с таким содержимым:

tench
goldfish
great white shark
tiger shark
...
Persian cat
Siamese cat
Egyptian cat
tabby
tiger cat
kitten
Chihuahua
Pekinese
Shih-Tzu
Blenheim spaniel
toy terrier
...

Полный список можно взять здесь или я могу сгенерировать тебе файл.

🏁 Запуск

python3 main.py

Во время работы камера будет включена. Когда появится кошка или собака — модель скажет результат голосом. Для выхода нажми клавишу q.

📁 Структура проекта

CatOrDogAI/
├── main.py
├── imagenet_classes.txt
├── requirements.txt
└── README.md

📸 Скриншот

Добавь сюда скриншот, если хочешь

🔧 Проблемы
	•	❗ Убедись, что ты запускаешь правильную версию Python (python3, а не python)
	•	❗ Убедись, что камера работает и не занята другими приложениями
