import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import pyttsx3

# Загружаем модель ResNet18
model = torch.hub.load('pytorch/vision', 'resnet18', weights='ResNet18_Weights.DEFAULT')
model.eval()

# Загружаем названия классов ImageNet
with open("imagenet_classes.txt") as f:
    categories = [line.strip() for line in f.readlines()]

# Настройка преобразования изображения
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Инициализация голосового движка
engine = pyttsx3.init()
already_said = None  # Чтобы не повторял фразу

# Запускаем камеру
cap = cv2.VideoCapture(0)
print("Нажмите 'q' чтобы выйти")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Преобразуем кадр OpenCV в PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Преобразуем изображение для модели
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # Получаем предсказание
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Получаем топ-1 категорию
    top1_prob, top1_catid = torch.topk(probabilities, 1)
    class_name = categories[top1_catid]

    # Проверяем — кошка или собака
    if "cat" in class_name.lower():
        if already_said != "cat":
            print("Обнаружена кошка 🐱")
            engine.say("Это кошка")
            engine.runAndWait()
            already_said = "cat"
    elif "dog" in class_name.lower():
        if already_said != "dog":
            print("Обнаружена собака 🐶")
            engine.say("Это собака")
            engine.runAndWait()
            already_said = "dog"
    else:
        already_said = None  # Сброс, если ни кошка, ни собака

    # Показываем видео
    cv2.imshow("Обнаружение кошки/собаки", frame)

    # Выход по клавише 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
