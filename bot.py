import os
import subprocess
import json
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.types import InputFile
from aiogram.utils import executor

# Задаем токен и инициализируем бота и диспетчер
API_TOKEN = os.getenv('DICE_BOT_TOKEN')
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# Настраиваем логирование
logging.basicConfig(level=logging.INFO)

def gen_result_path(output_path, prefix):
    i = 0
    while True:
        path = os.path.join(output_path, f"{prefix}_{i}")
        if not os.path.exists(path):
            return os.path.join(output_path, f"{prefix}_{i - 1}")
        i += 1

# Обработчик команды /start
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply("Отправьте мне картинку, и я обработаю её!")

# Обработчик для получения изображений
@dp.message_handler(content_types=['photo'])
async def handle_docs_photo(message: types.Message):
    # Сохранение изображения
    photo = message.photo[-1]
    img_path = 'img.jpg'
    await photo.download(img_path)

    # Запуск скрипта "./dice.py img.jpg" с помощью subprocess
    subprocess.run(["./dice.py", img_path])

    # Генерация пути для сохранения результата
    result_path = gen_result_path("res", "parse")
    print(result_path)

    # Ищем JSON и IMG файлы в сгенерированной директории
    result_json = None
    result_img = None

    for file in os.listdir(result_path):
        if file.endswith('.json'):
            result_json = os.path.join(result_path, file)
        elif file.endswith('.jpg') or file.endswith('.png'):
            result_img = os.path.join(result_path, file)

    # Отправка результатов пользователю
    if result_json and result_img:
        # Отправляем JSON файл как файл
        json_file = InputFile(result_json)
        await message.reply_document(json_file)

        # Отправляем изображение
        photo = InputFile(result_img)
        await message.reply_photo(photo)
    else:
        await message.reply("Ошибка: результаты не найдены.")

# Запуск бота
if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
