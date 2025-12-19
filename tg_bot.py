import logging
import cv2
import numpy as np
from PIL import Image
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# --- Конфигурация (ДОЛЖНА совпадать с обучением!) ---
MODEL_PATH = 'best_model.keras'
IMG_SIZE = (128, 128)  # Используйте (224, 224), если обучали на таком размере
model = load_model(MODEL_PATH)

# --- Инициализация детектора лиц ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Функция предсказания ---
def predict_mask(image):
    # Изменяем размер и подготавливаем изображение
    image = image.resize(IMG_SIZE)
    image_array = img_to_array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Делаем предсказание
    prediction = model.predict(image_array, verbose=0)
    class_idx = np.argmax(prediction[0])
    confidence = np.max(prediction[0])

    # Возвращаем результат
    labels = ['С маской', 'Без маски']
    return labels[class_idx], confidence

# --- Обработчики для Telegram ---
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text('Привет! Отправь мне фото лица, и я определю, есть ли на нём маска.')

async def handle_photo(update: Update, context: CallbackContext):
    try:
        # Скачиваем фото
        photo_file = await update.message.photo[-1].get_file()
        await photo_file.download_to_drive('user_photo.jpg')

        # Открываем изображение
        image = Image.open('user_photo.jpg').convert('RGB')
        
        # Конвертируем PIL в OpenCV формат
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
        # Детектируем лица
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        
        if len(faces) == 0:
            await update.message.reply_text("❌ Лицо не обнаружено. Отправьте более чёткое фото лица.")
            return
        
        # Берём первое найденное лицо (самое большое)
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)  # Сортируем по площади
        x, y, w, h = faces[0]
        
        # Вырезаем область лица с небольшим запасом
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image_cv.shape[1], x + w + padding)
        y2 = min(image_cv.shape[0], y + h + padding)
        
        # Конвертируем обратно в PIL Image
        face_roi = image.crop((x1, y1, x2, y2))
        
        # Определяем маску
        label, confidence = predict_mask(face_roi)
        
        # Определяем порог уверенности
        if confidence < 0.6:
            confidence_note = " (низкая уверенность)"
        elif confidence < 0.8:
            confidence_note = " (средняя уверенность)"
        else:
            confidence_note = " (высокая уверенность)"
        
        # Отправляем ответ
        response = f"📊 Результат: {label}{confidence_note}\n✅ Уверенность: {confidence:.2%}\n👤 Обнаружено лиц: {len(faces)}"
        
        # Опционально: отправляем обработанное фото с выделенным лицом
        if len(faces) <= 3:  # Если лиц не слишком много
            # Рисуем прямоугольники вокруг лиц
            for (fx, fy, fw, fh) in faces[:3]:  # Максимум 3 лица
                cv2.rectangle(image_cv, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)
            
            # Сохраняем обработанное фото
            processed_path = 'user_photo_processed.jpg'
            cv2.imwrite(processed_path, image_cv)
            
            # Отправляем фото с выделенными лицами
            with open(processed_path, 'rb') as photo:
                await update.message.reply_photo(photo, caption=response)
            
            # Удаляем временные файлы
            os.remove(processed_path)
        else:
            await update.message.reply_text(response)
            
    except Exception as e:
        logging.error(f"Ошибка обработки фото: {e}")
        await update.message.reply_text("⚠️ Произошла ошибка при обработке фото. Попробуйте другое изображение.")
    
    finally:
        # Удаляем временный файл
        if os.path.exists('user_photo.jpg'):
            os.remove('user_photo.jpg')

# --- Главная функция ---
def main():
    # ВАЖНО: Замените на ваш новый токен!
    TOKEN = "8230459480:AAHP99YpYbFRJ3IkTyImD1x8_i0_GKpvmwc"

    # Настройка логгирования
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    # Создание и запуск бота
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    application.run_polling()

if __name__ == '__main__':
    main()