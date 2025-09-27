import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

# Veri setinin bulunduğu dizin
base_dir = 'C:/Users/kgnct/PycharmProjects/silahdetect/silahtespit/resimler'

# Veri artırma (Image Data Generator ile)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Eğitim verisi
train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Kategori isimlerini öğrenme
class_indices = train_generator.class_indices
print(class_indices)

# Doğrulama verisi
validation_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Model oluşturma
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(train_generator.class_indices), activation='softmax'))

# Modeli derle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme (gerçek eğitim için epochs=10 yaptım)
history = model.fit(train_generator, validation_data=validation_generator, epochs=50)

model.save('citarsilahtespit.keras')

# Eğitim metriklerini yazdır
print(history.history.keys())

# Grafik çizimi
plt.figure(figsize=(8, 5))
if 'accuracy' in history.history:
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
if 'val_accuracy' in history.history:
    plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Model Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()

