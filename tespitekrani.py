from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Input(shape=(5,)))      # Input shape burada tanımlandı
model.add(Dense(10))              # Dense artık input_shape almaz
print("Başarılı!")



try:
    # Kaydedilen modeli yükle
    model = load_model('citarsilahtespit.keras')  # h5 and keras

    # Test etmek istediğin görsel yolu
    img_path = 'C:/Users\kgnct\Documents\modeldeneme/pompalideneme.jpg'

    # Görseli hazırla
    img = load_img(img_path, target_size=(150, 150
                                          ))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Tahmin yap
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    # Sınıf isimlerini al ve yazdır
    class_indices = {'bicak': 0, 'otomatiksilah': 1,'pompali': 2, 'smg': 3, 'tabanca': 4}
    index_to_class = {v: k for k, v in class_indices.items()}
    print(f'Tahmin Edilen Sınıf: {index_to_class[predicted_class]}')

except Exception as e:
    print(f"Hata oluştu: {e}")