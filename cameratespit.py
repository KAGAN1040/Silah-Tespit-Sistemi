import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 1. Modeli yükle
model = load_model('citarsilahtespit.keras')

# 2. Sınıf isimleri (train_generator.class_indices ile aynı olmalı)
class_names = ['bicak', 'otomatiksilah', 'pompali', 'smg', 'tabanca']
5
# 3. Kamerayı aç
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 4. Görüntüyü modele uygun şekilde hazırla (150x150, normalize)
    img = cv2.resize(frame, (150, 150))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # modelin beklediği format

    # 5. Tahmin yap
    preds = model.predict(img)
    class_idx = np.argmax(preds[0])
    confidence = preds[0][class_idx]

    # 6. Sonucu görüntüye yaz
    text = f"{class_names[class_idx]}: {confidence:.2f}"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # 7. Görüntüyü göster
    cv2.imshow('Canli Silah Tespiti', frame)

    # 8. 'q' tuşuna basınca çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
