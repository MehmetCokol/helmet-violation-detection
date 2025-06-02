from ultralytics import YOLO

# Eğitilen modeli yükle
model = YOLO("runs/helmet-detection/weights/best.pt")

# Test için bir görüntü kullan (örneğin kendi çektiğin bir motorlu görüntüsü)
results = model("frames/034_frame_0175.jpg", save=True, conf=0.5)

# Tahmin sonucu şu klasöre kaydedilir:
# runs/detect/predict/ornek.jpg


'''
Eğitilmiş modeli hızlıca test etmek için kullanılır.

Genelde tek bir görsel ya da kısa video üzerinde inference yapılır.
'''