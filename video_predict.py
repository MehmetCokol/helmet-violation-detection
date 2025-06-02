
import cv2
from ultralytics import YOLO
import os

# Model yükleniyor
model = YOLO('runs/helmet-detection/weights/best.pt')  # model yolu

# Video dosyasını aç
video_path = 'videos/010.mp4'  # giriş videosu
output_filename = os.path.basename(video_path)
output_path = f"results/output_{output_filename}"
cap = cv2.VideoCapture(video_path)

# Çıkış videosunu ayarla
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # YOLOv8 ile tespit yap
    results = model(frame)

    # Bounding box'ları çiz
    annotated_frame = results[0].plot()

    # Çıkış videosuna yaz
    out.write(annotated_frame)

   

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Inference complete. Saved as {output_path}")
