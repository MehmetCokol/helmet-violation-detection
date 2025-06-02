
if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"




    from ultralytics import YOLO

    model = YOLO("yolov8s.pt")

    model.train(
        data="data.yaml", # data.yaml dosyanın yolu
        epochs=50,  # eğitim turu sayısı
        imgsz=640,# resim boyutu
        batch=16,# aynı anda işlenen görüntü sayısı
        device=0,# 👈 bu satırı ekle! 0 = GPU (ilk kart)
        project="runs", # eğitim çıktıları nereye kaydedilecek
        name="helmet-detection",# sonuç klasörü ismi
        exist_ok=True,# klasör varsa üzerine yaz
        cos_lr=True # öğrenme oranını dalgalı hale getir, daha dengeli öğrenme sağlar
    )




'''

YOLOv8 modelini  veri ile eğitir.

yolov8s.pt pre-trained modelini kullanır.

data.yaml dosyasına göre eğitim verisini tanır.


'''