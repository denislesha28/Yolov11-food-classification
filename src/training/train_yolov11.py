from ultralytics import YOLO

def main():
    model = YOLO("yolo11s.pt")
    results = model.train(data = "../preprocessing/food_dataset/dataset.yaml",
                          epochs=150,
                          plots=True,
                          imgsz=1280,
                          device="cuda",
                          batch=8,
                          lr0=0.001,
                          mosaic=1.0,
                          mixup=0.1,
                          degrees=2,
                          translate=0.02,
                          scale=0.1,
                          fliplr=0.0,
                          hsv_v=0.2,
                          hsv_h=0.0,
                          hsv_s=0.1,
                          patience=50)

    metrics = model.val(data="../preprocessing/food_dataset/dataset.yaml",
                        imgsz=1280,
                        batch=8,
                        conf=0.25,
                        iou=0.6,
                        device="cuda")

    results = model("../preprocessing/label_studio_data/images/frame_000023.jpg")
    results[0].show()

if __name__ == "__main__":
     main()