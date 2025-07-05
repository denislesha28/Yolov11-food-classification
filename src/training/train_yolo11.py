from ultralytics import YOLO


def train_fixed_hyperparam():
    model = YOLO("yolo11s.pt")
    model.train(data="../preprocessing/food_dataset/dataset.yaml",
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

    model.val(data="../preprocessing/food_dataset/dataset.yaml",
                        imgsz=1280,
                        batch=8,
                        conf=0.25,
                        iou=0.6,
                        device="cuda")

    results = model("../preprocessing/food_dataset/images/test/frame_000217.jpg")
    results[0].show()


def explore_best_hyperparams():
    search_space = {
        'lr0': (0.0007, 0.0015),
        'lrf': (0.01, 0.03),
        'momentum': (0.92, 0.95),
        'weight_decay': (0.0003, 0.0007),
        'warmup_epochs': (2, 4),
        'box': (7.0, 8.0),
        'cls': (0.45, 0.55),
        'dfl': (1.4, 1.6),
        'mixup': (0.08, 0.12),
        'degrees': (1, 3),
        'translate': (0.015, 0.025),
        'scale': (0.08, 0.12),
        'hsv_v': (0.15, 0.25),
        'hsv_s': (0.08, 0.12),
        'mosaic': (1.0, 1.0),
        'fliplr': (0.0, 0.0),
        'hsv_h': (0.0, 0.0),
    }
    model = YOLO("yolo11s.pt")
    model.tune(
        space=search_space,
        data="../preprocessing/food_dataset/dataset.yaml",
        epochs=150,
        iterations=30,
        optimizer="AdamW",
        plots=True,
        save=True,
        val=True,
        imgsz=1280,
        batch=8,
        patience=25
    )


def eval_test_set():
    model = YOLO("../../runs/detect/train/weights/best.pt")
    model.val(
        data="../preprocessing/food_dataset/dataset.yaml",
        split="test",
        batch=8,
        conf=0.25,
        iou=0.6,
        device="cuda",
        plots=True
    )


if __name__ == "__main__":
    train_fixed_hyperparam()
    explore_best_hyperparams()
    eval_test_set()