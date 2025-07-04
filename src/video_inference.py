from ultralytics import YOLO
import cv2


def main():
    model = YOLO("../runs/detect/train15/weights/best.pt")
    video_path = "../dataset/3_1.MOV"
    cap = cv2.VideoCapture(video_path)

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale_factor = 0.5
    display_width = int(original_width * scale_factor)
    display_height = int(original_height * scale_factor)

    print(f"Original: {original_width}x{original_height}")
    print(f"Display: {display_width}x{display_height}")

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model(frame, conf=0.1, iou=0.6, imgsz=1280)
            annotated_frame = results[0].plot()
            resized_frame = cv2.resize(annotated_frame, (display_width, display_height))

            cv2.imshow("Food Detection", resized_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()