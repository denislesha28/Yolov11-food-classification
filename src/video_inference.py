from ultralytics import YOLO
import cv2
import os
from datetime import datetime


def main():
    model = YOLO("../runs/detect/train/weights/best.pt")
    video_path = "../dataset/2_1.MOV"
    cap = cv2.VideoCapture(video_path)

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    scale_factor = 0.5
    display_width = int(original_width * scale_factor)
    display_height = int(original_height * scale_factor)

    print(f"Original: {original_width}x{original_height}")
    print(f"Display: {display_width}x{display_height}")

    output_dir = "../runs/detect/inference"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"food_detection_{timestamp}.mp4")

    # Video writer with compression (H.264 codec, smaller file size)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (display_width, display_height))

    print(f"Saving video to: {output_path}")

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model(frame, imgsz=1280, conf=0.12, iou=0.4, max_det=300, augment=True)
            annotated_frame = results[0].plot()
            resized_frame = cv2.resize(annotated_frame, (display_width, display_height))

            # Save frame to video file
            out.write(resized_frame)

            cv2.imshow("Food Detection", resized_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Video saved successfully: {output_path}")

    # Print out file size for verification
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Output file size: {file_size_mb:.2f} MB")


if __name__ == '__main__':
    main()