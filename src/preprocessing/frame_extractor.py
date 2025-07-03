import os
import subprocess
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO


class FrameExtractor:
    def __init__(self, video_path: str, output_dir="label_studio_data"):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / "images"
        self.annotations_dir = self.output_dir / "labels"
        self.frame_counter = 0

        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)

    def extract_frames(self):
        if len(os.listdir(self.frames_dir)) == 0:
            for path in Path(self.video_path).rglob("*.MOV"):
                self.extract_frames_from_video(path)
        self.batch_yolo_inference()
        self.generate_label_studio_files()

    def batch_yolo_inference(self, model_name='yolo11s.pt', batch_size=16, confidence_threshold=0.3):
        print("Frame Labeling...")

        model = YOLO(model_name)
        frame_files = sorted(list(self.frames_dir.glob("*.jpg")))
        self.class_names = list(model.names.values())

        for i in tqdm(range(0, len(frame_files), batch_size)):
            batch_files = frame_files[i:i + batch_size]
            batch_paths = [str(f) for f in batch_files]

            results = model(batch_paths, conf=confidence_threshold, verbose=False)

            for j, result in enumerate(results):
                frame_name = batch_files[j].stem
                self._save_yolo_annotation(result, frame_name)

        print(f"Saved annotations to {self.annotations_dir}")
        self._save_classes_file()

    def _save_yolo_annotation(self, result, frame_name):
        annotation_file = self.annotations_dir / f"{frame_name}.txt"

        if result.boxes is not None:
            boxes = result.boxes.xywhn.cpu()
            classes = result.boxes.cls.cpu()

            with open(annotation_file, 'w') as f:
                for box, cls in zip(boxes, classes):
                    x_center, y_center, width, height = box
                    f.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        else:
            annotation_file.touch()

    def _save_classes_file(self):
        classes_file = self.output_dir / "classes.txt"
        with open(classes_file, 'w') as f:
            for class_name in self.class_names:
                f.write(f"{class_name}\n")

    def extract_frames_from_video(self, path, frame_interval=30, max_frames=None):
        print("Extracting frames...")

        cap = cv2.VideoCapture(str(path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"Video info: {total_frames} frames, {fps} FPS")
        print(f"Extracting every {frame_interval}th frame")
        print(f"Output directory: {self.frames_dir}")

        frame_count = 0
        saved_count = 0

        with tqdm(total=total_frames // frame_interval) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    frame_filename = f"frame_{self.frame_counter:06d}.jpg"
                    frame_path = self.frames_dir / frame_filename

                    if saved_count < 3:
                        print(f"Saving frame to: {frame_path}")

                    success = cv2.imwrite(str(frame_path), frame)
                    if not success:
                        print(f"Failed to save frame {saved_count}")

                    self.frame_counter += 1
                    saved_count += 1
                    pbar.update(1)

                    if max_frames and saved_count >= max_frames:
                        break

                frame_count += 1

        cap.release()
        print(f"Extracted {saved_count} frames to {self.frames_dir}")
        return saved_count

    def generate_label_studio_files(self):
        try:
            output_json = self.output_dir / "import.json"
            cmd = [
                "label-studio-converter", "import", "yolo",
                "-i", str(self.output_dir),
                "-o", str(output_json),
                "--image-root-url", "/data/local-files/?d=images"
            ]
            subprocess.run(cmd, check=True)
            print(f"Label Studio import file created: {output_json}")
        except:
            print("Install label-studio-converter for automatic conversion")


def main():
    FrameExtractor("../../dataset").extract_frames()


if __name__ == '__main__':
    main()