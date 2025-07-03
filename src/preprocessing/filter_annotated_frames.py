from pathlib import Path
import os


def create_labeled_dataset():
    print('Creating labeled dataset from manually labeled data...')
    images_dir = Path("food_dataset/images")
    annotations_dir = Path("food_dataset/labels")

    label_list = list(annotations_dir.glob("*.txt"))
    labeled_frames = []
    labeled_frame_stems = set()

    for label in label_list:
        frame_name = label.stem
        frame_path = images_dir / (frame_name + '.jpg')
        if frame_path.exists():
            labeled_frames.append(frame_path)
            labeled_frame_stems.add(frame_name)

    all_images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg')) + list(images_dir.glob('*.png'))

    unlabeled_images = []
    for img_path in all_images:
        if img_path.stem not in labeled_frame_stems:
            unlabeled_images.append(img_path)

    print(f"Found {len(unlabeled_images)} unlabeled images to delete")

    if unlabeled_images:
        confirm = input(f"Are you sure you want to delete {len(unlabeled_images)} unlabeled images? (y/N): ")
        if confirm.lower() == 'y':
            deleted_count = 0
            for img_path in unlabeled_images:
                try:
                    os.remove(img_path)
                    deleted_count += 1
                except OSError as e:
                    print(f"Error deleting {img_path}: {e}")

            print(f"Successfully deleted {deleted_count} unlabeled images")
            print(
                f"Remaining images: {len(list(images_dir.glob('*')))} (should match labeled frames: {len(labeled_frames)})")
        else:
            print("Deletion cancelled")
    else:
        print("No unlabeled images found to delete")


if __name__ == "__main__":
    create_labeled_dataset()