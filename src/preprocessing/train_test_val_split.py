import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split


def train_test_split_dataset(dataset_path, val_ratio=0.1, test_ratio=0.2, seed=42):
    """This method handles splitting dataset into train / test / val according to the defined ratios"""
    dataset_path = Path(dataset_path)
    images_path = dataset_path / "images"
    labels_path = dataset_path / "labels"

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in images_path.iterdir()
                   if f.suffix.lower() in image_extensions]

    # a directory is created for each split
    splits = ['train', 'val', 'test']
    for split in splits:
        (images_path / split).mkdir(exist_ok=True)
        (labels_path / split).mkdir(exist_ok=True)

    # train files are first split
    train_files, temp_files = train_test_split(
        image_files,
        test_size=(val_ratio + test_ratio),
        random_state=seed
    )
    # the remainder is then split into test and val
    val_files, test_files = train_test_split(
        temp_files,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=seed
    )

    file_splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    # files are then moved into new according directories
    for split_name, file_list in file_splits.items():
        for image_file in file_list:
            dest_image = images_path / split_name / image_file.name
            shutil.move(str(image_file), str(dest_image))

            label_file = labels_path / f"{image_file.stem}.txt"
            if label_file.exists():
                dest_label = labels_path / split_name / label_file.name
                shutil.move(str(label_file), str(dest_label))


if __name__ == "__main__":
    train_test_split_dataset("food_dataset")