# Yolov11_food_detection
## Step by Step guide
### The downloaded video data needs to be located in the project root in a folder called dataset and is not included in the repository!
### For convenience, the repo provides all results, fine-tuned models, plots and evaluations except the original video dataset
### Preprocessing -
 - `frame_extractor.py` handles frame extraction and baseline annotation with yolo11s
   - results are saved in a new directory under `preprocessing\label_studio_data` 
   - results are extracted in the necessary format for usage in label-studio
 - after manual labeling in label studio or the other tool of choice results should be saved under `preprocessing/food_dataset`
 - Optional Util tool - `filter_annotated_frames.py` is used to filter out images in `food_dataset` that have no corresponding labels
 - Optional Util tool - `class_label_cleaner.py` is used to refactor class numbers in `food_dataset` to start from 0 and go further up
 - `train_test_val_split.py` is then used to separate the data in train, test and val datasets

## Training
`train_yolov11.py` provides the following methods and saves all results in the root project dir under `runs\detect`:
- `train_fixed_hyperparam` fine tunes yolo11s with the given fixed hyperparams on the dataset
- `explore_best_hyperparams` explores hyper parameter optimization
- `eval_test_set` performs model evaluation on test set

## Live video inference
`video_inference.py` handles live video inference using fine-tuned model of choice
- it expects videos to be located in the project root dir under `dataset` dir
- it expects to find a model file under `runs/detect/..` under the project root dir