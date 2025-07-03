from yolov11_assignment.src.preprocessing.data_loader import FrameExtractor


def main():
    FrameExtractor().extract_frames_from_video("../../dataset/1.MOV")