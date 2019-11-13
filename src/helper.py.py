import os

PATH_TO_IMAGES = "..\images\"
EXTENSION = ".jpg"
TESTING = "test"
TRAINING = "train"

def extract_image_filenames():
    all_files = os.listdir(PATH_TO_IMAGES)
    return [filename for filename in all_files if filename.endswith(EXTENSION) != -1]

def extract_training_image_filenames():
    all_files = extract_image_filenames()
    return [filename for filename in all_files if filename.find(TRAINING) != -1]

def extract_testing_image_filenames():
    all_files = extract_image_filenames()
    return [filename for filename in all_files if filename.find(TESTING) != -1]
