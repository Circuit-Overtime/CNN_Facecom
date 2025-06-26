import os
from PIL import Image

def resize_images_in_folder(folder_path, size=(250, 250)):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                with Image.open(file_path) as img:
                    img = img.resize(size, Image.LANCZOS)
                    img.save(file_path)
            except Exception as e:
                print(f"Could not process {file_path}: {e}")

folders = [
    'Data/Task_A/TESTING/men',
    'Data/Task_A/TESTING/women',
]

for folder in folders:
    resize_images_in_folder(folder)