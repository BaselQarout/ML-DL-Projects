import os
import shutil

def separate_structured_data(split_path):
    images_dir = os.path.join(split_path, 'images')
    labels_dir = os.path.join(split_path, 'labels')
    
    if not os.path.exists(images_dir):
        print(f"Could not find images directory at {images_dir}")
        return

    os.makedirs(labels_dir, exist_ok=True)
    moved_count = 0

    # Look inside Data/train/images/ for folders like 0--Parade
    for folder_name in os.listdir(images_dir):
        subfolder_path = os.path.join(images_dir, folder_name)
        
        if os.path.isdir(subfolder_path) and not folder_name.startswith('.'):
            # Create a parallel folder inside Data/train/labels/0--Parade/
            target_label_subfolder = os.path.join(labels_dir, folder_name)
            os.makedirs(target_label_subfolder, exist_ok=True)
            
            # Move only the .txt files over
            for file_name in os.listdir(subfolder_path):
                if file_name.endswith('.txt'):
                    source_file = os.path.join(subfolder_path, file_name)
                    target_file = os.path.join(target_label_subfolder, file_name)
                    shutil.move(source_file, target_file)
                    moved_count += 1

    print(f"Successfully moved {moved_count} labels to {labels_dir}")

separate_structured_data('Data/train')
separate_structured_data('Data/val')