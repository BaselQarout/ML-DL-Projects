import os
import cv2

def convert_wider_structured(label_path, images_root):
    if not os.path.exists(label_path):
        print(f"Annotation file not found: {label_path}")
        return

    with open(label_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    i = 0
    total_lines = len(lines)
    match_count = 0
    
    while i < total_lines:
        line = lines[i]
        
        if line.endswith('.jpg'):
            # line is exactly '0--Parade/0_Parade_marchingband_1_5.jpg'
            img_rel_path = line
            img_full_path = os.path.join(images_root, img_rel_path)
            
            if os.path.exists(img_full_path):
                match_count += 1
                
                # Create the .txt file path at 'Data/train/images/0--Parade/0_Parade_marchingband_1_5.txt'
                label_output_path = os.path.splitext(img_full_path)[0] + '.txt'
                
                i += 1
                if i >= total_lines:
                    break
                
                next_line = lines[i]
                
                # Handle images with 0 faces
                if next_line.endswith('.jpg'):
                    open(label_output_path, 'w').close()
                    continue
                
                try:
                    num_faces = int(next_line)
                except ValueError:
                    i += 1
                    continue
                
                bboxes = []
                for _ in range(num_faces):
                    i += 1
                    if i >= total_lines:
                        break
                    
                    box_data = lines[i].split()
                    if len(box_data) < 4:
                        continue
                        
                    x = float(box_data[0])
                    y = float(box_data[1])
                    w = float(box_data[2])
                    h = float(box_data[3])
                    
                    invalid = int(box_data[7]) if len(box_data) > 7 else 0
                    if invalid == 1 or w <= 0 or h <= 0:
                        continue
                        
                    bboxes.append((x, y, w, h))
                
                # Write labels if faces exist
                if len(bboxes) > 0:
                    img = cv2.imread(img_full_path)
                    if img is not None:
                        img_h, img_w = img.shape[:2]
                        with open(label_output_path, 'w') as out_f:
                            for (x, y, w, h) in bboxes:
                                x_center = (x + w / 2.0) / img_w
                                y_center = (y + h / 2.0) / img_h
                                norm_w = w / img_w
                                norm_h = h / img_h
                                out_f.write(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                else:
                    open(label_output_path, 'w').close()
            else:
                # Skip face coordinate lines if image file doesn't exist
                i += 1
                while i < total_lines and not lines[i].endswith('.jpg'):
                    i += 1
                continue
        i += 1
        
    print(f"Successfully processed {match_count} images in {images_root}")

print("Converting Training annotations...")
convert_wider_structured('Data/train/wider_face_train_bbx_gt.txt', 'Data/train/images')

print("Converting Validation annotations...")
convert_wider_structured('Data/val/wider_face_val_bbx_gt.txt', 'Data/val/images')