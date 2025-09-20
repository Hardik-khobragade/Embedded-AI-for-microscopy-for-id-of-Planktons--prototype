import os
import random
import shutil

# Paths
base_path = "my_dataset"
img_dir = os.path.join(base_path, "images")
lbl_dir = os.path.join(base_path, "labels")

out_base = "datasets"
img_out = os.path.join(out_base, "images")
lbl_out = os.path.join(out_base, "labels")

# Create output dirs
for split in ["train", "val"]:
    os.makedirs(os.path.join(img_out, split), exist_ok=True)
    os.makedirs(os.path.join(lbl_out, split), exist_ok=True)

# Collect files
images = [f for f in os.listdir(img_dir) if f.endswith(".png")]
random.shuffle(images)

# 80:20 split
split_idx = int(0.8 * len(images))
train_files = images[:split_idx]
val_files = images[split_idx:]

def move_files(file_list, split):
    for img in file_list:
        label = img.replace(".png", ".txt")
        src_img = os.path.join(img_dir, img)
        src_lbl = os.path.join(lbl_dir, label)
        dst_img = os.path.join(img_out, split, img)
        dst_lbl = os.path.join(lbl_out, split, label)
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lbl, dst_lbl)

move_files(train_files, "train")
move_files(val_files, "val")

print("Done. Train:", len(train_files), " Val:", len(val_files))
