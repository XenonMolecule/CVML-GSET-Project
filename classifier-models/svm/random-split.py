import glob
import os
from shutil import copyfile
import random
from config import *

print("Clearing directories")
for i in range(0, 8):
    train_im_path = train_im_paths[i]
    test_im_path = test_im_paths[i]
    print("\t" + train_im_path.split("/")[-1].title())
    for file in glob.glob(os.path.join(train_im_path, "*")):
        os.remove(file)
    print("\t\t" + "Training Images")
    for file in glob.glob(os.path.join(test_im_path, "*")):
        os.remove(file)
    print("\t\t" + "Testing Images")
print("Completed clearing directories")


print("Randomly splitting images for training and testing")
for all_im_path in all_im_paths:
    print("\t" + all_im_path.split("/")[-1].title())
    images = [file for file in glob.glob(os.path.join(all_im_path, "*"))]
    random.shuffle(images)
    split_index = int(len(images) * 0.8)
    train_images = images[:split_index]
    test_images = images[split_index:]
    print("\t\tTraining images")
    for file in train_images:
        copyfile(str(file), str(file).replace("all-images", "train-images"))
    print("\t\tTesting images")
    for file in test_images:
        copyfile(str(file), str(file).replace("all-images", "test-images"))
