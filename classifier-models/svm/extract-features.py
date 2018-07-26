from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib
import glob
import os
from config import *

print ("Calculating and saving feature descriptors for each image")
for i in range(0, 8):
    all_im_path = all_im_paths[i]
    feat_path = feat_paths[i]
    print(feat_path.split("/")[-1].title())
    for file in glob.glob(os.path.join(all_im_path, "*")):
        im = imread(file, as_gray = True)
        fd = hog(im, orientations = orientations, pixels_per_cell = pixels_per_cell, cells_per_block = cells_per_block, visualize = False)
        fd_name = os.path.split(file)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(feat_path, fd_name)
        joblib.dump(fd, fd_path)
        print("\t" + str(file[str(file).find("\\") + 1:]))
print("Completed extracting feature descriptors from training images")
