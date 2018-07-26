from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib
from sklearn.metrics import classification_report, fbeta_score
import glob
import os
from config import *

fds = []
labels = []
print("Calculating feature descriptors for testing data ... ")
for i in range(0, 8):
    test_im_path = test_im_paths[i]
    feat_path = feat_paths[i]
    print("\t" + feat_path.split("/")[-1].title())
    for file in glob.glob(os.path.join(test_im_path, "*")):
        fd_name = os.path.split(file)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(feat_path, fd_name)
        fd = joblib.load(fd_path)
        fds.append(fd)
        labels.append(i + 1)
        print("\t\t" + str(file[str(file).find("\\") + 1:]))

print("Testing classifiers")
target_names = ["bottle", "can", "cardboard", "container", "cup", "paper", "scrap", "wrapper"]
print("\tLinear")
linear_clf = joblib.load(model_linear_path)
linear_pred = linear_clf.predict(fds)
print(classification_report(labels, linear_pred, target_names = target_names, digits = 3))
print(fbeta_score(labels, linear_pred, average = None, beta = 0.5))
print("\tRBF")
rbf_clf = joblib.load(model_rbf_path)
rbf_pred = rbf_clf.predict(fds)
print(classification_report(labels, rbf_pred, target_names = target_names, digits = 3))
print(fbeta_score(labels, rbf_pred, average = None, beta = 0.5))
print("\tLinearSVC")
linearsvc_clf = joblib.load(model_linearsvc_path)
linearsvc_pred = linearsvc_clf.predict(fds)
print(classification_report(labels, linearsvc_pred, target_names = target_names, digits = 3))
print(fbeta_score(labels, linearsvc_pred, average = None, beta = 0.5))
