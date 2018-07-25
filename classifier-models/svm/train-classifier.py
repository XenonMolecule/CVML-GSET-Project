from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier
import glob
import os
from config import *

fds = []
labels = []

print("Loading feature descriptors")
for i in range(0, 8):
    train_im_path = train_im_paths[i]
    feat_path = feat_paths[i]
    print("\t" + feat_path.split("/")[-1].title())
    for file in glob.glob(os.path.join(train_im_path, "*")):
        fd_name = os.path.split(file)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(feat_path, fd_name)
        fd = joblib.load(fd_path)
        fds.append(fd)
        labels.append(i + 1)
        print("\t\t" + str(file[str(file).find("\\") + 1:]) + str(len(fd)))
print("Completed loading feature descriptors")

print("Training classifiers ")
print("\tLinear")
linear_clf = SVC(kernel = "linear")
linear_clf.fit(fds, labels)
joblib.dump(linear_clf, model_linear_path)
print("\tRBF")
rbf_clf = SVC(kernel = "rbf")
rbf_clf.fit(fds, labels)
joblib.dump(rbf_clf, model_rbf_path)
print("\tLinearSVC")
linearsvc_clf = LinearSVC()
linearsvc_clf.fit(fds, labels)
joblib.dump(linearsvc_clf, model_linearsvc_path)
print("Saved classifiers")
