import sys
import os
from shutil import copyfile
from random import randint

TRAIN_SIZE = 0.8 # No greater than 1.0
DATASET_SIZE = 2842
ALL_ANNOTATION_PATH = "C:\\Users\\micha\\OneDrive\\Documents\\GitHub\\CVML-GSET-Project\\dataset\\total-dataset\\all-annotations\\"
TRAIN_ANNOTATION_PATH = "C:\\Users\\micha\\OneDrive\\Documents\\GitHub\\CVML-GSET-Project\\dataset\\total-dataset\\training-annotations\\"
TEST_ANNOTATION_PATH = "C:\\Users\\micha\\OneDrive\\Documents\\GitHub\\CVML-GSET-Project\\dataset\\total-dataset\\testing-annotations\\"

print("Splitting Dataset")

annotations = os.fsencode(ALL_ANNOTATION_PATH)

if(TRAIN_SIZE > 1.0):
    print("YOU GOOOOOOOBER")
    exit()

training_size = round(DATASET_SIZE * TRAIN_SIZE)
testing_size = DATASET_SIZE - training_size

i = 0
test_set = []
rand = randint(0, DATASET_SIZE-1)
while i < testing_size:
    while(rand in test_set):
        rand = randint(0, DATASET_SIZE-1)
    test_set.append(rand)
    i += 1

j = 0
for file in os.listdir(ALL_ANNOTATION_PATH):
    filename = str(os.fsencode(file))[2:-1]
    if(j % 100 == 0):
        print("Done: " + str(j))
    if(j in test_set):
        copyfile(ALL_ANNOTATION_PATH + filename, TEST_ANNOTATION_PATH + filename)
    else:
        copyfile(ALL_ANNOTATION_PATH + filename, TRAIN_ANNOTATION_PATH + filename)
    j += 1

print("DONE")
