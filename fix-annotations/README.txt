To use the fix-annotations script, locate the directory of the annotations you downloaded from another computer, then locate the directory of the trashnet directory (probably called dataset-resized or something if you didn't change it).

Run the script with those two paths as the arguments:

python3 fix-annotations.py <annotation-path> <dataset-path>

For example running it on my computer would be done like this:

python3 fix-annotations.py /Users/MichaelRyan/Documents/School/GSET/Project/Dataset/trashnet-annotations /Users/MichaelRyan/Documents/School/GSET/Project/Dataset/trashnet/

Do not worry that my image directory is called trashnet, I renamed it from dataset-resized

After you run the program, the xml files themselves will be modified and you can use them as you need



EXAMPLE FOR PASCAL=TO-TFRECORD

python pascal-to-tfrecord.py C:\\Users\\micha\\OneDrive\\Documents\\GitHub\\CVML-GSET-Project\\dataset\\total-dataset\\training-annotations\\ --output_path=C:/Users/micha/OneDrive/Documents/GitHub/CVML-GSET-Project/dataset/total-dataset/tensorflow/train.record
