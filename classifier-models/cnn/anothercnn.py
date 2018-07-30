import shutil
import numpy as np
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.models import Model
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten
from keras.models import load_model

batch_size = 8
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        '/Users/erikatan/Downloads/images',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # more than two classes

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        '/Users/erikatan/Downloads/trash_test',
        target_size=(150,150),
        batch_size=batch_size,
        class_mode='categorical')

# generator= train_datagen.flow_from_directory("train", batch_size=batch_size)
# label_map = (generator.class_indices)
# 0 --> bottle
# 1 --> can
# 2 --> cardboard
# 3 --> container
# 4 --> cup
# 5 --> paper
# 6 --> scrap
# 7 --> wrapper
# argmax gives the number of the class that it predicts

def get_model():
    # Get base model 
    base_model = VGG16(include_top=False, input_shape=(150,150,3))
    # Freeze the layers in base model
    for layer in base_model.layers:
        layer.trainable = False
    # Get base model output 
    base_model_ouput = base_model.output
    
    # Add new layers
    x = Flatten()(base_model.output)
    x = Dense(500, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(8, activation='softmax', name='fc2')(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    return model
    
model = VGG16(include_top=False, input_shape=(150,150,3))
for layer in model.layers[:-2]:
        layer.trainable = False
        

from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
     
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)  
        print(" -  val_f1: " + str(_val_f1) + " - val_precision: " + str(_val_precision))
        #print(“ — val_f1: %f — val_precision: %f — val_recall %f” %(_val_f1, _val_precision, _val_recall))
        return
 
#metrics = Metrics()       
        
        
# Compile it
opt = Adam(lr=1e-3, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=15,
        validation_data=validation_generator,
        validation_steps=150 // batch_size)

from keras.preprocessing import image
# prepare the test image by converting its resolution to 64 x 64
test_image = image.load_img('/Users/erikatan/Downloads/images/cup/glass2.jpg', target_size=(150,150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)
print(result)

model.save('vgg_cnn.h5')

# model = load_model('vgg_cnn.h5')