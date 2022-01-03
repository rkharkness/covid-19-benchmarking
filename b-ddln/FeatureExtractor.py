from loaddataset import processImages
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# Use GPU for training
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print("Number of GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

keras.backend.clear_session() # Clear session
workingDirectory = os.path.dirname(os.path.realpath(__file__))
imgDimensions = 224

img_height = imgDimensions
img_width = imgDimensions
img_channels = 3

images, labels, verImg, verLabels = processImages(workingDirectory, imgDimensions)# Use loaddataset.py

# Structure of Feature Extractor
def Extractor(x):
    def commonLayers(y):
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)
        return y

    def groupedConvolution(y, nb_channels, _strides):
        return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding="same")(y)

    def resBlock(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
        shortcut = y
        y = layers.Conv2D(nb_channels_in, kernel_size=(3, 3), strides=(1, 1), padding="same")(y)
        y = commonLayers(y)
        y = groupedConvolution(y, nb_channels_in, _strides=_strides)
        y = commonLayers(y)
        y = layers.BatchNormalization()(y)
        if _project_shortcut or _strides != (1, 1):
            shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding="same")(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        y = layers.add([shortcut, y])
        y = layers.LeakyReLU()(y)
        return y

    def AveragePooling(y):
        y = layers.GlobalAveragePooling2D()(y)
        return y

    # Feed-forward output of network
    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding="same")(x)
    x = commonLayers(x)
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
    for i in range(2):
        project_shortcut = True if i == 0 else False
        strides = (2, 2) if i == 0 else (1, 1)
        x = resBlock(x, 64, 64, _strides=strides, _project_shortcut=project_shortcut)
    for i in range(2):
        strides = (2, 2) if i == 0 else (1, 1)
        x = resBlock(x, 128, 128, _strides=strides)
    for i in range(2):
        strides = (2, 2) if i == 0 else (1, 1)
        x = resBlock(x, 256, 256, _strides=strides)
    for i in range(2):
        strides = (2, 2) if i == 0 else (1, 1)
        x = resBlock(x, 512, 512, _strides=strides)
    x_feature = AveragePooling(x)  # Produce input features
    print('%s = \r\n' % "Feature vector", x_feature)
    x = layers.Dense(1, activation="sigmoid")(x_feature)  # Full-connected layer, which can produce predicted labels
    return x_feature, x

# Confusion matrix
metrics = [
    keras.metrics.TruePositives(name="tp"),
    keras.metrics.FalsePositives(name="fp"),
    keras.metrics.TrueNegatives(name="tn"),
    keras.metrics.FalseNegatives(name="fn"),
    keras.metrics.BinaryAccuracy(name="accuracy"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
    keras.metrics.AUC(name="auc"),
]

image_tensor = layers.Input(shape=(img_height, img_width, img_channels)) # Network input
network_output = Extractor(image_tensor) # Network output
model = models.Model(inputs=[image_tensor], outputs=[network_output])

opt = tf.keras.optimizers.SGD(learning_rate=1, nesterov=True, momentum=0.9, decay=0.01) # Optimizer
model.compile(optimizer=opt, loss=keras.losses.BinaryCrossentropy(), metrics=metrics)

model.fit(images, labels, epochs=35, validation_data=(verImg, verLabels), batch_size=32) # Training model
model.save('model.h5') # Save model
print('Saved total model.')

del model
print('Load total model.')
model = tf.keras.models.load_model('model.h5')  # Load trained model
score = model.evaluate(verImg, verLabels, verbose=0) # Test model
print('Loss:', score[0])
print('Accuracy:', score[5])

trainingResult = model.predict(images) # Network output of training set
verResult = model.predict(verImg) # Network output of testing set

model_cut = models.Model(inputs=model.inputs, outputs=model.layers[-2].output) # Get the input of full-connected layer
training_features = model_cut.predict(images) # Obtain features of images from training set
verfeatures = model_cut.predict(verImg) # Obtain features of images from testing set

# Save features and labels for classification of CDNN
np.savetxt("training_features.csv", training_features)
np.savetxt("training_labels.csv", labels)
np.savetxt("verfeatures.csv", verfeatures)
np.savetxt("verlabels.csv", verLabels)
