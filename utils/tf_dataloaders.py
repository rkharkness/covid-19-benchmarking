import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
from keras.preprocessing.image import ImageDataGenerator

# update to be a shared hp ?
transforms = A.Compose([
                A.VerticalFlip(p=0.5),              
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                # A.Affine(translate_percent=10,p=0.5),
                A.CLAHE(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),    
                A.RandomGamma(p=0.5),
    ])
    
def aug_fn(image):
    data = {"image":image}
    aug_data = transforms(**data)
    aug_img = aug_data["image"]
    aug_img= tf.cast(aug_img/255.0, tf.float32)
    return image

def process_data(image, label, img_size):
    aug_img = tf.numpy_function(func=aug_fn, inp=[image, img_size], Tout=tf.float32)
    return aug_img, label

# create dataset
ds_alb = data.map(partial(process_data, img_size=120),
                  num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

print(ds_alb)  
def make_generators(train_df, val_df, batch_size, target, target_size, directory):
        def transform(image):
            image = image.astype(np.float64),
            return image

        train_datagen=ImageDataGenerator(
            horizontal_flip=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            # preprocessing_function=transform,
            rescale=1/255.0)

        test_datagen=ImageDataGenerator(
            # preprocessing_function=transform,
            rescale=1/255.0)

        train_generator=train_datagen.flow_from_dataframe(dataframe=train_df, directory=directory,
                                                    x_col="structured_path", y_col=target, class_mode="categorical", target_size=target_size, color_mode='grayscale',
                                                    batch_size=batch_size)


        val_generator=test_datagen.flow_from_dataframe(dataframe=val_df, directory=directory,
                                                    x_col="structured_path", y_col=target, class_mode="categorical", target_size=target_size, color_mode='grayscale',
                                                    batch_size=batch_size)

        return train_generator, val_generator