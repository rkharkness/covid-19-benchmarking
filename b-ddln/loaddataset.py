from cv2 import cv2
import numpy as np
from imutils import paths
import os
import pandas as pd

# Preprocessing of images
def processImages(workingDirectory, imageDimensions):
    images = []
    labels = []
    verImg = []
    verLabels = []
    covidPath = os.path.sep.join([f"{workingDirectory}", "Dataset", "COVID-19"])
    normalPath = os.path.sep.join([f"{workingDirectory}", "Dataset", "NORMAL"])
    verificationPath = os.path.sep.join([f"{workingDirectory}", "Dataset", "VERIFICATION"])

    # Preprocessing of images in training set
    covidImages = list(paths.list_images(f"{covidPath}"))
    normalImages = list(paths.list_images(f"{normalPath}"))
    for i in covidImages:
        label = i.split(os.path.sep)[-2]
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#改变通道
        image = cv2.resize(image, (imageDimensions, imageDimensions))#规范图像大小（224*224*3）
        images.append(image)
        labels.append(label)
    print("Finished copying COVID-19 images")

    for i in normalImages:
        label = i.split(os.path.sep)[-2]
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (imageDimensions, imageDimensions))
        images.append(image)
        labels.append(label)
    print("Finished copying normal images")

    # Preprocessing of images in testing set
    for (index, row) in pd.read_csv(os.path.sep.join([f"{workingDirectory}", "verification.csv"])).iterrows():
        verLabels.append(row["finding"])
        image = cv2.imread(
            os.path.sep.join(
                [
                    f"{workingDirectory}",
                    "Dataset",
                    "VERIFICATION",
                    str(row["filename"]),
                ]
            )
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (imageDimensions, imageDimensions))
        verImg.append(image)
    print("Finished copying verification images")

    images = np.asarray(images) # Images of training set
    labels = np.asarray(labels) # Labels of training set
    verImg = np.asarray(verImg) # Images of testing set
    verLabels = np.asarray(verLabels) # Labels of testing set
    labels = [1 if x == "COVID-19" else x for x in labels] # Labeling, 1: COVID-19, 0: Normal
    labels = [0 if x == "NORMAL" else x for x in labels]
    labels = np.asarray(labels)
    verLabels = [1 if x == "COVID-19" else x for x in verLabels]
    verLabels = [0 if x == "normal" else x for x in verLabels]
    verLabels = np.asarray(verLabels)
    images = images / 255.0 # Image normalization
    verImg = verImg / 255.0
    print("Number of COVID-19 training images:", str(len(covidImages)))
    print("Number of normal training images:", str(len(normalImages)))
    print("Number of verification images:",str(len(list(paths.list_images(f"{verificationPath}")))),)
    return images, labels, verImg, verLabels
