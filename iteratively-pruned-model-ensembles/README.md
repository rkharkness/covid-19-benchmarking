# Iteratively-pruned-model-ensembles-for-COVID-19-detection-in-CXRs
We demonstrate use of iteratively pruned deep learning (DL) model ensembles for detecting the “coronavirus disease 2019” (COVID-19) infection with chest X-rays (CXRs). The disease is caused by the novel Severe Acute Respiratory Syndrome Coronavirus 2 (SARS-CoV-2) virus, also known as the novel Coronavirus (2019-nCoV). A custom convolutional neural network (CNN) and a selection of pretrained CNN models are trained on publicly available CXR collections to learn CXR modality-specific feature representations and the learned knowledge is transferred and fine-tuned to improve performance and generalization in the related task of classifying normal, bacterial pneumonia, and CXRs exhibiting COVID-19 abnormalities. The best performing models are iteratively pruned to identify optimal number of neurons in the convolutional layers to reduce complexity and improve memory efficiency. 

The predictions of the best-performing pruned models are combined through different ensemble strategies to improve classification performance. The custom and pretrained CNNs are evaluated at the patient-level to alleviate issues due to information leakage and reduce generalization errors. 

Empirical evaluations demonstrate that the weighted average of the best-performing pruned models significantly improves performance resulting in an accuracy of 99.01% and area under the curve (AUC) of 0.9972 in detecting COVID-19 findings on CXRs as compared to the individual constituent models. The combined use of modality-specific knowledge transfer, iterative model pruning, and ensemble learning resulted in improved predictions. We expect that this model can be quickly adopted for COVID-19 screening using chest radiographs.

We used Grad-CAM and LIME visualizations to interpret the behavior of the learned top-3 pruned models in this study. 

The attached notebook discusses: preprocessing to convert DICOM and JPEG images to PNG, generation of lung masks using dropout-U-Net, segmentation and lung bounding box cropping, storing bounding boxes and disease ROI coordinates in the CSV files, performing modlaity-specific knowledge transfer and fine-tuning, iteative pruning, enmseble of the iteratively pruned models, and visualization of the learned behavior of the top-3 pruned models in this study, using Grad-CAM and LIME visualization techniques.

## Codes and Model weights

The weights of the best-performing, iteratively pruned DL models (VGG-16, VGG-19, and Inception-V3) are uploaded to https://drive.google.com/drive/folders/1ec894pKM4LUJx_m0rJmj2y7FgawphBhO?usp=sharing. We found that the VGG-16 model pruned to remove 20% of the filters with the highest APoZ from each layer, VGG-19 model pruned to remove 6% of the filters with the highest APoZ, and Inception-V3 model, pruned to remove 30% of the filters with the highest APoZ from each layer delivred superior classification performance toward classifying CXRs as normal or showing bacterial or COVID-19 viral pneumonia-related opacities. The performance of the pruned and unpruned, baseline models is discussed in the IEEE Access publication available at  https://ieeexplore.ieee.org/document/9121222. The percentage reduction in the computational parameters of the pruned models is as shown below:
```
Models	      % Reduction
VGG-16-P	      46.03
VGG-19-P	      16.13
Inception-V3-P	36.10

```

### Using the trained models

First, use the unet.hdf5 weights file in the shared google drive link to generate lung masks of 256×256 pixel resolution. Use the following code snippet to crop the lung boundaries using the generated lung masks and store them as a bounding box containing all the lung pixels, as shown below:

```
model = load_model("C:/Users/trained_model/unet.hdf5")
model.summary()
test_path = "C:/Users/data/test" #where your test data resides
save_path = "C:/Users/data/membrane/result" #where you want the generated lung masks to be stored

data_gen_args = dict(rotation_range=10.,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=5,
                    zoom_range=0.3,
                    horizontal_flip=True,
                    fill_mode='nearest') 

testGene = testGenerator(test_path)
results = model.predict_generator(testGene,steps per epoch=135,verbose=1, workers=1, use_multiprocessing=False) #steps per epoch is the no. of samples in test image.
saveResult(save_path, results, test_path)

#custom function to generate bounding boxes
def generate_bounding_box(image_dir: str, #containing images
                          mask_dir: str, #containing masks, images have same name as original images
                          dest_csv: str, #CSV file to write the bounding box coordinates
                          crop_save_dir: str): #save the cropped bounding box images
    """
    the orginal images are resized to 256 x 256
    the output crops are resized to 256 x 256
    """
    if not os.path.isdir(mask_dir):
        raise ValueError("mask_dir not existed")

    case_list = [f for f in os.listdir(mask_dir) if f.split(".")[-1] == 'png'] #all mask images are png files

    with open(dest_csv, 'w', newline='') as f:
        csv_writer = csv.writer(f)

        for j, case_name in enumerate(case_list):
            mask = cv2.imread(mask_dir + case_name)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            image = cv2.imread(image_dir + case_name, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (256, 256)) #original images are resized to 256 x 256
            if mask is None or image is None:
                raise ValueError("The image can not be read: " + case_name)

            reduce_col = np.sum(mask, axis=1)
            reduce_row = np.sum(mask, axis=0)
            # many 0s add up to none zero, we need to threshold it
            reduce_col = (reduce_col >= 255)*reduce_col
            reduce_row = (reduce_row >= 255)*reduce_row
            first_none_zero = None
            last_none_zero = None

            last = 0
            for i in range(reduce_col.shape[0]):
                current = reduce_col[i]
                if last == 0 and current != 0 and first_none_zero is None:
                    first_none_zero = i

                if current != 0:
                    last_none_zero = i

                last = reduce_col[i]

            up = first_none_zero
            down = last_none_zero

            first_none_zero = None
            last_none_zero = None
            last = 0
            for i in range(reduce_row.shape[0]):
                current = reduce_row[i]
                if last == 0 and current != 0 and first_none_zero is None:
                    first_none_zero = i

                if current != 0:
                    last_none_zero = i

                last = reduce_row[i]

            left = first_none_zero
            right = last_none_zero

            if up is None or down is None or left is None or right is None:
                raise ValueError("The border is not found: " + case_name)
            
            # new coordinates for image which is 1 times of mask, mask images are 256 x 256, 
            #so need to multiply 1 times to get 256 x 256, and relaxing the borders by 5% on all directions
            up_down_loose = int(1 * (down - up + 1) * 0.05)
            image_up = 1 * up - up_down_loose
            if image_up < 0:
                image_up = 0
            image_down = 1*(down+1)+up_down_loose
            if image_down > image.shape[0] + 1:
                image_down = image.shape[0]

            left_right_loose = int(1 * (right - left) * 0.05)
            image_left = 1 * left - left_right_loose
            if image_left < 0:
                image_left = 0
            image_right = 1*(right + 1)+left_right_loose
            if image_right > image.shape[1] + 1:
                image_right = image.shape[1]

            crop = image[image_up: image_down, image_left: image_right]
            crop = cv2.resize(crop, (256, 256)) #the cropped image is resized to 256 x 256

            cv2.imwrite(crop_save_dir + case_name, crop) # cropped images saved to crop directory

            # write new csv
            crop_width = image_right - image_left + 1
            crop_height = image_down - image_up + 1

            csv_writer.writerow([case_name,
                                 image_left,
                                 image_up,
                                 crop_width,
                                 crop_height]) #writes xmin, ymin, width, and height

            if j % 50 == 0:
                print(j, " images are processed!")

generate_bounding_box("C:/Users/data/test/",
                      "C:/Users/result/mask/",
                      'C:/Users/result/bounding_box.csv',
                      "C:/Users/result/cropped/")
```
Use the generated lung crops for your test data and then used the pruned weights available in the shared google drive link to predict on your data. Give the path to your directory where you have stored the images to be predicted. The data folder is organized as: 
data 
|-train
  |-normal
  |-bacterial
  |covid-viral
|-test
  |-normal
  |-bacterial
  |covid-viral
  
 We used Keras ImageDataGenerator to preprocess the test images as follows:
 
 ```
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

nb_test_samples = len(test_generator.filenames)
print(test_generator.class_indices)
#true labels
Y_test=test_generator.classes
print(Y_test.shape)
#convert test labels to categorical
Y_test1=to_categorical(Y_test, num_classes=num_classes, dtype='float32')
print(Y_test1.shape)
```
Predict on the test images as follows:

```
model = load_model('vgg16_pruning_20percent.h5') 
model.summary()
sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.95, nesterov=True) 
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
test_generator.reset()
gg16_y_pred = model.predict_generator(test_generator, 
                                              nb_test_samples/batch_size, workers=1)
#true labels
Y_test=test_generator.classes
#print the shape of y_pred and Y_test
print(vgg16_y_pred.shape)
print(Y_test.shape)
vgg16_model_accuracy=accuracy_score(Y_test,vgg16_y_pred.argmax(axis=-1))
print('The accuracy of custom VGG16 model is: ', vgg16_model_accuracy)
```

Repeat the above steps for each of the pruned models. You can further use the pruned model weights to perform ensembles at your preference. 

### Generate LIME-based decisions

To visualize the learned behavior of the pruned models, we shown an instance of how to use LIME visualization with the trained models:

```
model = load_model('vgg16_pruning_20percent.h5') 
model.summary()
sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.95, nesterov=True) 
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#path to image to visualize
img_path = 'image1.png'
img = image.load_img(img_path)
#preprocess the image
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255 
#predict on the image
preds = model.predict(x)[0]
print(preds)
#initialize the explainer
from lime import lime_image
from skimage.segmentation import mark_boundaries
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(x[0], 
                                         model.predict, top_labels=1, 
                                         hide_color=0, num_samples=42)
print(explanation.top_labels[0])
temp, mask = explanation.get_image_and_mask(0, #change the respective class index
                                            positive_only=False, 
                                            num_features=5, hide_rest=False) 
plt.figure()
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.figure()
plt.imshow(x[0] / 2 + 0.5) #this increases te brightness of the image

```

## Generate CAM based decision

Visualize the learned behavior using CAM-based ROI localization
```
model = load_model('vgg16_pruning_20percent.h5') 
model.summary()
sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.95, nesterov=True) 
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#path to image to visualize
img_path = 'image1.png'
img = image.load_img(img_path)
#preprocess the image
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255 
#predict on the image
preds = model.predict(x)[0]
print(preds)
#begin visualization
covid_output = model.output[:, 0] 
#Output feature map from the deepest convolutional layer
last_conv_layer = model.get_layer('block5_conv3')
#compute the Gradient of the expected class with regard to the output feature map of block5_conv3 or the deepst convolutional layer)
grads = K.gradients(covid_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input],[pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])
for i in range(512): #number of filters in the deepest convolutional layer
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
#For visualization purposes, we normalize the heatmap between 0 and 1.
heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
img = cv2.imread(img_path) #path to the image
#Resizes the heatmap to be the same size as the original image
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#Converts the heatmap to RGB 
heatmap = np.uint8(255 * heatmap)
#Applies the heatmap to the original image
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img # 0.4 here is a heatmap intensity factor.
#Saves the image to disk
cv2.imwrite('cam_image.png', superimposed_img)

```


  
