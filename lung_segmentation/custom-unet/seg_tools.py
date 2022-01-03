import matplotlib.pyplot as plt
import numpy as np
import itertools
from functools import reduce
import cv2


def mask_from_img(img):
    """create mask from prediction
    args:
        img: mask prediction - open-cv numpy array image"""
    return (img >= 0.5).astype(np.float32)

def mask_from_bbox(img, bbox):
    """segment image according to bbox coordinates
    args:
        img: image as open-cv numpy array
        bbox: numpy array - [xmin, xmax, ymin, ymax] coordinates"""
    img = img.copy()
    bbox_mask = np.zeros(img.shape,np.float32)
    xmin, xmax, ymin, ymax = bbox
    bbox_mask[ymin:ymax,xmin:xmax] = img[ymin:ymax,xmin:xmax]
    bbox_mask = bbox_mask.astype(np.float32)
    return bbox_mask
  
def bbox(img, input='prediction'):
    """extract bounding box coords from input
    args:
        img: image as open-cv numpy array
        input=mask: str arg - mask or prediction"""
    if input == 'mask':
        a = np.where(img == np.max(np.array(img)))
    elif input == 'prediction':
        a = np.where(img[0] >= 0.5)
    bbox = np.min(a[1]), np.max(a[1]), np.min(a[0]), np.max(a[0]) # extract coords - xmin, xmax, ymin, ymax
    return bbox

def draw_bbox(bbox_coords):
    bbox_mask = np.zeros((480,480), np.float32)
    xmin, xmax, ymin, ymax = bbox_coords
    bbox_mask[ymin:ymax,xmin:xmax] = 1
    return bbox_mask

def visualize_bbox(img, bbox, color=(201, 58, 64), thickness=5):  #https://www.kaggle.com/blondinka/how-to-do-augmentations-for-instance-segmentation
    """ add bboxes to images 
    args:
        img : image as open-cv numpy array
        bbox : boxes as a list or numpy array in pascal_voc format [x_min, y_min, x_max, y_max]  
        color=(255, 255, 0): boxes color 
        thickness=2 : boxes line thickness
    """

    # draw bbox on img
    img = img.copy()
    xmin, xmax, ymin, ymax = bbox
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=color, thickness=thickness)

    img = img.astype(np.float32)
    return img

def segment_mask(img, mask):
    """ extract region from predicted bbox
    args:
        img: image as open-cv numpy array
        mask: uint8 type numpy array
    """
    img = img.copy()
    mask = mask.transpose((1,2,0))
    img = img * mask

    return img.astype(np.float32)

def reverse_transform(inp, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    inp = inp.numpy().transpose((1, 2, 0))
    # inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp).astype(np.float32)/np.max(inp)
    return inp

def plot_img_array(img_array, idx, model_num, ncol=5, img_class=None):
    nrow = len(img_array) // ncol

    f, plots = plt.subplots(nrow, ncol, sharex='all', sharey='all', figsize=(ncol * 4, nrow * 4))
    print(len(img_array))
    print(img_class)
    for i in range(len(img_array)):
        plots[i // ncol, i % ncol]
        plots[i // ncol, i % ncol].imshow(img_array[i])
    
    if img_class != None:
        for j in range(4):
            plots[j, 0].set_ylabel(img_class[j])

    plt.savefig(f"/MULTIX/DATA/HOME/lung_segmentation/segmentation_results/seg_data_vggnestedunet_{model_num}_{idx}")

def plot_side_by_side(img_arrays, idx, model_num, img_class=None):
    flatten_list = reduce(lambda x,y: x+y, zip(*img_arrays))
    print(np.array(flatten_list).shape)

    plot_img_array(np.array(flatten_list),idx, model_num, ncol=len(img_arrays), img_class=img_class)

def plot_errors(results_dict, title):
    markers = itertools.cycle(('+', 'x', 'o'))

    plt.title('{}'.format(title))

    for label, result in sorted(results_dict.items()):
        plt.plot(result, marker=next(markers), label=label)
        plt.ylabel('dice_coef')
        plt.xlabel('epoch')
        plt.legend(loc=3, bbox_to_anchor=(1, 0))

    plt.show()

def masks_to_colorimg(masks):
    colors = np.asarray([(201, 58, 64)])

    colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    channels, height, width = masks.shape


    for y in range(height):
        for x in range(width):
            selected_colors = colors[masks[:,y,x] > 0.5]

            if len(selected_colors) > 0:
                colorimg[y,x,:] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.float32)/np.max(colorimg)