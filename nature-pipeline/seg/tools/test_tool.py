import logging
import os
import torch
from torch.autograd import Variable
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

def test(parameters, config, gpu_list, **kwargs):
    model = parameters["model"]
    dataset = parameters["test_dataset"]
    output_path = kwargs["output"]
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    model.eval()
    step = -1
    result = []

    for step, data in enumerate(dataset):
        if "empty" in data.keys():
            continue
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if len(gpu_list) > 0:
                    data[key] = Variable(data[key].cuda())
                else:
                    data[key] = Variable(data[key])

        results = model(data, config, gpu_list, [None, None], "test")

        if config.get("model", "model_name") != "Pipeline":
            prediction = results["prediction"].detach().cpu().numpy()
            file_list = data["file"]
            batch_size = prediction.shape[0]
            for bs in range(batch_size):
                output_dir = os.path.join(output_path, file_list[bs][:file_list[bs].rfind(".")] + ".npy")
                np.save(output_dir, prediction[bs])

        if config.get("model", "model_name") == "Pipeline":
            stage_1_prediction, stage_2_prediction = results["prediction"]
            location = results["location"]
            file_list = data["file"]

            batch_size = stage_1_prediction.shape[0]
            assert stage_2_prediction.shape[0] == batch_size
            assert len(location) == batch_size
            assert len(file_list) == batch_size

            stage_1_prediction = stage_1_prediction.astype(np.uint8)
            stage_2_prediction = stage_2_prediction.astype(np.uint8)
            stage_2_logits = results["logits"]

            for bs in range(batch_size):
                output_dir = os.path.join(output_path, file_list[bs][:file_list[bs].rfind(".")] + ".npy")
                origin_width = location[bs][1] - location[bs][0]
                origin_height = location[bs][3] - location[bs][2]
                lung_field = cv2.resize(stage_1_prediction[bs], (512, 512))
                background = 1 - lung_field
                bingzao = np.transpose(stage_2_prediction[bs], [1, 2, 0])
                bingzao = cv2.resize(bingzao, (origin_height, origin_width))
                annotation = np.zeros([512, 512, 7], np.uint8)
                annotation[:, :, 0] = background
                annotation[:, :, 1] = lung_field
                annotation[location[bs][0] : location[bs][1], location[bs][2] : location[bs][3], 2] = bingzao[:, :, 0]
                annotation[location[bs][0] : location[bs][1], location[bs][2] : location[bs][3], 3] = bingzao[:, :, 1]
                annotation[location[bs][0]: location[bs][1], location[bs][2]: location[bs][3], 4] = bingzao[:, :, 2]
                annotation[location[bs][0]: location[bs][1], location[bs][2]: location[bs][3], 5] = bingzao[:, :, 3]
                annotation[location[bs][0]: location[bs][1], location[bs][2]: location[bs][3], 6] = bingzao[:, :, 4]
                try:
                    np.save(output_dir, annotation)
                except:
                    pass

    if step == -1:
        logger.error("There is no data given to the model in this epoch, check your data.")
        raise NotImplementedError

    return result
