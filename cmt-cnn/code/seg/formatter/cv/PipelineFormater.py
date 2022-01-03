#coding
import torch
import cv2
import numpy as np

from formatter.Basic import BasicFormatter


class PipelineFormatter(BasicFormatter):
    #pipeline formatter
    # data["input_s1"] input_stage_of_stage_1    tensor   batch * channel * width * height
    # data["input_origin"] input_of_origin 512 * 512 image batch * width * height * channel
    # data["label_s1"] label of stage 1   tensor
    # data["label"] label of origin 512 * 512 image  np.array  batch * width * height
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)
        self.config = config
        self.mode = mode
        self.normalization = config.getboolean("data", "normalization")
        self.stage_1_input_size = config.getint("model", "stage_1_input_size")
        self.stage_2_output_class_num = config.getint("model", "stage_2_output_class_num")
        self.stae_2_input_size = config.getint("model", "stage_2_input_size")
        self.weights = [int(val) for val in config.get("model", "stage_2_weights").split(",")]
        assert len(self.weights) == self.stage_2_output_class_num
        self.use_weight_map = config.getboolean("data", "use_weight_map")

    def get_resize_image(self, image, size):
        image = cv2.resize(image, (size, size)).astype(np.float32)
        if self.normalization:
            image = image / 255.0
        return image

    def decorate_label(self, label):
        channel, width, height = label.shape
        outlabel = np.zeros([width, height], np.int32)
        for idx in range(1, channel):
            outlabel += (label[idx] == 1).astype(np.int32) * (idx)
        return outlabel

    def get_stage_1_label(self, label):
        lung_field = cv2.resize(label[0, :, :].astype(np.uint8), (self.stage_1_input_size, self.stage_1_input_size)).astype(np.int32)
        return lung_field

    def process_weight_map(self, weight_map, sigmoid_label):
        weight_map = weight_map.astype(np.float32)
        width, height = weight_map.shape
        weight_map = weight_map * self.w0
        #加class weights
        class_weights = np.ones([width, height], np.float32)
        class_num = sigmoid_label.shape[0]
        for idx in range(class_num):
            class_weights += self.weights[idx] * cv2.resize(sigmoid_label[idx], (256, 256))
        weight_map = weight_map + class_weights
        return weight_map

    def process(self, data, config, mode, *args, **params):
        out_input_s1 = []
        out_input_origin = []
        out_label_s1 = []
        out_sigmoid_label = []
        out_file = []
        for item in data:
            out_input_origin.append(item["data"].astype(np.float32) / 255.0)
            out_input_s1.append(self.get_resize_image(item["data"], self.stage_1_input_size))
            out_label_s1.append(self.get_stage_1_label(item["label"]))
            out_sigmoid_label.append(item["label"][1:])
            out_file.append(item["file"])

        out_input_s1 = torch.from_numpy(np.array(out_input_s1)).permute(0, 3, 1, 2)
        out_label_s1 = torch.from_numpy(np.array(out_label_s1))
        out_input_origin = np.array(out_input_origin)
        out_sigmoid_label = np.array(out_sigmoid_label)

        assert np.max(out_sigmoid_label) <= 1
        assert np.min(out_sigmoid_label) >= 0

        return {"input_s1": out_input_s1,
                "input_origin": out_input_origin,
                "label_s1": out_label_s1,
                "label": None,
                "sigmoid_label": out_sigmoid_label,
                "file": out_file,
                 }


class PipelineFormatter_test(BasicFormatter):
    #for test
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)
        self.config = config
        self.mode = mode
        self.normalization = config.getboolean("data", "normalization")
        self.stage_1_input_size = config.getint("model", "stage_1_input_size")
        self.stage_2_output_class_num = config.getint("model", "stage_2_output_class_num")
        self.stae_2_input_size = config.getint("model", "stage_2_input_size")

    def get_resize_image(self, image, size):
        image = cv2.resize(image, (size, size)).astype(np.float32)
        if self.normalization:
            image = image / 255.0
        return image

    def process(self, data, config, mode, *args, **params):
        out_input_s1 = []
        out_input_origin = []
        out_file = []
        for item in data:
            if "empty" in item.keys():
                continue
            if len(item["data"].shape) == 2:
                item["data"] = item["data"][:, :, np.newaxis]
            if item["data"].shape[2] == 1:
                item["data"] = item["data"].repeat(3, axis=2)
            out_input_origin.append(item["data"].astype(np.float32) / 255.0)
            out_input_s1.append(self.get_resize_image(item["data"], self.stage_1_input_size))
            out_file.append(item["file"])
        if len(out_input_origin) == 0:
            return {"empty": 0}
        out_input_s1 = torch.from_numpy(np.array(out_input_s1)).permute(0, 3, 1, 2)
        out_input_origin = np.array(out_input_origin)

        return {"input_s1": out_input_s1,
                "input_origin": out_input_origin,
                "label_s1": None,
                "label": None,
                "sigmoid_label": None,
                "file": out_file,
                "weight_map": None,
                }
