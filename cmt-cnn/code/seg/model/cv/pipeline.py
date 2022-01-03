#pipeline model

from __future__ import  print_function
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.autograd import Variable
from tools.accuracy_init import init_accuracy_function
import os

from model.cv.Unet import UNet

model_list = {
    "Unet": UNet,
}

class pipeline_model(nn.Module):
    # pipeline model
    # first lung field with 128 * 128, get result, get loss 1
    # second disease filed with 128 * 128 -> 256 * 256, get loss 2
    # bp to update
    def __init__(self, config, gpu_list, *args, **params):
        super(pipeline_model, self).__init__()

        #models
        self.stage_1_model = UNet(config, gpu_list, "stage_1", *args, **params)
        self.stage_2_model = UNet(config, gpu_list, "stage_2", *args, **params)

        #output size of two stages
        self.stage_1_input_size = config.getint("model", "stage_1_input_size")
        self.stage_2_input_size = config.getint("model", "stage_2_input_size")

    def get_boundary(self, prediction):
        batch_size, width, height = prediction.shape
        left_res, right_res, top_res, down_res = [], [], [], []
        for bs in range(batch_size):
            points_sum = np.sum(prediction[bs])
            threthold = 0.02 * points_sum
            left_sum, right_sum, top_sum, down_sum = 0, 0, 0, 0
            left, right, top, down = 0, width - 1, 0, height - 1
            while left_sum == 0 and left < 48:
                left_sum += np.sum(prediction[bs, left, :])
                left += 1
            while right_sum == 0 and right > 80:
                right_sum += np.sum(prediction[bs, right, :])
                right -= 1
            while top_sum == 0 and top < 48:
                top_sum += np.sum(prediction[bs, :, top])
                top += 1
            while down_sum == 0 and down > 80:
                down_sum += np.sum(prediction[bs, :, down])
                down -= 1
            left = max(0, int(0.9 * (512 / width) * left))
            right = min(512, int(1.1 * (512/ width) * right))
            top = max(0, int(0.9 * (512 / height) * top))
            down = min(512, int(1.1 * (512 / height) * down))
            left_res.append(left)
            right_res.append(right)
            top_res.append(top)
            down_res.append(down)
        return left_res, right_res, top_res, down_res

    def forward(self, data, config, gpu_list, acc_result, mode):
        #stage 1 logits and loss
        ret = self.stage_1_model(data = {"input": data["input_s1"],
                                         "label": data["label_s1"],
                                         },
                                 config = config,
                                 gpu_list = gpu_list,
                                 acc_result = acc_result[0],
                                 mode = mode
                                )
        if mode in ["train", "valid"]:
            stage_1_logits, stage_1_prediction, stage_1_loss, stage_1_acc_result = ret["logits"], ret["prediction"], \
                                                                                   ret["loss"], ret["acc_result"]
        else:
            stage_1_logits, stage_1_prediction = ret["logits"], ret["prediction"]

        stage_1_prediction = stage_1_prediction.detach().cpu().numpy()

        #crop and get image
        left_res, right_res, top_res, down_res = self.get_boundary(stage_1_prediction)
        stage_2_inputs = []
        stage_2_labels = []
        for bs in range(stage_1_prediction.shape[0]):
            left, right, top, down = left_res[bs], right_res[bs], top_res[bs], down_res[bs]
            cropped_image = data["input_origin"][bs, left:right, top:down, :]
            cropped_image = cv2.resize(cropped_image, (self.stage_2_input_size, self.stage_2_input_size))
            stage_2_inputs.append(torch.from_numpy(cropped_image))
            if config.get("model", "stage_2_activation") == "softmax":
                cropped_label = data["label"][bs, left:right, top:down].astype(np.uint8)
                cropped_label = cv2.resize(cropped_label, (self.stage_2_input_size, self.stage_2_input_size))
            else:
                cropped_label = data["sigmoid_label"][bs, :, left:right, top:down].astype(np.uint8)
                cropped_label = np.transpose(cropped_label, [1, 2, 0])
                cropped_label = cv2.resize(cropped_label, (self.stage_2_input_size, self.stage_2_input_size))
                cropped_label = np.transpose(cropped_label, [2, 0, 1])
            stage_2_labels.append(torch.from_numpy(cropped_label))

        stage_2_inputs = torch.stack(stage_2_inputs, dim=0).permute(0, 3, 1, 2)  # batch * channel * width * height
        stage_2_labels = torch.stack(stage_2_labels, dim=0)
        stage_2_labels = Variable(stage_2_labels).cuda()
        stage_2_inputs = Variable(stage_2_inputs).cuda()


        #stage_2_prediction, and loss
        ret_2 = self.stage_2_model(data = {"input": stage_2_inputs,
                                           "label": stage_2_labels,
                                           #"weight_map": data["weight_map"]
                                         },
                                 config = config,
                                 gpu_list = gpu_list,
                                 acc_result = acc_result[1],
                                 mode = mode
                                )

        if mode in ["train", "valid"]:
            stage_2_logits, stage_2_prediction, stage_2_loss, stage_2_acc_result = ret_2["logits"], ret_2["prediction"], \
                                                                                   ret_2["loss"], ret_2["acc_result"]
        else:
            stage_2_logits, stage_2_prediction = ret_2["logits"], ret_2["prediction"]

        stage_2_prediction = stage_2_prediction.detach().cpu().numpy()
        stage_2_logits = nn.Sigmoid()(stage_2_logits)
        stage_2_logits = stage_2_logits.detach().cpu().numpy()

        if mode in ["train", "valid"]:
            acc_result = [stage_1_acc_result, stage_2_acc_result]

        if mode in ["train", "valid"]:
            loss = [stage_1_loss, stage_2_loss]
            return {"loss": loss, "acc_result": acc_result, "prediction": [stage_1_prediction, stage_2_prediction]}

        #test mode return prediction and location
        location = []
        for idx in range(len(left_res)):
            location.append([left_res[idx], right_res[idx], top_res[idx], down_res[idx]])
        return {"prediction": [stage_1_prediction, stage_2_prediction], "location": location, "logits": stage_2_logits}







