from .cv.Unet import UNet
from .cv.pipeline import pipeline_model


model_list = {
    "Unet": UNet,
    "Pipeline": pipeline_model
}

def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
